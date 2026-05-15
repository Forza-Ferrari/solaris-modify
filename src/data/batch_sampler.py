import bisect
import json
import logging
import math
import os

import numpy as np
import torch

from .dataset import DatasetMultiplayer, DatasetMultiplayerN
from .segment import SegmentId, SegmentIdMultiplayer, SegmentIdMultiplayerN


def _load_actions(path):
    with open(path, "r") as f:
        return json.load(f)


def _strictly_increasing(times):
    return all(t_next > t_prev for t_prev, t_next in zip(times, times[1:]))


def _get_multiplayer_episode_info(dataset, episode_id, num_frames):
    episode_paths = dataset.get_episode_paths(episode_id)

    if "actions_paths" in episode_paths:
        action_paths = [
            dataset.directory / action_path
            for action_path in episode_paths["actions_paths"]
        ]
    else:
        action_paths = [
            dataset.directory / episode_paths["bot1_actions_path"],
            dataset.directory / episode_paths["bot2_actions_path"],
        ]

    actions_per_bot = [_load_actions(path) for path in action_paths]
    if any(len(actions) < 1 for actions in actions_per_bot):
        logging.warning(
            "Episode %s %s has less than 1 action, resampling",
            episode_id,
            episode_paths,
        )
        return None

    bot_times = [[action["renderTime"] for action in actions] for actions in actions_per_bot]
    if not all(_strictly_increasing(times) for times in bot_times):
        logging.warning(
            "Episode %s %s has non-increasing times, resampling",
            episode_id,
            episode_paths,
        )
        return None

    start_time = max(times[0] for times in bot_times)
    end_time = min(times[-1] for times in bot_times)

    start_indices = [bisect.bisect_left(times, start_time) for times in bot_times]
    end_indices = [bisect.bisect_right(times, end_time) - 1 - num_frames for times in bot_times]

    invalid = any(
        end_idx - start_idx < 1
        or end_idx < 0
        or start_idx >= len(actions)
        for start_idx, end_idx, actions in zip(start_indices, end_indices, actions_per_bot)
    )
    if invalid:
        logging.warning(
            "Episode %s %s doesn't have enough actions. starts=%s ends=%s lens=%s, resampling",
            episode_id,
            episode_paths,
            start_indices,
            end_indices,
            [len(actions) for actions in actions_per_bot],
        )
        return None

    start_time = min(
        actions[start_idx]["renderTime"]
        for actions, start_idx in zip(actions_per_bot, start_indices)
    )
    end_time = min(
        actions[end_idx]["renderTime"]
        for actions, end_idx in zip(actions_per_bot, end_indices)
    )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "bot_times": bot_times,
        "num_players": len(bot_times),
    }


class BatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()

    def __len__(self):
        raise NotImplementedError(
            "BatchSampler does not have a fixed length. Use __iter__ instead."
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def reset_rng(self):
        self.rng = np.random.default_rng(self._seed)

    def sample(self):
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        short_episode_ids = np.where(self.dataset.lengths < self.num_frames)[0]
        episodes_partition = episodes_partition[
            ~np.isin(episodes_partition, short_episode_ids)
        ]

        episode_ids = self.rng.choice(
            episodes_partition, size=self.batch_size, replace=True
        )
        starts = self.rng.integers(
            low=0,
            high=self.dataset.lengths[episode_ids] - self.num_frames + 1,
        )
        stops = starts + self.num_frames

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]


class BatchSamplerMultiplayer(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()
        self._episode_infos = {}

    def __len__(self):
        raise NotImplementedError(
            "BatchSampler does not have a fixed length. Use __iter__ instead."
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def reset_rng(self):
        self.rng = np.random.default_rng(self._seed)

    def get_episode_info(self, episode_id):
        if episode_id not in self._episode_infos:
            self._episode_infos[episode_id] = _get_multiplayer_episode_info(
                self.dataset, episode_id, self.num_frames
            )
        return self._episode_infos[episode_id]

    def sample(self):
        num_episodes = self.dataset.num_episodes
        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)

        segment_ids = []
        while len(segment_ids) < self.batch_size:
            episode_ids = self.rng.choice(
                episodes_partition,
                size=self.batch_size - len(segment_ids),
                replace=True,
            )
            for episode_id in episode_ids:
                episode_info = self.get_episode_info(episode_id)
                if episode_info is None:
                    continue
                start_time = self.rng.uniform(
                    episode_info["start_time"], episode_info["end_time"]
                )
                starts = [
                    bisect.bisect_left(times, start_time)
                    for times in episode_info["bot_times"]
                ]
                stops = [start + self.num_frames for start in starts]

                if len(starts) == 2:
                    segment_ids.append(
                        SegmentIdMultiplayer(
                            episode_id,
                            starts[0],
                            stops[0],
                            starts[1],
                            stops[1],
                        )
                    )
                else:
                    segment_ids.append(
                        SegmentIdMultiplayerN(episode_id, starts=starts, stops=stops)
                    )
                if len(segment_ids) >= self.batch_size:
                    break

        return segment_ids


class EvalBatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        num_global_samples=None,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        logging.info(f"Eval num_frames: {num_frames}")
        self._seed = seed
        self.reset_rng()

        self.ids = self._load_eval_ids()
        assert num_frames <= 1024, "num_frames must be at most 1024"
        num_global_samples = (
            min(num_global_samples, len(self.ids))
            if num_global_samples is not None
            else len(self.ids)
        )
        self.examples = self._build_examples(self.ids[:num_global_samples])
        self.examples = self.examples[self.rank :: self.world_size]
        self.num_batches = math.ceil(len(self.examples) / self.batch_size)

    def _load_eval_ids(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if "worldmem_demo" in str(self.dataset.directory):
            return [
                [0, 0, 1024],
                [1, 0, 1024],
                [2, 0, 1024],
                [3, 0, 1024],
                [4, 0, 1024],
                [5, 0, 1024],
            ]

        custom_eval_ids_path = getattr(self.dataset, "eval_ids_path", None)
        if custom_eval_ids_path and os.path.exists(custom_eval_ids_path):
            with open(custom_eval_ids_path, "r") as f:
                return json.load(f)

        default_eval_ids_path = os.path.join(
            base_dir, "eval_ids", f"eval_ids_{self.dataset.dataset_name}.json"
        )
        if os.path.exists(default_eval_ids_path):
            with open(default_eval_ids_path, "r") as f:
                return json.load(f)

        if isinstance(self.dataset, DatasetMultiplayerN):
            return self._generate_eval_ids_for_multiplayer_n()

        raise FileNotFoundError(
            f"Could not find eval ids for dataset {self.dataset.dataset_name}"
        )

    def _generate_eval_ids_for_multiplayer_n(self):
        ids = []
        for episode_id in range(self.dataset.num_episodes):
            episode_info = _get_multiplayer_episode_info(
                self.dataset, episode_id, self.num_frames
            )
            if episode_info is None:
                continue
            starts = [
                bisect.bisect_left(times, episode_info["start_time"])
                for times in episode_info["bot_times"]
            ]
            row = [episode_id]
            for start in starts:
                row.extend([start, start + self.num_frames])
            ids.append(row)
        return ids

    def _build_examples(self, ids):
        if isinstance(self.dataset, DatasetMultiplayerN):
            examples = []
            for row in ids:
                episode_id = row[0]
                player_ranges = row[1:]
                if len(player_ranges) % 2 != 0:
                    raise ValueError(
                        f"Invalid multiplayer eval id row for {self.dataset.dataset_name}: {row}"
                    )
                starts = player_ranges[0::2]
                examples.append(
                    SegmentIdMultiplayerN(
                        episode_id=episode_id,
                        starts=starts,
                        stops=[start + self.num_frames for start in starts],
                    )
                )
            return examples

        if isinstance(self.dataset, DatasetMultiplayer):
            return [
                SegmentIdMultiplayer(
                    episode_id,
                    bot1_start,
                    bot1_start + self.num_frames,
                    bot2_start,
                    bot2_start + self.num_frames,
                )
                for (episode_id, bot1_start, _, bot2_start, _) in ids
            ]

        return [
            SegmentId(episode_id, start, start + self.num_frames)
            for (episode_id, start, _) in ids
        ]

    def reset_rng(self):
        self.rng = np.random.default_rng(self._seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.examples))
            yield self.examples[start:end]
