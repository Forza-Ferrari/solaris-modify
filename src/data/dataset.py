import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset as TorchDataset

from . import minecraft
from .batch import Batch
from .segment import Segment


class InputConverter:
    def convert(self, act):
        return act


class CameraLinearConverterMatrixGame2(InputConverter):
    def convert(self, act):
        act[:, 23] = minecraft.compress_mouse_linear(act[:, 23])
        act[:, 24] = minecraft.compress_mouse_linear(act[:, 24])
        return act


input_converters = {
    "CameraLinearConverterMatrixGame2": CameraLinearConverterMatrixGame2(),
}


class VideoReadError(Exception):
    pass


def normalize_actions_path(video_path):
    actions_path = video_path.with_suffix(".json")
    if actions_path.name.endswith("_camera.json"):
        actions_path = actions_path.with_name(
            actions_path.name.replace("_camera.json", ".json")
        )
    return actions_path


def read_actions_slice_for_bot(actions_path, start, stop, bot_name):
    with open(actions_path, "r") as f:
        actions = json.load(f)
    return [{**a, "bot": bot_name} for a in actions[start:stop]]


def load_multiplayer_segment(
    *,
    directory,
    bot_names,
    video_paths,
    actions_paths,
    starts,
    stops,
    obs_resize,
    converters,
    shuffle_bots,
    rng,
):
    bot_obs = []
    bot_act = []
    for bot_name, video_name, actions_name, start, stop in zip(
        bot_names, video_paths, actions_paths, starts, stops
    ):
        video_path = directory / video_name
        actions_path = directory / actions_name
        try:
            decord_video = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}") from e

        try:
            actions = read_actions_slice_for_bot(actions_path, start, stop, bot_name)
        except Exception as e:
            raise ValueError(f"Error reading actions {actions_path}: {e}") from e

        try:
            obs = minecraft.read_obs_slice_decord(
                decord_video,
                start,
                stop,
                obs_resize,
            )
        except Exception as e:
            raise VideoReadError(f"Error reading segment slice: {e}") from e

        actions_one_hot = minecraft.convert_act_slice_mineflayer(actions)
        for converter in converters:
            actions_one_hot = converter.convert(actions_one_hot)

        bot_obs.append(obs)
        bot_act.append(actions_one_hot)

    if shuffle_bots:
        indices = list(range(len(bot_names)))
        rng.shuffle(indices)
        bot_obs = [bot_obs[i] for i in indices]
        bot_act = [bot_act[i] for i in indices]

    obs = np.stack(bot_obs, axis=1)
    act = np.stack(bot_act, axis=1)
    return Segment(obs, act)


class Dataset(TorchDataset):

    def __init__(
        self,
        data_dir,
        dataset_name,
        converters,
        obs_resize=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()

        with open(self.directory / "episodes_info.json", "r") as json_file:
            self.episodes_info = json.load(json_file)
        self._num_episodes = self.episodes_info["num_episodes"]
        self._lengths = np.array(
            [ep["length"] for ep in self.episodes_info["episodes"]]
        )
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]

    @property
    def num_episodes(self):
        return self._num_episodes

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def lengths(self):
        return self._lengths

    def __getitem__(self, segment_id):
        episode_info = self.episodes_info["episodes"][segment_id.episode_id]
        video_path = self.directory / episode_info["video_path"]
        actions_path = self.directory / episode_info["actions_path"]
        try:
            decord_video = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}") from e

        try:
            act = self.read_act_slice(actions_path, segment_id.start, segment_id.stop)
        except Exception as e:
            raise ValueError(
                f"Error reading episode actions {segment_id.episode_id}: {e}"
            ) from e

        try:
            obs_decord = minecraft.read_obs_slice_decord(
                decord_video,
                segment_id.start,
                segment_id.stop,
                self._obs_resize,
            )

        except Exception as e:
            raise VideoReadError(f"Error reading segment slice: {e}") from e

        for converter in self._converters:
            act = converter.convert(act)
        segment = Segment(obs_decord, act)

        return segment

    def read_act_slice(
        self,
        path,
        start,
        stop,
    ):
        full_actions_json = read_actions_json(path)
        return minecraft.read_act_slice_vpt(
            full_actions_json,
            start,
            stop,
        )


class DatasetMultiplayer(TorchDataset):
    def __init__(
        self,
        data_dir,
        dataset_name,
        bot1_name,
        bot2_name,
        converters,
        obs_resize=None,
        shuffle_bots=True,
        shuffle_bot_seed=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()
        self.bot1_name = bot1_name
        self.bot2_name = bot2_name
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]
        self._shuffle_bots = shuffle_bots
        self._shuffle_bot_seed = shuffle_bot_seed
        # Create a separate Random instance for shuffling
        self._rng = random.Random(shuffle_bot_seed)
        self._episodes = {}

        mp4_files = sorted(
            [p for p in self.directory.glob(f"*_{self.bot1_name}_*.mp4")],
            key=lambda p: p.name,
        )
        episode_id = 0

        for video_path_bot1 in mp4_files:
            name = video_path_bot1.name

            # Form the expected counterpart filename for bot2 by swapping bot name
            counterpart_name = name.replace(
                f"_{self.bot1_name}_", f"_{self.bot2_name}_", 1
            )
            video_path_bot2 = self.directory / counterpart_name

            # Corresponding action files must exist and share the same stem
            actions_path_bot1 = normalize_actions_path(video_path_bot1)
            actions_path_bot2 = normalize_actions_path(video_path_bot2)

            # Validate existence of all four files
            if (
                video_path_bot2.exists()
                and actions_path_bot1.exists()
                and actions_path_bot2.exists()
            ):
                self._episodes[episode_id] = {
                    "bot1_video_path": video_path_bot1.name,
                    "bot1_actions_path": actions_path_bot1.name,
                    "bot2_video_path": video_path_bot2.name,
                    "bot2_actions_path": actions_path_bot2.name,
                }
                episode_id += 1

        self._num_episodes = len(self._episodes)
        logging.info(
            f"Dataset {self.dataset_name} loaded {self._num_episodes} episodes. Data directory: {self.directory}"
        )

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def num_episodes(self):
        return self._num_episodes

    def get_episode_paths(self, episode_id):
        return self._episodes[episode_id]

    def __getitem__(self, segment_id):
        episode_paths = self.get_episode_paths(segment_id.episode_id)
        return load_multiplayer_segment(
            directory=self.directory,
            bot_names=[self.bot1_name, self.bot2_name],
            video_paths=[
                episode_paths["bot1_video_path"],
                episode_paths["bot2_video_path"],
            ],
            actions_paths=[
                episode_paths["bot1_actions_path"],
                episode_paths["bot2_actions_path"],
            ],
            starts=[segment_id.bot1_start, segment_id.bot2_start],
            stops=[segment_id.bot1_stop, segment_id.bot2_stop],
            obs_resize=self._obs_resize,
            converters=self._converters,
            shuffle_bots=self._shuffle_bots,
            rng=self._rng,
        )


class DatasetMultiplayerN(TorchDataset):
    def __init__(
        self,
        data_dir,
        dataset_name,
        bot_names,
        converters,
        obs_resize=None,
        shuffle_bots=False,
        shuffle_bot_seed=None,
        run_num_players=None,
        eval_ids_path=None,
    ):
        super().__init__()

        bot_names = list(bot_names)
        if run_num_players is not None:
            if run_num_players < 1:
                raise ValueError("run_num_players must be at least 1")
            if run_num_players > len(bot_names):
                raise ValueError(
                    f"run_num_players={run_num_players} exceeds available bot_names={len(bot_names)}"
                )
            bot_names = bot_names[:run_num_players]
        if len(bot_names) < 1:
            raise ValueError("bot_names must contain at least one bot")

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()
        self.bot_names = bot_names
        self.run_num_players = len(bot_names)
        self.eval_ids_path = (
            str(Path(eval_ids_path).expanduser()) if eval_ids_path is not None else None
        )
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]
        self._shuffle_bots = shuffle_bots
        self._rng = random.Random(shuffle_bot_seed)
        self._episodes = {}

        grouped_videos = defaultdict(dict)
        discovered_bots = set()
        for video_path in sorted(self.directory.glob("*.mp4"), key=lambda p: p.name):
            matched_bot = None
            template = None
            for bot_name in self.bot_names:
                token = f"_{bot_name}_"
                if token in video_path.name:
                    matched_bot = bot_name
                    template = video_path.name.replace(token, "_{BOT}_", 1)
                    break
            if matched_bot is None:
                continue
            grouped_videos[template][matched_bot] = video_path
            discovered_bots.add(matched_bot)

        episode_id = 0
        for template in sorted(grouped_videos.keys()):
            bot_to_video = grouped_videos[template]
            if not all(bot_name in bot_to_video for bot_name in self.bot_names):
                continue

            video_paths = []
            actions_paths = []
            valid = True
            for bot_name in self.bot_names:
                bot_video_path = bot_to_video[bot_name]
                bot_actions_path = normalize_actions_path(bot_video_path)
                if not bot_actions_path.exists():
                    valid = False
                    break
                video_paths.append(bot_video_path.name)
                actions_paths.append(bot_actions_path.name)

            if valid:
                self._episodes[episode_id] = {
                    "bot_names": list(self.bot_names),
                    "video_paths": video_paths,
                    "actions_paths": actions_paths,
                }
                episode_id += 1

        self._num_episodes = len(self._episodes)
        logging.info(
            f"Dataset {self.dataset_name} loaded {self._num_episodes} episodes for {self.run_num_players} players. Data directory: {self.directory}"
        )
        if self._num_episodes == 0:
            logging.warning(
                "No complete episodes found for bot_names=%s in %s. Discovered matching bots=%s",
                self.bot_names,
                self.directory,
                sorted(discovered_bots),
            )

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def num_episodes(self):
        return self._num_episodes

    def get_episode_paths(self, episode_id):
        return self._episodes[episode_id]

    def __getitem__(self, segment_id):
        episode_paths = self.get_episode_paths(segment_id.episode_id)
        if len(segment_id.starts) != self.run_num_players:
            raise ValueError(
                f"Segment player count {len(segment_id.starts)} does not match dataset player count {self.run_num_players}"
            )

        return load_multiplayer_segment(
            directory=self.directory,
            bot_names=episode_paths["bot_names"],
            video_paths=episode_paths["video_paths"],
            actions_paths=episode_paths["actions_paths"],
            starts=segment_id.starts,
            stops=segment_id.stops,
            obs_resize=self._obs_resize,
            converters=self._converters,
            shuffle_bots=self._shuffle_bots,
            rng=self._rng,
        )


def read_actions_json(actions_path):
    try:
        with open(actions_path) as json_file:
            json_lines = json_file.readlines()
    except Exception as e_utf:
        try:
            with open(actions_path, encoding="windows-1252") as json_file:
                json_lines = json_file.readlines()
        except Exception as e_win:
            raise ValueError(
                f"Error reading file {actions_path} in utf-8 and windows-1252 encodings: {e_utf} / {e_win}"
            ) from e_win

    json_data = "[" + ",".join(json_lines) + "]"
    json_data = json.loads(json_data)
    return json_data


def collate_segments_to_batch(sequence_length, pad_batch_to, segments):
    # Filter out None segments
    valid_segments = [s for s in segments if s]
    if not valid_segments:
        raise ValueError("No valid segments to collate.")

    # Find max length for obs and act
    # Pad obs and act to max length
    obs_padded = []
    act_padded = []
    for s in valid_segments:
        obs = s.obs
        act = s.act

        # Calculate pad widths to pad with zeros on the right along the first dimension
        # Also collect the real (unpadded) lengths for each segment
        # real_lengths will be used to indicate where padding starts for each segment
        # (i.e., the original length of each segment)
        # This array will be of size B (number of valid segments)
        obs_pad_width = [(0, sequence_length - len(s))] + [(0, 0)] * (obs.ndim - 1)
        act_pad_width = [(0, sequence_length - len(s))] + [(0, 0)] * (act.ndim - 1)
        obs_padded.append(np.pad(obs, obs_pad_width, mode="constant"))
        act_padded.append(np.pad(act, act_pad_width, mode="constant"))

    real_lengths = np.array([len(s) for s in valid_segments], dtype=np.int32)
    obs_batch = np.stack(obs_padded)
    act_batch = np.stack(act_padded)
    if pad_batch_to is not None and pad_batch_to > len(valid_segments):
        obs_batch = np.pad(
            obs_batch,
            [(0, pad_batch_to - len(valid_segments))] + [(0, 0)] * (obs_batch.ndim - 1),
            mode="constant",
        )
        act_batch = np.pad(
            act_batch,
            [(0, pad_batch_to - len(valid_segments))] + [(0, 0)] * (act_batch.ndim - 1),
            mode="constant",
        )
        real_lengths = np.pad(
            real_lengths, (0, pad_batch_to - len(valid_segments)), mode="constant"
        )

    return Batch(obs_batch, act_batch, real_lengths)
