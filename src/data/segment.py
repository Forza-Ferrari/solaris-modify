from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np


@dataclass
class SegmentId:
    episode_id: Union[int, str]
    start: int
    stop: int

    def to_list(self) -> list[int]:
        return [self.episode_id, self.start, self.stop]


@dataclass
class SegmentIdMultiplayer:
    episode_id: Union[int, str]
    bot1_start: int
    bot1_stop: int
    bot2_start: int
    bot2_stop: int

    def to_list(self) -> list[int]:
        return [
            self.episode_id,
            self.bot1_start,
            self.bot1_stop,
            self.bot2_start,
            self.bot2_stop,
        ]


@dataclass
class SegmentIdMultiplayerN:
    episode_id: Union[int, str]
    starts: list[int]
    stops: list[int]

    def to_list(self) -> list[int]:
        items = [self.episode_id]
        for start, stop in zip(self.starts, self.stops):
            items.extend([start, stop])
        return items


@dataclass
class Segment:
    obs: np.ndarray
    act: np.ndarray

    def __len__(self) -> int:
        return self.obs.shape[0]
