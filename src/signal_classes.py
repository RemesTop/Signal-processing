from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np


@dataclass(frozen=True)
class Slice:
    time: np.ndarray
    """In seconds, shape: (n,)"""
    data: np.ndarray
    """Shape: (n, channels)"""

    def __post_init__(self):
        time = np.asarray(self.time)
        data = np.asarray(self.data)
        if time.ndim != 1:
            raise ValueError("time must be a one-dimensional array")
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional array")
        if len(time) != len(data):
            raise ValueError("time and data must contain the same number of samples")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "data", data)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate the slice as (time, data)."""
        yield self.time
        yield self.data

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def sample_rate(self) -> float:
        if len(self.time) < 2 or self.time[-1] <= self.time[0]:
            return 1.0
        return (len(self.time) - 1) / (self.time[-1] - self.time[0])
