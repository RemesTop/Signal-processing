from dataclasses import dataclass, field
from collections.abc import Iterator

import numpy as np
from PySide6.QtGui import QColor


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


Color = int | tuple | str | QColor
"""pyqtgraph integer, tuple or string color"""


# Currently Line and PeakSet are redundantly the same, but maybe
# they can have some additional, distinct attributes later
@dataclass(frozen=True)
class Line:
    """A graphed line with x and y points, a label, and a color."""
    x: np.ndarray
    y: np.ndarray
    label: str
    color: Color


@dataclass(frozen=True)
class PeakSet:
    """A graphed set of points with x and y locations, a label, and a color."""
    x: np.ndarray
    y: np.ndarray
    label: str
    color: Color


@dataclass
class AnalysisResult:
    """Correlates a signal slice with multiple graphed lines and peak sets."""
    slice: Slice
    lines: list[Line]           = field(default_factory=list)
    peak_sets: list[PeakSet]    = field(default_factory=list)

    def line(self, x: np.ndarray, y: np.ndarray, label: str, color: Color = 0) -> None:
        """Adds a line to this analysis result. Using the same label for multiple
        slices allows the lines to connect when showing a graph.

        Args:
            x: One-dimensional array of x-coordinates.
            y: One-dimensional array of y-coordinates.
            label: Text label for the line.
            color: Line color.
        """
        self.lines.append(Line(x, y, label, color))

    # It might be more convenient if this didn't need the y parameter and just took the y values for graphing
    # from the line that's defined to correspond to these peaks. 
    def peaks(self, x: np.ndarray, y: np.ndarray, label: str = "peaks", color: Color = "#bb0000") -> None:
        """Adds a set of peaks to this analysis result. Using the same label for multiple
        slices allows the lines to connect when showing a graph.

        Args:
            x: One-dimensional array of peak x-coordinates.
            y: One-dimensional array of peak y-coordinates.
            label: Text label for the peak set.
            color: Point color.
        """
        self.peak_sets.append(PeakSet(x, y, label, color))

    @staticmethod
    def line_from_slice(slice: Slice):
        result = AnalysisResult(slice)
        for channel in range(slice.channels):
            result.line(f"Channel {channel + 1}", slice.time, slice.data[:, channel], color=channel)
        return result