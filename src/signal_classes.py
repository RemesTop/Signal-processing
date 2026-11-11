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
    channel_indices: tuple[int, ...] | None = None
    # Indices of channels to analyze/plot (allows isolating a single channel for performance)

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
        # Default to all channels when not explicitly restricted
        if self.channel_indices is None:
            object.__setattr__(self, "channel_indices", tuple(range(data.shape[1])))

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


@dataclass(frozen=True)
class Line:
    """A signal line with x and y points, a label, and a color."""
    x: np.ndarray
    y: np.ndarray
    label: str
    color: Color


@dataclass(frozen=True)
class PeakSet:
    """A set of signal peaks with x and y locations, properties, a label, and a color."""
    x: np.ndarray
    y: np.ndarray
    label: str
    color: Color
    properties: dict[str, np.ndarray] = field(default_factory=dict)


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
    def peaks(self, x: np.ndarray, y: np.ndarray, properties: dict[str, np.ndarray] | None = None, label: str = "peaks", color: Color = "#bb0000") -> None:
        """Adds a set of peaks to this analysis result. Using the same label for multiple
        slices allows the lines to connect when showing a graph.

        Args:
            x: One-dimensional array of peak x-coordinates.
            y: One-dimensional array of peak y-coordinates.
            properties: Dictionary containing arrays of properties for the peaks.
            label: Text label for the peak set.
            color: Point color.
        """
        self.peak_sets.append(PeakSet(x, y, label, color, properties or {}))

    @staticmethod
    def line_from_slice(slice: Slice):
        """Convenience method for creating an AnalysisResult from a slice without doing anything
        to the slice or finding any peaks. The AnalysisResult contains the slice as lines for each
        channel.
        """
        result = AnalysisResult(slice)
        for channel in slice.channel_indices:
            result.line(slice.time, slice.data[:, channel], label=f"Channel {channel + 1}", color=channel)
        return result