"""
Detection and reporting of "Suctions" based on the time gap between peaks.

Peaks are sorted by time, and a new suction starts whenever the time since the previous
peak is more than `max_gap` seconds. The signal itself is not used.
"""
from dataclasses import dataclass, field

import numpy as np

from signal_classes import AnalysisResult, PeakSet

DEFAULT_MAX_GAP = 10.0
DEFAULT_TOP_N = 3


#  Suction data class
# All "top" mean etc are based on DEFAULT_TOP_N highest peaks, not all peaks in the suction.


@dataclass(eq=False)
class Suction:
    """A single suction: a run of peaks where each is within `max_gap` of the previous."""
    number: int
    """1-based running number of the suction."""
    indices: np.ndarray
    """Indices of the member peaks in the original peak arrays."""
    times: np.ndarray
    """Peak times in seconds, ascending."""
    values: np.ndarray
    """Peak amplitudes (heights), in the same order as `times`."""
    properties: dict[str, np.ndarray] = field(default_factory=dict)
    """Per-peak properties (e.g. from `find_peaks`) restricted to the member peaks."""

    @property
    def n_peaks(self) -> int:
        return len(self.times)

    @property
    def start(self) -> float:
        """Time of the first peak."""
        return float(self.times[0])

    @property
    def end(self) -> float:
        """Time of the last peak."""
        return float(self.times[-1])

    @property
    def duration(self) -> float:
        """Seconds from the first to the last peak (0 for one peak)."""
        return self.end - self.start

    @property
    def largest_peak_index(self) -> int:
        """Position (within this suction) of the highest peak."""
        return int(np.argmax(self.values))

    @property
    def largest_peak_value(self) -> float:
        return float(self.values[self.largest_peak_index])

    @property
    def largest_peak_time(self) -> float:
        return float(self.times[self.largest_peak_index])

    def top_peak_indices(self, n: int = DEFAULT_TOP_N) -> np.ndarray:
        """Positions (within this suction) of the `n` highest peaks, highest first."""
        return np.argsort(-self.values, kind="stable")[:n]

    def top_mean_height(self, n: int = DEFAULT_TOP_N) -> float:
        """Average height of the `n` highest peaks."""
        return float(np.mean(self.values[self.top_peak_indices(n)]))

    def top_mean_prominence(self, n: int = DEFAULT_TOP_N) -> float:
        """Average prominence of the `n` highest peaks, NaN if prominences weren't computed."""
        prominences = self.properties.get("prominences")
        if prominences is None:
            return float("nan")
        return float(np.mean(prominences[self.top_peak_indices(n)]))

    @property
    def height_variance(self) -> float:
        """Population variance of the peak heights (0 for a single peak)."""
        return float(np.var(self.values))

    def to_dict(self) -> dict:
        """Basic information about this suction as plain Python types."""
        return {
            "number": self.number,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "n_peaks": self.n_peaks,
            "largest_peak_value": self.largest_peak_value,
            "largest_peak_time": self.largest_peak_time,
        }

def get_suction_info(self, number: int | None = None) -> list[dict] | dict:
        """Structured information about the suctions.

        Each entry is a dict with the keys: `number`, `start`, `end`, `duration`,
        `n_peaks`, `largest_peak_value`, `largest_peak_time`.

        Args:
            number: If given, return only the dict of the suction with this 1-based
                number (raises `KeyError` if it doesn't exist). Otherwise return a
                list of dicts, one per suction, in time order.
        """
        if number is None:
            return [s.to_dict() for s in self.suctions]
        if not 1 <= number <= self.count:
            raise KeyError(f"No suction number {number} (detected: {self.count})")
        return self.suctions[number - 1].to_dict()

def peak_labels(self) -> np.ndarray:
        """Suction number (1-based) of every peak, in the original peak order.

        Handy for graphing, e.g. `plt.scatter(times, values, c=detector.peak_labels())`.
        """
        labels = np.zeros(self.n_peaks_total, dtype=int)
        for s in self.suctions:
            labels[s.indices] = s.number
        return labels

def get_peak_info(self, peak_index: int | None = None) -> list[dict] | dict:
        """For each peak, which suction it belongs to, plus some information about that suction.

        Each entry is a dict with the keys: `peak_index` (index in the original peak
        arrays), `time`, `value`, `suction_number`, `position_in_suction` (1-based, by
        time), `is_largest_in_suction`, `suction_start`, `suction_end`, `suction_n_peaks`.

        Args:
            peak_index: If given, return only the dict of this peak (index in the
                original peak arrays; raises `IndexError` if out of range). Otherwise
                return a list of dicts, one per peak, in the original peak order.
        """
        rows: list[dict | None] = [None] * self.n_peaks_total
        for s in self.suctions:
            largest = s.largest_peak_index
            for pos, idx in enumerate(s.indices):
                rows[idx] = {
                    "peak_index": int(idx),
                    "time": float(s.times[pos]),
                    "value": float(s.values[pos]),
                    "suction_number": s.number,
                    "position_in_suction": pos + 1,
                    "is_largest_in_suction": pos == largest,
                    "suction_start": s.start,
                    "suction_end": s.end,
                    "suction_n_peaks": s.n_peaks,
                }
        if peak_index is None:
            return rows
        if not 0 <= peak_index < self.n_peaks_total:
            raise IndexError(f"Peak index {peak_index} out of range (0..{self.n_peaks_total - 1})")
        return rows[peak_index]

#  Suction detector
#  Groups peaks into suctions based on the time gap between them, and provides reporting.
#  


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else float("nan")


class SuctionDetector:
    """Groups peaks into suctions purely by the time between them.

    Args:
        max_gap: Peaks no more than this many seconds apart belong to the same suction.
            A suction can be longer than `max_gap` overall because peaks chain together.
    """

    def __init__(self, max_gap: float = DEFAULT_MAX_GAP):
        if max_gap <= 0:
            raise ValueError("max_gap must be positive")
        self.max_gap = float(max_gap)
        self.suctions: list[Suction] = []
        self.n_peaks_total = 0

    @classmethod
    def from_peak_set(cls, peak_set: PeakSet, **kwargs) -> "SuctionDetector":
        """Detect suctions directly from a `PeakSet` of an `AnalysisResult`."""
        return cls(**kwargs).detect(peak_set.x, peak_set.y, peak_set.properties)

    def detect(self, times, values, properties=None) -> "SuctionDetector":
        """Group peaks into suctions. Returns `self` so calls can be chained.

        Args:
            times: Peak times in seconds (need not be sorted, must be finite).
            values: Peak heights, same shape as `times`.
            properties: Optional per-peak property arrays (e.g. prominences).
        """
        times = np.asarray(times, dtype=float)
        values = np.asarray(values)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if values.shape != times.shape:
            raise ValueError("times and values must have the same shape")
        if not np.isfinite(times).all():
            raise ValueError("times must be finite")
        properties = properties or {}

        self.n_peaks_total = times.size
        self.suctions = []
        if times.size == 0:
            return self

        # New group wherever the gap to the previous peak is larger than max_gap
        order = np.argsort(times, kind="stable")
        boundaries = np.flatnonzero(np.diff(times[order]) > self.max_gap) + 1
        for number, group in enumerate(np.split(order, boundaries), start=1):
            group_props = {
                name: np.asarray(arr)[group]
                for name, arr in properties.items()
                if np.ndim(arr) > 0 and len(arr) == len(times)
            }
            self.suctions.append(Suction(number, group, times[group], values[group], group_props))
        return self

    #  Results

    @property
    def count(self) -> int:
        """Number of detected suctions."""
        return len(self.suctions)

    @property
    def mean_height_variance(self) -> float:
        """Average of the per-suction peak height variances."""
        return float(np.mean([s.height_variance for s in self.suctions])) if self.suctions else float("nan")

    def get_suction_info(self, number: int | None = None) -> list[dict] | dict:
        """Structured information about the suctions.

        Each entry is a dict with the keys: `number`, `start`, `end`, `duration`,
        `n_peaks`, `largest_peak_value`, `largest_peak_time`.

        Args:
            number: If given, return only the dict of the suction with this 1-based
                number (raises `KeyError` if it doesn't exist). Otherwise return a
                list of dicts, one per suction, in time order.
        """
        if number is None:
            return [s.to_dict() for s in self.suctions]
        if not 1 <= number <= self.count:
            raise KeyError(f"No suction number {number} (detected: {self.count})")
        return self.suctions[number - 1].to_dict()

    def add_to_result(self, out: AnalysisResult, channel: int, label_prefix: str = "Suction") -> None:
        """Add each suction's peaks to an `AnalysisResult` as its own `PeakSet`."""
        for s in self.suctions:
            out.peaks(s.times, s.values, channel, f"{label_prefix} {s.number}", s.properties)

    # ---- Reporting
    # There were mainly used for debugging, developement and seen from terminal
    # The could be used for something, idk.

    def report(self, title: str = "", top_n: int = DEFAULT_TOP_N) -> str:
        """Text report with a per-suction table followed by overall averages."""
        bar = "=" * 78
        lines = [bar, f"SUCTION REPORT{' - ' + title if title else ''}", bar]
        lines.append(f"Peaks: {self.n_peaks_total}   A new suction starts when the time since the "
                     f"previous peak is > {self.max_gap:g} s")
        lines.append(f"Suctions detected: {self.count}")

        if not self.suctions:
            lines.append(bar)
            return "\n".join(lines)

        lines.append("")
        lines.append(f"{'#':>3} {'peaks':>5} {'start (s)':>10} {'end (s)':>10} {'length (s)':>11} "
                     f"{f'top{top_n} height':>12} {f'top{top_n} prom.':>12} {'height var.':>13}")
        for s in self.suctions:
            lines.append(
                f"{s.number:>3} {s.n_peaks:>5} {s.start:>10.3f} {s.end:>10.3f} {s.duration:>11.3f} "
                f"{s.top_mean_height(top_n):>12.2f} {s.top_mean_prominence(top_n):>12.2f} "
                f"{s.height_variance:>13.2f}"
            )

        lines.append("")
        lines.append(f"Mean top-{top_n} peak height:       "
                     f"{_nanmean([s.top_mean_height(top_n) for s in self.suctions]):.2f}")
        lines.append(f"Mean top-{top_n} peak prominence:   "
                     f"{_nanmean([s.top_mean_prominence(top_n) for s in self.suctions]):.2f}")
        lines.append(f"Mean peak height variance:    {self.mean_height_variance:.2f}")
        lines.append(f"Mean suction length:          {np.mean([s.duration for s in self.suctions]):.3f} s")
        lines.append(bar)
        return "\n".join(lines)


def print_suction_report(result: AnalysisResult,
                         max_gap: float = DEFAULT_MAX_GAP,
                         top_n: int = DEFAULT_TOP_N) -> None:
    """Print a suction report for every peak set in an `AnalysisResult`
    (one report per channel / analysis label)."""
    if not result.peak_sets:
        print("Suction report: no peak sets in the analysis result.", flush=True)
        return
    for peak_set in result.peak_sets:
        detector = SuctionDetector.from_peak_set(peak_set, max_gap=max_gap)
        print(detector.report(f"Channel {peak_set.channel + 1} [{peak_set.label}]", top_n), flush=True)