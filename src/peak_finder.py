"""
Functions used to process a signal data set and find peaks in it.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, butter, sosfiltfilt, decimate
from dataclasses import dataclass
from collections.abc import Callable
from typing import TypeVar

from downsampler import downsample
from signal_constants import DEFAULT_SAMPLE_RATE
from signal_classes import AnalysisResult, Slice


# -------------------------------------------------- Analysis classes


@dataclass(frozen=True)
class AnalysisParameter:
    """A parameter of a signal analysis / peak finding function that's
    editable in the GUI.
    
    Defines the name and default value of the variable, and the label
    that's shown to the user.
    """
    name: str
    default: object
    label: str | None = None


@dataclass(frozen=True)
class AnalysisFunction:
    """A description of a signal analysis function including its unique
    string key, display title and parameters as `AnalysisParameter`s."""
    key: str
    title: str
    function: Callable[..., None]
    parameters: tuple[AnalysisParameter, ...] = ()

    @property
    def defaults(self) -> dict[str, object]:
        return {parameter.name: parameter.default for parameter in self.parameters}

    def parameters_for_call(self, values: dict[str, object]) -> dict[str, object]:
        """Return configured values, leaving unset optional parameters to the analysis function."""
        defaults = self.defaults
        return {
            name: value
            for name, value in values.items()
            if value is not None or defaults.get(name) is not None
        }


ANALYSIS_FUNCTIONS: dict[str, AnalysisFunction] = {}
"""Dictionary of all registered signal analysis functions.
The key is title of the `AnalysisFunction`.
"""

AnalysisFunctionType = TypeVar("AnalysisFunctionType", bound=Callable[..., None])
def analysis_function(
    title: str,
    parameters: tuple[AnalysisParameter, ...] = (),
) -> Callable[[AnalysisFunctionType], AnalysisFunctionType]:
    """Decorator for registering an analysis function by creating an `AnalysisFunction`
    instance for it and adding it to the `ANALYSIS_FUNCTIONS` dictionary.
    """
    def register(function: AnalysisFunctionType) -> AnalysisFunctionType:
        key = function.__name__
        ANALYSIS_FUNCTIONS[title] = AnalysisFunction(key, title, function, parameters)
        return function

    return register



# -------------------------------------------------- Filtering


def butterworth_lowpass_filter(
    x: np.ndarray,
    cutoff: float | None = None,
    fs: float = DEFAULT_SAMPLE_RATE,
    decimation_factor: int = 1,
    nyquist_cutoff_multiplier: float = 0.8,
    order: int = 4,
) -> np.ndarray:
    """
    Low-pass filter that keeps the cutoff below the the output Nyquist
    frequency based on `decimation_factor` and `nyquist_cutoff_multiplier`.
    """
    output_nyquist = fs / (2 * decimation_factor)
    max_cutoff = nyquist_cutoff_multiplier * output_nyquist
    cutoff = max_cutoff if cutoff is None else min(cutoff, max_cutoff)

    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    
    # Default signal padding from sosfiltfilt
    default_padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                                 (sos[:, 5] == 0).sum()))
    return sosfiltfilt(sos, x, padlen=min(default_padlen, x.size - 1))



# -------------------------------------------------- Peak finding


# --------- scipy.signal.find_peaks ---------

FIND_PEAKS_PARAMETERS = (
    AnalysisParameter("height", None),
    AnalysisParameter("threshold", None),
    AnalysisParameter("distance", None),
    AnalysisParameter("prominence", None),
    AnalysisParameter("width", None),
    AnalysisParameter("wlen", None),
    AnalysisParameter("rel_height", 0.5),
    AnalysisParameter("plateau_size", None),
)
"""All adjustable parameters of `scipy.signal.find_peaks` for GUI editing."""


def _find_peaks_parameters(**modified_defaults: object) -> tuple[AnalysisParameter, ...]:
    """Return the adjustable `scipy.signal.find_peaks` parameters with analysis-specific defaults."""
    return tuple(
        AnalysisParameter(
            parameter.name,
            modified_defaults.get(parameter.name, parameter.default),
            parameter.label,
        )
        for parameter in FIND_PEAKS_PARAMETERS
    )


def scipy_find_peaks(out: AnalysisResult,
                     time: np.ndarray,
                     x: np.ndarray,
                     channel: int,
                     peak_options: dict):
    """Convenience function for calling `scipy.signal.find_peaks`
    and adding the result to the AnalysisResult.
    """
    idx, properties = find_peaks(x, **peak_options)
    out.peaks(time[idx], x[idx], properties, f"Channel {channel + 1} peaks")



# -------------------------------------------------- Analysis functions


@analysis_function("Example analysis")
def example_analysis(s: Slice, out: AnalysisResult):
    """Example of a signal analysis function. Returns the unprocessed signal
    and peaks at points 10, 50 and 100.

    **Line:**<br>
    Unprocessed.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    # Loop through channels (columns) of the signal
    # Filter and detect peaks only for requested channel(s) to optimize performance
    for ch in s.channel_indices:
        y = s.data[:, ch]
        
        # Add line where
        # - the x value of each point is the time
        # - the y value of each point is the amplitude of the signal in this channel
        out.line(s.time, y,
                 f"Channel {ch + 1}", color=ch)
        
        # Example of the indices of found peaks
        idx = (10, 50, 100)
        # Add peaks, which is just taking the values of the line definition above at the peak indices
        out.peaks(s.time[idx], y[idx],
                  f"Channel {ch + 1} peaks")


@analysis_function(
    "Raw signal with peaks",
    (
        AnalysisParameter("used_downsample_factor", 1),
        *_find_peaks_parameters(
            prominence=40),
    ),
)
def raw_signal_with_peaks(s: Slice, out: AnalysisResult,
                          used_downsample_factor: int = 1,
                          **peak_options):
    """Basic peak detection without pre-processing. Specify the factor of any
    downsampling that has already been done.

    **Line:**<br>
    Unprocessed.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    # TODO improve: these dynamic default values are troublesome because they don't show up in the parameter dialog
    # This value is used if the corresponding parameter is set to "None" in the parameter dialog, and this value
    # isn't communicated to the user in any way.
    peak_options.setdefault("distance", max(1, int(0.005 * s.sample_rate) / used_downsample_factor))
    peak_options.setdefault("prominence", 40)

    # Filter and detect peaks only for requested channel(s) to optimize performance
    for ch in s.channel_indices:
        x = s.data[:, ch]
        
        # Create line
        out.line(s.time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        scipy_find_peaks(out, s.time, x, ch, peak_options)


@analysis_function(
    "SciPy decimation",
    (
        AnalysisParameter("decimation_factor", 10),
        *_find_peaks_parameters(
            prominence=40),
    ),
)
def scipy_decimation(s: Slice, out: AnalysisResult,
                     decimation_factor: int = 10,
                     **peak_options):
    """SciPy's decimation and peak finding. Don't downsample the signal
    before passing to this function.

    **Line:**<br>
    Decimated by a factor of 10 with `scipy.signal.decimate`.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    peak_options.setdefault("distance", max(1, int(0.005 * s.sample_rate) / decimation_factor))
    peak_options.setdefault("prominence", 40)
    
    # Filter and detect peaks only for requested channel(s) to optimize performance
    for ch in s.channel_indices:
        x = s.data[:, ch]
        
        time = s.time[::decimation_factor]
        x = decimate(x, decimation_factor)
        # x = downsample(x, downsample_factor)
        
        # Create line
        out.line(time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        scipy_find_peaks(out, time, x, ch, peak_options)


@analysis_function(
    "Custom decimation",
    (
        AnalysisParameter("decimation_factor", 10),
        *_find_peaks_parameters(
            prominence=40),
    ),
)
def custom_decimation(s: Slice, out: AnalysisResult,
                      decimation_factor: int = 10,
                      **peak_options):
    """Low-pass filter and downsample, then SciPy peak finding.
    Don't downsample the signal before passing to this function.

    **Line:**<br>
    Downsampled by a factor of 10 using `butterworth_lowpass_filter` and
    `downsampler.downsample`.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    peak_options.setdefault("distance", max(1, int(0.005 * s.sample_rate) / decimation_factor))
    peak_options.setdefault("prominence", 40)
    
    # Filter and detect peaks only for requested channel(s) to optimize performance
    for ch in s.channel_indices:
        x = s.data[:, ch]
        
        x = butterworth_lowpass_filter(
            x,
            decimation_factor=decimation_factor,
            cutoff=None,
            fs=s.sample_rate,
        )
        
        time = s.time[::decimation_factor]
        x = downsample(x, decimation_factor)
        
        out.line(time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        scipy_find_peaks(out, time, x, ch, peak_options)


@analysis_function(
    "Extra low-pass",
    (
        AnalysisParameter("decimation_factor", 10),
        AnalysisParameter("lowpass_cutoff", 250),
        *_find_peaks_parameters(
            prominence=40),
    ),
)
def extra_lowpass(s: Slice, out: AnalysisResult,
                  decimation_factor: int = 10,
                  lowpass_cutoff: float | None = 250,
                  **peak_options):
    """`custom_decimation` with an added low-pass filter after downsampling.
    Don't downsample the signal before passing to this function.

    **Line:**<br>
    Downsampled by a factor of 10 using `butterworth_lowpass_filter` and
    `downsampler.downsample`. Then low-pass filter at 250 Hz.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    peak_options.setdefault("distance", max(1, int(0.005 * s.sample_rate) / decimation_factor))
    peak_options.setdefault("prominence", 40)
    
    # Filter and detect peaks only for requested channel(s) to optimize performance
    for ch in s.channel_indices:
        x = s.data[:, ch]
        
        x = butterworth_lowpass_filter(
            x,
            decimation_factor=decimation_factor,
            fs=s.sample_rate,
        )
        
        time = s.time[::decimation_factor]
        x = downsample(x, decimation_factor)
        
        # More low-pass for removing noise
        x = butterworth_lowpass_filter(
            x,
            cutoff=lowpass_cutoff,
            fs=s.sample_rate / decimation_factor
        )
        
        out.line(time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        scipy_find_peaks(out, time, x, ch, peak_options)



# -------------------------------------------------- Peak finder


class peak_finder:
    def __init__(self, y, t=None):
        self.y = np.asarray(y)
        # If no time array is given, use sample indices as the time axis
        self.t = np.arange(len(self.y)) if t is None else np.asarray(t)

    def find(self, **kwargs):
        """
        Find peaks using scipy.signal.find_peaks.
        kwargs are passed straight through (height, distance, prominence, width, ...).
        Returns the time values of the peaks.
        """
        self.peaks, self.properties = find_peaks(self.y, **kwargs)
        return self.t[self.peaks]

    def time_frames(self, rel_height=0.5):
        """
        Return the time window (start, end) of each peak, measured at
        rel_height of the peak's prominence (0.5 = half prominence).
        Call find() first.
        """
        _, _, left, right = peak_widths(self.y, self.peaks, rel_height=rel_height)
        # left/right are fractional indices, so interpolate onto the time axis
        idx = np.arange(len(self.t))
        starts = np.interp(left, idx, self.t)
        ends = np.interp(right, idx, self.t)
        return np.column_stack((starts, ends))