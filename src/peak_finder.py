"""
Intented to be collection of functions to find peaks in a given data set.
Or just a single peak finding algorithm, wil have to see how it goes.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, butter, sosfiltfilt, decimate

from downsampler import downsample
from signal_constants import DEFAULT_SAMPLE_RATE
from signal_classes import AnalysisResult, Slice


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



# -------------------------------------------------- Analysis functions


def example_analysis(s: Slice, out: AnalysisResult):
    """Example of a signal analysis function. Returns the unprocessed signal
    and peaks at points 10, 50 and 100.

    **Line:**<br>
    Unprocessed.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    # Loop through channels (columns) of the signal
    for ch in range(s.channels):
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


def raw_signal_with_peaks(s: Slice, out: AnalysisResult, downsample_factor: int = 1):
    """Basic peak detection without pre-processing. Specify the factor of any
    downsampling that has already been done.

    **Line:**<br>
    Unprocessed.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    min_distance = max(1, int(0.005 * s.sample_rate) / downsample_factor) # Minimum distance between separate peaks

    for ch in range(s.channels):
        x = s.data[:, ch]
        
        # Create line
        out.line(s.time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        idx, properties = find_peaks(x, distance=min_distance, prominence=40)
        out.peaks(s.time[idx], x[idx],
                  f"Channel {ch + 1} peaks")


def scipy_decimation(s: Slice, out: AnalysisResult):
    """SciPy's decimation and peak finding. Don't downsample the signal
    before passing to this function.

    **Line:**<br>
    Decimated by a factor of 10 with `scipy.signal.decimate`.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    decimation_factor = 10
    min_distance = max(1, int((0.005 * s.sample_rate) / decimation_factor)) # Minimum distance between separate peaks
    
    for ch in range(s.channels):
        x = s.data[:, ch]
        
        time = s.time[::decimation_factor]
        x = decimate(x, decimation_factor)
        # x = downsample(x, downsample_factor)
        
        # Create line
        out.line(time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        idx, properties = find_peaks(x, distance=min_distance, prominence=40)
        out.peaks(time[idx], x[idx],
                  f"Channel {ch + 1} peaks")


def custom_decimation(s: Slice, out: AnalysisResult):
    """Low-pass filter and downsample, then SciPy peak finding.
    Don't downsample the signal before passing to this function.

    **Line:**<br>
    Downsampled by a factor of 10 using `butterworth_lowpass_filter` and
    `downsampler.downsample`.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    decimation_factor = 10
    min_distance = max(1, int((0.005 * s.sample_rate) / decimation_factor)) # Minimum distance between separate peaks
    
    for ch in range(s.channels):
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
        idx, properties = find_peaks(x, distance=min_distance, prominence=40)
        out.peaks(time[idx], x[idx],
                  f"Channel {ch + 1} peaks")


def extra_lowpass(s: Slice, out: AnalysisResult):
    """`custom_decimation` with an added low-pass filter after downsampling.
    Don't downsample the signal before passing to this function.

    **Line:**<br>
    Downsampled by a factor of 10 using `butterworth_lowpass_filter` and
    `downsampler.downsample`. Then low-pass filter at 250 Hz.

    **Peaks:**<br>
    `scipy.signal.find_peaks` with a 5 ms minimum peak distance and 40 prominence.
    """
    decimation_factor = 10
    min_distance = max(1, int((0.005 * s.sample_rate) / decimation_factor)) # Minimum distance between separate peaks
    
    cutoff = 250
    
    for ch in range(s.channels):
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
            cutoff=cutoff,
            fs=s.sample_rate / decimation_factor
        )
        
        out.line(time, x,
                 f"Channel {ch + 1}", color=ch)
        
        # Find peaks
        idx, properties = find_peaks(x, distance=min_distance, prominence=40)
        out.peaks(time[idx], x[idx],
                  f"Channel {ch + 1} peaks")



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