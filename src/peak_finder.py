"""
Intented to be collection of functions to find peaks in a given data set.
Or just a single peak finding algorithm, wil have to see how it goes.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, butter, sosfiltfilt

from downsampler import downsample
from signal_constants import DEFAULT_SAMPLE_RATE



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
    
    default_padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                                 (sos[:, 5] == 0).sum()))
    return sosfiltfilt(sos, x, padlen=min(default_padlen, x.size - 1))



# -------------------------------------------------- Finder functions


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