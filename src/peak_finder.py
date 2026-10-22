"""
Intented to be collection of functions to find peaks in a given data set.
Or just a single peak finding algorithm, wil have to see how it goes.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths


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