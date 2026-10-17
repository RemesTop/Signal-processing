import threading
import numpy as np
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThreadPool
from signal_converter import SignalConverter
from worker import Worker

import downsampler

# A class to read .dat binary files by slices and downsample them for easier usage
class SignalReader(QObject):
    ready = Signal()
    progress = Signal(float)

    # Does not use multithreading by default for the time being. 
    def __init__(self, file_path, sample_rate=50000, multithreading=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.data = None
        self.total_samples = 0
        file_path = Path(file_path)

        # If given a CSV file, check for cached .dat or convert it
        if file_path.suffix.lower() == ".csv":
            self.dat_path = file_path.with_suffix(".dat")
            if not self.dat_path.exists():
                print(f"Converting {file_path.name} to binary (.dat).. This is done once.")
                self.converter = SignalConverter()
                tmp_path = self.dat_path.with_suffix(".tmp")
                
                thread = threading.Thread(
                    target=self.converter.start, 
                    args=(file_path, tmp_path)
                )
                thread.start()
                thread.join()

                tmp_path.rename(self.dat_path)
        else:
            self.dat_path = file_path

        self._init_memmap()

    def _init_memmap(self):
        # Calculate row count from file size (For 2 channels = 4 bytes per row)
        file_bytes = self.dat_path.stat().st_size
        self.total_samples = file_bytes // 4

        # Read binary file without loading everything into memory
        self.data = np.memmap(
            self.dat_path,
            dtype=np.int16,
            mode="r",
            shape=(self.total_samples, 2)
        )
        self.ready.emit()

    def _downsample(self, raw_slice, downsample_factor, downsample):
        return downsampler.downsample(raw_slice, downsample_factor, method=downsample)

    # Get a slice of the signal, raw by default (downsampling can be optionally enabled)
    # downsample_factor dictates how many times smaller the data will be. Higher value: more downsampling, more inaccurate
    def get_slice(self, start_sec, end_sec, downsample_factor=1, downsample=None):
        # If data is not ready (e.g. conversion still running), return empty arrays
        if self.data is None:
            return np.array([]), np.array([])

        start_idx = max(0, int(start_sec * self.sample_rate))
        end_idx = min(self.total_samples, int(end_sec * self.sample_rate))

        raw_slice = self.data[start_idx:end_idx]
        total = len(raw_slice)

        # If no downsampling requested or factor <= 1, return raw data
        if downsample in (None, "none") or downsample_factor <= 1 or total <= downsample_factor:
            time_axis = np.linspace(start_sec, end_sec, total)
            return time_axis, raw_slice

        downsampled = self._downsample(raw_slice, downsample_factor, downsample)

        # Create time axis based on start/end and amount of points
        # Can be used for reading samples or working with matplotlib
        time_axis = np.linspace(start_sec, end_sec, len(downsampled))
        return time_axis, downsampled
