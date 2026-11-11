import numpy as np
from collections.abc import Iterable
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThreadPool
from signal_converter import SignalConverter
from worker import Worker

import downsampler
from signal_constants import DEFAULT_SAMPLE_RATE
from signal_classes import Slice


def convert_csv_files(file_paths: Iterable[str | Path], progress_callback: Signal | None = None) -> list[Path]:
    """Convert CSV signal files to DAT files and return the converted paths."""
    converted_paths = []
    for file_path in file_paths:
        source_path = Path(file_path)
        if source_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {source_path}")

        dat_path = source_path.with_suffix(".dat")
        if dat_path.exists():
            continue

        print(f"Converting {source_path.name} to binary (.dat).")
        tmp_path = dat_path.with_suffix(".tmp")
        converter = SignalConverter()
        try:
            success = converter.start(
                source_signal_path=source_path,
                target_signal_path=tmp_path,
                progress_callback=progress_callback)
            if not success:
                raise RuntimeError(f"Conversion failed for {source_path}")
            tmp_path.replace(dat_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        converted_paths.append(dat_path)

    return converted_paths


# A class to read .dat binary files by slices and downsample them for easier usage
class SignalReader(QObject):
    ready = Signal()
    progress = Signal(float)

    def __init__(self, file_path, sample_rate=DEFAULT_SAMPLE_RATE, multithreading=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.data = None
        self.total_samples = 0
        file_path = Path(file_path)

        # If given a CSV file, check for cached .dat or convert it
        if file_path.suffix.lower() == ".csv":
            self.dat_path = file_path.with_suffix(".dat")
            if not self.dat_path.exists():
                # If multithreading is enabled, run conversion in the background
                if multithreading:
                    worker = Worker(convert_csv_files, [file_path])
                    worker.signals.progress.connect(self.progress.emit)
                    worker.signals.result.connect(self.on_conversion_finished)
                    QThreadPool.globalInstance().start(worker)
                    return
                else:
                    # Otherwise convert directly (blocks until done)
                    convert_csv_files([file_path])
        else:
            self.dat_path = file_path

        self._init_memmap()

    def on_conversion_finished(self, _converted_paths):
        # Called when background worker finishes
        if self.dat_path in _converted_paths and self.dat_path.exists():
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
    def get_slice(self, start_sec, end_sec, downsample_factor=1, downsample=None) -> Slice:
        # If data is not ready (e.g. conversion still running), return empty arrays
        if self.data is None:
            return Slice(np.array([]), np.empty((0, 2), dtype=np.int16))

        start_idx = max(0, int(start_sec * self.sample_rate))
        end_idx = min(self.total_samples, int(end_sec * self.sample_rate))

        raw_slice = self.data[start_idx:end_idx]
        total = len(raw_slice)

        # If no downsampling requested or factor <= 1, return raw data
        if downsample in (None, "none") or downsample_factor <= 1 or total <= downsample_factor:
            time_axis = np.linspace(start_sec, end_sec, total)
            return Slice(time_axis, raw_slice)

        downsampled = self._downsample(raw_slice, downsample_factor, downsample)

        # Create time axis based on start/end and amount of points
        # Can be used for reading samples or working with matplotlib
        time_axis = np.linspace(start_sec, end_sec, len(downsampled))
        return Slice(time_axis, downsampled)
