import numpy as np
from pathlib import Path
from signal_converter import SignalConverter

# A class to read .dat binary files by slices and downsample them for easier usage
class SignalReader:
    def __init__(self, file_path, sample_rate=50000):
        self.sample_rate = sample_rate
        file_path = Path(file_path)

        # If given a CSV file, check for cached .dat or convert it
        if file_path.suffix.lower() == ".csv":
            self.dat_path = file_path.with_suffix(".dat")
            if not self.dat_path.exists():
                print(f"Converting {file_path.name} to binary (.dat).. This is done once.")
                converter = SignalConverter()
                converter.start(file_path, self.dat_path)
        else:
            self.dat_path = file_path

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

    # Get a slice of the signal, downsampled with decimation by default
    # max_points dictates the amount of downsampling. Higher value: more downsampling, more inaccurate
    def get_slice(self, start_sec, end_sec, max_points=5, downsample="decimation"):
        start_idx = max(0, int(start_sec * self.sample_rate))
        end_idx = min(self.total_samples, int(end_sec * self.sample_rate))

        raw_slice = self.data[start_idx:end_idx]
        total = len(raw_slice)

        # If the slice is already small enough, no downsampling needed
        # OR if downsample mode is set to none, return raw data
        if total <= max_points or downsample == "none":
            time_axis = np.linspace(start_sec, end_sec, total)
            return time_axis, raw_slice

        if downsample == "highest_points":
            # Highest points: split into buckets and find max values for each
            n_buckets = max_points
            bucket_size = total // n_buckets

            trimmed = raw_slice[:n_buckets * bucket_size]
            reshaped = trimmed.reshape(n_buckets, bucket_size, 2)

            downsampled = reshaped.max(axis=1)
        elif downsample == "decimation":
            # Another option for downsampling
            # Decimation: just take every Nth point. May lose important spikes
            step = total // max_points
            downsampled = raw_slice[::step]

        # Create time axis based on start/end and amount of points
        # Can be used for reading samples or working with matplotlib
        time_axis = np.linspace(start_sec, end_sec, len(downsampled))
        return time_axis, downsampled
