import numpy as np

# A collection of downsampling methods for the signal data
# downsample_factor controls the times the data is reduced
def downsample(raw_slice, downsample_factor=1, method="decimation"):
    total = len(raw_slice)

    # If no downsampling selected, return raw data
    if downsample_factor <= 1 or method in (None, "none"):
        return raw_slice

    if method == "max points":
        # Max points: split into buckets and find max values for each
        bucket_size = downsample_factor
        n_buckets = total // bucket_size

        trimmed = raw_slice[:n_buckets * bucket_size]
        reshaped = trimmed.reshape(n_buckets, bucket_size, 2)

        downsampled = reshaped.max(axis=1)
    elif method == "decimation":
        # Another option for downsampling
        # Decimation: just take every Nth point. May lose important spikes
        downsampled = raw_slice[::downsample_factor]
    else:
        downsampled = raw_slice

    return downsampled
