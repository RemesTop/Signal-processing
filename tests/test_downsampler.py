import numpy as np
import pytest

import downsampler

"""""
WRITTEN BY ARTICIAL INTELLIGENCE (AI) - CLAUDE SONNET 5.5
Tests for the downsampler. To test the downsampler, run the following command in the terminal:
pytest tests/test_downsampler.py
Last modified: 16.11.2026
"""""
@pytest.fixture
def signal_1d():
    return np.arange(10)  # [0 1 2 ... 9]


@pytest.fixture
def signal_2d():
    # 8 samples, 2 channels
    return np.arange(16).reshape(8, 2)


# ---------- No downsampling ----------

@pytest.mark.parametrize("factor", [1, 0, -3])
def test_factor_le_one_returns_raw(signal_1d, factor):
    result = downsampler.downsample(signal_1d, downsample_factor=factor)
    assert result is signal_1d


@pytest.mark.parametrize("method", [None, "none"])
def test_method_none_returns_raw(signal_1d, method):
    result = downsampler.downsample(signal_1d, downsample_factor=4, method=method)
    assert result is signal_1d


def test_unknown_method_returns_raw(signal_1d):
    result = downsampler.downsample(signal_1d, downsample_factor=4, method="banana")
    assert result is signal_1d


def test_default_arguments_return_raw(signal_1d):
    assert downsampler.downsample(signal_1d) is signal_1d


# ---------- Decimation ----------

def test_decimation_takes_every_nth_point(signal_1d):
    result = downsampler.downsample(signal_1d, 3, "decimation")
    np.testing.assert_array_equal(result, [0, 3, 6, 9])


def test_decimation_is_default_method(signal_1d):
    result = downsampler.downsample(signal_1d, 2)
    np.testing.assert_array_equal(result, [0, 2, 4, 6, 8])


def test_decimation_2d_keeps_channels(signal_2d):
    result = downsampler.downsample(signal_2d, 2, "decimation")
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result, signal_2d[::2])


def test_decimation_factor_larger_than_length(signal_1d):
    result = downsampler.downsample(signal_1d, 100, "decimation")
    np.testing.assert_array_equal(result, [0])


def test_decimation_can_miss_spike():
    data = np.zeros(10)
    data[1] = 99  # spike not on a sampled index
    result = downsampler.downsample(data, 5, "decimation")
    assert result.max() == 0


# ---------- Max points ----------

def test_max_points_takes_bucket_maximum():
    data = np.array([1, 5, 2, 8, 3, 4])
    result = downsampler.downsample(data, 2, "max points")
    np.testing.assert_array_equal(result, [5, 8, 4])


def test_max_points_keeps_spike():
    data = np.zeros(10)
    data[1] = 99
    result = downsampler.downsample(data, 5, "max points")
    assert result.max() == 99


def test_max_points_drops_incomplete_last_bucket():
    data = np.array([1, 2, 3, 4, 5, 6, 7])  # 7 samples, buckets of 3
    result = downsampler.downsample(data, 3, "max points")
    np.testing.assert_array_equal(result, [3, 6])  # the trailing 7 is discarded


def test_max_points_output_length(signal_1d):
    result = downsampler.downsample(signal_1d, 3, "max points")
    assert len(result) == len(signal_1d) // 3


def test_max_points_2d_is_per_channel(signal_2d):
    result = downsampler.downsample(signal_2d, 4, "max points")
    assert result.shape == (2, 2)
    # channel 0: [0,2,4,6] -> 6, [8,10,12,14] -> 14
    # channel 1: [1,3,5,7] -> 7, [9,11,13,15] -> 15
    np.testing.assert_array_equal(result, [[6, 7], [14, 15]])


def test_max_points_negative_values():
    data = np.array([-5, -2, -9, -1])
    result = downsampler.downsample(data, 2, "max points")
    np.testing.assert_array_equal(result, [-2, -1])


def test_max_points_factor_larger_than_length_gives_empty(signal_1d):
    result = downsampler.downsample(signal_1d, 100, "max points")
    assert len(result) == 0


# ---------- General ----------

@pytest.mark.parametrize("method", ["decimation", "max points"])
def test_input_is_not_modified(method):
    data = np.arange(12)
    original = data.copy()
    downsampler.downsample(data, 3, method)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("method", ["decimation", "max points"])
def test_dtype_is_preserved(method):
    data = np.arange(12, dtype=np.float32)
    assert downsampler.downsample(data, 3, method).dtype == np.float32


def test_max_points_rejects_python_list():
    # reshape() only exists on numpy arrays
    with pytest.raises(AttributeError):
        downsampler.downsample([1, 2, 3, 4], 2, "max points")