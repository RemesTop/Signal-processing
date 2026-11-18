import dataclasses

import numpy as np
import pytest

from signal_classes import AnalysisResult, Line, PeakSet, Slice

"""
WRITTEN BY ARTICIAL INTELLIGENCE (AI) - CLAUDE SONNET 5.5
Tests for the signal_classes module. To test the signal_classes, run the following command in the terminal:
pytest tests/test_signalClasses.py
Last modified: 17.11.2026
"""


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def time():
    return np.linspace(0.0, 1.0, 11)  # 11 samples, 10 Hz spacing -> 10 Hz rate


@pytest.fixture
def data():
    return np.arange(33, dtype=float).reshape(11, 3)  # 3 channels


@pytest.fixture
def sl(time, data):
    return Slice(time, data)


# ---------------------------------------------------------------- Slice

class TestSlice:
    def test_stores_time_and_data(self, sl, time, data):
        np.testing.assert_array_equal(sl.time, time)
        np.testing.assert_array_equal(sl.data, data)

    def test_converts_lists_to_arrays(self):
        s = Slice([0, 1, 2], [[1, 2], [3, 4], [5, 6]])
        assert isinstance(s.time, np.ndarray)
        assert isinstance(s.data, np.ndarray)
        assert s.data.shape == (3, 2)

    def test_default_channel_indices_is_all_channels(self, sl):
        assert sl.channel_indices == (0, 1, 2)

    def test_explicit_channel_indices_preserved(self, time, data):
        s = Slice(time, data, channel_indices=(2,))
        assert s.channel_indices == (2,)

    def test_empty_channel_indices_not_replaced(self, time, data):
        # () is not None, so it must be kept as-is
        s = Slice(time, data, channel_indices=())
        assert s.channel_indices == ()

    def test_time_must_be_1d(self, data):
        with pytest.raises(ValueError, match="one-dimensional"):
            Slice(np.zeros((11, 1)), data)

    def test_data_must_be_2d(self, time):
        with pytest.raises(ValueError, match="two-dimensional"):
            Slice(time, np.zeros(11))

    def test_length_mismatch(self, time):
        with pytest.raises(ValueError, match="same number of samples"):
            Slice(time, np.zeros((5, 2)))

    def test_is_frozen(self, sl):
        with pytest.raises(dataclasses.FrozenInstanceError):
            sl.time = np.zeros(11)

    def test_iter_yields_time_then_data(self, sl, time, data):
        t, d = sl
        np.testing.assert_array_equal(t, time)
        np.testing.assert_array_equal(d, data)
        assert len(list(sl)) == 2

    def test_channels_property(self, sl):
        assert sl.channels == 3

    def test_channels_ignores_channel_indices(self, time, data):
        assert Slice(time, data, channel_indices=(0,)).channels == 3

    def test_sample_rate(self, sl):
        assert sl.sample_rate == pytest.approx(10.0)

    def test_sample_rate_nonzero_start(self):
        s = Slice(np.array([5.0, 5.5, 6.0]), np.zeros((3, 1)))
        assert s.sample_rate == pytest.approx(2.0)

    def test_sample_rate_single_sample_defaults_to_one(self):
        assert Slice(np.array([0.0]), np.zeros((1, 1))).sample_rate == 1.0

    def test_sample_rate_empty_defaults_to_one(self):
        assert Slice(np.array([]), np.empty((0, 2))).sample_rate == 1.0

    def test_sample_rate_non_increasing_defaults_to_one(self):
        s = Slice(np.array([1.0, 1.0, 1.0]), np.zeros((3, 1)))
        assert s.sample_rate == 1.0


# ---------------------------------------------------------------- Line / PeakSet

class TestLine:
    def test_fields(self):
        x, y = np.arange(3), np.arange(3) * 2
        line = Line(x, y, "sig", "#ff0000")
        assert line.x is x
        assert line.y is y
        assert line.label == "sig"
        assert line.color == "#ff0000"

    def test_is_frozen(self):
        line = Line(np.zeros(1), np.zeros(1), "a", 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            line.label = "b"


class TestPeakSet:
    def test_fields(self):
        p = PeakSet(np.array([1, 2]), np.array([3, 4]), "pk", (255, 0, 0))
        assert p.label == "pk"
        assert p.color == (255, 0, 0)
        np.testing.assert_array_equal(p.x, [1, 2])
        np.testing.assert_array_equal(p.y, [3, 4])

    def test_properties_default_empty(self):
        p = PeakSet(np.zeros(1), np.zeros(1), "pk", 0)
        assert p.properties == {}

    def test_default_properties_not_shared_between_instances(self):
        a = PeakSet(np.zeros(1), np.zeros(1), "a", 0)
        b = PeakSet(np.zeros(1), np.zeros(1), "b", 0)
        a.properties["height"] = np.array([1.0])
        assert b.properties == {}

    def test_is_frozen(self):
        p = PeakSet(np.zeros(1), np.zeros(1), "pk", 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.label = "other"


# ---------------------------------------------------------------- AnalysisResult

class TestAnalysisResult:
    def test_starts_empty(self, sl):
        r = AnalysisResult(sl)
        assert r.slice is sl
        assert r.lines == []
        assert r.peak_sets == []

    def test_default_lists_not_shared(self, sl):
        a, b = AnalysisResult(sl), AnalysisResult(sl)
        a.line(np.zeros(2), np.zeros(2), "x")
        assert b.lines == []

    def test_line_appends_line(self, sl):
        r = AnalysisResult(sl)
        x, y = np.arange(3), np.arange(3)
        r.line(x, y, "foo", color="#123456")
        assert len(r.lines) == 1
        line = r.lines[0]
        assert isinstance(line, Line)
        assert line.x is x and line.y is y
        assert line.label == "foo"
        assert line.color == "#123456"

    def test_line_default_color_is_zero(self, sl):
        r = AnalysisResult(sl)
        r.line(np.zeros(1), np.zeros(1), "foo")
        assert r.lines[0].color == 0

    def test_line_returns_none(self, sl):
        assert AnalysisResult(sl).line(np.zeros(1), np.zeros(1), "foo") is None

    def test_multiple_lines_keep_order_and_allow_duplicate_labels(self, sl):
        r = AnalysisResult(sl)
        r.line(np.zeros(1), np.zeros(1), "same")
        r.line(np.ones(1), np.ones(1), "same")
        assert [l.label for l in r.lines] == ["same", "same"]
        assert r.lines[1].x[0] == 1

    def test_peaks_defaults(self, sl):
        r = AnalysisResult(sl)
        r.peaks(np.array([1.0]), np.array([2.0]))
        p = r.peak_sets[0]
        assert isinstance(p, PeakSet)
        assert p.label == "peaks"
        assert p.color == "#bb0000"
        assert p.properties == {}

    def test_peaks_with_properties_and_custom_args(self, sl):
        r = AnalysisResult(sl)
        props = {"prominence": np.array([0.5, 0.7])}
        r.peaks(np.array([1, 2]), np.array([3, 4]), props, label="mine", color=5)
        p = r.peak_sets[0]
        assert p.properties is props
        assert p.label == "mine"
        assert p.color == 5

    def test_peaks_none_properties_become_empty_dict(self, sl):
        r = AnalysisResult(sl)
        r.peaks(np.zeros(1), np.zeros(1), properties=None)
        assert r.peak_sets[0].properties == {}

    def test_peaks_returns_none(self, sl):
        assert AnalysisResult(sl).peaks(np.zeros(1), np.zeros(1)) is None

    def test_peaks_do_not_touch_lines(self, sl):
        r = AnalysisResult(sl)
        r.peaks(np.zeros(1), np.zeros(1))
        assert r.lines == []


class TestLineFromSlice:
    def test_one_line_per_channel(self, sl):
        r = AnalysisResult.line_from_slice(sl)
        assert isinstance(r, AnalysisResult)
        assert r.slice is sl
        assert len(r.lines) == 3
        assert r.peak_sets == []

    def test_labels_are_one_based(self, sl):
        r = AnalysisResult.line_from_slice(sl)
        assert [l.label for l in r.lines] == ["Channel 1", "Channel 2", "Channel 3"]

    def test_color_is_zero_based_channel_index(self, sl):
        r = AnalysisResult.line_from_slice(sl)
        assert [l.color for l in r.lines] == [0, 1, 2]

    def test_x_is_slice_time_and_y_is_channel_column(self, sl):
        r = AnalysisResult.line_from_slice(sl)
        for ch, line in enumerate(r.lines):
            np.testing.assert_array_equal(line.x, sl.time)
            np.testing.assert_array_equal(line.y, sl.data[:, ch])

    def test_respects_channel_indices(self, time, data):
        s = Slice(time, data, channel_indices=(2,))
        r = AnalysisResult.line_from_slice(s)
        assert len(r.lines) == 1
        assert r.lines[0].label == "Channel 3"
        assert r.lines[0].color == 2
        np.testing.assert_array_equal(r.lines[0].y, data[:, 2])

    def test_channel_order_follows_channel_indices(self, time, data):
        s = Slice(time, data, channel_indices=(2, 0))
        r = AnalysisResult.line_from_slice(s)
        assert [l.label for l in r.lines] == ["Channel 3", "Channel 1"]

    def test_empty_channel_indices_gives_no_lines(self, time, data):
        s = Slice(time, data, channel_indices=())
        assert AnalysisResult.line_from_slice(s).lines == []

    def test_works_when_called_via_instance_or_class(self, sl):
        # staticmethod: should behave identically either way
        r = AnalysisResult(sl).line_from_slice(sl)
        assert len(r.lines) == 3
