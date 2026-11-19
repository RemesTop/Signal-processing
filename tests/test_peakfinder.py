import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

import peak_finder as pf

FS = 1000  # sample rate used for synthetic test signals (Hz)
""""
WRITTEN BY ARTICIAL INTELLIGENCE (AI) - CLAUDE SONNET 5.5 AND GPT-6 LUNA
To test the peak finder, run the following command in the terminal:
pytest tests/test_peakfinder.py
Last modified: 17.11.2026
"""

# --------------------------------------------------------------- helpers


@dataclass
class FakeSlice:
    """Minimal stand-in for signal_classes.Slice."""
    data: np.ndarray
    time: np.ndarray
    sample_rate: float
    channel_indices: list


class FakeResult:
    """Minimal stand-in for signal_classes.AnalysisResult.

    Records every call to line() and peaks().
    """

    def __init__(self):
        self.lines = []
        self.peak_calls = []

    def line(self, time, y, label, color=None):
        self.lines.append({"time": time, "y": y, "label": label, "color": color})

    def peaks(self, *args):
        # Called as (t, y, label) or (t, y, properties, label)
        properties = args[2] if len(args) == 4 else None
        self.peak_calls.append({
            "time": args[0],
            "y": args[1],
            "properties": properties,
            "label": args[-1],
        })


def make_signal(peak_times=(0.5, 1.0, 1.5), duration=2.0, amplitude=100.0,
                sigma=0.02, offset=0.0, n_channels=1, seed=0):
    """Gaussian peaks on a flat baseline with a little noise.
    Returns (time, data) where data has shape (n_samples, n_channels)."""
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration * FS)) / FS
    signal = np.zeros_like(time)
    for t0 in peak_times:
        signal += amplitude * np.exp(-0.5 * ((time - t0) / sigma) ** 2)
    data = np.column_stack([signal + offset + rng.normal(0, 1, time.size)
                            for _ in range(n_channels)])
    return time, data


def make_slice(channel_indices=(0,), n_channels=1, **kwargs):
    time, data = make_signal(n_channels=n_channels, **kwargs)
    return FakeSlice(data, time, FS, list(channel_indices))


def assert_peaks_near(found_times, expected_times, tol=0.02):
    found_times = np.sort(np.asarray(found_times))
    assert len(found_times) == len(expected_times)
    np.testing.assert_allclose(found_times, expected_times, atol=tol)


@pytest.fixture
def out():
    return FakeResult()


@pytest.fixture
def clean_registry():
    """Restore ANALYSIS_FUNCTIONS after a test that registers functions."""
    saved = dict(pf.ANALYSIS_FUNCTIONS)
    yield pf.ANALYSIS_FUNCTIONS
    pf.ANALYSIS_FUNCTIONS.clear()
    pf.ANALYSIS_FUNCTIONS.update(saved)


# --------------------------------------------------------------- AnalysisParameter / AnalysisFunction


class TestAnalysisParameter:
    def test_label_defaults_to_none(self):
        p = pf.AnalysisParameter("height", 5)
        assert p.name == "height"
        assert p.default == 5
        assert p.label is None

    def test_is_frozen(self):
        p = pf.AnalysisParameter("height", 5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.default = 10


class TestAnalysisFunction:
    @pytest.fixture
    def func(self):
        return pf.AnalysisFunction(
            key="f",
            title="F",
            function=lambda s, out: None,
            parameters=(
                pf.AnalysisParameter("a", 1),
                pf.AnalysisParameter("b", None),
                pf.AnalysisParameter("c", 0.5),
            ),
        )

    def test_defaults(self, func):
        assert func.defaults == {"a": 1, "b": None, "c": 0.5}

    def test_defaults_empty_without_parameters(self):
        f = pf.AnalysisFunction("f", "F", lambda s, out: None)
        assert f.defaults == {}

    def test_parameters_for_call_keeps_set_values(self, func):
        assert func.parameters_for_call({"a": 2, "b": 3}) == {"a": 2, "b": 3}

    def test_parameters_for_call_drops_none_when_default_is_none(self, func):
        # "b" has default None -> an unset (None) value is left out
        assert func.parameters_for_call({"a": 2, "b": None}) == {"a": 2}

    def test_parameters_for_call_keeps_none_when_default_is_not_none(self, func):
        # "a" has a non-None default -> explicit None is the user's choice, keep it
        assert func.parameters_for_call({"a": None}) == {"a": None}

    def test_parameters_for_call_unknown_none_parameter_is_dropped(self, func):
        assert func.parameters_for_call({"zzz": None}) == {}

    def test_parameters_for_call_empty(self, func):
        assert func.parameters_for_call({}) == {}


# --------------------------------------------------------------- registry / decorator


class TestRegistry:
    EXPECTED = {
        "Raw signal with peaks": "raw_signal_with_peaks",
        "SciPy decimation": "scipy_decimation",
        "Custom decimation": "custom_decimation",
        "Extra low-pass": "extra_lowpass",
        "Baseline correction": "baseline_correction",
    }

    def test_builtin_functions_are_registered(self):
        for title, key in self.EXPECTED.items():
            entry = pf.ANALYSIS_FUNCTIONS[title]
            assert entry.key == key
            assert entry.title == title
            assert entry.function is getattr(pf, key)

    def test_example_analysis_is_not_registered(self):
        keys = {f.key for f in pf.ANALYSIS_FUNCTIONS.values()}
        assert "example_analysis" not in keys

    def test_decorator_returns_original_function(self, clean_registry):
        def my_func(s, out):
            pass

        assert pf.analysis_function("My title")(my_func) is my_func

    def test_decorator_registers_under_title(self, clean_registry):
        params = (pf.AnalysisParameter("x", 1),)

        @pf.analysis_function("Registered", params)
        def my_func(s, out):
            pass

        entry = clean_registry["Registered"]
        assert entry.key == "my_func"
        assert entry.function is my_func
        assert entry.parameters == params

    def test_same_title_overwrites(self, clean_registry):
        @pf.analysis_function("Dup")
        def first(s, out): ...

        @pf.analysis_function("Dup")
        def second(s, out): ...

        assert clean_registry["Dup"].function is second

    @pytest.mark.parametrize("title", list(EXPECTED))
    def test_every_registered_function_has_peak_parameters(self, title):
        names = {p.name for p in pf.ANALYSIS_FUNCTIONS[title].parameters}
        assert {p.name for p in pf.FIND_PEAKS_PARAMETERS} <= names

    @pytest.mark.parametrize("title", list(EXPECTED))
    def test_default_prominence_is_40(self, title):
        assert pf.ANALYSIS_FUNCTIONS[title].defaults["prominence"] == 40


class TestFindPeaksParameters:
    def test_names_match_scipy_find_peaks_options(self):
        names = [p.name for p in pf.FIND_PEAKS_PARAMETERS]
        assert names == ["height", "threshold", "distance", "prominence",
                         "width", "wlen", "rel_height", "plateau_size"]

    def test_rel_height_default(self):
        defaults = {p.name: p.default for p in pf.FIND_PEAKS_PARAMETERS}
        assert defaults["rel_height"] == 0.5
        assert defaults["prominence"] is None

    def test_modified_defaults_applied(self):
        params = {p.name: p.default for p in pf._find_peaks_parameters(prominence=40, height=3)}
        assert params["prominence"] == 40
        assert params["height"] == 3
        assert params["rel_height"] == 0.5  # untouched

    def test_no_overrides_equals_originals(self):
        assert pf._find_peaks_parameters() == pf.FIND_PEAKS_PARAMETERS

    def test_unknown_override_is_ignored(self):
        params = pf._find_peaks_parameters(not_a_param=1)
        assert params == pf.FIND_PEAKS_PARAMETERS


# --------------------------------------------------------------- butterworth_lowpass_filter


class TestButterworthLowpass:
    def test_output_has_same_shape(self):
        x = np.random.default_rng(0).normal(size=500)
        assert pf.butterworth_lowpass_filter(x, cutoff=50, fs=1000).shape == x.shape

    def test_removes_high_frequency_keeps_low(self):
        fs = 2000
        t = np.arange(4000) / fs
        low = np.sin(2 * np.pi * 10 * t)
        high = np.sin(2 * np.pi * 400 * t)
        y = pf.butterworth_lowpass_filter(low + high, cutoff=50, fs=fs)
        middle = slice(500, -500)  # ignore edge effects
        np.testing.assert_allclose(y[middle], low[middle], atol=0.05)

    def test_constant_signal_is_unchanged(self):
        y = pf.butterworth_lowpass_filter(np.full(300, 7.0), cutoff=50, fs=1000)
        np.testing.assert_allclose(y, 7.0, atol=1e-6)

    def test_zero_phase(self):
        # A peak should stay at the same index after filtering (filtfilt)
        x = np.zeros(1000)
        x[500] = 1.0
        y = pf.butterworth_lowpass_filter(x, cutoff=100, fs=1000)
        assert np.argmax(y) == 500

    @pytest.fixture
    def cutoff_spy(self, monkeypatch):
        """Record the cutoff passed to scipy's butter()."""
        calls = []
        real_butter = pf.butter

        def spy(order, cutoff, *args, **kwargs):
            calls.append({"order": order, "cutoff": cutoff})
            return real_butter(order, cutoff, *args, **kwargs)

        monkeypatch.setattr(pf, "butter", spy)
        return calls

    def test_cutoff_none_uses_max_cutoff(self, cutoff_spy):
        # fs=1000, factor 10 -> output Nyquist 50 Hz -> max cutoff 0.8 * 50 = 40
        pf.butterworth_lowpass_filter(np.zeros(500), cutoff=None, fs=1000, decimation_factor=10)
        assert cutoff_spy[0]["cutoff"] == pytest.approx(40)

    def test_cutoff_above_max_is_clamped(self, cutoff_spy):
        pf.butterworth_lowpass_filter(np.zeros(500), cutoff=1000, fs=1000, decimation_factor=10)
        assert cutoff_spy[0]["cutoff"] == pytest.approx(40)

    def test_cutoff_below_max_is_kept(self, cutoff_spy):
        pf.butterworth_lowpass_filter(np.zeros(500), cutoff=10, fs=1000, decimation_factor=10)
        assert cutoff_spy[0]["cutoff"] == pytest.approx(10)

    def test_custom_nyquist_multiplier(self, cutoff_spy):
        pf.butterworth_lowpass_filter(np.zeros(500), fs=1000, nyquist_cutoff_multiplier=0.5)
        assert cutoff_spy[0]["cutoff"] == pytest.approx(250)

    def test_order_is_forwarded(self, cutoff_spy):
        pf.butterworth_lowpass_filter(np.zeros(500), cutoff=50, fs=1000, order=2)
        assert cutoff_spy[0]["order"] == 2

    def test_default_sample_rate_is_used(self, cutoff_spy):
        pf.butterworth_lowpass_filter(np.zeros(500), cutoff=None)
        assert cutoff_spy[0]["cutoff"] == pytest.approx(0.8 * pf.DEFAULT_SAMPLE_RATE / 2)

    def test_short_signal_does_not_raise(self):
        # padlen is limited to x.size - 1
        y = pf.butterworth_lowpass_filter(np.arange(8, dtype=float), cutoff=50, fs=1000)
        assert y.shape == (8,)


# --------------------------------------------------------------- remove_baseline_morphological


class TestRemoveBaselineMorphological:
    def test_constant_signal_becomes_zero(self):
        y = pf.remove_baseline_morphological(np.full(1000, 100.0))
        np.testing.assert_allclose(y, 0, atol=1e-9)

    def test_explicit_bias_is_subtracted(self):
        # baseline == 100 everywhere, so output is x - (100 + bias)
        y = pf.remove_baseline_morphological(np.full(1000, 100.0), bias=5.0)
        np.testing.assert_allclose(y, -5.0, atol=1e-9)

    def test_bias_zero_is_respected_not_replaced_by_estimate(self):
        # Noisy data: the baseline is a lower envelope, so the estimated bias
        # (median residual) is clearly non-zero.
        x = 50 + np.random.default_rng(0).normal(size=1000)
        y_zero = pf.remove_baseline_morphological(x, bias=0.0)
        y_auto = pf.remove_baseline_morphological(x)
        assert np.median(y_auto) == pytest.approx(0.0, abs=1e-9)  # bias=None -> estimated
        assert abs(np.median(y_zero)) > 0.1                       # bias=0 is used as given

    @pytest.mark.parametrize("n", [1, 5, 10, 11, 99, 100, 1234])
    def test_output_length_matches_input(self, n):
        x = np.random.default_rng(0).normal(size=n)
        assert pf.remove_baseline_morphological(x).shape == (n,)

    def test_input_is_not_modified(self):
        x = np.random.default_rng(0).normal(size=500)
        original = x.copy()
        pf.remove_baseline_morphological(x)
        np.testing.assert_array_equal(x, original)

    def test_removes_slow_ramp_and_keeps_peak(self):
        n = 2000
        ramp = 0.01 * np.arange(n)           # slow drift, 0 -> 20
        peak = np.zeros(n)
        peak[995:1005] = 50                   # narrow peak
        y = pf.remove_baseline_morphological(ramp + peak, block_size=10, baseline_window=400)

        assert abs(np.median(y)) < 1.0                        # drift removed
        assert 990 <= np.argmax(y) <= 1010                    # peak position kept
        assert y.max() > 40                                   # peak height mostly kept

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "10", None])
    def test_invalid_block_size_raises(self, bad):
        with pytest.raises(ValueError, match="block_size"):
            pf.remove_baseline_morphological(np.zeros(100), block_size=bad)

    @pytest.mark.parametrize("bad", [0, -5, 2.5, "100", None])
    def test_invalid_baseline_window_raises(self, bad):
        with pytest.raises(ValueError, match="baseline_window"):
            pf.remove_baseline_morphological(np.zeros(100), baseline_window=bad)

    def test_numpy_integer_parameters_accepted(self):
        y = pf.remove_baseline_morphological(
            np.zeros(100), block_size=np.int64(10), baseline_window=np.int32(50))
        assert y.shape == (100,)

    def test_signal_shorter_than_block_size(self):
        y = pf.remove_baseline_morphological(np.array([1.0, 2.0, 3.0]), block_size=10)
        assert y.shape == (3,)


# --------------------------------------------------------------- scipy_find_peaks


class TestScipyFindPeaks:
    @pytest.fixture
    def simple(self):
        x = np.array([0, 1, 0, 3, 0, 2, 0], dtype=float)
        t = np.arange(len(x)) * 0.1
        return t, x

    def test_records_peaks_with_values(self, out, simple):
        t, x = simple
        pf.scipy_find_peaks(out, t, x, channel=0, peak_options={})
        call = out.peak_calls[0]
        np.testing.assert_allclose(call["time"], [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(call["y"], [1, 3, 2])

    def test_label_uses_one_based_channel_number(self, out, simple):
        t, x = simple
        pf.scipy_find_peaks(out, t, x, channel=2, peak_options={})
        assert out.peak_calls[0]["label"] == "Channel 3 peaks"

    def test_options_are_forwarded(self, out, simple):
        t, x = simple
        pf.scipy_find_peaks(out, t, x, 0, {"height": 2.5})
        np.testing.assert_allclose(out.peak_calls[0]["time"], [0.3])

    def test_unknown_options_are_ignored(self, out, simple):
        t, x = simple
        pf.scipy_find_peaks(out, t, x, 0, {"used_downsample_factor": 5, "foo": "bar"})
        assert len(out.peak_calls[0]["time"]) == 3

    def test_properties_are_passed_on(self, out, simple):
        t, x = simple
        pf.scipy_find_peaks(out, t, x, 0, {"prominence": 0})
        assert "prominences" in out.peak_calls[0]["properties"]

    def test_no_peaks_found(self, out):
        pf.scipy_find_peaks(out, np.arange(5.0), np.zeros(5), 0, {})
        assert len(out.peak_calls[0]["time"]) == 0


# --------------------------------------------------------------- analysis functions


class TestExampleAnalysis:
    def test_outputs_line_and_fixed_peaks(self, out):
        s = make_slice()
        pf.example_analysis(s, out)
        assert out.lines[0]["label"] == "Channel 1"
        np.testing.assert_array_equal(out.peak_calls[0]["time"], s.time[[10, 50, 100]])
        assert out.peak_calls[0]["label"] == "Channel 1 peaks"


class TestRawSignalWithPeaks:
    def test_line_is_unprocessed_signal(self, out):
        s = make_slice()
        pf.raw_signal_with_peaks(s, out)
        np.testing.assert_array_equal(out.lines[0]["y"], s.data[:, 0])
        np.testing.assert_array_equal(out.lines[0]["time"], s.time)

    def test_finds_known_peaks(self, out):
        pf.raw_signal_with_peaks(make_slice(), out)
        assert_peaks_near(out.peak_calls[0]["time"], [0.5, 1.0, 1.5], tol=0.005)

    def test_only_requested_channels_are_processed(self, out):
        s = make_slice(channel_indices=(1,), n_channels=3)
        pf.raw_signal_with_peaks(s, out)
        assert [l["label"] for l in out.lines] == ["Channel 2"]
        assert [c["label"] for c in out.peak_calls] == ["Channel 2 peaks"]

    def test_multiple_channels_get_separate_lines_and_colors(self, out):
        s = make_slice(channel_indices=(0, 1), n_channels=2)
        pf.raw_signal_with_peaks(s, out)
        assert [l["label"] for l in out.lines] == ["Channel 1", "Channel 2"]
        assert [l["color"] for l in out.lines] == [0, 1]
        assert len(out.peak_calls) == 2

    def test_user_prominence_overrides_default(self, out):
        pf.raw_signal_with_peaks(make_slice(), out, prominence=1000)
        assert len(out.peak_calls[0]["time"]) == 0

    def test_low_prominence_finds_noise_peaks_too(self, out):
        pf.raw_signal_with_peaks(make_slice(), out, prominence=0.1, distance=1)
        assert len(out.peak_calls[0]["time"]) > 3

    def test_default_distance_is_5ms_in_samples(self, out):
        # Two peaks 3 samples apart (3 ms): the smaller must be suppressed by
        # the default 5-sample minimum distance
        x = np.zeros(200)
        x[100] = 100
        x[103] = 90
        s = FakeSlice(x[:, None], np.arange(200) / FS, FS, [0])
        pf.raw_signal_with_peaks(s, out)
        assert len(out.peak_calls[0]["time"]) == 1

    def test_used_downsample_factor_scales_default_distance(self, out):
        # With factor 5 the default distance becomes max(1, 5/5) = 1, so both survive
        x = np.zeros(200)
        x[100] = 100
        x[103] = 90
        s = FakeSlice(x[:, None], np.arange(200) / FS, FS, [0])
        pf.raw_signal_with_peaks(s, out, used_downsample_factor=5)
        assert len(out.peak_calls[0]["time"]) == 2


class TestScipyDecimation:
    def test_line_is_decimated(self, out):
        s = make_slice()
        pf.scipy_decimation(s, out, decimation_factor=10)
        assert len(out.lines[0]["y"]) == len(s.time) // 10
        np.testing.assert_array_equal(out.lines[0]["time"], s.time[::10])

    def test_finds_known_peaks(self, out):
        pf.scipy_decimation(make_slice(), out, decimation_factor=10)
        assert_peaks_near(out.peak_calls[0]["time"], [0.5, 1.0, 1.5])

    def test_only_requested_channels_are_processed(self, out):
        s = make_slice(channel_indices=(2,), n_channels=3)
        pf.scipy_decimation(s, out)
        assert [l["label"] for l in out.lines] == ["Channel 3"]

    def test_input_data_is_not_modified(self, out):
        s = make_slice()
        original = s.data.copy()
        pf.scipy_decimation(s, out)
        np.testing.assert_array_equal(s.data, original)


class TestCustomDecimation:
    def test_line_is_downsampled(self, out):
        s = make_slice()
        pf.custom_decimation(s, out, decimation_factor=10)
        assert len(out.lines[0]["y"]) == len(s.time) // 10
        assert len(out.lines[0]["time"]) == len(out.lines[0]["y"])

    def test_finds_known_peaks(self, out):
        pf.custom_decimation(make_slice(), out, decimation_factor=10)
        assert_peaks_near(out.peak_calls[0]["time"], [0.5, 1.0, 1.5])

    def test_filter_reduces_noise(self, out):
        s = make_slice()
        pf.custom_decimation(s, out, decimation_factor=10)
        raw_noise = np.std(np.diff(s.data[:, 0][:300]))
        filtered_noise = np.std(np.diff(out.lines[0]["y"][:30]))
        assert filtered_noise < raw_noise

    def test_multiple_channels(self, out):
        s = make_slice(channel_indices=(0, 1), n_channels=2)
        pf.custom_decimation(s, out)
        assert len(out.lines) == 2
        assert len(out.peak_calls) == 2


class TestExtraLowpass:
    def test_line_is_downsampled(self, out):
        s = make_slice()
        pf.extra_lowpass(s, out, decimation_factor=10)
        assert len(out.lines[0]["y"]) == len(s.time) // 10

    def test_finds_known_peaks(self, out):
        pf.extra_lowpass(make_slice(), out, decimation_factor=10, lowpass_cutoff=40)
        assert_peaks_near(out.peak_calls[0]["time"], [0.5, 1.0, 1.5])

    def test_extra_filter_smooths_more_than_custom_decimation(self, out):
        s = make_slice()
        out2 = FakeResult()
        pf.custom_decimation(s, out, decimation_factor=10)
        pf.extra_lowpass(s, out2, decimation_factor=10, lowpass_cutoff=20)
        assert np.std(np.diff(out2.lines[0]["y"])) < np.std(np.diff(out.lines[0]["y"]))


class TestBandPassFilter:
    def test_line_is_downsampled_and_known_peaks_are_found(self, out):
        s = make_slice()
        pf.band_pass_filter(s, out, decimation_factor=5, low_cutoff=2, high_cutoff=60)
        assert len(out.lines[0]["y"]) == len(s.time) // 5
        np.testing.assert_array_equal(out.lines[0]["time"], s.time[::5])
        assert_peaks_near(out.peak_calls[0]["time"], [0.5, 1.0, 1.5])

    def test_only_requested_channels_are_processed(self, out):
        s = make_slice(channel_indices=(1,), n_channels=2)
        pf.band_pass_filter(s, out, decimation_factor=1, low_cutoff=2, high_cutoff=100)

        assert [line["label"] for line in out.lines] == ["Channel 2"]
        assert len(out.peak_calls) == 1


class TestBaselineCorrection:
    KWARGS = dict(decimation_factor=10, block_size=5, baseline_window=100)

    def test_removes_offset(self, out):
        s = make_slice(offset=500.0)
        pf.baseline_correction(s, out, **self.KWARGS)
        assert abs(np.median(out.lines[0]["y"])) < 5

    def test_finds_known_peaks_despite_offset(self, out):
        s = make_slice(offset=500.0)
        pf.baseline_correction(s, out, **self.KWARGS)
        assert_peaks_near(out.peak_calls[0]["time"], [0.5, 1.0, 1.5])

    def test_line_is_downsampled(self, out):
        s = make_slice()
        pf.baseline_correction(s, out, **self.KWARGS)
        assert len(out.lines[0]["y"]) == len(s.time) // 10
        assert len(out.lines[0]["time"]) == len(out.lines[0]["y"])

    def test_baseline_window_none_uses_default(self, out):
        s = make_slice()
        pf.baseline_correction(s, out, decimation_factor=10, block_size=5, baseline_window=None)
        assert len(out.lines[0]["y"]) == len(s.time) // 10

    def test_explicit_bias_shifts_line(self, out):
        s = make_slice(offset=500.0)
        out2 = FakeResult()
        pf.baseline_correction(s, out, bias=0.0, **self.KWARGS)
        pf.baseline_correction(s, out2, bias=10.0, **self.KWARGS)
        diff = out.lines[0]["y"] - out2.lines[0]["y"]
        np.testing.assert_allclose(diff, 10.0, atol=1e-9)

    def test_decimation_factor_one_without_downsampling(self, out):
        s = make_slice()
        pf.baseline_correction(s, out, decimation_factor=1, block_size=5, baseline_window=100)
        assert len(out.lines[0]["y"]) == len(s.time)


class TestBaselineCorrectionPybaselines:
    @pytest.mark.parametrize(("half_window", "expected_half_window"), [(17, 17), (None, 3000)])
    def test_selected_channel_is_corrected_and_half_window_is_forwarded(
        self, out, monkeypatch, half_window, expected_half_window
    ):
        s = make_slice(channel_indices=(1,), n_channels=2, offset=20.0)
        original = s.data.copy()
        estimator_inputs = []
        estimator_half_windows = []

        def estimate_baseline(x, half_window):
            estimator_inputs.append(x.copy())
            estimator_half_windows.append(half_window)
            return np.full_like(x, 4.0), {"ok": True}

        monkeypatch.setattr(pf.pybaselines.morphological, "jbcd", estimate_baseline)
        pf.baseline_correction_pybaselines(s, out, decimation_factor=2, half_window=half_window)
        assert estimator_half_windows == [expected_half_window]
        assert len(out.lines) == 1
        assert out.lines[0]["label"] == "Channel 2"
        assert out.lines[0]["color"] == 1
        np.testing.assert_array_equal(out.lines[0]["time"], s.time[::2])
        np.testing.assert_allclose(out.lines[0]["y"], estimator_inputs[0] - 4.0)
        np.testing.assert_array_equal(s.data, original)
        assert len(out.peak_calls) == 1

    def test_returns_without_processing_when_pybaselines_is_unavailable(self, out, monkeypatch, capsys):
        monkeypatch.delitem(pf.sys.modules, "pybaselines", raising=False)
        s = make_slice()
        pf.baseline_correction_pybaselines(s, out)
        assert out.lines == []
        assert out.peak_calls == []
        assert "pybaselines package is required" in capsys.readouterr().out


# --------------------------------------------------------------- peak_finder class


class TestPeakFinderClass:
    @pytest.fixture
    def triangle(self):
        # single triangular peak at index 3 (value 3), prominence 3
        return np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)

    def test_default_time_axis_is_sample_index(self, triangle):
        pfinder = pf.peak_finder(triangle)
        np.testing.assert_array_equal(pfinder.t, np.arange(7))

    def test_accepts_lists(self):
        pfinder = pf.peak_finder([0, 1, 0], [0.0, 0.1, 0.2])
        assert isinstance(pfinder.y, np.ndarray)
        assert isinstance(pfinder.t, np.ndarray)

    def test_find_returns_peak_times(self, triangle):
        t = np.arange(7) * 0.1
        times = pf.peak_finder(triangle, t).find()
        np.testing.assert_allclose(times, [0.3])

    def test_find_returns_indices_when_no_time_given(self, triangle):
        np.testing.assert_array_equal(pf.peak_finder(triangle).find(), [3])

    def test_find_stores_peaks_and_properties(self, triangle):
        pfinder = pf.peak_finder(triangle)
        pfinder.find(prominence=1)
        np.testing.assert_array_equal(pfinder.peaks, [3])
        assert "prominences" in pfinder.properties

    def test_find_forwards_kwargs(self):
        y = np.array([0, 1, 0, 5, 0, 2, 0], dtype=float)
        pfinder = pf.peak_finder(y)
        np.testing.assert_array_equal(pfinder.find(), [1, 3, 5])
        np.testing.assert_array_equal(pfinder.find(height=3), [3])

    def test_find_no_peaks_returns_empty(self):
        assert len(pf.peak_finder(np.zeros(10)).find()) == 0

    def test_time_frames_half_prominence(self, triangle):
        t = np.arange(7) * 0.1
        pfinder = pf.peak_finder(triangle, t)
        pfinder.find()
        # half prominence = 1.5 -> crossings at index 1.5 and 4.5
        np.testing.assert_allclose(pfinder.time_frames(), [[0.15, 0.45]])

    def test_time_frames_full_prominence(self, triangle):
        pfinder = pf.peak_finder(triangle, np.arange(7) * 0.1)
        pfinder.find()
        np.testing.assert_allclose(pfinder.time_frames(rel_height=1.0), [[0.0, 0.6]])

    def test_time_frames_one_row_per_peak(self):
        y = np.array([0, 1, 0, 5, 0, 2, 0], dtype=float)
        pfinder = pf.peak_finder(y)
        pfinder.find()
        frames = pfinder.time_frames()
        assert frames.shape == (3, 2)
        assert np.all(frames[:, 0] < frames[:, 1])

    def test_time_frames_window_contains_peak(self):
        time, data = make_signal()
        pfinder = pf.peak_finder(data[:, 0], time)
        peak_times = pfinder.find(prominence=40)
        frames = pfinder.time_frames()
        assert np.all(frames[:, 0] < peak_times)
        assert np.all(peak_times < frames[:, 1])

    def test_time_frames_no_peaks_gives_empty_2d_array(self):
        pfinder = pf.peak_finder(np.zeros(10))
        pfinder.find()
        assert pfinder.time_frames().shape == (0, 2)

    def test_time_frames_before_find_raises(self, triangle):
        with pytest.raises(AttributeError):
            pf.peak_finder(triangle).time_frames()