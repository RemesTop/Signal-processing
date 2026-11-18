from types import SimpleNamespace

import numpy as np
import pytest

import signal_reader
from signal_classes import Slice
from signal_constants import DEFAULT_SAMPLE_RATE
from signal_reader import SignalReader, convert_csv_files

"""
WRITTEN BY ARTIFICIAL INTELLIGENCE (AI) - CLAUDE SONNET 5.5
Tests for the signal_reader module. To run these tests, execute the following command in the terminal:
pytest tests/test_signalReader.py
Last modified: 18.11.2026"""

# ---------------------------------------------------------------- helpers

def write_csv(path, rows, header="adc1,adc2"):
    lines = [header] + [f"{a},{b}" for a, b in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def make_samples(n):
    """n samples, two channels: [i, -i]."""
    i = np.arange(n, dtype=np.int16)
    return np.column_stack([i, -i]).astype(np.int16)


def write_dat(path, n=100):
    samples = make_samples(n)
    samples.tofile(path)
    return samples


class FakeConverter:
    """Replacement for SignalConverter. Behavior is configured per test."""

    outcome = "success"  # "success" | "fail" | "raise"
    calls = []

    def start(self, source_signal_path, target_signal_path, progress_callback=None):
        FakeConverter.calls.append(
            SimpleNamespace(
                source=source_signal_path,
                target=target_signal_path,
                progress_callback=progress_callback,
            )
        )
        target_signal_path.write_bytes(b"partial")
        if FakeConverter.outcome == "raise":
            raise OSError("disk exploded")
        return FakeConverter.outcome == "success"


@pytest.fixture
def fake_converter(monkeypatch):
    FakeConverter.outcome = "success"
    FakeConverter.calls = []
    monkeypatch.setattr(signal_reader, "SignalConverter", FakeConverter)
    return FakeConverter


class _Slot:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


@pytest.fixture
def fake_threads(monkeypatch):
    """Replace Worker and QThreadPool so the multithreaded path never starts a thread."""
    created, started = [], []

    class FakeWorker:
        def __init__(self, fn, *args, **kwargs):
            self.fn, self.args, self.kwargs = fn, args, kwargs
            self.signals = SimpleNamespace(progress=_Slot(), result=_Slot())
            created.append(self)

    class FakePool:
        def start(self, worker):
            started.append(worker)

    monkeypatch.setattr(signal_reader, "Worker", FakeWorker)
    monkeypatch.setattr(
        signal_reader, "QThreadPool", SimpleNamespace(globalInstance=lambda: FakePool())
    )
    return SimpleNamespace(created=created, started=started)


@pytest.fixture
def fake_downsample(monkeypatch):
    """Replace downsampler.downsample with a recorder that keeps every n-th row."""
    calls = []

    def fake(raw_slice, factor, method=None):
        calls.append(SimpleNamespace(raw=raw_slice, factor=factor, method=method))
        return np.asarray(raw_slice[::factor])

    monkeypatch.setattr(signal_reader.downsampler, "downsample", fake)
    return calls


@pytest.fixture
def reader(tmp_path):
    """Reader over 100 samples at 10 Hz (10 seconds of data)."""
    samples = write_dat(tmp_path / "sig.dat", 100)
    r = SignalReader(tmp_path / "sig.dat", sample_rate=10)
    r.expected = samples
    yield r
    r.data = None  # release the memmap (matters on Windows)


# ---------------------------------------------------------------- convert_csv_files

class TestConvertCsvFiles:
    def test_empty_input_returns_empty_list(self):
        assert convert_csv_files([]) == []

    def test_non_csv_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Expected a CSV file"):
            convert_csv_files([tmp_path / "signal.txt"])

    def test_dat_file_is_rejected_too(self, tmp_path):
        with pytest.raises(ValueError):
            convert_csv_files([tmp_path / "signal.dat"])

    def test_converts_csv_to_dat(self, tmp_path):
        src = write_csv(tmp_path / "sig.csv", [(1, 2), (3, 4), (5, 6)])
        result = convert_csv_files([src])
        assert result == [tmp_path / "sig.dat"]
        data = np.fromfile(result[0], dtype=np.int16).reshape(-1, 2)
        np.testing.assert_array_equal(data, [[1, 2], [3, 4], [5, 6]])

    def test_no_tmp_file_left_after_success(self, tmp_path):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        convert_csv_files([src])
        assert not (tmp_path / "sig.tmp").exists()

    def test_accepts_string_paths(self, tmp_path):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        assert convert_csv_files([str(src)]) == [tmp_path / "sig.dat"]

    def test_uppercase_extension_is_accepted(self, tmp_path):
        src = write_csv(tmp_path / "SIG.CSV", [(1, 2)])
        assert convert_csv_files([src]) == [tmp_path / "SIG.dat"]

    def test_converts_multiple_files_in_order(self, tmp_path):
        a = write_csv(tmp_path / "a.csv", [(1, 1)])
        b = write_csv(tmp_path / "b.csv", [(2, 2)])
        result = convert_csv_files([a, b])
        assert result == [tmp_path / "a.dat", tmp_path / "b.dat"]

    def test_existing_dat_is_skipped_and_not_overwritten(self, tmp_path):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        dat = tmp_path / "sig.dat"
        dat.write_bytes(b"cached")
        assert convert_csv_files([src]) == []
        assert dat.read_bytes() == b"cached"

    def test_only_unconverted_files_are_returned(self, tmp_path):
        a = write_csv(tmp_path / "a.csv", [(1, 1)])
        b = write_csv(tmp_path / "b.csv", [(2, 2)])
        (tmp_path / "a.dat").write_bytes(b"cached")
        assert convert_csv_files([a, b]) == [tmp_path / "b.dat"]

    def test_bad_extension_stops_after_earlier_files_converted(self, tmp_path):
        a = write_csv(tmp_path / "a.csv", [(1, 1)])
        with pytest.raises(ValueError):
            convert_csv_files([a, tmp_path / "b.txt"])
        assert (tmp_path / "a.dat").exists()

    def test_passes_paths_and_progress_callback_to_converter(self, tmp_path, fake_converter):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        progress = object()
        convert_csv_files([src], progress_callback=progress)
        call = fake_converter.calls[0]
        assert call.source == src
        assert call.target == tmp_path / "sig.tmp"
        assert call.progress_callback is progress

    def test_converter_failure_raises_runtime_error_and_cleans_tmp(self, tmp_path, fake_converter):
        fake_converter.outcome = "fail"
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        with pytest.raises(RuntimeError, match="Conversion failed"):
            convert_csv_files([src])
        assert not (tmp_path / "sig.tmp").exists()
        assert not (tmp_path / "sig.dat").exists()

    def test_converter_exception_propagates_and_cleans_tmp(self, tmp_path, fake_converter):
        fake_converter.outcome = "raise"
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        with pytest.raises(OSError, match="disk exploded"):
            convert_csv_files([src])
        assert not (tmp_path / "sig.tmp").exists()
        assert not (tmp_path / "sig.dat").exists()


# ---------------------------------------------------------------- SignalReader construction

class TestSignalReaderInit:
    def test_reads_dat_file_directly(self, reader):
        assert reader.dat_path.suffix == ".dat"
        assert reader.total_samples == 100
        assert reader.data.shape == (100, 2)
        assert reader.data.dtype == np.int16
        np.testing.assert_array_equal(reader.data, reader.expected)

    def test_default_sample_rate(self, tmp_path):
        write_dat(tmp_path / "sig.dat", 10)
        r = SignalReader(tmp_path / "sig.dat")
        assert r.sample_rate == DEFAULT_SAMPLE_RATE
        r.data = None

    def test_accepts_string_path(self, tmp_path):
        write_dat(tmp_path / "sig.dat", 10)
        r = SignalReader(str(tmp_path / "sig.dat"))
        assert r.total_samples == 10
        r.data = None

    def test_partial_trailing_bytes_are_ignored(self, tmp_path):
        path = tmp_path / "sig.dat"
        write_dat(path, 10)
        with path.open("ab") as f:
            f.write(b"\x01\x02\x03")  # incomplete row
        r = SignalReader(path)
        assert r.total_samples == 10
        assert r.data.shape == (10, 2)
        r.data = None

    def test_csv_input_is_converted_blocking(self, tmp_path):
        src = write_csv(tmp_path / "sig.csv", [(1, 2), (3, 4), (5, 6)])
        r = SignalReader(src)
        assert r.dat_path == tmp_path / "sig.dat"
        assert (tmp_path / "sig.dat").exists()
        assert r.total_samples == 3
        np.testing.assert_array_equal(r.data, [[1, 2], [3, 4], [5, 6]])
        r.data = None

    def test_csv_input_uses_cached_dat_without_reconverting(self, tmp_path):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])  # would give 1 sample
        write_dat(tmp_path / "sig.dat", 7)               # cache holds 7 samples
        r = SignalReader(src)
        assert r.total_samples == 7
        r.data = None

    def test_uppercase_csv_suffix_is_recognised(self, tmp_path):
        src = write_csv(tmp_path / "SIG.CSV", [(1, 2)])
        r = SignalReader(src)
        assert r.dat_path == tmp_path / "SIG.dat"
        assert r.total_samples == 1
        r.data = None


class TestSignalReaderMultithreading:
    def test_background_worker_is_started_and_data_not_ready(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        assert len(fake_threads.created) == 1
        assert fake_threads.started == fake_threads.created
        worker = fake_threads.created[0]
        assert worker.fn is signal_reader.convert_csv_files
        assert worker.args == ([src],)
        assert r.data is None
        assert r.total_samples == 0
        assert not (tmp_path / "sig.dat").exists()

    def test_worker_progress_is_forwarded_to_reader_progress(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        received = []
        r.progress.connect(received.append)
        fake_threads.created[0].signals.progress.slots[0](42.0)
        assert received == [42.0]

    def test_worker_result_is_wired_to_on_conversion_finished(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        slots = fake_threads.created[0].signals.result.slots
        assert len(slots) == 1
        assert slots[0] == r.on_conversion_finished

    def test_no_worker_when_dat_already_cached(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        write_dat(tmp_path / "sig.dat", 5)
        r = SignalReader(src, multithreading=True)
        assert fake_threads.created == []
        assert r.total_samples == 5
        r.data = None

    def test_no_worker_for_dat_input(self, tmp_path, fake_threads):
        write_dat(tmp_path / "sig.dat", 5)
        r = SignalReader(tmp_path / "sig.dat", multithreading=True)
        assert fake_threads.created == []
        r.data = None

    def test_on_conversion_finished_loads_data_and_emits_ready(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        ready = []
        r.ready.connect(lambda: ready.append(True))

        samples = write_dat(tmp_path / "sig.dat", 6)  # simulate the worker finishing
        r.on_conversion_finished([tmp_path / "sig.dat"])

        assert ready == [True]
        assert r.total_samples == 6
        np.testing.assert_array_equal(r.data, samples)
        r.data = None

    def test_on_conversion_finished_ignores_other_paths(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        write_dat(tmp_path / "sig.dat", 6)
        ready = []
        r.ready.connect(lambda: ready.append(True))
        r.on_conversion_finished([tmp_path / "other.dat"])
        r.on_conversion_finished([])
        assert ready == []
        assert r.data is None

    def test_on_conversion_finished_requires_file_to_exist(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        r.on_conversion_finished([tmp_path / "sig.dat"])  # listed but never written
        assert r.data is None

    def test_get_slice_before_ready_returns_empty_slice(self, tmp_path, fake_threads):
        src = write_csv(tmp_path / "sig.csv", [(1, 2)])
        r = SignalReader(src, multithreading=True)
        s = r.get_slice(0, 10)
        assert isinstance(s, Slice)
        assert len(s.time) == 0
        assert s.data.shape == (0, 2)
        assert s.data.dtype == np.int16


# ---------------------------------------------------------------- get_slice (raw)

class TestGetSliceRaw:
    def test_returns_slice_instance_with_two_channels(self, reader):
        s = reader.get_slice(0, 1)
        assert isinstance(s, Slice)
        assert s.channels == 2
        assert s.channel_indices == (0, 1)

    def test_returns_samples_between_start_and_end(self, reader):
        s = reader.get_slice(1, 2)  # indices 10..20 at 10 Hz
        np.testing.assert_array_equal(s.data, reader.expected[10:20])

    def test_time_axis_length_matches_data_and_spans_requested_range(self, reader):
        s = reader.get_slice(1, 2)
        assert len(s.time) == len(s.data) == 10
        assert s.time[0] == pytest.approx(1.0)
        assert s.time[-1] == pytest.approx(2.0)

    def test_whole_signal(self, reader):
        s = reader.get_slice(0, 10)
        assert len(s.data) == 100
        np.testing.assert_array_equal(s.data, reader.expected)

    def test_fractional_seconds_truncate_to_sample_index(self, reader):
        s = reader.get_slice(0.25, 0.55)  # int(2.5)=2 .. int(5.5)=5
        np.testing.assert_array_equal(s.data, reader.expected[2:5])

    def test_negative_start_is_clamped_to_zero(self, reader):
        s = reader.get_slice(-5, 1)
        np.testing.assert_array_equal(s.data, reader.expected[0:10])

    def test_end_beyond_signal_is_clamped(self, reader):
        s = reader.get_slice(8, 20)
        np.testing.assert_array_equal(s.data, reader.expected[80:100])

    def test_range_completely_outside_signal_is_empty(self, reader):
        s = reader.get_slice(50, 60)
        assert len(s.data) == 0
        assert len(s.time) == 0

    def test_start_equal_to_end_is_empty(self, reader):
        s = reader.get_slice(5, 5)
        assert len(s.data) == 0
        assert s.data.shape == (0, 2)

    def test_start_after_end_is_empty(self, reader):
        s = reader.get_slice(6, 4)
        assert len(s.data) == 0

    @pytest.mark.parametrize(
        "downsample, factor",
        [
            (None, 5),      # downsampling not requested
            ("none", 5),    # explicit "none"
            ("mean", 1),    # factor of 1
            ("mean", 0),    # factor below 1
            ("mean", -3),   # negative factor
            ("mean", 10),   # total (10) <= factor
            ("mean", 100),  # factor larger than the slice
        ],
    )
    def test_raw_data_returned_when_downsampling_not_applicable(
        self, reader, fake_downsample, downsample, factor
    ):
        s = reader.get_slice(0, 1, downsample_factor=factor, downsample=downsample)
        assert fake_downsample == []
        np.testing.assert_array_equal(s.data, reader.expected[0:10])
        assert len(s.time) == 10

    def test_slice_data_is_int16(self, reader):
        assert reader.get_slice(0, 1).data.dtype == np.int16


# ---------------------------------------------------------------- get_slice (downsampled)

class TestGetSliceDownsampled:
    def test_downsampler_called_with_slice_factor_and_method(self, reader, fake_downsample):
        reader.get_slice(0, 10, downsample_factor=10, downsample="mean")
        assert len(fake_downsample) == 1
        call = fake_downsample[0]
        assert call.factor == 10
        assert call.method == "mean"
        np.testing.assert_array_equal(call.raw, reader.expected)

    def test_returns_downsampled_data(self, reader, fake_downsample):
        s = reader.get_slice(0, 10, downsample_factor=10, downsample="mean")
        np.testing.assert_array_equal(s.data, reader.expected[::10])
        assert len(s.data) == 10

    def test_time_axis_matches_downsampled_length(self, reader, fake_downsample):
        s = reader.get_slice(0, 10, downsample_factor=10, downsample="mean")
        assert len(s.time) == len(s.data) == 10
        assert s.time[0] == pytest.approx(0.0)
        assert s.time[-1] == pytest.approx(10.0)

    def test_downsampling_applies_when_total_just_exceeds_factor(self, reader, fake_downsample):
        # 10 samples with factor 9 -> total > factor, so downsampling happens
        reader.get_slice(0, 1, downsample_factor=9, downsample="mean")
        assert len(fake_downsample) == 1

    def test_only_requested_range_is_passed_to_downsampler(self, reader, fake_downsample):
        reader.get_slice(2, 6, downsample_factor=4, downsample="max")
        np.testing.assert_array_equal(fake_downsample[0].raw, reader.expected[20:60])

    def test_downsample_method_delegates_to_downsampler_module(self, reader, fake_downsample):
        raw = reader.expected[:20]
        result = reader._downsample(raw, 5, "mean")
        assert fake_downsample[0].method == "mean"
        assert fake_downsample[0].factor == 5
        np.testing.assert_array_equal(result, raw[::5])


# ---------------------------------------------------------------- known issues in time axis

class TestTimeAxisKnownIssues:
    """These document behavior that looks wrong; strict xfail flips to a failure once fixed."""

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Time axis is built with np.linspace(start, end, n) (endpoint included), so "
        "spacing is (end-start)/(n-1) instead of 1/sample_rate.",
    )
    def test_slice_sample_rate_matches_reader_sample_rate(self, reader):
        s = reader.get_slice(1, 2)
        assert s.sample_rate == pytest.approx(reader.sample_rate)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="Data is clamped to the file's range but the time axis still uses the "
        "unclamped start/end, so timestamps no longer line up with the samples.",
    )
    def test_time_axis_stays_within_signal_when_range_is_clamped(self, reader):
        s = reader.get_slice(-5, 1)
        assert s.time[0] >= 0.0