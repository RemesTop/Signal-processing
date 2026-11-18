import numpy as np
import pytest

import signal_converter
from signal_converter import SignalConverter

""""
WRITTEN BY ARTICIAL INTELLIGENCE (AI) - CLAUDE SONNET 5.5
Tests for the signal_converter module. To test the signal_converter, run the following command in the terminal:
pytest tests/test_signal_converter.py
Last modified: 18.11.2026
"""


# ---------------------------------------------------------------- helpers

class FakeSignal:
    """Stand-in for a Qt Signal: records every emitted value.

    If cancel_on is given, the converter is cancelled during that (1-based) emit,
    which lets tests cancel at precise points of the conversion.
    """

    def __init__(self, converter=None, cancel_on=None):
        self.values = []
        self.converter = converter
        self.cancel_on = cancel_on

    def emit(self, value):
        self.values.append(value)
        if self.cancel_on is not None and len(self.values) == self.cancel_on:
            self.converter.cancel()


def write_csv(path, rows, header="adc1,adc2", raw_lines=None):
    lines = [header] if header is not None else []
    if raw_lines is not None:
        lines += raw_lines
    else:
        lines += [f"{a},{b}" for a, b in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_binary(path):
    return np.fromfile(path, dtype=np.int16).reshape(-1, 2)


@pytest.fixture
def source(tmp_path):
    return write_csv(tmp_path / "signal.csv", [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])


@pytest.fixture
def target(tmp_path):
    return tmp_path / "signal.bin"


@pytest.fixture
def small_rate(monkeypatch):
    """Use a tiny sample rate so progress/flush logic is exercised with few rows."""
    monkeypatch.setattr(signal_converter, "DEFAULT_SAMPLE_RATE", 2)


# ---------------------------------------------------------------- conversion output

class TestConversionOutput:
    def test_returns_true_on_success(self, source, target):
        assert SignalConverter().start(source, target) is True

    def test_binary_contents_match_csv(self, source, target):
        SignalConverter().start(source, target)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int16)
        np.testing.assert_array_equal(read_binary(target), expected)

    def test_file_size_is_rows_times_two_int16(self, source, target):
        SignalConverter().start(source, target)
        assert target.stat().st_size == 5 * 2 * np.dtype(np.int16).itemsize

    def test_header_row_is_skipped(self, tmp_path, target):
        src = write_csv(tmp_path / "s.csv", [(11, 22)], header="adc1,adc2")
        SignalConverter().start(src, target)
        np.testing.assert_array_equal(read_binary(target), [[11, 22]])

    def test_negative_and_extreme_int16_values(self, tmp_path, target):
        rows = [(-1, 1), (-32768, 32767), (0, 0)]
        src = write_csv(tmp_path / "s.csv", rows)
        SignalConverter().start(src, target)
        np.testing.assert_array_equal(read_binary(target), np.array(rows, dtype=np.int16))

    def test_memmap_layout_is_c_order_row_major(self, source, target):
        SignalConverter().start(source, target)
        raw = np.fromfile(target, dtype=np.int16)
        # Row-major: adc1, adc2 interleaved sample by sample
        np.testing.assert_array_equal(raw, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_trailing_blank_lines_are_ignored(self, tmp_path, target):
        src = tmp_path / "s.csv"
        src.write_text("adc1,adc2\n1,2\n3,4\n\n\n")
        assert SignalConverter().start(src, target) is True
        np.testing.assert_array_equal(read_binary(target), [[1, 2], [3, 4]])

    def test_overwrites_existing_target_file(self, source, target):
        target.write_bytes(b"x" * 1000)
        SignalConverter().start(source, target)
        assert target.stat().st_size == 5 * 2 * 2
        assert read_binary(target)[0].tolist() == [1, 2]

    def test_larger_file_round_trip(self, tmp_path, target):
        rng = np.random.default_rng(0)
        values = rng.integers(-2000, 2000, size=(1000, 2))
        src = write_csv(tmp_path / "big.csv", values.tolist())
        assert SignalConverter().start(src, target) is True
        np.testing.assert_array_equal(read_binary(target), values.astype(np.int16))

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="Known bug: blank rows in the middle are excluded from row_count, but "
        "enumerate() still counts them, so the write index runs past the end of the memmap.",
    )
    def test_blank_row_in_middle_of_file(self, tmp_path, target):
        src = write_csv(tmp_path / "s.csv", None, raw_lines=["1,2", "", "3,4"])
        SignalConverter().start(src, target)
        np.testing.assert_array_equal(read_binary(target), [[1, 2], [3, 4]])


# ---------------------------------------------------------------- error handling

class TestErrors:
    def test_empty_file_without_header_raises_stop_iteration(self, tmp_path, target):
        src = tmp_path / "empty.csv"
        src.write_text("")
        with pytest.raises(StopIteration):
            SignalConverter().start(src, target)

    def test_header_only_file_produces_no_samples(self, tmp_path, target):
        src = write_csv(tmp_path / "s.csv", [])
        progress = FakeSignal()
        assert SignalConverter().start(src, target, progress_callback=progress) is True
        assert target.exists()
    # numpy may pad an empty memmap file to 1 byte, so check samples, not file size
        assert np.fromfile(target, dtype=np.int16).size == 0
        assert progress.values[0] == 0
        assert progress.values[-1] == 100

    def test_non_numeric_value_raises_value_error(self, tmp_path, target):
        src = write_csv(tmp_path / "s.csv", None, raw_lines=["1,2", "a,b"])
        with pytest.raises(ValueError):
            SignalConverter().start(src, target)

    def test_missing_source_file_raises(self, tmp_path, target):
        with pytest.raises(FileNotFoundError):
            SignalConverter().start(tmp_path / "nope.csv", target)


# ---------------------------------------------------------------- progress reporting

class TestProgress:
    def test_works_without_progress_callback(self, source, target):
        assert SignalConverter().start(source, target, progress_callback=None) is True

    def test_first_emit_is_zero_and_last_is_hundred(self, source, target):
        progress = FakeSignal()
        SignalConverter().start(source, target, progress_callback=progress)
        assert progress.values[0] == 0
        assert progress.values[-1] == 100

    def test_exact_progress_sequence(self, source, target, small_rate):
        # 5 rows, rate 2 -> updates at indices 0, 2, 4
        progress = FakeSignal()
        SignalConverter().start(source, target, progress_callback=progress)
        assert progress.values == [0, 0.0, 40.0, 80.0, 100]

    def test_progress_is_monotonic_and_bounded(self, tmp_path, target, small_rate):
        src = write_csv(tmp_path / "s.csv", [(i, i) for i in range(101)])
        progress = FakeSignal()
        SignalConverter().start(src, target, progress_callback=progress)
        assert progress.values == sorted(progress.values)
        assert all(0 <= v <= 100 for v in progress.values)

    def test_progress_emitted_only_once_per_sample_rate_rows(self, tmp_path, target, small_rate):
        src = write_csv(tmp_path / "s.csv", [(i, i) for i in range(10)])
        progress = FakeSignal()
        SignalConverter().start(src, target, progress_callback=progress)
        # initial 0 + indices 0,2,4,6,8 + final 100
        assert len(progress.values) == 1 + 5 + 1


# ---------------------------------------------------------------- cancellation

class TestCancellation:
    def test_cancel_sets_flag(self):
        c = SignalConverter()
        assert c.cancelled is False
        c.cancel()
        assert c.cancelled is True

    def test_cancel_is_per_instance(self):
        a, b = SignalConverter(), SignalConverter()
        a.cancel()
        assert a.cancelled is True
        assert b.cancelled is False
        assert SignalConverter.cancelled is False

    def test_start_resets_previous_cancel(self, source, target):
        c = SignalConverter()
        c.cancel()
        assert c.start(source, target) is True
        assert c.cancelled is False
        assert target.exists()

    def test_cancel_while_counting_rows(self, source, target):
        c = SignalConverter()
        progress = FakeSignal(c, cancel_on=1)  # cancel during the initial emit(0)
        assert c.start(source, target, progress_callback=progress) is False
        assert not target.exists()  # binary file was never created
        assert progress.values == [0]

    def test_cancel_while_writing_removes_target_file(self, source, target, small_rate):
        c = SignalConverter()
        progress = FakeSignal(c, cancel_on=2)  # first emit inside the write loop
        assert c.start(source, target, progress_callback=progress) is False
        assert not target.exists()
        assert 100 not in progress.values

    def test_cancel_during_final_progress_emit(self, source, target, small_rate):
        c = SignalConverter()
        # emits: 0, idx0, idx2, idx4, final 100 -> cancel on the 5th (final) one
        progress = FakeSignal(c, cancel_on=5)
        assert c.start(source, target, progress_callback=progress) is False
        assert not target.exists()
        assert progress.values[-1] == 0  # progress is reset after late cancellation

    def test_can_run_again_after_cancel(self, source, target, small_rate):
        c = SignalConverter()
        c.start(source, target, progress_callback=FakeSignal(c, cancel_on=2))
        assert not target.exists()
        assert c.start(source, target) is True
        np.testing.assert_array_equal(read_binary(target)[0], [1, 2])
