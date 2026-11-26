import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QThreadPool
from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog, QToolButton

from signal_app import SignalApp
from signal_classes import AnalysisResult, Slice


@pytest.fixture
def app_window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = SignalApp()
    # Isolate QSettings from repo settings.ini during tests
    window.settings = QSettings(str(tmp_path / "test_settings.ini"), QSettings.IniFormat)
    yield window
    window.pan_timer.stop()
    window._invalidate_total_peak_count()
    QThreadPool.globalInstance().waitForDone(2000)
    window.close()
    app.processEvents()


def _attach_fake_plot(window: SignalApp, total_duration: float = 3600.0, sample_rate: float = 100.0):
    total_samples = int(total_duration * sample_rate)
    window.reader = SimpleNamespace(
        total_samples=total_samples,
        sample_rate=sample_rate,
        data=np.zeros((10, 1)),
        get_slice=lambda s, e: Slice(np.array([s, e]), np.zeros((2, 1))),
    )
    # Pre-populate peak count completion so _on_analysis_finished doesn't spawn background counting
    window._peak_count_completed = True
    time = np.linspace(0.0, min(total_duration, window.visible_slice_seconds * 2.0), 100)
    data = np.zeros((100, 1))
    result = AnalysisResult(Slice(time, data))
    result.line(time, data[:, 0], 0, "Signal")
    window._analysis_request_id += 1
    window._on_analysis_finished(
        (window._analysis_request_id, result, (0.0, float(time[-1])), True, True, False)
    )


def test_jump_controls_hidden_until_graph_is_shown(app_window):
    app_window.show()
    QApplication.processEvents()
    assert app_window.jump_controls.isHidden()
    assert not app_window.jump_controls.isVisible()

    run_geometry_before = app_window.btn.geometry()
    file_input_geometry_before = app_window.file_input.geometry()

    _attach_fake_plot(app_window, total_duration=1800.0)
    QApplication.processEvents()
    assert not app_window.jump_controls.isHidden()
    assert app_window.jump_controls.isVisible()
    assert app_window.btn.geometry() == run_geometry_before
    assert app_window.file_input.geometry() == file_input_geometry_before
    assert app_window.jump_controls.x() >= app_window.btn.geometry().right() + 12


def test_run_from_time_runs_file_then_jumps_after_plot(app_window, monkeypatch):
    app_window.show()
    app_window.file_input.setText("signal.dat")
    run_calls = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args: ("20:00", True),
    )
    monkeypatch.setattr(app_window, "visualize", lambda: run_calls.append(True))

    assert app_window.btn.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
    assert app_window.run_from_time_action.text() == "Run at time..."

    app_window.run_from_time_action.trigger()

    assert run_calls == [True]
    assert app_window.jump_input.text() == "20:00"
    assert app_window._jump_after_run

    reloads = _stub_reload(app_window, monkeypatch)
    span = app_window.visible_slice_seconds
    _attach_fake_plot(app_window, total_duration=3600.0)

    x_min, x_max = app_window.plot_widget.getViewBox().viewRange()[0]
    assert np.allclose([x_min, x_max], [1200.0 - span / 2, 1200.0 + span / 2])
    assert not app_window._jump_after_run
    assert reloads == [{"force": True}]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0:00", 0.0),
        ("5:30", 330.0),
        ("90:15", 5415.0),
        ("1:05:30", 3930.0),
        ("45", 45.0),
        ("12.5s", 12.5),
        (" 1:00.5 ", 60.5),
    ],
)
def test_parse_jump_time_valid(text, expected):
    assert SignalApp._parse_jump_time(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "   ", "abc", "-1:00", "1:60", "1:60:00", "1:2:3:4", "-5"])
def test_parse_jump_time_invalid(text):
    with pytest.raises(ValueError):
        SignalApp._parse_jump_time(text)


def _stub_reload(window: SignalApp, monkeypatch, total_duration: float = 3600.0):
    reloads = []

    def fake_reload(**kw):
        reloads.append(kw)
        x0, x1 = window.plot_widget.getViewBox().viewRange()[0]
        center = (x0 + x1) / 2
        window.loaded_range = (
            max(0.0, center - window.visible_slice_seconds),
            min(total_duration, center + window.visible_slice_seconds),
        )

    monkeypatch.setattr(window, "_reload_visible_slice", fake_reload)
    return reloads


def test_jump_input_enter_moves_graph_and_reloads_immediately(app_window, monkeypatch):
    app_window.show()
    _attach_fake_plot(app_window, total_duration=3600.0)
    reloads = _stub_reload(app_window, monkeypatch)

    app_window.jump_input.setText("20:00")
    app_window.jump_input.returnPressed.emit()

    x_min, x_max = app_window.plot_widget.getViewBox().viewRange()[0]
    span = app_window.visible_slice_seconds
    assert np.allclose([x_min, x_max], [1200.0 - span / 2, 1200.0 + span / 2])
    assert not app_window.pan_timer.isActive()
    assert reloads == [{"force": True}]


def test_jump_arrows_move_10_minutes_instantly(app_window, monkeypatch):
    app_window.show()
    _attach_fake_plot(app_window, total_duration=3600.0)
    reloads = _stub_reload(app_window, monkeypatch)

    app_window.jump_forward_btn.click()
    x_min, x_max = app_window.plot_widget.getViewBox().viewRange()[0]
    assert np.allclose([x_min, x_max], [600.0, 600.0 + app_window.visible_slice_seconds])
    assert not app_window.pan_timer.isActive()
    assert len(reloads) == 1

    app_window.jump_back_btn.click()
    x_min, x_max = app_window.plot_widget.getViewBox().viewRange()[0]
    assert np.allclose([x_min, x_max], [0.0, app_window.visible_slice_seconds])
    assert not app_window.pan_timer.isActive()
    assert len(reloads) == 2


def test_jump_within_loaded_slice_does_not_force_reload(app_window, monkeypatch):
    app_window.show()
    _attach_fake_plot(app_window, total_duration=3600.0)
    reloads = _stub_reload(app_window, monkeypatch)

    app_window.jump_input.setText("2:30")
    app_window.jump_input.returnPressed.emit()

    x_min, x_max = app_window.plot_widget.getViewBox().viewRange()[0]
    span = app_window.visible_slice_seconds
    assert np.allclose([x_min, x_max], [150.0 - span / 2, 150.0 + span / 2])
    assert reloads == []


def test_invalid_jump_input_shows_warning(app_window, monkeypatch):
    app_window.show()
    _attach_fake_plot(app_window, total_duration=3600.0)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    app_window.jump_input.setText("not-a-time")
    app_window.jump_input.returnPressed.emit()

    assert len(warnings) == 1


def _legend_texts(window: SignalApp) -> list[str]:
    legend = window.plot_widget.getPlotItem().legend
    return [label.text for _, label in legend.items]


def test_title_and_per_function_peak_labels_update_on_zoom_and_counting(app_window, monkeypatch):
    from peak_finder import ANALYSIS_FUNCTIONS

    app_window.show()
    # Prevent background worker from actually running on QThreadPool so we can step progress manually
    started_workers = []
    monkeypatch.setattr(
        QThreadPool.globalInstance(), "start", lambda worker: started_workers.append(worker)
    )

    raw_fn = ANALYSIS_FUNCTIONS["Raw signal with peaks"]
    dec_fn = ANALYSIS_FUNCTIONS["SciPy decimation"]
    app_window.active_analyses = (raw_fn, dec_fn)
    app_window.analysis_parameters = {
        raw_fn.key: raw_fn.defaults,
        dec_fn.key: dec_fn.defaults,
    }
    app_window.visible_slice_seconds = 240

    total_duration = 3600.0
    sample_rate = 10.0
    app_window.reader = SimpleNamespace(
        total_samples=int(total_duration * sample_rate),
        sample_rate=sample_rate,
        data=np.zeros((10, 1)),
    )

    # Loaded slice [0, 480], initial visible window [0, 240]
    time = np.linspace(0.0, 480.0, 481)
    data = np.zeros((481, 1))
    result = AnalysisResult(Slice(time, data))
    result.line(time, data[:, 0], 0, "Raw")
    result.line(time, data[:, 0], 0, "Decimated SciPy")

    # Raw has 5 peaks in [0, 240] (3 in [0, 100]) and 8 total in [0, 480]
    raw_peaks_x = np.array([10.0, 50.0, 90.0, 150.0, 200.0, 300.0, 350.0, 420.0])
    # Decimated SciPy has 2 peaks in [0, 240] (1 in [0, 100]) and 4 total in [0, 480]
    dec_peaks_x = np.array([50.0, 180.0, 310.0, 450.0])
    result.peaks(raw_peaks_x, np.ones_like(raw_peaks_x), 0, "Raw")
    result.peaks(dec_peaks_x, np.ones_like(dec_peaks_x), 0, "Decimated SciPy")

    app_window._invalidate_total_peak_count()
    app_window._analysis_request_id += 1
    app_window._on_analysis_finished(
        (app_window._analysis_request_id, result, (0.0, 480.0), True, True, False)
    )

    # 1. While counting: title shows "Counting peaks.." without "Peaks:" or "Functions:"
    title_during = app_window.plot_widget.getPlotItem().titleLabel.text
    assert title_during == "File duration: 1:00:00, Counting peaks.."
    assert len(started_workers) == 1

    # Both functions show (visible/loaded+) initially
    labels = _legend_texts(app_window)
    assert "Peaks: Raw ch.1 (5/8+)" in labels
    assert "Peaks: Decimated SciPy ch.1 (2/4+)" in labels

    # 2. Zooming in to [0, 100] updates the visible count immediately
    app_window.plot_widget.getViewBox().setXRange(0.0, 100.0, padding=0)
    labels = _legend_texts(app_window)
    assert "Peaks: Raw ch.1 (3/8+)" in labels
    assert "Peaks: Decimated SciPy ch.1 (1/4+)" in labels

    # 3. Raw function finishes counting (total 200), while Decimated SciPy is still counting
    count_id = app_window._peak_count_request_id
    app_window._on_total_peak_count_progress(count_id, {("Raw", 0): 200}, True)
    app_window._on_total_peak_count_progress(count_id, {("Decimated SciPy", 0): 90}, False)

    labels = _legend_texts(app_window)
    assert "Peaks: Raw ch.1 (3/200)" in labels
    assert "Peaks: Decimated SciPy ch.1 (1/90+)" in labels
    assert app_window.plot_widget.getPlotItem().titleLabel.text == "File duration: 1:00:00, Counting peaks.."

    # 4. Whole-file counting finishes for all functions -> '+' removed and Functions shown in title
    app_window._on_total_peak_count_finished(
        (count_id, {("Raw", 0): 200, ("Decimated SciPy", 0): 150})
    )
    labels = _legend_texts(app_window)
    assert "Peaks: Raw ch.1 (3/200)" in labels
    assert "Peaks: Decimated SciPy ch.1 (1/150)" in labels
    assert (
        app_window.plot_widget.getPlotItem().titleLabel.text
        == "File duration: 1:00:00, Functions: Raw signal with peaks, SciPy decimation"
    )


def test_count_total_peaks_runs_per_function_across_chunks(app_window):
    from peak_finder import ANALYSIS_FUNCTIONS

    raw_fn = ANALYSIS_FUNCTIONS["Raw signal with peaks"]
    sample_rate = 1000.0
    total_samples = 6000  # 6.0 s total
    t_full = np.arange(total_samples) / sample_rate
    sig = np.zeros((total_samples, 1))
    # 4 sharp spikes > prominence 40 at 0.5s, 1.5s, 3.5s, 5.0s
    for peak_t in (0.5, 1.5, 3.5, 5.0):
        sig[int(peak_t * sample_rate), 0] = 100.0

    def get_slice(start_sec, end_sec):
        i0 = max(0, int(round(start_sec * sample_rate)))
        i1 = min(total_samples, int(round(end_sec * sample_rate)))
        return Slice(t_full[i0:i1], sig[i0:i1])

    reader = SimpleNamespace(
        total_samples=total_samples,
        sample_rate=sample_rate,
        data=sig,
        get_slice=get_slice,
    )

    progress_events = []
    app_window._peak_count_progress.connect(
        lambda cid, counts, done: progress_events.append((cid, dict(counts), done))
    )

    count_id = app_window._peak_count_request_id
    # Assume [0, 2.0) already counted 2 peaks; count remaining [2.0, 6.0] in 2.0s chunks
    cid_out, totals = app_window._count_total_peaks(
        count_id,
        reader,
        2.0,
        {("Raw", 0): 2},
        "Channel 1",
        (raw_fn,),
        {raw_fn.key: raw_fn.defaults},
        2.0,
    )
    QApplication.processEvents()

    assert cid_out == count_id
    assert totals == {("Raw", 0): 4}
    assert any(done and counts == {("Raw", 0): 4} for _, counts, done in progress_events)


