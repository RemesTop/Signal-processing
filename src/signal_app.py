import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QToolButton, QFileDialog, QMenuBar, QMessageBox
)
from PySide6.QtCore import Qt, QSettings, QThreadPool, QTimer, Signal
import numpy as np
import pyqtgraph as pg
from signal_reader import SignalReader, convert_csv_files
from worker import Worker
from visualize_data import (
    PeakKey, TimeAxisItem, plot_signal, set_plot_downsampling, update_peak_legend_labels
)
from signal_classes import AnalysisResult, Line, PeakSet, Slice
from functions_parameters_dialog import FunctionsParametersDialog
from peak_finder import ANALYSIS_FUNCTIONS, AnalysisFunction


# Main app window for the project
class SignalApp(QWidget):
    """Main app window."""

    _peak_count_progress = Signal(int, object, bool)
    
    # Analysis function selected by default if there are no saved selections
    DEFAULT_ANALYSIS = ANALYSIS_FUNCTIONS["Raw signal with peaks"]
    # Default length of the visible slice in seconds (loaded slice is 2x for off-screen buffering)
    DEFAULT_VISIBLE_SLICE_SECONDS = 240
    # Start prefetching next slice when visible edge is within 20% of the loaded slice boundary
    PREFETCH_MARGIN_RATIO = 0.2
    # Extend each analyzed slice on both sides to avoid boundary artifacts, then trim back
    SLICE_OVERLAP_SECONDS = 1.0
    # Step size in seconds for the quick jump arrow buttons (10 minutes)
    JUMP_STEP_SECONDS = 600.0
    
    def __init__(self):
        super().__init__()
        # Initialize stylesheet
        try:
            stylesheet_path = Path(__file__).resolve().with_name("style.qss")
            self.setStyleSheet(stylesheet_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"Stylesheet file {stylesheet_path} not found")
            
        self.setWindowTitle("Signal app")
        self.resize(1200, 700)

        # Options are saved to settings.ini and persist when restarting the app
        settings_file = Path(__file__).resolve().parent / "settings.ini"
        self.settings = QSettings(str(settings_file), QSettings.IniFormat)

        # Analysis functions that are run for each slice
        self.active_analyses = (self.DEFAULT_ANALYSIS,)
        # Editable parameters for each analysis function
        self.analysis_parameters = {analysis.key: analysis.defaults for analysis in self.active_analyses}
        # General slicing and display options
        self.visible_slice_seconds = self.DEFAULT_VISIBLE_SLICE_SECONDS
        self.remove_invisible_points = True
        # Dialog for editing the parameters
        self.parameters_dialog = FunctionsParametersDialog(
            tuple(ANALYSIS_FUNCTIONS.values()),
            self.active_analyses,
            self,
            default_visible_slice_seconds=self.visible_slice_seconds,
            default_remove_invisible_points=self.remove_invisible_points,
        )
        self.parameters_dialog.changes_applied.connect(self.update_analysis_functions)
        self.parameters_dialog.closed.connect(lambda: self.action_param_window.setChecked(False))


        # Menu bar
        self.menu_bar = QMenuBar()
        
        file_menu = self.menu_bar.addMenu("File")
        self.action_convert_files = file_menu.addAction("Convert CSV files to DAT")
        self.action_convert_files.triggered.connect(self.convert_files)
        
        view_menu = self.menu_bar.addMenu("View")
        self.action_param_window = view_menu.addAction("Analysis functions and parameters")
        self.action_param_window.setCheckable(True)
        self.action_param_window.setChecked(False)
        self.action_param_window.toggled.connect(self.parameters_dialog.setVisible)
        

        # 1. Text input for file path and browse button
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("File path")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)

        # 2. Dropdown for channel selection (restricting channels improves performance)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Both channels", "Channel 1", "Channel 2"])
        self.channel_combo.setCurrentText("Both channels")
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)

        # 3. Visualize Data button (When the button is pressed)
        self.btn = QPushButton("Run")
        self.btn.setObjectName("runButton")
        self.btn.clicked.connect(self.visualize)

        # 4. Graph jump navigation controls (hidden until a graph is displayed)
        self.jump_controls = QWidget()
        jump_layout = QHBoxLayout(self.jump_controls)
        jump_layout.setContentsMargins(0, 0, 0, 0)
        jump_layout.setSpacing(4)

        self.jump_label = QLabel("Jump to:")
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("0:00")
        self.jump_input.setFixedWidth(75)
        self.jump_input.setToolTip("Enter target time (m:ss, h:mm:ss, or seconds) and press Enter")
        self.jump_input.returnPressed.connect(self._on_jump_entered)

        self.jump_back_btn = QToolButton()
        self.jump_back_btn.setArrowType(Qt.ArrowType.LeftArrow)
        self.jump_back_btn.setToolTip("Jump 10 minutes backward")
        self.jump_back_btn.setAccessibleName("Jump 10 minutes backward")
        self.jump_back_btn.clicked.connect(lambda: self.jump_by(-self.JUMP_STEP_SECONDS))

        self.jump_forward_btn = QToolButton()
        self.jump_forward_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.jump_forward_btn.setToolTip("Jump 10 minutes forward")
        self.jump_forward_btn.setAccessibleName("Jump 10 minutes forward")
        self.jump_forward_btn.clicked.connect(lambda: self.jump_by(self.JUMP_STEP_SECONDS))

        jump_layout.addWidget(self.jump_label)
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_back_btn)
        jump_layout.addWidget(self.jump_forward_btn)
        jump_size_policy = self.jump_controls.sizePolicy()
        jump_size_policy.setRetainSizeWhenHidden(True)
        self.jump_controls.setSizePolicy(jump_size_policy)
        self.jump_controls.setVisible(False)

        # Plot widget
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("bottom", "Time")
        self.plot_widget.setLabel("left", "Reading")
        self.plot_widget.setTitle(" ")
        view_box: pg.ViewBox = self.plot_widget.getViewBox()
        # Set grid opacity
        self.plot_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        # State for slice-based loading and whole-file peak counting on large files
        self.loaded_range: tuple[float, float] | None = None
        self._analysis_request_id = 0
        self._analysis_running = False
        self._analysis_worker = None
        self.total_peak_counts: dict[PeakKey, int] = {}
        self.peak_counts_so_far: dict[PeakKey, int] = {}
        self.peaks_counting = False
        self._peak_count_completed = False
        self._peak_count_request_id = 1
        self._peak_count_started_id = 0
        self._peak_count_worker = None
        self._peak_count_progress.connect(self._on_total_peak_count_progress)
        self.pan_timer = QTimer(self)
        self.pan_timer.setSingleShot(True)
        self.pan_timer.setInterval(50)
        self.pan_timer.timeout.connect(self._reload_visible_slice)

        # Keep y-axis fixed while panning x; prefetch slices on large files
        view_box.setMouseEnabled(x=True, y=False)
        view_box.sigXRangeChanged.connect(self._on_x_range_changed)

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_input, 1)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.btn)
        top_layout.addSpacing(12)
        top_layout.addWidget(self.jump_controls)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menu_bar)
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_widget)

        # Restore saved settings on startup
        self.load_settings()

    def _slice_reload_needed(self, x_range) -> bool:
        """Return whether the given visible X range requires loading a new slice."""
        if (
            not hasattr(self, "reader")
            or self.reader is None
            or self.reader.data is None
            or self.loaded_range is None
        ):
            return False

        total_duration = self.reader.total_samples / self.reader.sample_rate
        loaded_window_sec = self.visible_slice_seconds * 2.0
        if total_duration <= loaded_window_sec + 1e-9:
            return False

        window_sec = min(total_duration, loaded_window_sec)
        margin = window_sec * self.PREFETCH_MARGIN_RATIO
        view_min = max(0.0, x_range[0])
        view_max = min(total_duration, x_range[1])
        loaded_start, loaded_end = self.loaded_range

        need_left = loaded_start > 1e-3 and view_min < loaded_start + margin
        need_right = loaded_end < total_duration - 1e-3 and view_max > loaded_end - margin
        return need_left or need_right

    def _on_x_range_changed(self, _view_box: pg.ViewBox, x_range):
        """Schedule a slice prefetch before the user reaches the edge of the loaded range."""
        if self._slice_reload_needed(x_range) and not self.pan_timer.isActive():
            self.pan_timer.start()

    @staticmethod
    def _parse_jump_time(text: str) -> float:
        """Parse a time string (`m:ss`, `h:mm:ss`, or seconds) into seconds."""
        s = text.strip()
        if not s:
            raise ValueError("Enter a time! (eg. 5:00)")
        if s.endswith("s"):
            s = s[:-1].strip()
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                m_str, sec_str = parts
                minutes = int(m_str)
                secs = float(sec_str)
                if minutes < 0 or not (0.0 <= secs < 60.0):
                    raise ValueError(f"Invalid time '{text.strip()}'.")
                return minutes * 60.0 + secs
            if len(parts) == 3:
                h_str, m_str, sec_str = parts
                hours = int(h_str)
                minutes = int(m_str)
                secs = float(sec_str)
                if hours < 0 or not (0 <= minutes < 60) or not (0.0 <= secs < 60.0):
                    raise ValueError(f"Invalid time '{text.strip()}'.")
                return hours * 3600.0 + minutes * 60.0 + secs
            raise ValueError(f"Invalid time '{text.strip()}'.")
        seconds = float(s)
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid time '{text.strip()}'.")
        return seconds

    def _has_loaded_signal(self) -> bool:
        """Return whether a signal reader is currently loaded with data."""
        return (
            hasattr(self, "reader")
            and self.reader is not None
            and self.reader.data is not None
        )

    def _apply_view_x_range(self, new_x_min: float, new_x_max: float):
        """Move the X view range immediately and reload the slice if needed."""
        self.plot_widget.getViewBox().setXRange(new_x_min, new_x_max, padding=0)
        self.pan_timer.stop()
        if self._slice_reload_needed((new_x_min, new_x_max)):
            self._reload_visible_slice(force=True)

    def _on_jump_entered(self):
        """Jump to the timestamp entered in the jump input field."""
        if not self._has_loaded_signal():
            return
        try:
            target_sec = self._parse_jump_time(self.jump_input.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid jump time", str(exc))
            return
        self.jump_to(target_sec)

    def jump_to(self, target_sec: float):
        """Center the visible graph window around `target_sec` immediately."""
        if not self._has_loaded_signal():
            return
        total_duration = self.reader.total_samples / self.reader.sample_rate
        if total_duration <= 0:
            return
        clamped_target = max(0.0, min(float(target_sec), total_duration))
        x_min, x_max = self.plot_widget.getViewBox().viewRange()[0]
        span = min(total_duration, max(1e-6, x_max - x_min))
        new_x_min = max(0.0, min(clamped_target - span / 2.0, total_duration - span))
        self._apply_view_x_range(new_x_min, new_x_min + span)

    def jump_by(self, delta_sec: float):
        """Shift the visible graph window by `delta_sec` seconds immediately."""
        if not self._has_loaded_signal():
            return
        total_duration = self.reader.total_samples / self.reader.sample_rate
        if total_duration <= 0:
            return
        x_min, x_max = self.plot_widget.getViewBox().viewRange()[0]
        span = min(total_duration, max(1e-6, x_max - x_min))
        new_x_min = max(0.0, min(x_min + delta_sec, total_duration - span))
        if abs(new_x_min - x_min) < 1e-9:
            return
        self._apply_view_x_range(new_x_min, new_x_min + span)

    def _reload_visible_slice(
        self,
        force: bool = False,
        reset_y_range: bool = False,
        fit_x_range: bool = False,
        show_loading: bool = False,
    ):
        """Load a new slice centered on the current X view range."""
        if (
            (self._analysis_running and not force)
            or not hasattr(self, "reader")
            or self.reader is None
            or self.reader.data is None
        ):
            return
        total_duration = self.reader.total_samples / self.reader.sample_rate
        window_sec = min(total_duration, self.visible_slice_seconds * 2.0)
        x_min, x_max = self.plot_widget.getViewBox().viewRange()[0]
        center = (x_min + x_max) / 2
        start_sec = max(0.0, min(center - window_sec / 2, total_duration - window_sec))
        end_sec = min(total_duration, start_sec + window_sec)
        self.plot_data(
            start_sec=start_sec,
            end_sec=end_sec,
            reset_x_range=False,
            reset_y_range=reset_y_range,
            fit_x_range=fit_x_range,
            show_loading=show_loading,
        )

    def load_settings(self):
        """Load the saved settings ini file."""
        saved_file = self.settings.value("file_path", "")
        if saved_file:
            self.file_input.setText(str(saved_file))

        saved_channel = self.settings.value("channel_selection", "Both channels")
        if saved_channel in ["Both channels", "Channel 1", "Channel 2"]:
            self.channel_combo.setCurrentText(str(saved_channel))
        else:
            self.channel_combo.setCurrentText("Both channels")

        # Selected analysis functions and their parameters
        saved_analyses = self.settings.value("analysis_functions", "")
        analyses = ()
        parameters = {}
        if saved_analyses:
            # The setting is stored as JSON
            try:
                functions_json = json.loads(str(saved_analyses))
            except json.JSONDecodeError as exc:
                QMessageBox.warning(self, "Invalid saved analysis settings",
                                    f"Could not load saved analysis functions and parameters:\n{exc}")
                functions_json = {}

            if not isinstance(functions_json, dict):
                functions_json = {}
                
            # Saved functions
            functions_keys = functions_json.get("functions", [])
            if not isinstance(functions_keys, list):
                functions_keys = []
            analyses = tuple(
                analysis
                for _, analysis in ANALYSIS_FUNCTIONS.items()
                if analysis.key in functions_keys
            )
            
            # Saved parameters
            parameters = functions_json.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            parameters = {
                key: value
                for key, value in parameters.items()
                if isinstance(value, dict)
            }

        if not analyses:
            analyses = (self.DEFAULT_ANALYSIS,)
        
        # General slicing and display options
        saved_visible_seconds = self.settings.value(
            "visible_slice_seconds", self.DEFAULT_VISIBLE_SLICE_SECONDS
        )
        try:
            visible_slice_seconds = int(saved_visible_seconds)
            if visible_slice_seconds <= 0:
                visible_slice_seconds = self.DEFAULT_VISIBLE_SLICE_SECONDS
        except (TypeError, ValueError):
            visible_slice_seconds = self.DEFAULT_VISIBLE_SLICE_SECONDS
        self.visible_slice_seconds = visible_slice_seconds

        saved_remove_invisible = self.settings.value("remove_invisible_points", True)
        if isinstance(saved_remove_invisible, bool):
            self.remove_invisible_points = saved_remove_invisible
        else:
            self.remove_invisible_points = (
                str(saved_remove_invisible).strip().lower() not in ("false", "0", "no")
            )
        self.parameters_dialog.set_general_options(
            self.visible_slice_seconds,
            self.remove_invisible_points,
        )

        # Populate the dialog
        self.parameters_dialog.set_selected_functions_and_parameters(analyses, parameters)
        self.active_analyses = analyses
        # First set the default parameters
        self.analysis_parameters = {analysis.key: analysis.defaults for analysis in analyses}
        # Then use the dialog to parse the saved string values
        try:
            self.active_analyses, self.analysis_parameters = self.parameters_dialog.get_selected_functions_and_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid saved analysis settings",
                                f"Could not apply saved analysis parameters:\n{exc}")

    def save_settings(self):
        """Save recently used signal processing configurations into an ini file.
        Doesn't include saving the selected analysis functions and parameters.
        For those, use `save_analysis_functions_and_parameters`.
        """
        self.settings.setValue("file_path", self.file_input.text())
        self.settings.setValue("channel_selection", self.channel_combo.currentText())
        self.settings.sync()

    def save_analysis_functions_and_parameters(self, evaluate: bool = True):
        """Save selected analysis functions, parameters, and general options into an ini file.
        If `evaluate` is true, the parameters are first evaluated and if there is an
        invalid value, the saving is aborted and a message box is shown.
        """
        if evaluate:
            try:
                self.parameters_dialog.get_general_options()
                self.parameters_dialog.get_selected_functions_and_parameters()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid analysis parameter", str(exc))
                return
        
        # Get the text values of the parameters for saving
        functions, parameters = self.parameters_dialog.get_selected_functions_and_parameter_texts()
        configuration = {
            "functions": [analysis.key for analysis in functions],
            "parameters": parameters,
        }
        # Stored as JSON
        self.settings.setValue("analysis_functions", json.dumps(configuration))
        self.settings.setValue("visible_slice_seconds", self.visible_slice_seconds)
        self.settings.setValue("remove_invisible_points", self.remove_invisible_points)
        self.settings.sync()

    def on_channel_changed(self):
        """When selecting displayed channel(s), plot the data if it has been read before."""
        if hasattr(self, "reader") and self.reader is not None and self.reader.data is not None:
            self._invalidate_total_peak_count()
            if self.loaded_range is not None:
                self.plot_data(*self.loaded_range, reset_x_range=False)
            else:
                self.plot_data()

    def closeEvent(self, event):
        """Save settings on closing the application."""
        self.save_settings()
        super().closeEvent(event)

    def browse_file(self):
        """Open a file selection dialog and populate the file input field with the selected file's name."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file",
            "",
            "Data File (*.csv *.dat);;All Files (*)"
        )
        if file_path:
            self.file_input.setText(file_path)

    def visualize(self):
        """Read, convert and plot the selected file and save the settings."""
        self.save_settings()
        file_path = self.file_input.text().strip()
        if not file_path:
            return

        self.pan_timer.stop()
        self.loaded_range = None
        self._invalidate_total_peak_count()
        self.btn.setEnabled(False)
        self.reader = SignalReader(file_path, multithreading=True)

        # If data is already converted, plot it
        if self.reader.data is not None:
            self.plot_data()
        else:
            # Wait for background conversion to finish
            self.plot_widget.setTitle("Converting csv to binary, please wait")
            self.reader.progress.connect(self.update_progress)
            self.reader.ready.connect(self.plot_data)

    def update_progress(self, percent):
        """Update csv conversion progress label."""
        self.plot_widget.setTitle(f"Converting csv to binary, please wait {percent:.0f}%")

    def _channel_indices_for(self, channel_choice: str) -> tuple[int, ...]:
        """Map the channel dropdown selection to channel index tuple."""
        if channel_choice == "Channel 1":
            return (0,)
        if channel_choice == "Channel 2":
            return (1,)
        return (0, 1)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds as m:ss (or h:mm:ss)."""
        total_secs = int(round(seconds))
        hours, rem = divmod(total_secs, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    def _update_plot_title(self):
        """Update the plot title with total file duration and peak counting / active functions."""
        total_duration = self.reader.total_samples / self.reader.sample_rate
        duration_str = self._format_duration(total_duration)
        if self.peaks_counting:
            self.plot_widget.setTitle(f"File duration: {duration_str}, Counting peaks..")
        else:
            analysis_names = ", ".join(analysis.title for analysis in self.active_analyses)
            self.plot_widget.setTitle(
                f"File duration: {duration_str}, Functions: {analysis_names}"
            )

    def _displayed_peak_totals(self) -> dict[PeakKey, int]:
        """Return current per-function/channel peak totals (live or finalized)."""
        return dict(self.peak_counts_so_far) | dict(self.total_peak_counts)

    def _current_loading_peak_keys(self, peak_sets=()) -> set[PeakKey]:
        """Return `(label, channel)` keys whose whole-file peak counts are still loading."""
        if not self.peaks_counting:
            return set()
        known_keys = set(self.peak_counts_so_far.keys()) | {
            (ps.label, ps.channel) for ps in peak_sets
        }
        return {key for key in known_keys if key not in self.total_peak_counts}

    def _refresh_peak_legend_labels(self):
        """Push latest peak totals and loading flags to the plot legend."""
        entries = getattr(self.plot_widget, "_peak_legend_entries", ())
        peak_sets = [ps for _, ps in entries]
        update_peak_legend_labels(
            self.plot_widget,
            peak_totals=self._displayed_peak_totals(),
            loading_keys=self._current_loading_peak_keys(peak_sets),
        )

    def _invalidate_total_peak_count(self):
        """Reset cached whole-file peak counts and cancel any in-flight counting worker."""
        self.total_peak_counts.clear()
        self.peak_counts_so_far.clear()
        self.peaks_counting = False
        self._peak_count_completed = False
        self._peak_count_request_id += 1

    def _analyze_slice(
        self, reader, start_sec, end_sec, channel_indices, analyses, parameters,
    ) -> AnalysisResult:
        """Analyze `[start_sec, end_sec]` using 1 s overlap on both sides, then trim results back."""
        total_duration = reader.total_samples / reader.sample_rate
        padded_start = max(0.0, start_sec - self.SLICE_OVERLAP_SECONDS)
        padded_end = min(total_duration, end_sec + self.SLICE_OVERLAP_SECONDS)

        raw_slice = reader.get_slice(padded_start, padded_end)
        padded_slice = Slice(raw_slice.time, raw_slice.data, channel_indices=channel_indices)
        result = AnalysisResult(padded_slice)
        for analysis in analyses:
            analysis.function(
                padded_slice,
                result,
                **analysis.parameters_for_call(parameters[analysis.key])
            )

        if padded_start == start_sec and padded_end == end_sec:
            return result

        # Trim slice, lines, and peaks back to [start_sec, end_sec) (closed on the final chunk)
        is_last = end_sec >= total_duration - 1e-9
        slice_mask = (padded_slice.time >= start_sec) & (padded_slice.time <= end_sec)
        trimmed_slice = Slice(
            padded_slice.time[slice_mask],
            padded_slice.data[slice_mask],
            channel_indices=channel_indices,
        )
        trimmed_lines = []
        for line in result.lines:
            mask = (line.x >= start_sec) & (line.x <= end_sec)
            trimmed_lines.append(Line(line.x[mask], line.y[mask], line.channel, line.label))

        trimmed_peaks = []
        for peak_set in result.peak_sets:
            upper_mask = (peak_set.x <= end_sec) if is_last else (peak_set.x < end_sec)
            mask = (peak_set.x >= start_sec) & upper_mask
            trimmed_props = {k: np.asarray(v)[mask] for k, v in peak_set.properties.items()}
            trimmed_peaks.append(
                PeakSet(peak_set.x[mask], peak_set.y[mask], peak_set.channel, peak_set.label, trimmed_props)
            )

        return AnalysisResult(trimmed_slice, lines=trimmed_lines, peak_sets=trimmed_peaks)

    def _count_total_peaks(
        self, count_id, reader, start_from_sec, initial_counts,
        channel_choice, analyses, parameters, chunk_sec,
    ):
        """Count total peaks per function across the entire file in bounded chunks."""
        total_duration = reader.total_samples / reader.sample_rate
        channel_indices = self._channel_indices_for(channel_choice)
        all_counts: dict[PeakKey, int] = dict(initial_counts)

        for analysis in analyses:
            if count_id != self._peak_count_request_id:
                return count_id, None
            analysis_counts: dict[PeakKey, int] = {}
            t = start_from_sec
            while t < total_duration - 1e-9:
                if count_id != self._peak_count_request_id:
                    return count_id, None
                end = min(total_duration, t + chunk_sec)
                result = self._analyze_slice(reader, t, end, channel_indices, (analysis,), parameters)
                for peak_set in result.peak_sets:
                    key = (peak_set.label, peak_set.channel)
                    if key not in analysis_counts:
                        analysis_counts[key] = initial_counts.get(key, 0)
                    analysis_counts[key] += len(peak_set.x)
                if analysis_counts:
                    self._peak_count_progress.emit(count_id, dict(analysis_counts), False)
                t = end

            all_counts.update(analysis_counts)
            if analysis_counts:
                self._peak_count_progress.emit(count_id, dict(analysis_counts), True)

        return count_id, all_counts

    def _on_total_peak_count_progress(self, count_id: int, counts: dict, analysis_finished: bool):
        """Update live per-function peak counts in the graph legend as chunks/functions finish."""
        if count_id != self._peak_count_request_id or self._peak_count_completed:
            return
        for key, val in counts.items():
            self.peak_counts_so_far[key] = max(self.peak_counts_so_far.get(key, 0), int(val))
            if analysis_finished:
                self.total_peak_counts[key] = int(val)
        self._refresh_peak_legend_labels()

    def _on_total_peak_count_finished(self, payload):
        """Finalize whole-file peak counts and show active functions in the plot title."""
        count_id, total_counts = payload
        if count_id != self._peak_count_request_id or total_counts is None:
            return
        for key, val in total_counts.items():
            self.total_peak_counts[key] = int(val)
            self.peak_counts_so_far[key] = int(val)
        self.peaks_counting = False
        self._peak_count_completed = True
        self._peak_count_worker = None
        self._update_plot_title()
        self._refresh_peak_legend_labels()

    def _on_total_peak_count_error(self, _err):
        """Clear counting state if background peak counting fails."""
        self.peaks_counting = False
        self._peak_count_worker = None
        if self._has_loaded_signal():
            self._update_plot_title()
            self._refresh_peak_legend_labels()

    def _run_analysis(
        self, request_id, start_sec, end_sec, reset_x_range, reset_y_range,
        fit_x_range, channel_choice, parameters,
    ):
        """Run the selected signal analysis functions for a slice of the current `SignalReader`'s data."""
        # Background worker task: slices data (with boundary overlap) and runs signal analysis
        channel_indices = self._channel_indices_for(channel_choice)
        result = self._analyze_slice(
            self.reader, start_sec, end_sec, channel_indices, self.active_analyses, parameters
        )
        return request_id, result, (start_sec, end_sec), reset_x_range, reset_y_range, fit_x_range

    def plot_data(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
        reset_x_range: bool = True,
        reset_y_range: bool = True,
        fit_x_range: bool = False,
        show_loading: bool | None = None,
    ):
        """Run signal analysis and plotting in a background thread."""
        channel_choice = self.channel_combo.currentText()
        parameters = self.analysis_parameters

        # Cap loaded slice length to 2x visible slice length for off-screen buffering
        total_duration = self.reader.total_samples / self.reader.sample_rate
        max_window_sec = min(total_duration, self.visible_slice_seconds * 2.0)
        if start_sec is None or end_sec is None:
            start_sec = 0.0
            end_sec = max_window_sec

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._analysis_running = True

        if show_loading is None:
            show_loading = reset_x_range
        if show_loading:
            self.btn.setEnabled(False)
            self.plot_widget.setTitle("Analyzing signal, please wait...")

        # Run the peak analysis in a background worker thread to keep GUI responsive
        worker = Worker(
            self._run_analysis,
            request_id, start_sec, end_sec, reset_x_range, reset_y_range,
            fit_x_range, channel_choice, parameters,
        )
        worker.signals.result.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_analysis_error)
        self._analysis_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_analysis_finished(self, payload):
        """Plot the signal and update the peak label and plot title."""
        request_id, result, loaded_range, reset_x_range, reset_y_range, fit_x_range = payload
        # Ignore results from outdated pan/analysis requests
        if request_id != self._analysis_request_id:
            return
        self._analysis_running = False
        self._analysis_worker = None
        self.loaded_range = loaded_range

        slice_counts = {(ps.label, ps.channel): len(ps.x) for ps in result.peak_sets}
        total_duration = self.reader.total_samples / self.reader.sample_rate
        visible_sec = float(self.visible_slice_seconds)
        loaded_window_sec = visible_sec * 2.0
        count_worker = None

        if total_duration <= loaded_window_sec + 1e-9:
            self.total_peak_counts = dict(slice_counts)
            self.peak_counts_so_far = dict(slice_counts)
            self.peaks_counting = False
            self._peak_count_completed = True
        else:
            for key, count in slice_counts.items():
                self.peak_counts_so_far[key] = max(self.peak_counts_so_far.get(key, 0), count)
            if not self._peak_count_completed and self._peak_count_started_id != self._peak_count_request_id:
                if not self.active_analyses:
                    self.peaks_counting = False
                    self._peak_count_completed = True
                else:
                    count_id = self._peak_count_request_id
                    self._peak_count_started_id = count_id
                    self.peaks_counting = True
                    start_from = loaded_range[1] if loaded_range[0] <= 1e-9 else 0.0
                    initial_counts = dict(slice_counts) if loaded_range[0] <= 1e-9 else {}
                    count_worker = Worker(
                        self._count_total_peaks,
                        count_id,
                        self.reader,
                        start_from,
                        initial_counts,
                        self.channel_combo.currentText(),
                        self.active_analyses,
                        {k: dict(v) for k, v in self.analysis_parameters.items()},
                        loaded_window_sec,
                    )
                    count_worker.signals.result.connect(self._on_total_peak_count_finished)
                    count_worker.signals.error.connect(self._on_total_peak_count_error)
                    self._peak_count_worker = count_worker

        self._update_plot_title()
        # Show the parameter editing dialog on initial load
        if reset_x_range:
            self.action_param_window.setChecked(True)

        x_limits = (0.0, total_duration) if total_duration > 0 else None
        max_x_range = min(total_duration, visible_sec) if total_duration > 0 else None

        plot_signal(
            self.plot_widget,
            result,
            x_limits=x_limits,
            max_x_range=max_x_range,
            reset_x_range=reset_x_range,
            reset_y_range=reset_y_range,
            remove_invisible_points=self.remove_invisible_points,
            fit_x_range=fit_x_range,
            peak_totals=self._displayed_peak_totals(),
            loading_peak_keys=self._current_loading_peak_keys(result.peak_sets),
        )
        if count_worker is not None:
            QThreadPool.globalInstance().start(count_worker)
        self.jump_controls.setVisible(True)
        self.btn.setEnabled(True)

        # Re-check in case the user panned further while the worker was running
        view_box = self.plot_widget.getViewBox()
        self._on_x_range_changed(view_box, view_box.viewRange()[0])

    def _on_analysis_error(self, err):
        """Show an error message for an error in signal analysis."""
        self._analysis_running = False
        self.plot_widget.setTitle("Peaks: unavailable.")
        QMessageBox.critical(self, "Error in signal analysis", f"{err[0].__name__}: {err[1]}")
        self.btn.setEnabled(True)

    def update_analysis_functions(
        self,
        functions: tuple[AnalysisFunction, ...],
        parameters: dict[str, dict],
        visible_slice_seconds: int,
        remove_invisible_points: bool,
    ):
        """Update and save selected analysis functions, parameters, and general options,
        then update the plot if data is available.
        """
        new_parameters = {
            analysis.key: dict(parameters.get(analysis.key, analysis.defaults))
            for analysis in functions
        }
        analyses_changed = (
            functions != self.active_analyses
            or new_parameters != self.analysis_parameters
        )
        visible_slice_changed = visible_slice_seconds != self.visible_slice_seconds

        self.active_analyses = functions
        self.analysis_parameters = new_parameters
        self.visible_slice_seconds = visible_slice_seconds
        self.remove_invisible_points = remove_invisible_points

        # Save selected functions, parameters, and general options
        self.save_analysis_functions_and_parameters(evaluate=False)

        # Update plot if reader and data are available
        if hasattr(self, "reader") and self.reader is not None and self.reader.data is not None:
            if analyses_changed or visible_slice_changed:
                self._invalidate_total_peak_count()
                if visible_slice_changed and self.loaded_range is not None:
                    self._reload_visible_slice(
                        force=True,
                        reset_y_range=analyses_changed,
                        fit_x_range=True,
                        show_loading=True,
                    )
                elif self.loaded_range is not None:
                    self.plot_data(
                        *self.loaded_range,
                        reset_x_range=False,
                        show_loading=True,
                    )
                else:
                    self.plot_data()
            else:
                set_plot_downsampling(self.plot_widget, self.remove_invisible_points)
    
    
    # CSV to DAT conversion
    
    def convert_files(self):
        """Convert selected CSV files to DAT without loading or plotting them."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV files to convert",
            "",
            "CSV files (*.csv)",
        )
        if not file_paths:
            return

        self.action_convert_files.setEnabled(False)
        
        # Convert in a background worker
        worker = Worker(convert_csv_files, file_paths)
        worker.signals.result.connect(self._on_batch_conversion_finished)
        worker.signals.error.connect(self._on_batch_conversion_error)
        worker.signals.finished.connect(self._on_batch_conversion_worker_finished)
        self.batch_conversion_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_batch_conversion_finished(self, converted_paths):
        """Show a message box saying how many CSV files were converted."""
        count = len(converted_paths)
        if count:
            QMessageBox.information(
                self,
                "CSV conversion complete",
                f"Converted {count} CSV file(s) to DAT.",
            )
        else:
            QMessageBox.information(
                self,
                "CSV conversion complete",
                "All selected CSV files already have DAT files.",
            )

    def _on_batch_conversion_error(self, error):
        """Show a message box for an error during batch conversion."""
        self.plot_widget.setTitle("CSV conversion failed")
        QMessageBox.critical(
            self,
            "CSV conversion failed",
            f"{error[0].__name__}: {error[1]}",
        )

    def _on_batch_conversion_worker_finished(self):
        """Re-enable batch conversion menu item after the worker is finished."""
        self.action_convert_files.setEnabled(True)
        self.batch_conversion_worker = None

def main():
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
