import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QPushButton, QSpinBox, QFileDialog, QMenuBar, QMessageBox, QLabel
)
from PySide6.QtCore import QSettings, QThreadPool, QTimer
import pyqtgraph as pg
from signal_reader import SignalReader, convert_csv_files
from worker import Worker
from visualize_data import plot_signal
from signal_classes import AnalysisResult, Slice
from functions_parameters_dialog import FunctionsParametersDialog
from peak_finder import ANALYSIS_FUNCTIONS, AnalysisFunction


# Main app window for the project
class SignalApp(QWidget):
    """Main app window."""
    
    # Analysis function selected by default if there are no saved selections
    DEFAULT_ANALYSIS = ANALYSIS_FUNCTIONS["Raw signal with peaks"]
    # Maximum raw samples loaded and analyzed per slice on large files
    MAX_SLICE_POINTS = 20_000_000
    # Start prefetching next slice when visible edge is within 20% of the loaded slice boundary
    PREFETCH_MARGIN_RATIO = 0.2
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal app")
        self.resize(1200, 700)

        # Options are saved to settings.ini and persist when restarting the app
        settings_file = Path(__file__).resolve().parent / "settings.ini"
        self.settings = QSettings(str(settings_file), QSettings.IniFormat)

        # Analysis functions that are run for each slice
        self.active_analyses = (self.DEFAULT_ANALYSIS,)
        # Editable parameters for each analysis function
        self.analysis_parameters = {analysis.key: analysis.defaults for analysis in self.active_analyses}
        # Dialog for editing the parameters
        self.parameters_dialog = FunctionsParametersDialog(tuple(ANALYSIS_FUNCTIONS.values()), self.active_analyses, self)
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

        # 2. Dropdown for downsample method
        self.combo = QComboBox()
        self.combo.addItems(["max points", "decimation", "none"])

        # 3. Number input for downsample factor
        self.factor_input = QSpinBox()
        self.factor_input.setRange(1, 100000)
        self.factor_input.setValue(5)

        # 4. Dropdown for channel selection (restricting channels improves performance)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Both channels", "Channel 1", "Channel 2"])
        self.channel_combo.setCurrentText("Both channels")
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)

        # 5. Visualize Data button (When the button is pressed)
        self.btn = QPushButton("Run")
        self.btn.clicked.connect(self.visualize)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Reading")
        self.plot_widget.setTitle(" ")
        view_box: pg.ViewBox = self.plot_widget.getViewBox()
        # State for slice-based loading and whole-file peak counting on large files
        self.loaded_range: tuple[float, float] | None = None
        self._analysis_request_id = 0
        self._analysis_running = False
        self.total_peak_count: int | None = None
        self._peak_count_request_id = 1
        self._peak_count_started_id = 0
        self.pan_timer = QTimer(self)
        self.pan_timer.setSingleShot(True)
        self.pan_timer.setInterval(50)
        self.pan_timer.timeout.connect(self._reload_visible_slice)

        # Keep y-axis fixed while panning x; prefetch slices on large files
        view_box.setMouseEnabled(x=True, y=False)
        view_box.sigXRangeChanged.connect(self._on_x_range_changed)
        self.factor_input.setPrefix("Factor: ")

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.combo)
        top_layout.addWidget(self.factor_input)
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.btn)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menu_bar)
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_widget)

        # Restore saved settings on startup
        self.load_settings()

    def _on_x_range_changed(self, _view_box: pg.ViewBox, x_range):
        """Schedule a slice prefetch before the user reaches the edge of the loaded range."""
        if (
            not hasattr(self, "reader")
            or self.reader is None
            or self.reader.data is None
            or self.loaded_range is None
            or self.reader.total_samples <= self.MAX_SLICE_POINTS
        ):
            return

        total_duration = self.reader.total_samples / self.reader.sample_rate
        window_sec = min(total_duration, self.MAX_SLICE_POINTS / self.reader.sample_rate)
        margin = window_sec * self.PREFETCH_MARGIN_RATIO
        view_min = max(0.0, x_range[0])
        view_max = min(total_duration, x_range[1])
        loaded_start, loaded_end = self.loaded_range

        # Prefetch early while buffered data is still ahead so the graph stays connected
        need_left = loaded_start > 1e-3 and view_min < loaded_start + margin
        need_right = loaded_end < total_duration - 1e-3 and view_max > loaded_end - margin
        if (need_left or need_right) and not self.pan_timer.isActive():
            self.pan_timer.start()

    def _reload_visible_slice(self):
        """Load a new slice centered on the current X view range."""
        if (
            self._analysis_running
            or not hasattr(self, "reader")
            or self.reader is None
            or self.reader.data is None
        ):
            return
        total_duration = self.reader.total_samples / self.reader.sample_rate
        window_sec = min(total_duration, self.MAX_SLICE_POINTS / self.reader.sample_rate)
        x_min, x_max = self.plot_widget.getViewBox().viewRange()[0]
        center = (x_min + x_max) / 2
        start_sec = max(0.0, min(center - window_sec / 2, total_duration - window_sec))
        end_sec = min(total_duration, start_sec + window_sec)
        self.plot_data(start_sec=start_sec, end_sec=end_sec, reset_x_range=False, reset_y_range=False)

    def load_settings(self):
        """Load the saved settings ini file."""
        saved_file = self.settings.value("file_path", "")
        if saved_file:
            self.file_input.setText(str(saved_file))

        saved_method = self.settings.value("downsample_method", "")
        if saved_method:
            self.combo.setCurrentText(str(saved_method))

        saved_factor = self.settings.value("downsample_factor", 2)
        self.factor_input.setValue(int(saved_factor))

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
        self.settings.setValue("downsample_method", self.combo.currentText())
        self.settings.setValue("downsample_factor", self.factor_input.value())
        self.settings.setValue("channel_selection", self.channel_combo.currentText())
        self.settings.sync()

    def save_analysis_functions_and_parameters(self, evaluate: bool = True):
        """Save selected analysis functions and parameters into an ini file.
        If `evaluate` is true, the parameters are first evaluated and if there is an
        invalid value, the saving is aborted and a message box is shown.
        """
        if evaluate:
            try:
                self.parameters_dialog.get_selected_functions_and_parameters()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid analysis parameter", str(exc))
        
        # Get the text values of the parameters for saving
        functions, parameters = self.parameters_dialog.get_selected_functions_and_parameter_texts()
        configuration = {
            "functions": [analysis.key for analysis in functions],
            "parameters": parameters,
        }
        # Stored as JSON
        self.settings.setValue("analysis_functions", json.dumps(configuration))
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
        """Update the plot title with total file duration, total peaks, and active functions."""
        total_duration = self.reader.total_samples / self.reader.sample_rate
        duration_str = self._format_duration(total_duration)
        peaks_str = str(self.total_peak_count) if self.total_peak_count is not None else "counting..."
        analysis_names = ", ".join(analysis.title for analysis in self.active_analyses)
        self.plot_widget.setTitle(
            f"File duration: {duration_str}, Peaks: {peaks_str}, Functions: {analysis_names}"
        )

    def _invalidate_total_peak_count(self):
        """Reset cached whole-file peak count and cancel any in-flight counting worker."""
        self.total_peak_count = None
        self._peak_count_request_id += 1

    def _count_total_peaks(
        self, count_id, reader, start_from_sec, initial_count,
        method, factor, channel_choice, analyses, parameters,
    ):
        """Count total peaks across the entire file in bounded chunks."""
        total_duration = reader.total_samples / reader.sample_rate
        chunk_sec = self.MAX_SLICE_POINTS / reader.sample_rate
        channel_indices = self._channel_indices_for(channel_choice)
        total_peaks = initial_count
        t = start_from_sec

        while t < total_duration - 1e-9:
            if count_id != self._peak_count_request_id:
                return count_id, None
            end = min(total_duration, t + chunk_sec)
            signal_slice = reader.get_slice(t, end, downsample_factor=factor, downsample=method)
            signal_slice = Slice(signal_slice.time, signal_slice.data, channel_indices=channel_indices)
            result = AnalysisResult(signal_slice)
            for analysis in analyses:
                analysis.function(
                    signal_slice,
                    result,
                    **analysis.parameters_for_call(parameters[analysis.key])
                )
            total_peaks += sum(len(peak_set.x) for peak_set in result.peak_sets)
            t = end

        return count_id, total_peaks

    def _on_total_peak_count_finished(self, payload):
        """Update the title once whole-file peak counting completes."""
        count_id, total_peaks = payload
        if count_id != self._peak_count_request_id or total_peaks is None:
            return
        self.total_peak_count = total_peaks
        self._update_plot_title()

    def _run_analysis(
        self, request_id, start_sec, end_sec, reset_x_range, reset_y_range,
        method, factor, channel_choice, parameters,
    ):
        """Run the selected signal analysis functions for a slice of the current `SignalReader`'s data."""
        # Background worker task: slices data and runs signal analysis
        signal_slice = self.reader.get_slice(start_sec, end_sec, downsample_factor=factor, downsample=method)
        channel_indices = self._channel_indices_for(channel_choice)
        signal_slice = Slice(signal_slice.time, signal_slice.data, channel_indices=channel_indices)

        result = AnalysisResult(signal_slice)
        for analysis in self.active_analyses:
            analysis.function(
                signal_slice,
                result,
                **analysis.parameters_for_call(parameters[analysis.key])
            )
        return request_id, result, (start_sec, end_sec), reset_x_range, reset_y_range

    def plot_data(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
        reset_x_range: bool = True,
        reset_y_range: bool = True,
    ):
        """Run signal analysis and plotting in a background thread."""
        method = self.combo.currentText()
        factor = self.factor_input.value()
        channel_choice = self.channel_combo.currentText()
        parameters = self.analysis_parameters

        # Cap slice length to MAX_SLICE_POINTS raw samples for large files
        total_duration = self.reader.total_samples / self.reader.sample_rate
        max_window_sec = min(total_duration, self.MAX_SLICE_POINTS / self.reader.sample_rate)
        if start_sec is None or end_sec is None:
            start_sec = 0.0
            end_sec = max_window_sec

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._analysis_running = True

        if reset_x_range:
            self.btn.setEnabled(False)
            self.plot_widget.setTitle("Analyzing signal, please wait...")

        # Run the peak analysis in a background worker thread to keep GUI responsive
        worker = Worker(
            self._run_analysis,
            request_id, start_sec, end_sec, reset_x_range, reset_y_range,
            method, factor, channel_choice, parameters,
        )
        worker.signals.result.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_analysis_error)
        QThreadPool.globalInstance().start(worker)

    def _on_analysis_finished(self, payload):
        """Plot the signal and update the peak label and plot title."""
        request_id, result, loaded_range, reset_x_range, reset_y_range = payload
        # Ignore results from outdated pan/analysis requests
        if request_id != self._analysis_request_id:
            return
        self._analysis_running = False
        self.loaded_range = loaded_range

        slice_peak_count = sum(len(peak_set.x) for peak_set in result.peak_sets)
        total_duration = self.reader.total_samples / self.reader.sample_rate

        if self.reader.total_samples <= self.MAX_SLICE_POINTS:
            self.total_peak_count = slice_peak_count
        elif self.total_peak_count is None and self._peak_count_started_id != self._peak_count_request_id:
            # Count remaining chunks in the background for the total whole-file peak count
            self._peak_count_started_id = self._peak_count_request_id
            start_from = loaded_range[1] if loaded_range[0] <= 1e-9 else 0.0
            initial_count = slice_peak_count if loaded_range[0] <= 1e-9 else 0
            count_worker = Worker(
                self._count_total_peaks,
                self._peak_count_request_id,
                self.reader,
                start_from,
                initial_count,
                self.combo.currentText(),
                self.factor_input.value(),
                self.channel_combo.currentText(),
                self.active_analyses,
                {k: dict(v) for k, v in self.analysis_parameters.items()},
            )
            count_worker.signals.result.connect(self._on_total_peak_count_finished)
            QThreadPool.globalInstance().start(count_worker)

        self._update_plot_title()
        # Show the parameter editing dialog on initial load
        if reset_x_range:
            self.action_param_window.setChecked(True)

        x_limits = (0.0, total_duration) if total_duration > 0 else None
        # Keep half of the loaded slice off-screen as buffer on large files
        if total_duration > 0 and self.reader.total_samples > self.MAX_SLICE_POINTS:
            max_x_range = (self.MAX_SLICE_POINTS / self.reader.sample_rate) * 0.5
        else:
            max_x_range = total_duration if total_duration > 0 else None

        plot_signal(
            self.plot_widget,
            result,
            x_limits=x_limits,
            max_x_range=max_x_range,
            reset_x_range=reset_x_range,
            reset_y_range=reset_y_range,
        )
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

    def update_analysis_functions(self, functions: tuple[AnalysisFunction, ...], parameters: dict[str, dict]):
        """Update and saveselected analysis functions and their parameters,
        then update the plot if data is available.
        """
        self.active_analyses = functions
        self.analysis_parameters = {
            analysis.key: dict(parameters.get(analysis.key, analysis.defaults))
            for analysis in self.active_analyses
        }
        # Save selected functions and parameters
        self.save_analysis_functions_and_parameters(evaluate=False)
        # Plot if reader and data available
        if hasattr(self, "reader") and self.reader is not None and self.reader.data is not None:
            self._invalidate_total_peak_count()
            if self.loaded_range is not None:
                self.plot_data(*self.loaded_range, reset_x_range=False)
            else:
                self.plot_data()
    
    
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
