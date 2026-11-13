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


# Analysis functions that are currently run for each slice
ACTIVE_ANALYSES: tuple[AnalysisFunction, ...] = (
    ANALYSIS_FUNCTIONS["Raw signal with peaks"],
)


# Main app window for the project
class SignalApp(QWidget):
    """Main app window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal app")
        self.resize(800, 600)

        # Options are saved to settings.ini and persist when restarting the app
        settings_file = Path(__file__).resolve().parent / "settings.ini"
        self.settings = QSettings(str(settings_file), QSettings.IniFormat)

        # Analysis functions that are run for each slice
        self.active_analyses = ACTIVE_ANALYSES
        # Editable parameters for each analysis function
        self.analysis_parameters = {analysis.key: analysis.defaults for analysis in self.active_analyses}
        # Dialog for editing the parameters
        self.parameters_dialog = FunctionsParametersDialog(tuple(ANALYSIS_FUNCTIONS.values()), self.active_analyses, self)
        self.parameters_dialog.changes_applied.connect(self.apply_analysis_parameters)
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
        # Label to display the number of detected peaks
        self.peak_count_label = QLabel("Peaks: not detected")

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
        # Auto fit y-axis
        view_box.setMouseEnabled(x=True, y=False)
        view_box.setAutoVisible(y=True)
        view_box.sigXRangeChanged.connect(self._fit_plot_y)
        self.factor_input.setPrefix("Factor: ")

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.combo)
        top_layout.addWidget(self.factor_input)
        top_layout.addWidget(self.channel_combo)
        top_layout.addWidget(self.peak_count_label)
        top_layout.addWidget(self.btn)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menu_bar)
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_widget)

        # Restore saved settings on startup
        self.load_settings()

    def _fit_plot_y(self, view_box: pg.ViewBox, _):
        """Refresh the auto-ranging of the pyqtgraph ViewBox."""
        QTimer.singleShot(0, view_box.updateAutoRange)

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

    def save_settings(self):
        """Save recently used signal processing configurations into an ini file."""
        self.settings.setValue("file_path", self.file_input.text())
        self.settings.setValue("downsample_method", self.combo.currentText())
        self.settings.setValue("downsample_factor", self.factor_input.value())
        self.settings.setValue("channel_selection", self.channel_combo.currentText())
        self.settings.sync()

    def on_channel_changed(self):
        """When selecting displayed channel(s), plot the data if it has been read before."""
        if hasattr(self, "reader") and self.reader is not None and self.reader.data is not None:
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

    def _run_analysis(self, method, factor, channel_choice, parameters):
        """Run the selected signal analysis functions for the current `SignalReader`'s data."""
        # Background worker task: slices data and runs signal analysis
        total_duration = self.reader.total_samples / self.reader.sample_rate
        signal_slice = self.reader.get_slice(0, total_duration, downsample_factor=factor, downsample=method)

        # Restrict processing to the chosen channel (or both) to optimize performance
        if channel_choice == "Channel 1":
            channel_indices = (0,)
        elif channel_choice == "Channel 2":
            channel_indices = (1,)
        else:
            channel_indices = (0, 1)

        signal_slice = Slice(signal_slice.time, signal_slice.data, channel_indices=channel_indices)

        result = AnalysisResult(signal_slice)
        for analysis in self.active_analyses:
            analysis.function(
                signal_slice,
                result,
                **analysis.parameters_for_call(parameters[analysis.key])
            )
        return result

    def plot_data(self):
        """Run signal analysis and plotting in a background thread."""
        method = self.combo.currentText()
        factor = self.factor_input.value()
        channel_choice = self.channel_combo.currentText()
        parameters = self.analysis_parameters

        self.btn.setEnabled(False)
        self.plot_widget.setTitle("Analyzing signal, please wait...")

        # Run the peak analysis in a background worker thread to keep GUI responsive
        worker = Worker(self._run_analysis, method, factor, channel_choice, parameters)
        worker.signals.result.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_analysis_error)
        QThreadPool.globalInstance().start(worker)

    def _on_analysis_finished(self, result):
        """Plot the signal and update the peak label and plot title."""
        # Displays the number of detected peaks in the label 
        peak_count = sum(len(peak_set.x) for peak_set in result.peak_sets)
        self.peak_count_label.setText(f"Peaks: {peak_count}")

        # This displaying of names can maybe be replaced with just labels in the graph or multiple graphs
        analysis_names = ", ".join(analysis.title for analysis in self.active_analyses)
        points_count = len(result.lines[0].x) if result.lines else 0
        self.plot_widget.setTitle(f"Points: {points_count}, Functions: {analysis_names}")
        # Show the parameter editing dialog
        self.action_param_window.setChecked(True)

        plot_signal(self.plot_widget, result)
        self.btn.setEnabled(True)

    def _on_analysis_error(self, err):
        """Show an error message for an error in signal analysis."""
        self.peak_count_label.setText("Peaks: unavailable")
        QMessageBox.critical(self, "Error in signal analysis", f"{err[0].__name__}: {err[1]}")
        self.btn.setEnabled(True)

    def apply_analysis_parameters(self, functions: tuple[AnalysisFunction, ...], parameters: dict[str, dict]):
        """Apply selected analysis functions and their parameters, then update the plot if data is available."""
        self.active_analyses = functions
        self.analysis_parameters = {
            analysis.key: dict(parameters.get(analysis.key, analysis.defaults))
            for analysis in self.active_analyses
        }
        if hasattr(self, "reader") and self.reader is not None and self.reader.data is not None:
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
