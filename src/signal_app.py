import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QPushButton, QSpinBox, QFileDialog, QCheckBox, QMenuBar, QMessageBox, QLabel
)
from PySide6.QtCore import QSettings
import pyqtgraph as pg
from signal_reader import SignalReader
from visualize_data import plot_signal
from signal_classes import AnalysisResult, Slice
from analysis_parameters_dialog import AnalysisParametersDialog
from peak_finder import ANALYSIS_FUNCTIONS, AnalysisFunction


# Analysis functions that are currently run for each slice
ACTIVE_ANALYSES: tuple[AnalysisFunction, ...] = (
    ANALYSIS_FUNCTIONS["Custom decimation"],
)


# Main app window for the project
class SignalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal app")
        self.resize(800, 600)

        # Cached result and per-label checkboxes, no need to re-run analysis when toggling visibility of lines/peaks
        self.result = None
        self.line_checks = {}
        self.peak_checks = {}
        self.checks_layout = QHBoxLayout()

        # Options are saved to settings.ini and persist when restarting the app
        settings_file = Path(__file__).resolve().parent / "settings.ini"
        self.settings = QSettings(str(settings_file), QSettings.IniFormat)

        # Analysis functions that are run for each slice
        self.active_analyses = ACTIVE_ANALYSES
        # Editable parameters for each analysis function
        self.analysis_parameters = {analysis.key: analysis.defaults for analysis in self.active_analyses}
        # Dialog for editing the parameters
        self.parameters_dialog = AnalysisParametersDialog(self.active_analyses, self)
        self.parameters_dialog.changes_applied.connect(self.apply_analysis_parameters)
        self.parameters_dialog.closed.connect(lambda: self.action_param_window.setChecked(False))


        # Menu bar
        self.menu_bar = QMenuBar()
        view_menu = self.menu_bar.addMenu("View")
        self.action_param_window = view_menu.addAction("Analysis parameters window")
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

        # 4. Checkbox for enabling peak detection
        self.peaks_checkbox = QCheckBox("Detect peaks")
        self.peaks_checkbox.setChecked(True)

        # 5. Visualize Data button (When the button is pressed)
        self.btn = QPushButton("Run")
        self.btn.clicked.connect(self.visualize)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Reading")
        self.plot_widget.setTitle(" ")
        self.factor_input.setPrefix("Factor: ")

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.combo)
        top_layout.addWidget(self.factor_input)
        top_layout.addWidget(self.peaks_checkbox)
        top_layout.addWidget(self.btn)


        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menu_bar)
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_widget)

        layout.addLayout(top_layout)
        layout.addLayout(self.checks_layout)
        layout.addWidget(self.plot_widget)

        # Restore saved settings on startup
        self.load_settings()

    def load_settings(self):
        saved_file = self.settings.value("file_path", "")
        if saved_file:
            self.file_input.setText(str(saved_file))

        saved_method = self.settings.value("downsample_method", "")
        if saved_method:
            self.combo.setCurrentText(str(saved_method))

        saved_factor = self.settings.value("downsample_factor", 2)
        self.factor_input.setValue(int(saved_factor))

        saved_peaks = self.settings.value("detect_peaks", True, type=bool)
        self.peaks_checkbox.setChecked(saved_peaks)

    def save_settings(self):
        self.settings.setValue("file_path", self.file_input.text())
        self.settings.setValue("downsample_method", self.combo.currentText())
        self.settings.setValue("downsample_factor", self.factor_input.value())
        self.settings.setValue("detect_peaks", self.peaks_checkbox.isChecked())
        self.settings.sync()

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file",
            "",
            "Data File (*.csv *.dat);;All Files (*)"
        )
        if file_path:
            self.file_input.setText(file_path)

    def visualize(self):
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
        self.plot_widget.setTitle(f"Converting csv to binary, please wait {percent:.0f}%")

    def plot_data(self):
        method = self.combo.currentText()
        factor = self.factor_input.value()
        
        # Get full slice from the data
        total_duration = self.reader.total_samples / self.reader.sample_rate
        signal_slice = self.reader.get_slice(0, total_duration, downsample_factor=factor, downsample=method)
        
        if self.peaks_checkbox.isChecked():
            # Do signal analysis using all the active functions
            try:
                result = AnalysisResult(signal_slice)
                for analysis in self.active_analyses:
                    analysis.function(
                        signal_slice,
                        result,
                        **analysis.parameters_for_call(self.analysis_parameters[analysis.key])
                    )
            except Exception as exc:
                QMessageBox.critical(self, "Error in signal analysis", f"{type(exc).__name__}: {exc}")
                self.btn.setEnabled(True)
                return
            
            # This displaying of names can maybe be replaced with just labels in the graph or multiple graphs
            analysis_names = ", ".join(analysis.title for analysis in self.active_analyses)
            self.plot_widget.setTitle(f"Points: {len(result.lines[0].x) if result.lines else 0}, Functions: {analysis_names}")
            # Show the parameter editing dialog
            self.action_param_window.setChecked(True)
        else:
            result = AnalysisResult.line_from_slice(signal_slice)
            
            self.plot_widget.setTitle(f"Points: {len(signal_slice.time)}, Downsampling method: {method}")

        # Cache the result and rebuild the checkboxes for lines and peaks
        self.result = result
        self.rebuild_checkboxes(result)
        self.redraw()

        self.btn.setEnabled(True)

    def rebuild_checkboxes(self, result):
        """Rebuilds the checkboxes for lines and peaks based on the provided result.
        """

        # Save the current state of checkboxes to restore after rebuilding
        old_lines = {k: cb.isChecked() for k, cb in self.line_checks.items()}
        old_peaks = {k: cb.isChecked() for k, cb in self.peak_checks.items()}

        # Clears existing checkboxes
        while self.checks_layout.count():
            item = self.checks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.line_checks, self.peak_checks = {}, {}

        #Checkboxes for lines
        self.checks_layout.addWidget(QLabel("Signals:"))
        for label in dict.fromkeys(t.label for t in result.lines): 
            cb = QCheckBox(str(label))
            cb.setChecked(old_lines.get(label, True))
            cb.toggled.connect(self.redraw)
            self.checks_layout.addWidget(cb)
            self.line_checks[label] = cb

        #Checkboxes for peaks
        peak_labels = list(dict.fromkeys(p.label for p in result.peak_sets))
        if peak_labels:
            self.checks_layout.addWidget(QLabel("Peaks:"))
        for label in peak_labels:
            cb = QCheckBox(str(label))
            cb.setChecked(old_peaks.get(label, True))
            cb.toggled.connect(self.redraw)
            self.checks_layout.addWidget(cb)
            self.peak_checks[label] = cb

        self.checks_layout.addStretch()

    def redraw(self):
        """Redraws the plot based on the current result and checkbox states. 
        Uses cached result and checkbox states to avoid re-running analysis. 
        See rebuild_checkboxes() for how checkboxes are built."""
    
        if self.result is None:
            return
        hidden_lines = {k for k, cb in self.line_checks.items() if not cb.isChecked()}
        hidden_peaks = {k for k, cb in self.peak_checks.items() if not cb.isChecked()}
        plot_signal(self.plot_widget, self.result, hidden_lines, hidden_peaks)

    def apply_analysis_parameters(self, parameters: dict[str, dict]):
        """Apply the provided parameters to all the active analyses and run `plot_data`."""
        self.analysis_parameters = {
            analysis.key: dict(parameters.get(analysis.key, analysis.defaults))
            for analysis in self.active_analyses
        }
        self.plot_data()
    

def main():
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
