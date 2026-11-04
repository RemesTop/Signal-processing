import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QPushButton, QSpinBox, QFileDialog, QCheckBox, QMenuBar, QMessageBox
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

    def load_settings(self):
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
        self.settings.setValue("file_path", self.file_input.text())
        self.settings.setValue("downsample_method", self.combo.currentText())
        self.settings.setValue("downsample_factor", self.factor_input.value())
        self.settings.setValue("channel_selection", self.channel_combo.currentText())
        self.settings.sync()

    def on_channel_changed(self):
        if hasattr(self, "reader") and self.reader is not None and self.reader.data is not None:
            self.plot_data()

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
        channel_choice = self.channel_combo.currentText()
        
        # Get full slice from the data
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
        points_count = len(result.lines[0].x) if result.lines else 0
        self.plot_widget.setTitle(f"Points: {points_count}, Functions: {analysis_names}")
        # Show the parameter editing dialog
        self.action_param_window.setChecked(True)
        
        plot_signal(self.plot_widget, result)
        self.btn.setEnabled(True)

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
