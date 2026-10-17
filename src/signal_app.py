import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QComboBox, QPushButton, QSpinBox
)
from PySide6.QtCore import QSettings
import pyqtgraph as pg
from signal_reader import SignalReader
from visualize_data import plot_signal

# Main app window for the project
class SignalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal app")
        self.resize(800, 600)

        # Options are saved to settings.ini and persist when restarting the app
        settings_file = Path(__file__).resolve().parent / "settings.ini"
        self.settings = QSettings(str(settings_file), QSettings.IniFormat)

        # 1. Text input for file path
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("File path")

        # 2. Dropdown for downsample method
        self.combo = QComboBox()
        self.combo.addItems(["max points", "decimation", "none"])

        # 3. Number input for downsample factor
        self.factor_input = QSpinBox()
        self.factor_input.setRange(1, 100000)
        self.factor_input.setValue(5)

        # 4. Visualize Data button (When the button is pressed)
        self.btn = QPushButton("Visualize Data")
        self.btn.clicked.connect(self.visualize)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Reading")
        self.factor_input.setPrefix("Factor: ")

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_input)
        top_layout.addWidget(self.combo)
        top_layout.addWidget(self.factor_input)
        top_layout.addWidget(self.btn)

        layout = QVBoxLayout(self)
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

    def save_settings(self):
        self.settings.setValue("file_path", self.file_input.text())
        self.settings.setValue("downsample_method", self.combo.currentText())
        self.settings.setValue("downsample_factor", self.factor_input.value())
        self.settings.sync()

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def visualize(self):
        self.save_settings()
        file_path = self.file_input.text().strip()
        if not file_path:
            return

        reader = SignalReader(file_path)
        method = self.combo.currentText()
        factor = self.factor_input.value()
        time, signal = reader.get_slice(0, 5, downsample_factor=factor, downsample=method)
        plot_signal(self.plot_widget, time, signal)
        self.plot_widget.setTitle(f"Points: {len(time)}, Downsampling method: {method}")


def main():
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
