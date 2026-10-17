import sys
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg
from signal_reader import SignalReader


def plot_signal(plot_widget, time, signal):
    # Plots both channels of signal data into a pyqtgraph
    plot_widget.clear()
    if len(time) > 0 and len(signal) > 0:
        plot_widget.plot(time, signal[:, 0], pen="y", name="Channel 1")
        if signal.ndim > 1 and signal.shape[1] > 1:
            plot_widget.plot(time, signal[:, 1], pen="c", name="Channel 2")

# Main method for testing the visualizer only
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Manually set file path
    reader = SignalReader("data/IDP_signal_example_[sami_dmitry_september2023].csv")
    time, signal = reader.get_slice(0, 5, downsample_factor=5)

    win = pg.plot(title="Signal Visualizer test")
    win.showGrid(x=True, y=True)
    win.setLabel("bottom", "Time", units="s")
    win.setLabel("left", "Amplitude")

    plot_signal(win, time, signal)

    sys.exit(app.exec())
