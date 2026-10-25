import sys
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg
from signal_reader import SignalReader

from signal_classes import Slice


def plot_signal(plot_widget, signal_slice: Slice):
    # Plots both channels of signal data into a pyqtgraph
    plot_widget.clear()
    if len(signal_slice.time) > 0 and len(signal_slice.data) > 0:
        plot_widget.plot(signal_slice.time, signal_slice.data[:, 0], pen="y", name="Channel 1")
        if signal_slice.data.ndim > 1 and signal_slice.channels > 1:
            plot_widget.plot(signal_slice.time, signal_slice.data[:, 1], pen="c", name="Channel 2")

# Main method for testing the visualizer only
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Manually set file path
    reader = SignalReader("data/IDP_signal_example_[sami_dmitry_september2023].csv")
    signal_slice = reader.get_slice(0, 5, downsample_factor=5)

    win = pg.plot(title="Signal Visualizer test")
    win.showGrid(x=True, y=True)
    win.setLabel("bottom", "Time", units="s")
    win.setLabel("left", "Amplitude")

    plot_signal(win, signal_slice)

    sys.exit(app.exec())
