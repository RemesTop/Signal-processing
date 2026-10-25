import sys
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg
from signal_reader import SignalReader

from signal_classes import AnalysisResult

def plot_signal(plot_widget, result: AnalysisResult):
    plot_widget.clear()
    plot_item = plot_widget.getPlotItem()
    if plot_item.legend is None:
        plot_item.addLegend()

    for trace in result.lines:
        plot_widget.plot(trace.x, trace.y,
                         pen=pg.mkPen(trace.color, width=1),
                         name=trace.label)
    for peak_set in result.peak_sets:
        plot_widget.plot(peak_set.x, peak_set.y,
                         pen=None,
                         symbol="o",
                         symbolBrush=pg.mkBrush(peak_set.color), symbolPen=pg.mkPen(peak_set.color),
                         symbolSize=6,
                         name=peak_set.label)

# Main method for testing the visualizer only
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Manually set file path
    reader = SignalReader("data/IDP_signal_example_[sami_dmitry_september2023].csv")
    signal_slice = reader.get_slice(0, 5, downsample_factor=5)
    result = AnalysisResult.line_from_slice(signal_slice)

    win = pg.plot(title="Signal Visualizer test")
    win.showGrid(x=True, y=True)
    win.setLabel("bottom", "Time", units="s")
    win.setLabel("left", "Amplitude")

    plot_signal(win, result)

    sys.exit(app.exec())
