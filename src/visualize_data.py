import sys
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg
from signal_reader import SignalReader

from signal_classes import AnalysisResult

def plot_signal(plot_widget: pg.PlotWidget, result: AnalysisResult):
    """Populate `plot_widget` with the lines and peak sets from `result`."""
    plot_widget.clear()
    plot_item = plot_widget.getPlotItem()
    view_box = plot_item.getViewBox()

    # Update the zoom limits of the plot widget
    time = result.slice.time
    if time.size:
        view_box.setLimits(xMin=time.min(), xMax=time.max(), maxXRange=time.max() - time.min(), minXRange=2e-4)
    else:
        view_box.setLimits(xMin=None, xMax=None, maxXRange=None, minXRange=2e-4)

    # Clip to visible view and auto-downsample to screen resolution for performance
    plot_item.setClipToView(True)
    plot_item.setDownsampling(auto=True, mode="peak")

    if plot_item.legend is None:
        plot_item.addLegend()
    else:
        plot_item.legend.clear()

    # Plot all lines as lines (obviously)
    for line in result.lines:
        plot_widget.plot(line.x, line.y,
                         pen=pg.mkPen(line.color, width=1),
                         name=line.label)
    # Plot all peaks as points
    for peak_set in result.peak_sets:
        plot_widget.plot(peak_set.x, peak_set.y,
                         pen=None,
                         symbol="o",
                         symbolBrush=pg.mkBrush(None), symbolPen=pg.mkPen(peak_set.color, width=1.5),
                         symbolSize=7,
                         name=peak_set.label)

    # Center and fit the graph view to the currently plotted channel(s)
    plot_widget.autoRange()
    view_box.enableAutoRange(axis=pg.ViewBox.YAxis)


# Main method for testing the visualizer only
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Manually set file path
    reader = SignalReader("data/IDP_signal_example_[sami_dmitry_september2023].csv")
    signal_slice = reader.get_slice(0, 30, downsample_factor=5)
    result = AnalysisResult.line_from_slice(signal_slice)

    win = pg.plot(title="Signal Visualizer test")
    win.showGrid(x=True, y=True)
    win.setLabel("bottom", "Time", units="s")
    win.setLabel("left", "Amplitude")

    plot_signal(win, result)

    sys.exit(app.exec())
