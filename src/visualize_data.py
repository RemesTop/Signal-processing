import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg

from signal_reader import SignalReader
from signal_classes import AnalysisResult

DEFAULT_LINE_COLORS = (
    "#ffaa00", "#00aaff", "#f34141", "#06cf17",
    "#bc56ec", "#abd47b", "#2743e0", "#8c564b"
)
DEFAULT_PEAKSET_COLORS = ("#bb0000",)
DEFAULT_PEAKSET_SYMBOLS = ("o", "t1", "s", "d")


def make_peak_tip(peak_set):
    """Return a tooltip function for one peak set.

    The peak is found by matching the hovered x to peak_set.x, so it also
    works when pyqtgraph clips or downsamples the points.
    """
    props = getattr(peak_set, "properties", None) or {}

    def tip(x, y, data):
        i = int(np.argmin(np.abs(peak_set.x - x)))
        lines = [peak_set.label, f"Time: {x:.4f} s", f"Value: {y:.4f}"]
        for name, values in props.items():
            lines.append(f"{name}: {values[i]:.4g}")
        return "\n".join(lines)
    return tip

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
    for index, line in enumerate(result.lines):
        plot_widget.plot(line.x, line.y,
                         pen=pg.mkPen(
                             DEFAULT_LINE_COLORS[index % len(DEFAULT_LINE_COLORS)],
                             width=1
                         ),
                         name=f"{line.label} ch.{line.channel+1}")
     # Plot all peaks as points, with a hover tooltip
    for index, peak_set in enumerate(result.peak_sets):
        item = plot_widget.plot(peak_set.x, peak_set.y,
                                pen=None,
                                symbol=DEFAULT_PEAKSET_SYMBOLS[index % len(DEFAULT_PEAKSET_SYMBOLS)],
                                symbolBrush=pg.mkBrush(None),
                                symbolPen=pg.mkPen(DEFAULT_PEAKSET_COLORS[index % len(DEFAULT_PEAKSET_COLORS)], width=1.5),
                                symbolSize=9,
                                name=f"Peaks: {peak_set.label} ch.{peak_set.channel+1} ({len(peak_set.x)})")
        item.scatter.opts["hoverable"] = True
        item.scatter.opts["hoverSize"] = 13
        item.scatter.opts["tip"] = make_peak_tip(peak_set)

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
