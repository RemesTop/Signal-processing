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
DEFAULT_PEAKSET_SYMBOLS = {
    "brush": (False, False, True, False),
    "symbol": ("o", "s", "x", "d")
}


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

def plot_signal(plot_widget: pg.PlotWidget, result: AnalysisResult, preserve_view_range: bool = False):
    """Populate `plot_widget` with the lines and peak sets from `result`.
    
    If `preserve_view_range` is `True`, the current zoom and positioning
    of the graph are preserved (unless the data doesn't reach there).
    Otherwise the graph is fitted to show the entire data.
    """
    plot_item = plot_widget.getPlotItem()
    view_box = plot_item.getViewBox()
    old_view_range = view_box.viewRange()
    plot_widget.clear()

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
        symbol_idx = index % len(DEFAULT_PEAKSET_SYMBOLS["symbol"])
        item = plot_widget.plot(peak_set.x, peak_set.y,
                                pen=None,
                                symbol=DEFAULT_PEAKSET_SYMBOLS["symbol"][symbol_idx],
                                symbolBrush=(
                                    pg.mkBrush(DEFAULT_PEAKSET_COLORS[index % len(DEFAULT_PEAKSET_COLORS)])
                                    if DEFAULT_PEAKSET_SYMBOLS["brush"][symbol_idx]
                                    else pg.mkBrush(None)
                                ),
                                symbolPen=(
                                    pg.mkPen(DEFAULT_PEAKSET_COLORS[index % len(DEFAULT_PEAKSET_COLORS)], width=2)
                                    if not DEFAULT_PEAKSET_SYMBOLS["brush"][symbol_idx]
                                    else pg.mkPen(None)
                                ),
                                symbolSize=9,
                                name=f"Peaks: {peak_set.label} ch.{peak_set.channel+1} ({len(peak_set.x)})")
        item.scatter.opts["hoverable"] = True
        item.scatter.opts["hoverSize"] = 13
        item.scatter.opts["tip"] = make_peak_tip(peak_set)

    if not preserve_view_range:
        # Show the entire result
        if time.size:
            view_box.disableAutoRange()
            view_box.setRange(
                xRange=(time.min(), time.max()),
                padding=0,
            )
        else:
            plot_widget.autoRange()
    else:
        # Keep the zoom and position
        (x_range, y_range) = old_view_range
        view_box.disableAutoRange()
        view_box.setRange(
            xRange=(x_range[0], x_range[1]),
            yRange=(y_range[0], y_range[1]),
            padding=0,
        )
    
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
