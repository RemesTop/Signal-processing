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

def set_plot_downsampling(plot_widget: pg.PlotWidget, enabled: bool):
    """Enable or disable automatic peak downsampling on `plot_widget`."""
    plot_item = plot_widget.getPlotItem()
    if enabled:
        plot_item.setDownsampling(auto=True, mode="peak")
    else:
        plot_item.setDownsampling(ds=False, auto=False)
    plot_item.updateDownsampling()
    plot_widget.update()


def plot_signal(
    plot_widget: pg.PlotWidget,
    result: AnalysisResult,
    x_limits: tuple[float, float] | None = None,
    max_x_range: float | None = None,
    reset_x_range: bool = True,
    reset_y_range: bool = True,
    remove_invisible_points: bool = True,
    fit_x_range: bool = False,
):
    """Populate `plot_widget` with the lines and peak sets from `result`."""
    plot_widget.clear()
    plot_item = plot_widget.getPlotItem()
    view_box = plot_item.getViewBox()
    prev_x_min, prev_x_max = view_box.viewRange()[0]

    # Update the zoom/pan limits (use full file bounds when loading in slices)
    time = result.slice.time
    if x_limits is not None:
        x_min, x_max = x_limits
        max_range = max_x_range if max_x_range is not None else (x_max - x_min)
        min_range = min(2e-4, max_range) if max_range > 0 else 2e-4
        view_box.setLimits(xMin=x_min, xMax=x_max, maxXRange=max_range, minXRange=min_range)
    elif time.size:
        view_box.setLimits(xMin=time.min(), xMax=time.max(), maxXRange=time.max() - time.min(), minXRange=2e-4)
    else:
        view_box.setLimits(xMin=None, xMax=None, maxXRange=None, minXRange=2e-4)

    # Clip to visible view and optionally auto-downsample to screen resolution for performance
    plot_item.setClipToView(True)
    set_plot_downsampling(plot_widget, remove_invisible_points)

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

    if reset_x_range:
        # Fit initial X range (showing up to max_x_range so extra slice data is buffered off-screen)
        if x_limits is not None and max_x_range is not None:
            view_box.setXRange(x_limits[0], min(x_limits[1], x_limits[0] + max_x_range), padding=0)
        else:
            view_box.enableAutoRange(axis=pg.ViewBox.XAxis)
            view_box.updateAutoRange()
            view_box.disableAutoRange(axis=pg.ViewBox.XAxis)
    elif x_limits is not None and max_x_range is not None and (
        fit_x_range or (prev_x_max - prev_x_min) > max_x_range
    ):
        # Resize X range around the previous center when max_slice_points changed or max_x_range shrunk
        center = (prev_x_min + prev_x_max) / 2
        new_min = max(x_limits[0], min(center - max_x_range / 2, x_limits[1] - max_x_range))
        view_box.setXRange(new_min, min(x_limits[1], new_min + max_x_range), padding=0)
    if reset_y_range:
        # Fit Y once and lock it so the Y axis does not rescale while panning
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis)
        view_box.updateAutoRange()
        view_box.disableAutoRange(axis=pg.ViewBox.YAxis)


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
