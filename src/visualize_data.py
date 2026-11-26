from collections.abc import Collection, Mapping
import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg

from signal_reader import SignalReader
from signal_classes import AnalysisResult, PeakSet

PeakKey = tuple[str, int]

DEFAULT_LINE_COLORS = (
    "#ffaa00", "#00aaff", "#f34141", "#06cf17",
    "#bc56ec", "#abd47b", "#2743e0", "#8c564b"
)
DEFAULT_LINE_WIDTH = 2
SINGLE_PEAKSET_COLOR = "#bb0000"
MULTI_PEAKSET_COLORS = (
    "#bb0000", "#c04b4b", "#ffc6b8", "#f54076",
    "#FF7265", "#B61277", "#AF4178", "#C759C7",
)
DEFAULT_PEAKSET_COLORS = [SINGLE_PEAKSET_COLOR]
DEFAULT_PEAKSET_SYMBOLS = {
    "brush": (False, False, True, False),
    "symbol": ("o", "s", "x", "d")
}


def set_peakset_colors(multi_color: bool):
    """Select the default peak-set palette for future plots."""
    DEFAULT_PEAKSET_COLORS[:] = (
        MULTI_PEAKSET_COLORS if multi_color else (SINGLE_PEAKSET_COLOR,)
    )


def update_peakset_colors(plot_widget: pg.PlotWidget):
    """Update peak marker colors without rebuilding plotted data."""
    for index, (item, _peak_set) in enumerate(
        getattr(plot_widget, "_peak_legend_entries", ())
    ):
        symbol_idx = index % len(DEFAULT_PEAKSET_SYMBOLS["symbol"])
        color = DEFAULT_PEAKSET_COLORS[index % len(DEFAULT_PEAKSET_COLORS)]
        if DEFAULT_PEAKSET_SYMBOLS["brush"][symbol_idx]:
            item.setSymbolBrush(pg.mkBrush(color))
        else:
            item.setSymbolPen(pg.mkPen(color, width=2))


class TimeAxisItem(pg.AxisItem):
    """Bottom time axis showing seconds for short ranges and m:ss / h:mm:ss for longer ranges."""

    _TIME_STEPS = (15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickSpacing(self, minVal: float, maxVal: float, size: float):
        span = abs(maxVal - minVal)
        if span < 60:
            return super().tickSpacing(minVal, maxVal, size)
        target = span / max(3.0, size / 90.0)
        major = next((s for s in self._TIME_STEPS if s >= target), self._TIME_STEPS[-1])
        minor = major / 5 if major in (15, 300, 600, 900) else major / 4
        return [(major, 0), (minor, 0)]

    def tickStrings(self, values, scale, spacing):
        if not values:
            return []
        decimals = max(0, min(6, int(np.ceil(-np.log10(spacing))))) if spacing > 0 else 0
        max_abs = max(abs(v) for v in values)
        if max_abs < 60:
            return [f"{v:.{decimals}f} s" for v in values]

        show_hours = max_abs >= 3600
        width = 2 + (decimals + 1 if decimals else 0)
        labels = []
        for v in values:
            sign = "-" if v < 0 else ""
            total = round(abs(v), decimals)
            hours, rem = divmod(total, 3600)
            minutes, secs = divmod(rem, 60)
            sec_str = f"{secs:0{width}.{decimals}f}"
            if show_hours:
                labels.append(f"{sign}{int(hours)}:{int(minutes):02d}:{sec_str}")
            else:
                labels.append(f"{sign}{int(total // 60)}:{sec_str}")
        return labels


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


def format_peak_legend_name(
    peak_set: PeakSet,
    visible_count: int,
    total_count: int,
    is_loading: bool = False,
) -> str:
    """Format the legend label for a `PeakSet`."""
    plus = "+" if is_loading else ""
    return f"Peaks: {peak_set.label} ch.{peak_set.channel + 1} ({visible_count}/{total_count}{plus})"


def update_peak_legend_labels(
    plot_widget: pg.PlotWidget,
    peak_totals: Mapping[PeakKey, int] | None = None,
    loading_keys: Collection[PeakKey] | None = None,
):
    """Refresh peak legend labels on `plot_widget` for the current X view range."""
    if peak_totals is not None:
        plot_widget._peak_totals = dict(peak_totals)
    if loading_keys is not None:
        plot_widget._peak_loading_keys = set(loading_keys)

    entries = getattr(plot_widget, "_peak_legend_entries", ())
    plot_item = plot_widget.getPlotItem()
    legend = plot_item.legend
    if not entries or legend is None:
        return

    x_min, x_max = plot_item.getViewBox().viewRange()[0]
    low, high = (x_min, x_max) if x_min <= x_max else (x_max, x_min)
    totals = getattr(plot_widget, "_peak_totals", None)
    loading = getattr(plot_widget, "_peak_loading_keys", set())

    for item, peak_set in entries:
        key = (peak_set.label, peak_set.channel)
        visible_count = int(np.count_nonzero((peak_set.x >= low) & (peak_set.x <= high)))
        slice_count = len(peak_set.x)
        if totals is None:
            total_count = slice_count
        else:
            total_count = max(totals.get(key, 0), slice_count)
        is_loading = key in loading
        label_item = legend.getLabel(item)
        if label_item is not None:
            label_item.setText(format_peak_legend_name(peak_set, visible_count, total_count, is_loading))

    if hasattr(legend, "updateSize"):
        legend.updateSize()


def plot_signal(
    plot_widget: pg.PlotWidget,
    result: AnalysisResult,
    x_limits: tuple[float, float] | None = None,
    max_x_range: float | None = None,
    reset_x_range: bool = True,
    reset_y_range: bool = True,
    remove_invisible_points: bool = True,
    fit_x_range: bool = False,
    auto_y_range: bool = False,
    peak_totals: Mapping[PeakKey, int] | None = None,
    loading_peak_keys: Collection[PeakKey] | None = None,
):
    """Populate `plot_widget` with the lines and peak sets from `result`."""
    plot_widget._peak_legend_entries = ()
    plot_widget.clear()
    plot_item = plot_widget.getPlotItem()
    view_box = plot_item.getViewBox()
    if not getattr(plot_widget, "_peak_legend_connected", False):
        view_box.sigXRangeChanged.connect(
            lambda _vb, _rng, pw=plot_widget: update_peak_legend_labels(pw)
        )
        plot_widget._peak_legend_connected = True
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
                             width=DEFAULT_LINE_WIDTH,
                             cosmetic=True,
                         ),
                         name=f"{line.label} ch.{line.channel+1}")
    # Plot all peaks as points, with a hover tooltip
    peak_entries = []
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
                                name=format_peak_legend_name(peak_set, 0, len(peak_set.x)))
        item.scatter.opts["hoverable"] = True
        item.scatter.opts["hoverSize"] = 13
        item.scatter.opts["tip"] = make_peak_tip(peak_set)
        peak_entries.append((item, peak_set))

    plot_widget._peak_legend_entries = tuple(peak_entries)
    plot_widget._peak_totals = dict(peak_totals) if peak_totals is not None else None
    plot_widget._peak_loading_keys = set(loading_peak_keys) if loading_peak_keys is not None else set()

    if reset_x_range:
        # Fit initial X range (showing up to max_x_range so extra slice data is buffered off-screen)
        if x_limits is not None and max_x_range is not None:
            view_box.setXRange(x_limits[0], min(x_limits[1], x_limits[0] + max_x_range), padding=0)
        elif time.size:
            view_box.setXRange(float(time.min()), float(time.max()), padding=0)
        else:
            view_box.enableAutoRange(axis=pg.ViewBox.XAxis)
            view_box.updateAutoRange()
            view_box.disableAutoRange(axis=pg.ViewBox.XAxis)
    elif x_limits is not None and max_x_range is not None and (
        fit_x_range or (prev_x_max - prev_x_min) > max_x_range
    ):
        center = (prev_x_min + prev_x_max) / 2
        new_min = max(x_limits[0], min(center - max_x_range / 2, x_limits[1] - max_x_range))
        view_box.setXRange(new_min, min(x_limits[1], new_min + max_x_range), padding=0)
    # Automatic y-range
    view_box.setAutoVisible(y=auto_y_range)
    if auto_y_range:
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis)
    elif reset_y_range:
        # Fit Y once and lock it so the Y axis does not rescale while panning
        view_box.enableAutoRange(axis=pg.ViewBox.YAxis)
        view_box.updateAutoRange()
        view_box.disableAutoRange(axis=pg.ViewBox.YAxis)
    else:
        view_box.disableAutoRange(axis=pg.ViewBox.YAxis)

    update_peak_legend_labels(plot_widget)

'''
# Main method for testing the visualizer only
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set file path
    reader = SignalReader("data/IDP_signal_example_[sami_dmitry_september2023].csv")
    signal_slice = reader.get_slice(0, 30, downsample_factor=5)
    result = AnalysisResult.line_from_slice(signal_slice)

    win = pg.plot(title="Signal Visualizer test")
    win.showGrid(x=True, y=True)
    win.setLabel("bottom", "Time", units="s")
    win.setLabel("left", "Amplitude")

    plot_signal(win, result)

    sys.exit(app.exec())'''
