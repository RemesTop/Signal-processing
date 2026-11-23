import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from signal_classes import AnalysisResult, Slice
from visualize_data import plot_signal

""""
WRITTEN BY ARTICIAL INTELLIGENCE (AI) - MAI Code 2 Flash
Tests for data visualization. To test the visualization, run the following command in the terminal:
pytest tests/test_visualize_data.py
Last modified: 23.11.2026
"""


def make_result():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    data = np.array([[1.0], [3.0], [2.0], [4.0]])
    result = AnalysisResult(Slice(time, data))
    result.line(time, data[:, 0], 0, "Signal")
    return result


def test_plot_signal_preserves_view_range():
    app = QApplication.instance() or QApplication([])
    widget = pg.PlotWidget()

    plot_signal(widget, make_result())
    widget.setXRange(1.0, 2.0, padding=0)
    widget.setYRange(1.5, 3.5, padding=0)
    expected = widget.getViewBox().viewRange()

    plot_signal(widget, make_result(), expected)

    actual = widget.getViewBox().viewRange()
    assert np.allclose(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1])
    widget.close()
    app.processEvents()


def test_plot_signal_resets_view_for_new_file():
    app = QApplication.instance() or QApplication([])
    widget = pg.PlotWidget()

    plot_signal(widget, make_result())
    widget.setXRange(1.0, 2.0, padding=0)

    plot_signal(widget, make_result())

    view_range = widget.getViewBox().viewRange()
    assert np.allclose(view_range[0], [0.0, 3.0])
    widget.close()
    app.processEvents()
