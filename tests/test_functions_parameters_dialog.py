import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from functions_parameters_dialog import FunctionsParametersDialog
from peak_finder import AnalysisFunction, AnalysisParameter


def test_parameter_reset_button():
    app = QApplication.instance() or QApplication([])

    def example_function(value=40):
        return value

    analysis = AnalysisFunction(
        "example",
        "Example",
        example_function,
        (AnalysisParameter("value", 40, "Value"),),
    )
    dialog = FunctionsParametersDialog((analysis,), (analysis,))
    dialog.show()
    app.processEvents()
    field = dialog.fields["example"]["value"]
    reset_button = dialog.reset_buttons["example"]["value"]

    assert reset_button.text() == "\u21ba"
    assert reset_button.width() == reset_button.height()
    assert not reset_button.isVisible()

    field.setText("50")
    assert reset_button.isVisible()

    reset_button.click()
    assert field.text() == "40"
    assert not reset_button.isVisible()
    dialog.close()
    app.processEvents()
