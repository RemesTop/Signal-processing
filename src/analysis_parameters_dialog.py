import ast
from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget
)

from peak_finder import AnalysisFunction


class AnalysisParametersDialog(QDialog):
    """Dialog for editing the parameters of one or more signal analysis functions."""
    changes_applied = Signal(dict) # Emits a dict[str, dict] of parameters when the Apply button is pressed
    closed = Signal()

    def __init__(self, analyses: Sequence[AnalysisFunction], parent=None):
        super().__init__(parent)
        self.analyses = tuple(analyses)
        self.fields: dict[str, dict[str, QLineEdit]] = {}
        self.setWindowTitle("Analysis parameters")
        self.setMinimumWidth(200)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # A separate boxed section for each active analysis function
        for analysis in self.analyses:
            section = QGroupBox(analysis.title)
            form = QFormLayout(section)
            self.fields[analysis.key] = {}
            
            # A QLineEdit for each parameter
            for parameter in analysis.parameters:
                field = QLineEdit("None" if parameter.default is None else str(parameter.default))
                field.setToolTip("Enter a number, None, or a tuple for a range-valued parameter")
                self.fields[analysis.key][parameter.name] = field
                form.addRow(parameter.label or parameter.name, field)
                
            content_layout.addWidget(section)
        content_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_parameters)

        hbox_buttons = QHBoxLayout()
        hbox_buttons.addStretch()
        hbox_buttons.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)
        layout.addLayout(hbox_buttons)

    def apply_parameters(self):
        """Go through the fields for each analysis function's parameters,
        evaluate their values and emit `apply_called` with the dictionary
        of those values.
        """
        parameters = {}
        for analysis in self.analyses:
            analysis_parameters = {}
            
            # Go through all fields of the parameters of this analysis function
            for name, field in self.fields[analysis.key].items():
                try:
                    # Surely fine to use literal_eval
                    # So that we can parse values like (5, None)
                    evaluated_value = ast.literal_eval(field.text().strip())
                    # Type and bounds checking can be added here if needed
                    analysis_parameters[name] = evaluated_value
                except (ValueError, SyntaxError) as exc:
                    QMessageBox.warning(self, "Invalid analysis parameter",
                                        f"{analysis.title} — {name}: enter a number, None, or a tuple for a range-valued parameter.\n{exc}")
                    field.setFocus()
                    return
            
            parameters[analysis.key] = analysis_parameters

        self.changes_applied.emit(parameters)

    def showEvent(self, event):
        parent = self.parentWidget()
        if parent is not None:
            parent_frame = parent.frameGeometry()
            # Show the dialog to the right of the main window
            self.move(parent_frame.right() + 10, parent_frame.top())
        super().showEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
