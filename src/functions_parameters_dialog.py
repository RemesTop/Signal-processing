import ast
from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QFrame,
    QMessageBox, QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget
)

from peak_finder import AnalysisFunction

class FunctionsParametersDialog(QDialog):
    """Dialog for selecting signal analysis functions and editing their parameters."""
    changes_applied = Signal(tuple, dict)
    closed = Signal()

    FUNCS_AREA_WIDTH = 180
    PARAMS_AREA_WIDTH = 350

    def __init__(self, analyses: Sequence[AnalysisFunction], selected_analyses: Sequence[AnalysisFunction], parent=None):
        super().__init__(parent)

        self.analyses = tuple(analyses)
        selected_keys = {analysis.key for analysis in selected_analyses}

        self.fields: dict[str, dict[str, QLineEdit]] = {}
        self.sections: dict[str, QGroupBox] = {}
        self.checkboxes: dict[str, QCheckBox] = {}

        self.setWindowTitle("Analysis functions and parameters")
        self.resize(self.FUNCS_AREA_WIDTH + self.PARAMS_AREA_WIDTH, 600)
        self.setMinimumWidth(self.FUNCS_AREA_WIDTH + self.PARAMS_AREA_WIDTH)

        funcs_widget = QWidget()
        funcs_layout = QVBoxLayout(funcs_widget)
        funcs_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add checkbox for each function
        for analysis in self.analyses:
            checkbox = QCheckBox(analysis.title)
            checkbox.setChecked(analysis.key in selected_keys)
            checkbox.setToolTip(analysis.function.__doc__) # Show the docstring as the tooltip
            
            self.checkboxes[analysis.key] = checkbox
            funcs_layout.addWidget(checkbox)
        
        funcs_layout.addStretch()

        funcs_scroll = QScrollArea()
        funcs_scroll.setWidgetResizable(True)
        funcs_scroll.setFrameShape(QFrame.NoFrame)
        funcs_scroll.setFixedWidth(self.FUNCS_AREA_WIDTH)
        funcs_scroll.setWidget(funcs_widget)

        self.btn_toggle_funcs = QToolButton()
        self.btn_toggle_funcs.setText("Hide function selection")
        self.btn_toggle_funcs.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_toggle_funcs.setCheckable(True)
        self.btn_toggle_funcs.setChecked(True)
        self.btn_toggle_funcs.setArrowType(Qt.RightArrow)
        self.btn_toggle_funcs.setAutoRaise(True)
        self.btn_toggle_funcs.setStyleSheet("QToolButton:checked { background-color: transparent; }")

        params_panel = QWidget()
        params_panel_layout = QVBoxLayout(params_panel)
        params_panel_layout.setContentsMargins(0, 0, 0, 0)
        params_panel_layout.addWidget(self.btn_toggle_funcs)

        def on_funcs_toggled(visible: bool):
            """Toggle visibility of the function selection section and resize the window."""
            funcs_scroll.setVisible(visible)
            self.btn_toggle_funcs.setArrowType(Qt.RightArrow if visible else Qt.LeftArrow)
            self.btn_toggle_funcs.setText("Hide function selection" if visible else "Show function selection")
            
            if visible:
                self.resize(self.size() + QSize(self.FUNCS_AREA_WIDTH, 0))
            self.setMinimumWidth(self.FUNCS_AREA_WIDTH + self.PARAMS_AREA_WIDTH if visible else self.PARAMS_AREA_WIDTH)
            if not visible:
                self.resize(self.size() + QSize(-self.FUNCS_AREA_WIDTH, 0))

        self.btn_toggle_funcs.toggled.connect(on_funcs_toggled)

        params_content = QWidget()
        params_layout = QVBoxLayout(params_content)
        params_layout.setContentsMargins(0, 0, 6, 0)

        # Forms for editing the parameters
        for analysis in self.analyses:
            section = QGroupBox(analysis.title)
            form = QFormLayout(section)
            self.fields[analysis.key] = {}
            self.sections[analysis.key] = section
            
            # A QLineEdit for each parameter
            for parameter in analysis.parameters:
                field = QLineEdit("None" if parameter.default is None else str(parameter.default))
                field.setToolTip("Enter a number, None, or a tuple for a range-valued parameter")
                self.fields[analysis.key][parameter.name] = field
                form.addRow(parameter.label or parameter.name, field)

            section.setVisible(analysis.key in selected_keys)
            params_layout.addWidget(section)
            self.checkboxes[analysis.key].toggled.connect(section.setVisible)
        
        params_layout.addStretch()

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_content)
        params_scroll.setFrameShape(QFrame.NoFrame)
        params_scroll.setStyleSheet("QScrollArea { padding-top: 6; padding-bottom: 6; }")
        params_scroll.setContentsMargins(0, 6, 0, 6)
        params_panel_layout.addWidget(params_scroll, 1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_parameters)

        hbox_buttons = QHBoxLayout()
        hbox_buttons.addStretch()
        hbox_buttons.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        content_layout.addWidget(funcs_scroll)
        content_layout.addWidget(params_panel, 1)
        layout.addLayout(content_layout, 1)
        layout.addLayout(hbox_buttons)

    def apply_parameters(self):
        """Evaluate the parameters for each selected function
        and emit `apply_called` with the dictionary of those values.
        """
        parameters = {}
        selected_analyses = tuple(
            analysis
            for analysis in self.analyses
            if self.checkboxes[analysis.key].isChecked()
        )
        for analysis in selected_analyses:
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

        self.changes_applied.emit(selected_analyses, parameters)

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
