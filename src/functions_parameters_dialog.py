import ast
from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QFrame, QMessageBox, QPushButton, QScrollArea, QSpinBox, QToolButton,
    QVBoxLayout, QWidget
)

from peak_finder import AnalysisFunction


class _DurationSpinBox(QSpinBox):
    """QSpinBox displaying seconds in `m:ss` format (e.g. `3:00`)."""

    def textFromValue(self, value: int) -> str:
        minutes, secs = divmod(max(0, value), 60)
        return f"{minutes}:{secs:02d}"

    def valueFromText(self, text: str) -> int:
        s = text.strip()
        if ":" in s:
            m_str, sec_str = s.split(":", 1)
            minutes = int(m_str) if m_str else 0
            secs = int(sec_str) if sec_str else 0
            return minutes * 60 + secs
        return int(s) * 60

    def validate(self, text: str, pos: int):
        s = text.strip()
        if not s:
            return (QValidator.State.Intermediate, text, pos)
        if s.isdigit():
            return (QValidator.State.Acceptable, text, pos)
        if ":" in s:
            parts = s.split(":")
            if (
                len(parts) == 2
                and (parts[0] == "" or parts[0].isdigit())
                and (parts[1] == "" or (parts[1].isdigit() and len(parts[1]) <= 2))
            ):
                if parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) < 60:
                    return (QValidator.State.Acceptable, text, pos)
                return (QValidator.State.Intermediate, text, pos)
        return (QValidator.State.Invalid, text, pos)


class _CheckboxLabel(QLabel):
    """QLabel that emits a `clicked` signal when it's clicked so it can be
    connected to a CheckBox.
    """
    clicked = Signal()

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class FunctionsParametersDialog(QDialog):
    """Dialog for selecting signal analysis functions and editing their parameters.
    The changes made here are only saved once "Apply" is clicked.
    """
    changes_applied = Signal(tuple, dict, int, bool)
    closed = Signal()

    FUNCS_AREA_WIDTH = 180
    PARAMS_AREA_WIDTH = 350

    def __init__(self,
                 analyses: Sequence[AnalysisFunction],
                 selected_analyses: Sequence[AnalysisFunction],
                 parent=None,
                 default_visible_slice_seconds: int = 240,
                 default_remove_invisible_points: bool = True):
        super().__init__(parent)

        self.analyses = tuple(analyses)
        selected_keys = {analysis.key for analysis in selected_analyses}

        self.fields: dict[str, dict[str, QLineEdit]] = {}
        self.sections: dict[str, QGroupBox] = {}
        self.checkboxes: dict[str, QCheckBox] = {}
        self.stored_parameters: Mapping[str, Mapping[str, object]] = {}

        self.setWindowTitle("Analysis functions and parameters")
        self.resize(self.FUNCS_AREA_WIDTH + self.PARAMS_AREA_WIDTH, 600)
        self.setMinimumWidth(self.FUNCS_AREA_WIDTH + self.PARAMS_AREA_WIDTH)

        funcs_widget = QWidget()
        funcs_layout = QVBoxLayout(funcs_widget)
        funcs_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add checkbox for each function
        for analysis in self.analyses:
            checkbox = QCheckBox()
            checkbox.setChecked(analysis.key in selected_keys)
            checkbox.setToolTip(analysis.function.__doc__) # Show the docstring as the tooltip

            # Separate label so the text can be wrapped
            label = _CheckboxLabel(analysis.title)
            label.setWordWrap(True)
            label.setBuddy(checkbox)
            label.setToolTip(analysis.function.__doc__)
            label.clicked.connect(checkbox.click)
            checkbox.setAccessibleName(analysis.title)

            checkbox_row = QWidget()
            checkbox_row_layout = QHBoxLayout(checkbox_row)
            checkbox_row_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_row_layout.addWidget(checkbox, 0, Qt.AlignmentFlag.AlignTop)
            checkbox_row_layout.addWidget(label, 1)

            self.checkboxes[analysis.key] = checkbox
            funcs_layout.addWidget(checkbox_row)
        
        funcs_layout.addStretch()

        funcs_scroll = QScrollArea()
        funcs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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

        # General slicing and display options at the top of the parameters panel
        general_section = QGroupBox("General graph options")
        general_form = QFormLayout(general_section)

        self.visible_slice_spinbox = _DurationSpinBox()
        self.visible_slice_spinbox.setRange(1, 600_000)
        self.visible_slice_spinbox.setSingleStep(60)
        self.visible_slice_spinbox.setValue(
            max(1, min(600_000, int(default_visible_slice_seconds)))
        )
        self.visible_slice_spinbox.setToolTip(
            "Length of the visible slice in m:ss (arrows step by ±1 min)"
        )
        general_form.addRow("Length of visible slice", self.visible_slice_spinbox)

        remove_invisible_tooltip = (
            "Auto-downsample plotted curves to screen resolution (peak mode) for better performance"
        )
        self.remove_invisible_points_checkbox = QCheckBox()
        self.remove_invisible_points_checkbox.setChecked(default_remove_invisible_points)
        self.remove_invisible_points_checkbox.setToolTip(remove_invisible_tooltip)
        self.remove_invisible_points_checkbox.setAccessibleName("Remove invisible points")

        remove_invisible_label = _CheckboxLabel("Remove invisible points")
        remove_invisible_label.setBuddy(self.remove_invisible_points_checkbox)
        remove_invisible_label.setToolTip(remove_invisible_tooltip)
        remove_invisible_label.clicked.connect(self.remove_invisible_points_checkbox.click)
        general_form.addRow(remove_invisible_label, self.remove_invisible_points_checkbox)

        params_layout.addWidget(general_section)

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
        """Evaluate the general options and parameters for each selected function and
        emit `changes_applied` if all values were valid.
        """
        try:
            visible_slice_seconds, remove_invisible_points = self.get_general_options()
            selected_analyses, parameters = self.get_selected_functions_and_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid analysis parameter", str(exc))
            return

        self.changes_applied.emit(
            selected_analyses, parameters, visible_slice_seconds, remove_invisible_points
        )

    def set_general_options(self, visible_slice_seconds: int, remove_invisible_points: bool):
        """Set the displayed general slicing and plot options."""
        clamped = max(self.visible_slice_spinbox.minimum(), min(self.visible_slice_spinbox.maximum(), int(visible_slice_seconds)))
        self.visible_slice_spinbox.setValue(clamped)
        self.remove_invisible_points_checkbox.setChecked(bool(remove_invisible_points))

    def get_general_options(self) -> tuple[int, bool]:
        """Return evaluated `(visible_slice_seconds, remove_invisible_points)`."""
        self.visible_slice_spinbox.interpretText()
        return self.visible_slice_spinbox.value(), self.remove_invisible_points_checkbox.isChecked()

    def set_selected_functions_and_parameters(self,
                                              functions: Sequence[AnalysisFunction],
                                              parameters: Mapping[str, Mapping[str, object]]):
        """Set the selected functions and their displayed parameter values.
        Store all the parameters as strings for later saving.
        """
        # Store parameters even for functions that are not selected
        self.stored_parameters = parameters
        
        selected_keys = {analysis.key for analysis in functions}
        
        for analysis in self.analyses:
            self.checkboxes[analysis.key].setChecked(analysis.key in selected_keys)
            
            for name, field in self.fields[analysis.key].items():
                value = parameters.get(analysis.key, {}).get(name, analysis.defaults[name])
                field.setText(str(value))

    def get_selected_functions_and_parameters(self) -> tuple[tuple[AnalysisFunction, ...], dict[str, dict[str, object]]]:
        """Get the selected functions and their evaluated parameter values.
        Raises a ValueError if a parameter was invalid.
        """
        selected_analyses = tuple(
            analysis
            for analysis in self.analyses
            if self.checkboxes[analysis.key].isChecked()
        )
        
        parameters = {}
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
                    field.setFocus()
                    raise ValueError(f"{analysis.title} - {name}: enter a number, None, or a tuple for a range-valued parameter.\n{exc}") from exc
            
            parameters[analysis.key] = analysis_parameters
        
        return selected_analyses, parameters

    def get_selected_functions_and_parameter_texts(self) -> tuple[tuple[AnalysisFunction, ...], dict[str, dict[str, str]]]:
        """Get the selected functions and their parameter field contents
        plus previously saved parameters as strings.
        """
        selected_analyses = tuple(
            analysis
            for analysis in self.analyses
            if self.checkboxes[analysis.key].isChecked()
        )
        
        # Combine the previously saved parameters and the latest inputs.
        # The reason we combine these like this is because parameters
        # for functions that are not selected are not evaluated
        # (it might be weird to evaluate and show errors for parameters
        # that are not visible)
        parameters = self.stored_parameters | {
            analysis.key: {
                name: field.text()
                for name, field in self.fields[analysis.key].items()
            }
            for analysis in selected_analyses
        }
        
        return selected_analyses, parameters

    def showEvent(self, event):
        """Display the dialog to the right of the parent widget
        and move the parent to the left if there's no space.
        """
        parent = self.parentWidget()
        if parent is not None:
            parent_frame = parent.frameGeometry()
            dialog_width = self.frameGeometry().width()
            dialog_x = parent_frame.right() + 10
            
            # Move parent window if this dialog would go off screen
            overflow = dialog_x + dialog_width - parent.screen().availableGeometry().right()
            if overflow > 0:
                parent.move(parent.x() - overflow, parent.y())
                parent_frame = parent.frameGeometry()
                dialog_x = parent_frame.right() + 10
            
            self.move(dialog_x, parent_frame.top())
        super().showEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
