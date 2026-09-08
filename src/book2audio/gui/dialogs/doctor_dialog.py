"""System / Diagnostics screen. Read-only -- never downloads anything."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class DoctorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System / Diagnostics")
        self.resize(560, 420)

        layout = QVBoxLayout(self)

        self.summary_label = QLabel("Checking...")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Check", "Result"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        rerun_button = QPushButton("Run Diagnostics")
        rerun_button.clicked.connect(self.run_diagnostics)
        layout.addWidget(rerun_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.run_diagnostics()

    def run_diagnostics(self) -> None:
        from book2audio.core.doctor import run_doctor

        report = run_doctor()
        categories = (("system", "SYSTEM"), ("text", "TEXT"), ("tts", "TTS"))
        total_rows = len(report.checks) + len(categories)
        self.table.setRowCount(total_rows)

        row = 0
        for category, label in categories:
            header_item = QTableWidgetItem(label)
            header_item.setFont(_bold_font(header_item))
            self.table.setItem(row, 0, header_item)
            self.table.setItem(row, 1, QTableWidgetItem(""))
            row += 1

            for check in report.by_category(category):
                name_item = QTableWidgetItem(f"    {check.name}")
                if check.ok:
                    prefix, color = "OK -- ", _green()
                elif check.optional:
                    prefix, color = "OPTIONAL -- ", _neutral()
                else:
                    prefix, color = "MISSING -- ", _red()
                result_item = QTableWidgetItem(prefix + check.detail)
                result_item.setForeground(color)
                self.table.setItem(row, 0, name_item)
                self.table.setItem(row, 1, result_item)
                row += 1

        self.table.resizeColumnsToContents()

        if report.all_ok:
            self.summary_label.setText("All checks passed.")
        else:
            missing = [c.name for c in report.checks if not c.ok and not c.optional]
            self.summary_label.setText(
                "Missing: " + ", ".join(missing) + ". Run `book2audio setup-models` in a terminal if a model is missing."
            )


def _green():
    from PySide6.QtGui import QColor

    return QColor("#2e7d32")


def _red():
    from PySide6.QtGui import QColor

    return QColor("#c62828")


def _neutral():
    from PySide6.QtGui import QColor

    return QColor("#9e9e9e")


def _bold_font(item):
    font = item.font()
    font.setBold(True)
    return font
