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

        self.table.setRowCount(len(report.checks))
        for row, check in enumerate(report.checks):
            name_item = QTableWidgetItem(check.name)
            result_item = QTableWidgetItem(("OK -- " if check.ok else "MISSING -- ") + check.detail)
            result_item.setForeground(_green() if check.ok else _red())
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, result_item)
        self.table.resizeColumnsToContents()

        if report.all_ok:
            self.summary_label.setText("All checks passed.")
        else:
            missing = [c.name for c in report.checks if not c.ok]
            self.summary_label.setText(
                "Missing: " + ", ".join(missing) + ". Run `book2audio setup-models` in a terminal if a model is missing."
            )


def _green():
    from PySide6.QtGui import QColor

    return QColor("#2e7d32")


def _red():
    from PySide6.QtGui import QColor

    return QColor("#c62828")
