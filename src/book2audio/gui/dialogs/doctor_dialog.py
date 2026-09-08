"""System / Diagnostics screen. The checks themselves are read-only; the
"Download Missing Models" button is the one place this screen can trigger a
download, and only on explicit click -- nothing here contacts the internet
on its own."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
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

        button_row = QHBoxLayout()
        rerun_button = QPushButton("Run Diagnostics")
        rerun_button.clicked.connect(self.run_diagnostics)
        button_row.addWidget(rerun_button)

        self.download_button = QPushButton("Download Missing Models...")
        self.download_button.clicked.connect(self._open_model_setup)
        self.download_button.setVisible(False)
        button_row.addWidget(self.download_button)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.run_diagnostics()

    def _open_model_setup(self) -> None:
        from book2audio.gui.dialogs.model_setup_dialog import ModelSetupDialog

        ModelSetupDialog(tts="all", parent=self).exec()
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

        # Only Chatterbox/Kokoro/OpenOCR are things this dialog can fetch --
        # a missing FFmpeg or an unavailable device is a system-level issue
        # setup_models() can't do anything about, so the button only shows
        # up when it would actually help.
        downloadable_names = {"Chatterbox Multilingual V3", "Kokoro", "OpenOCR models"}
        missing = [c.name for c in report.checks if not c.ok and not c.optional]
        missing_downloadable = [n for n in missing if n in downloadable_names]
        self.download_button.setVisible(bool(missing_downloadable))

        if report.all_ok:
            self.summary_label.setText("All checks passed.")
        elif missing_downloadable:
            self.summary_label.setText("Missing: " + ", ".join(missing) + ".")
        else:
            self.summary_label.setText(
                "Missing: " + ", ".join(missing) + " -- not something this app can download for you."
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
