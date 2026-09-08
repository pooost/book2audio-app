"""OUTPUT: directory, filename, format (m4b only -- see module docstring below)."""

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QWidget

# The backend (book2audio.audio.mux) only builds .m4b -- ffmpeg could target
# other containers, but nothing in the pipeline exposes that today. One real
# option beats a dropdown with fake alternatives.
OUTPUT_FORMATS = ["m4b"]


class OutputSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QFormLayout(self)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(str(Path.home()))
        dir_row.addWidget(self.dir_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._choose_dir)
        dir_row.addWidget(browse_button)
        layout.addRow("Output directory:", dir_row)

        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("Defaults to the book's filename")
        layout.addRow("Output filename:", self.filename_edit)

        self.format_combo = QComboBox()
        self.format_combo.addItems(OUTPUT_FORMATS)
        layout.addRow("Format:", self.format_combo)

    def output_path(self, input_path: Path | None) -> Path:
        stem = self.filename_edit.text().strip() or (input_path.stem if input_path else "audiobook")
        ext = self.format_combo.currentText()
        return Path(self.dir_edit.text()).expanduser() / f"{stem}.{ext}"

    def _choose_dir(self) -> None:
        path_str = QFileDialog.getExistingDirectory(self, "Choose output directory", self.dir_edit.text())
        if path_str:
            self.dir_edit.setText(path_str)
