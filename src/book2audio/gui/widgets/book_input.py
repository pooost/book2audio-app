"""BOOK INPUT: drag-and-drop area + Select Book button + selected filename."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from book2audio.ingest.markitdown_backend import SUPPORTED_SUFFIXES as MARKITDOWN_SUFFIXES
from book2audio.ocr.backend import IMAGE_SUFFIXES

SUPPORTED_SUFFIXES = MARKITDOWN_SUFFIXES | IMAGE_SUFFIXES


class BookInputWidget(QWidget):
    file_selected = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._selected_path: Path | None = None

        layout = QVBoxLayout(self)

        self.drop_label = QLabel("Drag a book here\n(.pdf, .epub, .docx, .txt, or a scanned image)")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet(
            "QLabel { border: 2px dashed palette(mid); border-radius: 8px; padding: 24px; }"
        )
        self.drop_label.setMinimumHeight(100)
        layout.addWidget(self.drop_label)

        self.select_button = QPushButton("Select Book...")
        self.select_button.clicked.connect(self._open_file_dialog)
        layout.addWidget(self.select_button)

        self.filename_label = QLabel("No book selected")
        self.filename_label.setStyleSheet("QLabel { color: palette(mid); }")
        layout.addWidget(self.filename_label)

    def selected_path(self) -> Path | None:
        return self._selected_path

    def set_path(self, path: Path) -> None:
        self._selected_path = path
        self.filename_label.setText(str(path))
        self.filename_label.setStyleSheet("")
        self.file_selected.emit(path)

    def _open_file_dialog(self) -> None:
        patterns = " ".join(f"*{s}" for s in sorted(SUPPORTED_SUFFIXES))
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select a book", "", f"Supported books ({patterns});;All files (*)"
        )
        if path_str:
            self.set_path(Path(path_str))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() in SUPPORTED_SUFFIXES or path.is_dir():
            self.set_path(path)
        else:
            self.filename_label.setText(f"Unsupported file type: {path.suffix}")
