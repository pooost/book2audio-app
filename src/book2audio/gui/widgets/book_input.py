"""BOOK INPUT: drag-and-drop area + Select Book button + selected filename +
optional page range (PDF only)."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

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

        page_row = QHBoxLayout()
        self.page_range_label = QLabel("Pages:")
        page_row.addWidget(self.page_range_label)
        self.page_range_edit = QLineEdit()
        self.page_range_edit.setPlaceholderText('All pages, or e.g. "1-10,15,20-25"')
        page_row.addWidget(self.page_range_edit, stretch=1)
        self.page_count_label = QLabel("")
        self.page_count_label.setStyleSheet("QLabel { color: palette(mid); }")
        page_row.addWidget(self.page_count_label)
        layout.addLayout(page_row)
        self._set_page_range_visible(False)

    def selected_path(self) -> Path | None:
        return self._selected_path

    def page_range(self) -> str | None:
        text = self.page_range_edit.text().strip()
        return text or None

    def set_path(self, path: Path) -> None:
        self._selected_path = path
        self.filename_label.setText(str(path))
        self.filename_label.setStyleSheet("")
        self.page_range_edit.clear()

        is_pdf = path.suffix.lower() == ".pdf"
        self._set_page_range_visible(is_pdf)
        if is_pdf:
            self._update_page_count(path)

        self.file_selected.emit(path)

    def _set_page_range_visible(self, visible: bool) -> None:
        self.page_range_label.setVisible(visible)
        self.page_range_edit.setVisible(visible)
        self.page_count_label.setVisible(visible)

    def _update_page_count(self, path: Path) -> None:
        try:
            from book2audio.ingest.page_select import get_page_count

            n = get_page_count(path)
            self.page_count_label.setText(f"({n} pages)")
        except Exception:
            self.page_count_label.setText("")

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
