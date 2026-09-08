"""ADVANCED: settings that map onto real backend options only.

Chunk size, bitrate, and cache directory are genuine ConversionRequest
fields. OCR mode / TTS engine / language / offline mode already have
first-class controls in their own sections above, so they aren't repeated
here -- duplicate controls for the same setting is worse UX, not better.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from book2audio.processing.chunker import DEFAULT_MAX_CHARS


class AdvancedPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.toggle_button = QToolButton()
        self.toggle_button.setText("Advanced")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.clicked.connect(self._on_toggled)
        outer.addWidget(self.toggle_button)

        self.content = QWidget()
        self.content.setVisible(False)
        form = QFormLayout(self.content)

        self.max_chars_spin = QSpinBox()
        self.max_chars_spin.setRange(50, 5000)
        self.max_chars_spin.setSingleStep(50)
        self.max_chars_spin.setValue(DEFAULT_MAX_CHARS)
        form.addRow("Max characters per chunk:", self.max_chars_spin)

        self.bitrate_edit = QLineEdit("64k")
        form.addRow("Output bitrate:", self.bitrate_edit)

        cache_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setPlaceholderText("Defaults to <output>_cache next to the .m4b")
        cache_row.addWidget(self.cache_dir_edit, stretch=1)
        cache_browse = QPushButton("Browse...")
        cache_browse.clicked.connect(self._choose_cache_dir)
        cache_row.addWidget(cache_browse)
        form.addRow("Cache directory:", cache_row)

        outer.addWidget(self.content)

    def _on_toggled(self, checked: bool) -> None:
        self.content.setVisible(checked)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def max_chars(self) -> int:
        return self.max_chars_spin.value()

    def bitrate(self) -> str:
        return self.bitrate_edit.text().strip() or "64k"

    def cache_dir(self) -> Path | None:
        text = self.cache_dir_edit.text().strip()
        return Path(text).expanduser() if text else None

    def _choose_cache_dir(self) -> None:
        path_str = QFileDialog.getExistingDirectory(self, "Choose cache directory", str(Path.home()))
        if path_str:
            self.cache_dir_edit.setText(path_str)
