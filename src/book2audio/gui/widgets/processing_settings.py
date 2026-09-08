"""PROCESSING: OCR mode, chapter detection, offline mode, optional narration
cleanup.

Deliberately does NOT include footnote/table/figure-caption toggles: the
backend has no such controls (footnote-marker stripping in
processing/clean.py is unconditional, and there is no table extraction at
all) -- exposing switches for either would be a GUI setting that does
nothing, which the spec for this app explicitly rules out.

The narration-cleanup control is deliberately NOT framed as "visual OCR
verification" or anything image-related -- it's a local, text-only
cleanup pass (processing/narration_cleanup.py). No page image is ever
sent anywhere.
"""

from PySide6.QtWidgets import QComboBox, QFormLayout, QCheckBox, QLabel, QLineEdit, QWidget


class ProcessingSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QFormLayout(self)

        self.ocr_mode_combo = QComboBox()
        self.ocr_mode_combo.addItem("Auto (OCR only if the text layer looks bad)", userData="auto")
        self.ocr_mode_combo.addItem("Force (always OCR)", userData="force")
        self.ocr_mode_combo.addItem("Never (only use the text layer)", userData="never")
        layout.addRow("OCR mode:", self.ocr_mode_combo)

        self.preserve_chapters_check = QCheckBox("Detect and preserve chapters")
        self.preserve_chapters_check.setChecked(True)
        layout.addRow("", self.preserve_chapters_check)

        self.offline_check = QCheckBox("Fully offline (never download a missing model)")
        self.offline_check.setChecked(True)
        layout.addRow("", self.offline_check)

        self.save_text_check = QCheckBox("Save readable text files (.raw.md / .cleaned.md / .narration.md)")
        self.save_text_check.setChecked(True)
        layout.addRow("", self.save_text_check)

        self.cleanup_combo = QComboBox()
        self.cleanup_combo.addItem("Off", userData="off")
        self.cleanup_combo.addItem("Conservative", userData="conservative")
        self.cleanup_combo.setCurrentIndex(0)
        self.cleanup_combo.currentIndexChanged.connect(self._on_cleanup_level_changed)
        layout.addRow("Qwen Cleanup:", self.cleanup_combo)

        self.cleanup_hint = QLabel("Local • Text only — no page images are analyzed")
        self.cleanup_hint.setStyleSheet("QLabel { color: palette(mid); }")
        layout.addRow("", self.cleanup_hint)

        self.cleanup_model_edit = QLineEdit("qwen3-vl:4b-instruct")
        self.cleanup_model_edit.setEnabled(False)
        layout.addRow("Ollama model:", self.cleanup_model_edit)

    def _on_cleanup_level_changed(self, _index: int) -> None:
        self.cleanup_model_edit.setEnabled(self.cleanup_combo.currentData() != "off")

    def ocr_mode(self) -> str:
        return self.ocr_mode_combo.currentData()

    def preserve_chapters(self) -> bool:
        return self.preserve_chapters_check.isChecked()

    def allow_download(self) -> bool:
        return not self.offline_check.isChecked()

    def save_text_outputs(self) -> bool:
        return self.save_text_check.isChecked()

    def narration_cleanup(self) -> bool:
        return self.cleanup_combo.currentData() != "off"

    def narration_cleanup_model(self) -> str:
        return self.cleanup_model_edit.text().strip() or "qwen3-vl:4b-instruct"
