"""PROCESSING: OCR mode, chapter detection, offline mode, optional AI review.

Deliberately does NOT include footnote/table/figure-caption toggles: the
backend has no such controls (footnote-marker stripping in
processing/clean.py is unconditional, and there is no table extraction at
all) -- exposing switches for either would be a GUI setting that does
nothing, which the spec for this app explicitly rules out.
"""

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QWidget


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

        self.ai_review_check = QCheckBox("AI review: fix OCR errors with a local model before narrating")
        self.ai_review_check.setChecked(False)
        self.ai_review_check.toggled.connect(self._on_ai_review_toggled)
        layout.addRow("", self.ai_review_check)

        self.ai_review_model_edit = QLineEdit("llama3.2")
        self.ai_review_model_edit.setEnabled(False)
        layout.addRow("Ollama model:", self.ai_review_model_edit)

    def _on_ai_review_toggled(self, checked: bool) -> None:
        self.ai_review_model_edit.setEnabled(checked)

    def ocr_mode(self) -> str:
        return self.ocr_mode_combo.currentData()

    def preserve_chapters(self) -> bool:
        return self.preserve_chapters_check.isChecked()

    def allow_download(self) -> bool:
        return not self.offline_check.isChecked()

    def ai_review(self) -> bool:
        return self.ai_review_check.isChecked()

    def ai_review_model(self) -> str:
        return self.ai_review_model_edit.text().strip() or "llama3.2"
