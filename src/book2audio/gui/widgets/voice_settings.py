"""VOICE: TTS engine (Chatterbox only, for now), language, voice reference file."""

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from book2audio.tts.chatterbox_backend import ENGINE_NAME, supported_languages


class VoiceSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._voice_path: Path | None = None

        layout = QFormLayout(self)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem(ENGINE_NAME)
        self.engine_combo.setEnabled(False)  # only one engine exists -- no point offering a fake choice
        layout.addRow("TTS engine:", self.engine_combo)

        self.language_combo = QComboBox()
        langs = supported_languages()
        for code in sorted(langs, key=lambda c: langs[c]):
            self.language_combo.addItem(f"{langs[code]} ({code})", userData=code)
        self.language_combo.setCurrentIndex(self.language_combo.findData("en"))
        layout.addRow("Language:", self.language_combo)

        voice_row = QHBoxLayout()
        self.voice_label = QLabel("Default voice (no reference clip)")
        self.voice_label.setStyleSheet("QLabel { color: palette(mid); }")
        voice_row.addWidget(self.voice_label, stretch=1)
        self.voice_button = QPushButton("Choose Reference...")
        self.voice_button.clicked.connect(self._choose_voice)
        voice_row.addWidget(self.voice_button)
        self.clear_button = QPushButton("Use Default")
        self.clear_button.clicked.connect(self._clear_voice)
        self.clear_button.setEnabled(False)
        voice_row.addWidget(self.clear_button)
        layout.addRow("Voice:", voice_row)

    def language_code(self) -> str:
        return self.language_combo.currentData()

    def voice_path(self) -> Path | None:
        return self._voice_path

    def _choose_voice(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Choose a reference voice clip", "", "Audio (*.wav *.mp3 *.flac);;All files (*)"
        )
        if path_str:
            self._voice_path = Path(path_str)
            self.voice_label.setText(self._voice_path.name)
            self.voice_label.setStyleSheet("")
            self.clear_button.setEnabled(True)

    def _clear_voice(self) -> None:
        self._voice_path = None
        self.voice_label.setText("Default voice (no reference clip)")
        self.voice_label.setStyleSheet("QLabel { color: palette(mid); }")
        self.clear_button.setEnabled(False)
