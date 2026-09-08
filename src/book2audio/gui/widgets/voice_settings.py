"""VOICE: TTS engine selector (Chatterbox/Kokoro) with per-backend controls.

Chatterbox's reference-clip picker and Kokoro's named-voice dropdown are
mutually exclusive rows -- only the selected backend's actual controls are
shown, per "do not show Chatterbox-specific voice-reference controls if
Kokoro cannot use them."
"""

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

from book2audio.tts.chatterbox_backend import ENGINE_ID as CHATTERBOX_ID
from book2audio.tts.chatterbox_backend import ENGINE_NAME as CHATTERBOX_NAME
from book2audio.tts.chatterbox_backend import supported_languages as chatterbox_languages
from book2audio.tts.kokoro_backend import DEFAULT_VOICE as KOKORO_DEFAULT_VOICE
from book2audio.tts.kokoro_backend import ENGINE_ID as KOKORO_ID
from book2audio.tts.kokoro_backend import ENGINE_NAME as KOKORO_NAME
from book2audio.tts.kokoro_backend import VOICES as KOKORO_VOICES
from book2audio.tts.kokoro_backend import supported_languages as kokoro_languages


class VoiceSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._voice_reference: Path | None = None

        layout = QFormLayout(self)
        self._layout = layout

        self.engine_combo = QComboBox()
        self.engine_combo.addItem(f"{CHATTERBOX_NAME}  •  Local, Quality", userData=CHATTERBOX_ID)
        self.engine_combo.addItem(f"{KOKORO_NAME}  •  Local, Fast", userData=KOKORO_ID)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        layout.addRow("TTS engine:", self.engine_combo)

        self.language_combo = QComboBox()
        layout.addRow("Language:", self.language_combo)

        self.voice_row = QHBoxLayout()
        self.voice_label = QLabel("Default voice (no reference clip)")
        self.voice_label.setStyleSheet("QLabel { color: palette(mid); }")
        self.voice_row.addWidget(self.voice_label, stretch=1)
        self.voice_button = QPushButton("Choose Reference...")
        self.voice_button.clicked.connect(self._choose_voice)
        self.voice_row.addWidget(self.voice_button)
        self.clear_button = QPushButton("Use Default")
        self.clear_button.clicked.connect(self._clear_voice)
        self.clear_button.setEnabled(False)
        self.voice_row.addWidget(self.clear_button)
        layout.addRow("Voice:", self.voice_row)

        self.kokoro_voice_combo = QComboBox()
        for voice_id, label in KOKORO_VOICES.items():
            self.kokoro_voice_combo.addItem(f"{label}", userData=voice_id)
        default_idx = self.kokoro_voice_combo.findData(KOKORO_DEFAULT_VOICE)
        self.kokoro_voice_combo.setCurrentIndex(max(default_idx, 0))
        layout.addRow("Kokoro voice:", self.kokoro_voice_combo)

        self._on_engine_changed(0)

    def _on_engine_changed(self, _index: int) -> None:
        backend = self.engine_combo.currentData()
        is_chatterbox = backend == CHATTERBOX_ID

        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        langs = chatterbox_languages() if is_chatterbox else kokoro_languages()
        for code in sorted(langs, key=lambda c: langs[c]):
            self.language_combo.addItem(f"{langs[code]} ({code})", userData=code)
        default_idx = self.language_combo.findData("en")
        self.language_combo.setCurrentIndex(default_idx if default_idx >= 0 else 0)
        self.language_combo.blockSignals(False)

        # setRowVisible needs the exact widget/layout object originally
        # passed to addRow() -- a widget merely nested inside that layout
        # (e.g. voice_label inside voice_row) logs "Invalid widget" and
        # behaves unreliably; confirmed by testing both directly.
        self._layout.setRowVisible(self.voice_row, is_chatterbox)
        self._layout.setRowVisible(self.kokoro_voice_combo, not is_chatterbox)

    def backend(self) -> str:
        return self.engine_combo.currentData()

    def language_code(self) -> str:
        return self.language_combo.currentData()

    def voice_path(self) -> Path | None:
        return self._voice_reference

    def kokoro_voice(self) -> str:
        return self.kokoro_voice_combo.currentData()

    def _choose_voice(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Choose a reference voice clip", "", "Audio (*.wav *.mp3 *.flac);;All files (*)"
        )
        if path_str:
            self._voice_reference = Path(path_str)
            self.voice_label.setText(self._voice_reference.name)
            self.voice_label.setStyleSheet("")
            self.clear_button.setEnabled(True)

    def _clear_voice(self) -> None:
        self._voice_reference = None
        self.voice_label.setText("Default voice (no reference clip)")
        self.voice_label.setStyleSheet("QLabel { color: palette(mid); }")
        self.clear_button.setEnabled(False)
