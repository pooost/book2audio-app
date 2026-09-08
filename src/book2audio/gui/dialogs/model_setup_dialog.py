"""First-time / on-demand model download screen -- lets someone with no
terminal get the app fully set up by clicking a button and waiting. Wraps
core.models.setup_models() (the same explicit, user-requested download path
`book2audio setup-models` uses) on a background thread so the window stays
responsive during multi-GB downloads."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from book2audio.gui.workers import ModelSetupWorker


class ModelSetupDialog(QDialog):
    def __init__(self, tts: str = "all", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Models")
        self.resize(520, 360)
        self.setModal(True)

        self._tts = tts
        self._worker: ModelSetupWorker | None = None

        layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "This downloads the text-to-speech and OCR models this app needs. "
            "It's a one-time download (a few GB) -- after this, the app works "
            "fully offline. Requires an internet connection now."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate -- setup_models() has no byte-level progress
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.start_button = QPushButton("Download Now")
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)
        layout.addWidget(self.buttons)

    def _start(self) -> None:
        self.start_button.setEnabled(False)
        self.buttons.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log.clear()

        self._worker = ModelSetupWorker(self._tts)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _on_finished(self) -> None:
        self.progress_bar.setVisible(False)
        self.log.appendPlainText("")
        self.log.appendPlainText("Done. Models are cached for offline use.")
        self.buttons.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.log.appendPlainText("")
        self.log.appendPlainText(f"Failed: {message}")
        self.start_button.setEnabled(True)
        self.buttons.setEnabled(True)

    def closeEvent(self, event) -> None:
        # Same reasoning as MainWindow.closeEvent: destroying a QThread
        # wrapper before its OS thread actually returns aborts the process.
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        event.accept()
