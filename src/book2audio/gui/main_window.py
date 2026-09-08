"""The main Book2Audio window. Wires widgets to book2audio.pipeline via a
background worker thread -- no conversion logic lives here, only UI glue."""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from book2audio.gui.dialogs.doctor_dialog import DoctorDialog
from book2audio.gui.widgets.advanced_panel import AdvancedPanel
from book2audio.gui.widgets.book_input import BookInputWidget
from book2audio.gui.widgets.output_settings import OutputSettingsWidget
from book2audio.gui.widgets.processing_settings import ProcessingSettingsWidget
from book2audio.gui.widgets.progress_panel import ProgressPanel
from book2audio.gui.widgets.voice_settings import VoiceSettingsWidget
from book2audio.gui.workers import ConversionWorker
from book2audio.pipeline.convert import ConversionRequest


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Book2Audio")
        self.resize(560, 780)

        self._worker: ConversionWorker | None = None
        self._last_plan = None
        self._last_output_path: Path | None = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        self.book_input = BookInputWidget()
        layout.addWidget(self._boxed("Book Input", self.book_input))

        self.voice_settings = VoiceSettingsWidget()
        layout.addWidget(self._boxed("Voice", self.voice_settings))

        self.processing_settings = ProcessingSettingsWidget()
        layout.addWidget(self._boxed("Processing", self.processing_settings))

        self.output_settings = OutputSettingsWidget()
        layout.addWidget(self._boxed("Output", self.output_settings))

        self.advanced_panel = AdvancedPanel()
        layout.addWidget(self.advanced_panel)

        action_row = QHBoxLayout()
        self.convert_button = QPushButton("Convert to Audiobook")
        self.convert_button.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 12px; }")
        self.convert_button.clicked.connect(self._start_conversion)
        action_row.addWidget(self.convert_button, stretch=1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_conversion)
        self.cancel_button.setVisible(False)
        action_row.addWidget(self.cancel_button)
        layout.addLayout(action_row)

        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        results_row = QHBoxLayout()
        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self._open_output_folder)
        results_row.addWidget(self.open_folder_button)
        self.open_text_button = QPushButton("Open Final Text")
        self.open_text_button.clicked.connect(self._open_final_text)
        results_row.addWidget(self.open_text_button)
        self.open_audiobook_button = QPushButton("Open Audiobook")
        self.open_audiobook_button.clicked.connect(self._open_audiobook)
        results_row.addWidget(self.open_audiobook_button)
        layout.addLayout(results_row)
        self._set_results_buttons_visible(False)

        diagnostics_button = QPushButton("System / Diagnostics...")
        diagnostics_button.clicked.connect(self._open_diagnostics)
        layout.addWidget(diagnostics_button)

        layout.addStretch(1)

    @staticmethod
    def _boxed(title: str, widget: QWidget) -> QGroupBox:
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(widget)
        return box

    def _open_diagnostics(self) -> None:
        DoctorDialog(self).exec()

    def _build_request(self) -> ConversionRequest | None:
        input_path = self.book_input.selected_path()
        if input_path is None:
            QMessageBox.warning(self, "No book selected", "Select or drop a book first.")
            return None

        page_range = self.book_input.page_range()
        if page_range and input_path.suffix.lower() == ".pdf":
            from book2audio.ingest.page_select import PageRangeError, get_page_count, parse_page_range

            try:
                parse_page_range(page_range, get_page_count(input_path))
            except PageRangeError as e:
                QMessageBox.warning(self, "Invalid page range", str(e))
                return None

        if self.processing_settings.narration_cleanup():
            from book2audio.processing.narration_cleanup import DEFAULT_MODEL, is_ollama_available

            if not is_ollama_available():
                QMessageBox.warning(
                    self, "Ollama not reachable",
                    "Narration cleanup is on, but no local Ollama server was found at "
                    "http://localhost:11434.\n\nInstall Ollama and run "
                    f"`ollama pull {DEFAULT_MODEL}`, or turn narration cleanup off.",
                )
                return None

        backend = self.voice_settings.backend()
        if backend == "kokoro":
            from book2audio.tts.kokoro_backend import is_model_cached as kokoro_cached

            if not kokoro_cached(self.voice_settings.kokoro_voice()):
                QMessageBox.warning(
                    self, "Kokoro not set up",
                    f"Kokoro voice {self.voice_settings.kokoro_voice()!r} (or the core model) isn't "
                    "downloaded yet.\n\nRun `book2audio setup-models --tts kokoro` in a terminal, "
                    "then try again.",
                )
                return None
        elif backend == "chatterbox":
            from book2audio.tts.chatterbox_backend import is_model_cached as chatterbox_cached

            if not chatterbox_cached():
                QMessageBox.warning(
                    self, "Chatterbox not set up",
                    "Chatterbox model files aren't downloaded yet.\n\n"
                    "Run `book2audio setup-models` in a terminal, then try again.",
                )
                return None

        output_path = self.output_settings.output_path(input_path, page_range)

        return ConversionRequest(
            input_path=input_path,
            output_path=output_path,
            voice=self.voice_settings.voice_path(),
            kokoro_voice=self.voice_settings.kokoro_voice(),
            tts_backend=backend,
            language=self.voice_settings.language_code(),
            device="auto",
            max_chars=self.advanced_panel.max_chars(),
            title=None,
            author=None,
            cache_dir=self.advanced_panel.cache_dir(),
            ocr_mode=self.processing_settings.ocr_mode(),
            preserve_chapters=self.processing_settings.preserve_chapters(),
            allow_download=self.processing_settings.allow_download(),
            bitrate=self.advanced_panel.bitrate(),
            page_range=page_range,
            narration_cleanup=self.processing_settings.narration_cleanup(),
            narration_cleanup_model=self.processing_settings.narration_cleanup_model(),
            save_text_outputs=self.processing_settings.save_text_outputs(),
        )

    def _start_conversion(self) -> None:
        request = self._build_request()
        if request is None:
            return

        self._set_running(True)
        self.progress_panel.start()
        self._set_results_buttons_visible(False)

        self._last_plan = None
        self._last_output_path = request.output_path
        self._worker = ConversionWorker(request)
        self._worker.progress.connect(self.progress_panel.update_from_event)
        self._worker.plan_ready.connect(self._on_plan_ready)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_plan_ready(self, plan) -> None:
        self._last_plan = plan

    def _cancel_conversion(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)

    def closeEvent(self, event) -> None:
        # A QThread whose run() hasn't actually returned yet when its Python
        # wrapper gets garbage-collected aborts the process ("QThread:
        # Destroyed while thread is still running"). Signal emission from
        # run() (e.g. self.cancelled.emit()) happens before run() returns,
        # so closing the window right after a cancel/finish can race this.
        # QThread.wait() blocks until the OS thread has actually stopped.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        event.accept()

    def _set_running(self, running: bool) -> None:
        self.convert_button.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(True)

    def _on_finished(self, output_path: str) -> None:
        self._set_running(False)
        self.progress_panel.stop()
        self._last_output_path = Path(output_path)
        self._set_results_buttons_visible(True)

        message = f"Audiobook saved to:\n{output_path}"
        QMessageBox.information(self, "Done", message)

    def _on_failed(self, message: str, failure) -> None:
        self._set_running(False)
        self.progress_panel.stop()
        detail = ""
        if failure is not None:
            detail = (
                f"\n\nChapter {failure.chapter_index} ({failure.chapter_title!r}), "
                f"chunk {failure.chunk_index} failed.\n"
                "Earlier chunks are cached -- fix the issue and click Convert again to resume."
            )
        QMessageBox.critical(self, "Conversion failed", message + detail)

    def _on_cancelled(self) -> None:
        self._set_running(False)
        self.progress_panel.stop()
        QMessageBox.information(self, "Cancelled", "Conversion cancelled. Completed chunks are cached -- click Convert again to resume.")

    def _set_results_buttons_visible(self, visible: bool) -> None:
        self.open_folder_button.setVisible(visible)
        self.open_text_button.setVisible(visible)
        self.open_audiobook_button.setVisible(visible)

    def _open_output_folder(self) -> None:
        if self._last_output_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_path.parent)))

    def _open_final_text(self) -> None:
        # Prefer narration > cleaned > raw, per spec.
        if self._last_plan is not None and self._last_plan.text_outputs_saved:
            by_name = {p.name: p for p in self._last_plan.text_outputs_saved}
            for suffix in ("narration.md", "cleaned.md", "raw.md"):
                for name, path in by_name.items():
                    if name.endswith(suffix) and path.exists():
                        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
                        return
        QMessageBox.information(self, "No text file", "No saved text output is available for this run.")

    def _open_audiobook(self) -> None:
        if self._last_output_path is not None and self._last_output_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_path)))
        else:
            QMessageBox.information(self, "No audiobook", "No audiobook file is available for this run.")
