"""Runs the conversion pipeline on a background thread.

The Qt UI thread must never block on the pipeline (a book is thousands of
GPU-bound TTS calls). This is the only place the GUI touches
book2audio.pipeline -- everything else talks to this worker via signals.
"""

import threading

from PySide6.QtCore import QThread, Signal

from book2audio.pipeline.convert import (
    ChunkSynthesisError,
    ConversionCancelled,
    ConversionRequest,
    ProgressEvent,
    plan_conversion,
    run_conversion,
)


class ConversionWorker(QThread):
    progress = Signal(object)  # ProgressEvent
    plan_ready = Signal(object)  # ConversionPlan -- includes AI-review flags/skip reason
    finished_ok = Signal(str)  # output path
    failed = Signal(str, object)  # message, ChunkFailure | None
    cancelled = Signal()

    def __init__(self, request: ConversionRequest, parent=None):
        super().__init__(parent)
        self.request = request
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        on_progress = lambda e: self.progress.emit(e)  # noqa: E731
        try:
            plan, chapter_chunks = plan_conversion(self.request, on_progress=on_progress)
            self.plan_ready.emit(plan)

            output = run_conversion(
                self.request,
                chapter_chunks=chapter_chunks,
                on_progress=on_progress,
                should_cancel=self._cancel_event.is_set,
            )
            self.finished_ok.emit(str(output))
        except ConversionCancelled:
            self.cancelled.emit()
        except ChunkSynthesisError as e:
            self.failed.emit(str(e), e.failure)
        except Exception as e:  # noqa: BLE001 -- surface anything to the UI, never crash silently
            self.failed.emit(f"{type(e).__name__}: {e}", None)


class ModelSetupWorker(QThread):
    """Runs core.models.setup_models() on a background thread so the GUI
    doesn't freeze during multi-GB downloads. Lets a friend who doesn't use
    a terminal get set up entirely from the app."""

    progress = Signal(str)  # human-readable status line
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, tts: str, parent=None):
        super().__init__(parent)
        self.tts = tts

    def run(self) -> None:
        from book2audio.core.models import setup_models

        try:
            setup_models(progress=lambda msg: self.progress.emit(msg), tts=self.tts)
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001 -- surface anything to the UI, never crash silently
            self.failed.emit(f"{type(e).__name__}: {e}")


__all__ = ["ConversionWorker", "ModelSetupWorker", "ProgressEvent"]
