"""PROGRESS DISPLAY: stage, chapter/chunk counters, bar, device, elapsed time."""

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

STAGE_LABELS = {
    "extracting": "Extracting text",
    "ocr": "Running OCR",
    "cleaning": "Cleaning text",
    "chapter_detection": "Detecting chapters",
    "ai_review": "AI review (local model)",
    "tts": "Narrating",
    "assembling": "Assembling .m4b",
    "finished": "Finished",
}


class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.stage_label = QLabel("Idle")
        self.stage_label.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(self.stage_label)

        self.chapter_label = QLabel("")
        layout.addWidget(self.chapter_label)

        self.chunk_label = QLabel("")
        layout.addWidget(self.chunk_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("QLabel { color: palette(mid); }")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        self._start_time: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._timer.start()
        self.bar.setRange(0, 0)  # indeterminate until we know a total
        self.stage_label.setText("Starting...")
        self.chapter_label.setText("")
        self.chunk_label.setText("")
        self.detail_label.setText("")

    def stop(self) -> None:
        self._timer.stop()

    def update_from_event(self, event) -> None:
        self.stage_label.setText(STAGE_LABELS.get(event.stage, event.stage))

        if event.chapter_total:
            self.chapter_label.setText(f"Chapter {event.chapter_index} / {event.chapter_total}")
        if event.chunk_total:
            self.chunk_label.setText(f"Chunk {event.chunk_index} / {event.chunk_total}")
            self.bar.setRange(0, event.chunk_total)
            self.bar.setValue(event.chunk_index)

        details = []
        if event.device:
            details.append(f"device: {event.device}")
        if event.message:
            details.append(event.message)
        self.detail_label.setText(" -- ".join(details))

        if event.stage == "finished":
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            self.stop()

    def _tick_elapsed(self) -> None:
        if self._start_time is None:
            return
        elapsed = int(time.monotonic() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        current = self.detail_label.text()
        stamp = f"{hours:02d}:{mins:02d}:{secs:02d} elapsed"
        # Replace a previous elapsed stamp rather than piling them up.
        parts = [p for p in current.split(" -- ") if not p.endswith("elapsed")]
        parts.append(stamp)
        self.detail_label.setText(" -- ".join(parts))
