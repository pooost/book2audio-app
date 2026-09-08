"""GUI entry point: `book2audio-gui` on Linux/Windows/macOS (CUDA/MPS/CPU)."""

import sys

from PySide6.QtWidgets import QApplication

from book2audio.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Book2Audio")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
