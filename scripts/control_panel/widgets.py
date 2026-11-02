"""Small reusable pieces of the window."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

GREEN, ORANGE, RED, GREY = "#3fb950", "#d29922", "#f85149", "#8b949e"

STYLE = """
QWidget { background: #0d1117; color: #e6edf3; font-size: 10pt; }
QGroupBox { border: 1px solid #30363d; border-radius: 8px; margin-top: 14px;
            padding: 10px 8px 8px 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #58a6ff; }
QPushButton { background: #21262d; border: 1px solid #30363d; border-radius: 6px;
              padding: 6px 12px; }
QPushButton:hover { background: #30363d; }
QPushButton:disabled { color: #6e7681; background: #161b22; }
QPushButton#start { background: #238636; border-color: #2ea043; font-weight: bold; }
QPushButton#start:hover { background: #2ea043; }
QPushButton#stop { background: #b62324; border-color: #da3633; font-weight: bold; }
QPushButton#stop:hover { background: #da3633; }
QPushButton#start:disabled, QPushButton#stop:disabled {
    background: #161b22; border-color: #30363d; color: #6e7681; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #161b22; border: 1px solid #30363d; border-radius: 5px; padding: 4px; }
QPlainTextEdit { background: #010409; border: 1px solid #30363d; border-radius: 6px;
                 font-family: Consolas, monospace; font-size: 9pt; }
QLabel#hint { color: #8b949e; }
"""


def dot(colour: str, text: str) -> str:
    """Rich text: a coloured status bullet followed by `text`."""
    return f'<span style="color:{colour}">●</span> {text}'


class Background(QObject):
    """Runs one call at a time off the UI thread; the result arrives as `done`.

    adb and the process table can each take a noticeable fraction of a second,
    which would freeze the window if run on the UI thread. A call that is
    still in flight makes `submit` a no-op rather than queue a backlog. A
    raised exception is delivered as the result so the receiver can show it.
    """

    done = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._busy = False

    def submit(self, call: Callable, *args) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def run() -> None:
            try:
                result = call(*args)
            except Exception as exc:        # delivered, not swallowed
                result = exc
            with self._lock:
                self._busy = False
            self.done.emit(result)

        threading.Thread(target=run, daemon=True).start()
        return True


class ScreenPreview(QLabel):
    """The emulator's screen, scaled to fit, or a placeholder message."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(216, 384)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#010409; border:1px solid #30363d; border-radius:6px;")
        self._pixmap: Optional[QPixmap] = None
        self.show_message("Aperçu désactivé")

    def show_message(self, text: str) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def show_image(self, image: QImage) -> None:
        """`image` was decoded off the UI thread; only the cheap conversion is here."""
        self._pixmap = QPixmap.fromImage(image)
        self._rescale()

    def resizeEvent(self, event) -> None:      # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            # Fast scaling: smooth filtering of a 1080x1920 frame on the UI
            # thread costs more than the preview is worth.
            self.setPixmap(self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))
