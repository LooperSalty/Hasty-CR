"""Entry point: `python -m control_panel` from scripts/, or panel.cmd at the repo root."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from .widgets import STYLE
from .window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
CRASH_LOG = ROOT / "tmp" / "live" / "control_panel.log"


def _report(exc_type, exc, tb) -> None:
    """Log and show an uncaught exception instead of dying silently.

    PyQt6 aborts on an exception escaping a slot, and under pythonw there is
    no console, so without this the window just vanishes with no trace.
    """
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CRASH_LOG.open("a", encoding="utf-8") as stream:
            stream.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S}\n{text}\n")
    except OSError:
        pass
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "Erreur du panneau",
                             f"{exc_type.__name__}: {exc}\n\nDétails dans {CRASH_LOG}")


def main() -> int:
    sys.excepthook = _report
    app = QApplication(sys.argv)
    app.setApplicationName("Hasty-CR Control Panel")
    app.setStyleSheet(STYLE)
    window = MainWindow(ROOT)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
