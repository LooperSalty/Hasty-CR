"""Starting, stopping and finding bot processes.

Liveness comes from the process table, not only from the handle the panel
kept: a bot started from a terminal or `bot.cmd` is still a bot that will
tap the screen, and the panel must not start a second one on top of it.

On Windows a venv's `python.exe` is a launcher that re-executes the base
interpreter with the same arguments, so every bot shows up twice. The scan
keeps the outermost of each pair, and stopping kills the whole tree.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import psutil
from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

from .bots import bot_for_process

ROOT = Path(__file__).resolve().parents[2]
UNREADABLE = Path("<unreadable cwd>")      # makes relative script paths never match
CREATE_TIME_TOLERANCE = 0.01


@dataclass(frozen=True)
class RunningBot:
    pid: int
    bot: str
    started: float


def _bot_of(process: psutil.Process, root: Path) -> Optional[str]:
    try:
        name = (process.name() or "").lower()
        if not name.startswith("python"):
            return None
        cmdline = process.cmdline()
        try:
            cwd = Path(process.cwd())
        except (psutil.AccessDenied, OSError):
            cwd = UNREADABLE
        return bot_for_process(cmdline, cwd, root)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def find_running_bots(root: Path = ROOT) -> list[RunningBot]:
    found = []
    for process in psutil.process_iter():
        bot = _bot_of(process, root)
        if bot is None:
            continue
        try:
            parent = process.parent()
            if parent is not None and _bot_of(parent, root) is not None:
                continue            # the inner half of a venv launcher pair
            found.append(RunningBot(process.pid, bot, process.create_time()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def kill_tree(pid: int) -> list[int]:
    """Kill `pid` and its children; returns the pids still alive afterwards."""
    if pid <= 4 or pid == os.getpid():
        return []
    try:
        root = psutil.Process(pid)
        victims = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return []
    for victim in victims:
        try:
            victim.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(victims, timeout=5)
    return [process.pid for process in alive]


def kill_bot(bot: RunningBot, root: Path = ROOT) -> list[int]:
    """Kill a bot found by a scan, but only if its pid still belongs to it.

    A scan can be seconds old and Windows reuses pids quickly, so the process
    behind the pid must still have the bot's start time and command line.
    """
    try:
        process = psutil.Process(bot.pid)
        if (abs(process.create_time() - bot.started) > CREATE_TIME_TOLERANCE
                or _bot_of(process, root) is None):
            return []
    except psutil.NoSuchProcess:
        return []
    return kill_tree(bot.pid)


class BotRunner(QObject):
    """One bot started by the panel, its output streamed line by line."""

    line = pyqtSignal(str)
    started = pyqtSignal(int)
    finished = pyqtSignal(int)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._buffer = ""

    @property
    def running(self) -> bool:
        return (self._process is not None
                and self._process.state() != QProcess.ProcessState.NotRunning)

    @property
    def pid(self) -> Optional[int]:
        return int(self._process.processId()) if self.running else None

    def start(self, command: Sequence[str], cwd: Path, env: dict[str, str]) -> None:
        if self.running:
            raise RuntimeError("un bot est déjà lancé depuis le panneau")
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in env.items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._buffer = ""
        process.start(command[0], list(command[1:]))
        if process.waitForStarted(5000):
            self.started.emit(int(process.processId()))

    def stop(self) -> None:
        """Kill the bot without blocking the UI; `finished` fires once it is gone.

        The QProcess keeps the process handle open until then, so the pid
        cannot be recycled under the kill.
        """
        if self.running:
            pid = int(self._process.processId())
            threading.Thread(target=self._kill, args=(pid,), daemon=True).start()

    def _kill(self, pid: int) -> None:
        alive = kill_tree(pid)
        if alive:
            self.line.emit(f"[panneau] impossible d'arrêter les processus {alive} "
                           f"(lancés en administrateur ?)")

    def _read(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        self._buffer += data.replace("\r\n", "\n")
        *complete, self._buffer = self._buffer.split("\n")
        for text in complete:
            if text.strip():
                self.line.emit(text)

    def _on_finished(self, code: int, _status) -> None:
        if self._buffer.strip():
            self.line.emit(self._buffer)
        self._buffer = ""
        self.finished.emit(code)

    def _on_error(self, error) -> None:
        self.line.emit(f"[panneau] erreur de processus : {error.name}")


def default_env(serial: str) -> dict[str, str]:
    env = {"HASTYCR_SERIAL": serial, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    instance = os.environ.get("HASTYCR_INSTANCE")
    if instance:
        env["HASTYCR_INSTANCE"] = instance
    return env
