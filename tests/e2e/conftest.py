"""Drives the recorder as a real process under a pseudo-terminal.

curses only initialises against a real tty, and pytest's capture replaces
sys.stdout while curses writes to the fd directly. Running the recorder in a
child process attached to its own pty sidesteps both, and is the only way to
exercise what the stub-screen tests cannot: that curses actually translates
keystrokes and paints the screen.
"""

import contextlib
import os
import pty
import re
import select
import signal
import time

import pytest

import whisper_pipeline as wp

# The pty buffer must be drained continuously. A harness that stops reading
# while the child is still writing blocks the child on a full buffer, which
# looks exactly like an application hang.
POLL_SECONDS = 0.05
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][B0]|\x1b[>=]")

KEY_DOWN = b"\x1b[B"
KEY_UP = b"\x1b[A"
SPACE = b" "


SGR = re.compile(r"\x1b\[([0-9;]*)m")


def strip_ansi(text):
    return ANSI.sub("", text)


class RecorderProcess:
    """A recorder running on its own pty, driven by timed keystrokes."""

    def __init__(self, pid, fd):
        self.pid = pid
        self.fd = fd
        self.output = ""
        self._exited = False
        self._status = None

    def _pump(self):
        """Read whatever is pending and reap the child if it has finished."""
        ready, _, _ = select.select([self.fd], [], [], POLL_SECONDS)
        if ready:
            with contextlib.suppress(OSError):
                self.output += os.read(self.fd, 65536).decode(errors="replace")
        if not self._exited:
            done, status = os.waitpid(self.pid, os.WNOHANG)
            if done:
                self._exited, self._status = True, status

    def settle(self, seconds=1.5):
        """Let the child draw, draining throughout."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._pump()
        return self

    def press(self, key, settle=1.0):
        os.write(self.fd, key)
        return self.settle(settle)

    def wait_for_exit(self, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._exited:
            self._pump()
        return self._exited

    def kill(self):
        if not self._exited:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
            self._exited = True

    @property
    def screen(self):
        return strip_ansi(self.output)

    def styles_before(self, needle):
        """Every SGR sequence issued since the last one, for each time `needle`
        was drawn. Colour is the whole point of some assertions, so those cannot
        use the ANSI-stripped screen."""
        found = []
        for chunk in self.output.split(needle)[:-1]:
            codes = SGR.findall(chunk)
            found.append(codes[-1] if codes else "")
        return found


@pytest.fixture
def recorder(tmp_path):
    """Launch record_data.py against a temporary script and dataset."""
    import sys

    running = []

    def launch(text, lang="es", theme=None, columns=80, lines=24):
        script = tmp_path / "script.txt"
        script.write_text(text, encoding="utf8")

        argv = [
            sys.executable, str(wp.PROJECT_ROOT / "record_data.py"),
            "--text", str(script), "--lang", lang,
            "--out-dir", str(tmp_path / "data"),
            "--csv", str(tmp_path / "dataset.csv"),
        ]
        if theme is not None:
            argv += ["--theme", str(theme)]

        pid, fd = pty.fork()
        if pid == 0:                      # child: becomes the recorder
            os.environ["TERM"] = "xterm-256color"
            os.environ["LINES"], os.environ["COLUMNS"] = str(lines), str(columns)
            os.execv(argv[0], argv)

        process = RecorderProcess(pid, fd)
        running.append(process)
        return process.settle(2.0)

    yield launch

    for process in running:
        process.kill()


@pytest.fixture
def dataset_rows(tmp_path):
    """Read back whatever the recorder committed."""
    import csv

    def read():
        path = tmp_path / "dataset.csv"
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf8") as handle:
            return list(csv.DictReader(handle))

    return read


@pytest.fixture
def clips(tmp_path):
    def find():
        directory = tmp_path / "data"
        return sorted(p.name for p in directory.glob("*.wav")) if directory.is_dir() else []

    return find
