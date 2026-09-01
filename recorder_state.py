"""Which chunks are recorded, and where the cursor should sit.

Pure data model: no curses, no audio, no microphone. The recorder UI renders
what these functions return, and the controller mutates a cursor against them.
"""

import contextlib
import csv
import fcntl
import os
import re
import tempfile
from pathlib import Path

import whisper_pipeline as wp

RECORDED = "recorded"
SELECTED = "selected"
RECORDED_SELECTED = "recorded_selected"
PENDING = "pending"


LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def dataset_lock(csv_path):
    """Serialise one read-modify-write of the dataset against every other.

    fcntl.flock rather than a threading.Lock because the writers are in
    different *processes*: record_data.py drives the terminal recorder while
    recorder_server.py answers the phone, and both upsert into the same
    dataset.csv. An in-process lock cannot see the other process at all, and a
    ThreadingHTTPServer alone would still lose rows to it.

    The lock lives in a sidecar file rather than on dataset.csv itself because
    every write replaces that inode (see _write_rows); a descriptor held on the
    replaced file would guard a path nobody writes to any more. flock is
    advisory and inherited across fork, and it is honoured between processes
    sharing a kernel: host-to-host, container-to-container, and between two
    containers sharing the bind mount. That last case is the deployment the
    guard exists for - on the DietPi box the terminal recorder and the
    container run against one Linux kernel, so both sides lock the same inode.

    It does NOT serialise a macOS host process against a container on Docker
    Desktop. That bind mount is `fakeowner` over VirtioFS and the container
    runs under a separate linuxkit VM kernel, so the two sides lock different
    inodes in different kernels: a containerised writer completes a save well
    inside a lock the host is still holding. Recording from the browser and the
    terminal at the same time is therefore safe on the box and unguarded on a
    Mac - run one front end at a time when developing locally.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + LOCK_SUFFIX)

    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def clip_path(audio_dir, language, index):
    """The one filename a chunk's audio may occupy, for any index."""
    return Path(audio_dir) / f"{language}_{index:05d}.wav"


def recorded_indices(csv_path, audio_dir, language):
    """Indices backed by BOTH a CSV row and a wav still on disk.

    Requiring both means deleting a clip re-opens that line for recording,
    rather than leaving a green row pointing at a file that no longer exists.
    """
    path = Path(csv_path)
    if not path.exists():
        return set()

    found = set()
    with path.open(newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if row["language"] != language:
                continue
            match = re.search(rf"{language}_(\d+)\.wav$", row["audio_path"])
            if match and clip_path(audio_dir, language, int(match.group(1))).exists():
                found.add(int(match.group(1)))
    return found


def first_unrecorded(total, recorded):
    """The lowest gap, so a skipped chunk is revisited rather than lost."""
    for index in range(total):
        if index not in recorded:
            return index
    return max(total - 1, 0)


def chunk_statuses(total, recorded, cursor):
    """Per-line status.

    A finished line under the cursor is its own status rather than plain
    SELECTED: collapsing the two left a line yellow once its take was saved,
    so the screen said 'read this next' about work already done.
    """
    return [_status(index in recorded, index == cursor) for index in range(total)]


def _status(is_recorded, is_cursor):
    if is_recorded:
        return RECORDED_SELECTED if is_cursor else RECORDED
    return SELECTED if is_cursor else PENDING


def _key(audio_path):
    """Compare clips by filename, so a row written on another machine and one
    written here name the same clip rather than becoming two rows.

    Every clip lives directly in one audio directory, so the filename already
    identifies it uniquely; comparing resolved paths instead would split a
    dataset the moment it moved between the container and the laptop.
    """
    return wp.dataset_audio_path(audio_path)


def prune_missing(csv_path, audio_dir):
    """Drop rows whose audio file is gone, returning how many were removed.

    recorded_indices deliberately reopens a line when its wav is deleted; without
    this the dataset would keep a row pointing at a missing file and train.py
    would fail on it.
    """
    path = Path(csv_path)
    with dataset_lock(path):
        if not path.exists():
            return 0

        rows = _read_rows(path)
        kept = [
            row for row in rows
            if wp.resolve_audio_path(row["audio_path"], audio_dir).exists()
        ]
        removed = len(rows) - len(kept)
        if removed:
            _write_rows(path, kept)
        return removed


def upsert_row(csv_path, audio_path, text, language):
    """Write this clip's row, replacing any existing row for the same file.

    Re-recording a line must not append a second row: train.py expects one row
    per chunk. The rewrite goes through a temp file and os.replace so an
    interrupted write cannot truncate the dataset, and the whole
    read-modify-write is held under dataset_lock so a concurrent save cannot
    read the same 'before' image and overwrite this one's row.
    """
    path = Path(csv_path)
    target = wp.dataset_audio_path(audio_path)

    with dataset_lock(path):
        rows = _read_rows(path) if path.exists() else []

        replacement = dict(zip(wp.CSV_COLUMNS, (target, text, language), strict=True))
        for position, row in enumerate(rows):
            if _key(row["audio_path"]) == _key(target):
                rows[position] = replacement
                break
        else:
            rows.append(replacement)

        _write_rows(path, rows)


def _read_rows(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path, rows):
    """Rewrite the dataset atomically so an interrupted write cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf8", dir=path.parent, delete=False
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(wp.CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        os.unlink(handle.name)
        raise
