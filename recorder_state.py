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
    advisory and inherited across fork, and it is honoured on both macOS and
    Linux, including a Docker bind mount, where it is held by the kernel on the
    underlying inode rather than by the filesystem being mounted.
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
    """Compare clips by resolved path so an absolute and a relative reference
    to the same file are one row, not two."""
    return str(Path(audio_path).resolve())


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
        kept = [row for row in rows if Path(row["audio_path"]).exists()]
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
    target = str(Path(audio_path).resolve())

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
