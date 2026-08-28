"""Which chunks are recorded, and where the cursor should sit.

Pure data model: no curses, no audio, no microphone. The recorder UI renders
what these functions return, and the controller mutates a cursor against them.
"""

import csv
import os
import re
import tempfile
from pathlib import Path

import whisper_pipeline as wp

RECORDED = "recorded"
SELECTED = "selected"
PENDING = "pending"


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
    """Per-line status. The cursor outranks 'recorded' so it stays visible."""
    return [
        SELECTED if index == cursor else RECORDED if index in recorded else PENDING
        for index in range(total)
    ]


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
    if not path.exists():
        return 0

    with path.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))

    kept = [row for row in rows if Path(row["audio_path"]).exists()]
    removed = len(rows) - len(kept)
    if removed:
        _write_rows(path, kept)
    return removed


def upsert_row(csv_path, audio_path, text, language):
    """Write this clip's row, replacing any existing row for the same file.

    Re-recording a line must not append a second row: train.py expects one row
    per chunk. The rewrite goes through a temp file and os.replace so an
    interrupted write cannot truncate the dataset.
    """
    path = Path(csv_path)
    target = str(Path(audio_path).resolve())

    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf8") as handle:
            rows = [row for row in csv.DictReader(handle)]

    replacement = dict(zip(wp.CSV_COLUMNS, (target, text, language)))
    for position, row in enumerate(rows):
        if _key(row["audio_path"]) == _key(target):
            rows[position] = replacement
            break
    else:
        rows.append(replacement)

    _write_rows(path, rows)


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
