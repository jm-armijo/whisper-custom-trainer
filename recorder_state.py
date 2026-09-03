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


def script_slug(script):
    """A script's name reduced to something a filename may contain.

    A qualified name is "en/general.txt": it carries a separator, a suffix, and
    whatever punctuation the file was given. Everything outside [A-Za-z0-9] is
    collapsed to a dash so the result is safe on any filesystem, and the
    language prefix and .txt suffix are dropped because clip_path already
    writes the language and every script ends in the same suffix.
    """
    stem = Path(str(script)).name
    if stem.endswith(".txt"):
        stem = stem[: -len(".txt")]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")
    # A name made entirely of punctuation would otherwise slug to "", putting
    # two such scripts back on one key - the collision this whole change exists
    # to stop.
    return slug or "script"


def clip_path(audio_dir, language, index, script=None):
    """The one filename a chunk's audio may occupy, for any index.

    Scoped by script as well as language: keyed on (language, index) alone,
    line 5 of en/general.txt and line 5 of en/software_engineering.txt named
    the same file, so recording either one made the other report progress it
    had never made - and a second take silently overwrote the first.

    `script` is optional so a caller with no script in hand (a legacy dataset,
    a test) still names the historical file; see recorded_indices for how those
    older names keep counting.
    """
    stem = f"{language}_{index:05d}" if script is None \
        else f"{language}_{script_slug(script)}_{index:05d}"
    return Path(audio_dir) / f"{stem}.wav"


def existing_clip_path(csv_path, audio_dir, language, index, script, texts=None):
    """The clip already on disk for this line, or where a new one would go.

    Playback and deletion must reach a take recorded before clips were scoped
    by script, or the user's existing library would be unplayable and
    undeletable while still occupying the line. A legacy file is only offered
    when the dataset shows it belongs to this script - the same provenance rule
    recorded_indices applies, so what the picker counts and what playback finds
    cannot disagree. A new take always lands on the scoped name.
    """
    scoped = clip_path(audio_dir, language, index, script)
    if scoped.exists():
        return scoped

    legacy = clip_path(audio_dir, language, index)
    if legacy.exists() and index in recorded_indices(
        csv_path, audio_dir, language, script, texts
    ):
        return legacy
    return scoped


def recorded_indices(csv_path, audio_dir, language, script=None, texts=None):
    """Indices backed by BOTH a CSV row and a wav still on disk.

    Requiring both means deleting a clip re-opens that line for recording,
    rather than leaving a green row pointing at a file that no longer exists.

    A row counts for this script when its filename carries this script's slug.
    Clips recorded before clips were scoped have no slug, and the dataset holds
    no other record of which script they came from - so their only remaining
    evidence of provenance is the row's own text, which `texts` supplies as
    {index: chunk text}. A legacy clip counts only where the text still matches
    the line it sits on, which keeps the user's existing takes green under the
    script they were read from and hides them from every sibling script.
    """
    path = Path(csv_path)
    if not path.exists():
        return set()

    found = set()
    with path.open(newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if row["language"] != language:
                continue
            index = _index_for(row, language, script, texts)
            if index is not None and \
                    wp.resolve_audio_path(row["audio_path"], audio_dir).exists():
                found.add(index)
    return found


def _index_for(row, language, script, texts):
    """The chunk this row occupies in the given script, or None if it is not ours."""
    name = wp.dataset_audio_path(row["audio_path"])

    if script is not None:
        prefix = f"{re.escape(language)}_{re.escape(script_slug(script))}"
        scoped = re.fullmatch(rf"{prefix}_(\d+)\.wav", name)
        if scoped:
            return int(scoped.group(1))

    legacy = re.fullmatch(rf"{re.escape(language)}_(\d+)\.wav", name)
    if not legacy:
        return None

    index = int(legacy.group(1))
    # No script asked for, or no chunk text to check against: the caller is
    # working on a single script's dataset the old way, so keep the old answer.
    if script is None or texts is None:
        return index
    return index if texts.get(index) == row["text"] else None


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
