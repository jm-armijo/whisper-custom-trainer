"""Which scripts are available to read, and how far each one has got.

Pure data model, in the same spirit as recorder_state: no HTTP, no JSON, no
framework. The web server in recorder_server.py is the only caller that knows a
request exists, so this file stays testable without a socket - and a second
front end could reuse it unchanged.

recorder_state owns the per-chunk bookkeeping and is used as-is; this module
adds only what a multi-script picker needs on top of it.
"""

from pathlib import Path

import recorder_state as rs
import whisper_pipeline as wp

SCRIPT_SUFFIX = ".txt"


class ScriptNotFound(wp.PipelineError):
    """Named something that does not exist, as opposed to something invalid.

    An exception type rather than a message a caller has to match on: an HTTP
    adapter needs to tell 'gone' from 'malformed' to pick a status, and doing
    that by string-sniffing would turn a reworded error into a wrong code.
    """


def list_scripts(scripts_dir):
    """The readable scripts in a directory, sorted by name.

    A missing directory lists nothing rather than raising: scripts/ is
    gitignored and user-managed, so a fresh clone legitimately has none.
    """
    directory = Path(scripts_dir)
    if not directory.is_dir():
        return []

    # The language directory is the label, so a file loose at the top level or
    # under an unsupported language has nowhere to be filed and is skipped
    # rather than offered as an unrecordable script.
    return [
        {"name": f"{language}/{path.name}", "path": path, "language": language}
        for language in sorted(wp.SUPPORTED_LANGUAGES)
        for path in sorted((directory / language).glob(f"*{SCRIPT_SUFFIX}"))
        if path.is_file()
    ]


def resolve_script(scripts_dir, name):
    """The path a requested script name refers to, or an error if it escapes.

    The server binds to a LAN address, so a name arriving over the wire is
    untrusted. Both halves matter: a bare name never containing a separator,
    and a resolved path still inside the directory, which is what stops a
    symlink pointing outside from being read.
    """
    directory = Path(scripts_dir).resolve()
    language, _, filename = name.partition("/")
    if language not in wp.SUPPORTED_LANGUAGES or _is_unsafe_name(filename):
        raise wp.PipelineError(f"Invalid script name: {name!r}")

    # Resolved before comparing, so a symlink pointing out of the tree is
    # caught by the parent check rather than followed.
    path = (directory / language / filename).resolve()
    if path.parent != (directory / language) or path.suffix != SCRIPT_SUFFIX:
        raise wp.PipelineError(f"Invalid script name: {name!r}")
    if not path.is_file():
        raise ScriptNotFound(f"Script not found: {name}")
    return path


def _is_unsafe_name(name):
    """Reject anything that is not a bare filename before touching the disk."""
    return (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    )


def script_progress(path, csv_path, audio_dir, language, name=None):
    """Chunks, which are recorded, and the counts a picker needs to show.

    Chunking here rather than caching it means the dataset and the script file
    stay the only sources of truth, so a script edited between sessions is
    picked up without a sidecar state file - the same property recorder_state
    gets by deriving 'recorded' from the CSV and the wavs.
    """
    script = Path(path)
    if not script.is_file():
        raise ScriptNotFound(f"Script not found: {script}")

    chunks = wp.chunk_text(script.read_text(encoding="utf8"))
    # The qualified name is what scopes a clip to this script; falling back to
    # the filename keeps a caller that passes only a path working, and two
    # languages cannot collide because the language is in the key too.
    recorded = rs.recorded_indices(
        csv_path, audio_dir, language, name or script.name, dict(enumerate(chunks))
    )
    # Only indices the script still has count: a shortened script would
    # otherwise report more takes recorded than there are lines to read.
    recorded = {index for index in recorded if index < len(chunks)}

    return {
        # The caller's qualified name ("es/a.txt"), not the bare filename: it
        # is the identifier the client sends back on every subsequent request,
        # and two languages may hold the same filename.
        "name": name or script.name,
        "language": language,
        "chunks": chunks,
        "recorded": recorded,
        "total": len(chunks),
        "recorded_count": len(recorded),
        "next_index": rs.first_unrecorded(len(chunks), recorded),
        # An empty script is not 'done': zero of zero would otherwise hide a
        # file whose text failed to chunk behind a complete badge.
        "complete": bool(chunks) and len(recorded) == len(chunks),
    }


def chunk_view(chunks, recorded):
    """Each chunk with its state, as data - the caller decides how it looks."""
    return [
        {"index": index, "text": text, "recorded": index in recorded}
        for index, text in enumerate(chunks)
    ]
