"""HTTP recorder: read a script and record takes from a phone browser.

The adapter layer, in the same role record_data.py plays for curses. Routing,
request parsing and audio decoding live here; every rule about which chunks
exist, which are recorded and what a dataset row looks like belongs to
recorder_scripts and recorder_state, which this module only calls.

Takes recorded here are indistinguishable from terminal ones: the same
clip_path, the same PCM_16 at wp.SAMPLE_RATE, the same upsert_row. That is the
point - one dataset, two front ends.

Stdlib only by design: setup.sh is the dependency contract, and a recorder that
needs a web framework installed on a phone-facing box is a worse recorder.
"""

import argparse
import email.parser
import email.policy
import io
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlparse

import recorder_scripts as rsc
import recorder_state as rs
import whisper_pipeline as wp

STATIC_DIR = wp.PROJECT_ROOT / "static"

# How long a peer may take to complete a TLS handshake before its worker
# thread is released. Generous for a phone on slow wifi, finite for a
# connection that will never finish one.
HANDSHAKE_TIMEOUT_SECONDS = 15
# Reading material ships with the repo so a fresh clone (or the Pi) has
# something to record; scripts/ was gitignored and arrived empty.
SCRIPTS_DIR = wp.PROJECT_ROOT / "training-text"

# index.html addresses its stylesheet and modules as /static/<name>, which keeps
# every asset in one namespace that cannot collide with /api/ or with a script
# name. SimpleHTTPRequestHandler resolves paths against static_dir as its
# document root, so the prefix is stripped before it runs.
STATIC_PREFIX = "/static/"
DEFAULT_PORT = 8080

# 0.0.0.0 by default: the recorder is useless unless the phone can reach it,
# and the LAN is the deployment target rather than an accident.
DEFAULT_HOST = "0.0.0.0"

# A take is at most a few minutes of Opus; anything larger is not a recording
# and reading it would let one request exhaust the box's memory.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


# The domain's own "names something absent" type, reused rather than redefined
# so one exception class decides the 404s.
NotFound = rsc.ScriptNotFound


class Config(NamedTuple):
    """Where the server reads scripts and writes the dataset."""

    scripts_dir: Path
    csv_path: Path
    audio_dir: Path
    static_dir: Path


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def ffmpeg_command():
    """ffmpeg beside this interpreter first, mirroring export.converter_command.

    A venv on the box may ship its own binary, and `which` only finds one when
    the venv is on PATH - the same trap ct2-transformers-converter fell into.
    """
    local = Path(sys.executable).parent / "ffmpeg"
    if local.exists():
        return str(local)
    return shutil.which("ffmpeg")


def decode_audio(payload):
    """Decode an uploaded blob to mono float32 at wp.SAMPLE_RATE.

    soundfile is tried first because a WAV upload needs no subprocess. Browsers
    record WebM/Opus, which libsndfile cannot open at all, and librosa 1.x
    dropped the audioread fallback that used to cover it - so ffmpeg is the
    only remaining decoder for the format MediaRecorder actually produces.
    """
    if not payload:
        raise wp.PipelineError("Uploaded audio was empty.")

    samples = _decode_with_soundfile(payload)
    if samples is None:
        samples = _decode_with_ffmpeg(payload)
    return samples


def _decode_with_soundfile(payload):
    """The uploaded samples, or None when libsndfile does not know the format."""
    import numpy as np
    import soundfile as sf

    try:
        samples, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        return None

    mono = samples.mean(axis=1).astype(np.float32)
    return _resample(mono, rate)


def _resample(samples, rate):
    if rate == wp.SAMPLE_RATE:
        return samples

    import librosa

    return librosa.resample(samples, orig_sr=rate, target_sr=wp.SAMPLE_RATE)


def _decode_with_ffmpeg(payload):
    import numpy as np

    command = ffmpeg_command()
    if command is None:
        raise wp.PipelineError(
            "Could not decode the upload: libsndfile does not know this format "
            "and ffmpeg is not installed. Install ffmpeg, or record WAV."
        )

    result = subprocess.run(
        [command, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "1", "-ar", str(wp.SAMPLE_RATE), "pipe:1"],
        input=payload, capture_output=True, check=False,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf8", "replace").strip().splitlines()
        raise wp.PipelineError(
            f"Could not decode the uploaded audio: {detail[-1] if detail else 'unknown format'}"
        )

    # frombuffer views the pipe's bytes; copy so the array owns writable memory.
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def clip_seconds(samples):
    return len(samples) / wp.SAMPLE_RATE


def audio_from_body(body, content_type):
    """The audio bytes in a request, whether posted raw or as a form field.

    The browser client uses FormData so the blob keeps its MIME type; curl and
    the tests post the blob raw. Supporting both keeps the API usable by hand.

    Parsed through email.parser rather than cgi, which was removed in Python
    3.13; the body is fed as bytes throughout so a 0x0d 0x0a run inside the
    audio is not mistaken for a line ending and rewritten.
    """
    if not (content_type or "").lower().startswith("multipart/"):
        return body

    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("utf8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    for part in message.iter_parts():
        if _part_name(part) == "audio":
            return part.get_payload(decode=True) or b""
    raise wp.PipelineError("The upload carried no 'audio' part.")


def _part_name(part):
    disposition = part.get("Content-Disposition", "")
    match = re.search(r'name="?([^";]+)"?', disposition)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Operations - each one a request's worth of work, returning plain data
# ---------------------------------------------------------------------------


def _progress(config, name):
    """The requested script's progress, or an error naming what went wrong."""
    path = rsc.resolve_script(config.scripts_dir, name)
    # resolve_script has already rejected any name whose directory is not a
    # supported language, so the prefix is the label - nothing to infer.
    language = name.split("/", 1)[0]
    return path, language, rsc.script_progress(
        path, config.csv_path, config.audio_dir, language, name
    )


def scripts_payload(config):
    """Every script with its counts, for the picker.

    The chunk text is deliberately left out: the picker renders counts, and
    sending every script's prose would make this response grow with the corpus.
    """
    rows = []
    for item in rsc.list_scripts(config.scripts_dir):
        rows.append(_summary(config, item))
    return {"scripts": rows}


# What both responses report about a script. script_progress already returns
# every one of these under these names, so this is a projection of it rather
# than a second calculation - the picker row and the script response cannot
# disagree about a count.
PROGRESS_FIELDS = ("name", "language", "total", "recorded_count", "next_index", "complete")


def _progress_payload(progress):
    """The JSON-safe view of one script_progress result.

    Only `recorded` needs converting: it is a set, which json cannot encode,
    and sorting it makes the response byte-identical between runs.
    """
    payload = {field: progress[field] for field in PROGRESS_FIELDS}
    payload["recorded"] = sorted(progress["recorded"])
    return payload


def _summary(config, item):
    """One picker row. list_scripts only yields files under a language
    directory, so every item here has a language and is recordable."""
    progress = rsc.script_progress(
        item["path"], config.csv_path, config.audio_dir, item["language"], item["name"]
    )
    return _progress_payload(progress)


def script_payload(config, name):
    """One script's chunks, each with whether it already has a take."""
    _, _, progress = _progress(config, name)
    return _progress_payload(progress) | {
        "chunks": rsc.chunk_view(progress["chunks"], progress["recorded"]),
    }


def save_chunk(config, name, index, payload):
    """Decode a take and commit it, or reject it without touching the dataset.

    The minimum-length check happens before anything is written, so a rejected
    take cannot clobber a good one already on that line - the terminal recorder
    keeps the previous take on the same rule.
    """
    import soundfile as sf

    path, language, progress = _progress(config, name)
    _check_index(index, progress["total"])

    samples = decode_audio(payload)
    seconds = clip_seconds(samples)
    if seconds < wp.MIN_CLIP_SECONDS:
        raise wp.PipelineError(
            f"Discarded: {seconds:.2f}s is too short to use "
            f"(minimum {wp.MIN_CLIP_SECONDS}s)."
        )

    Path(config.audio_dir).mkdir(parents=True, exist_ok=True)
    # Re-recording overwrites the take on this line whatever it is named, so a
    # legacy clip is replaced rather than shadowed by a second file the picker
    # would then have to choose between.
    destination = rs.existing_clip_path(
        config.csv_path, config.audio_dir, language, index, name,
        dict(enumerate(progress["chunks"])),
    )
    # PCM_16 at wp.SAMPLE_RATE matches record_data.write_clip exactly; a second
    # format in the dataset would surface as a decode surprise inside train.py.
    sf.write(str(destination), samples, wp.SAMPLE_RATE, subtype="PCM_16")
    rs.upsert_row(config.csv_path, destination, progress["chunks"][index], language)

    return {
        "name": path.name,
        "index": index,
        "language": language,
        "recorded": True,
        "seconds": round(seconds, 2),
        # Over Whisper's window the take still saves, as in the terminal
        # recorder; the client decides whether to suggest a redo.
        "too_long": seconds > wp.MAX_CLIP_SECONDS,
    }


def delete_chunk(config, name, index):
    """Drop a take so the line re-opens, dataset row included.

    Deleting only the wav would leave a row pointing at a missing file, which
    train.py cannot load; prune_missing is what record_data runs at startup for
    the same reason, applied here at the moment of deletion instead.
    """
    _, language, progress = _progress(config, name)
    _check_index(index, progress["total"])

    destination = rs.existing_clip_path(
        config.csv_path, config.audio_dir, language, index, name,
        dict(enumerate(progress["chunks"])),
    )
    # missing_ok: the browser may retry a delete, and a second one is a no-op
    # rather than an error - the line is already open either way.
    Path(destination).unlink(missing_ok=True)
    rs.prune_missing(config.csv_path, config.audio_dir)

    return {"name": name, "index": index, "recorded": False}


def clip_bytes(config, name, index):
    """The stored wav for playback."""
    _, language, progress = _progress(config, name)
    _check_index(index, progress["total"])

    destination = rs.existing_clip_path(
        config.csv_path, config.audio_dir, language, index, name,
        dict(enumerate(progress["chunks"])),
    )
    if not Path(destination).exists():
        raise NotFound(f"No recording on line {index + 1} yet.")
    return Path(destination).read_bytes()


def _check_index(index, total):
    if not 0 <= index < total:
        raise wp.PipelineError(f"Chunk index {index} is outside this script.")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_SCRIPTS = re.compile(r"^/api/scripts/?$")
# A script name is "<language>/<file>", but the name group also matches a bare
# single segment: an unqualified or traversing name must still reach
# resolve_script, which is the single guard and answers 400 "invalid name".
# Matching only the two-segment form made those a 404 from no route at all,
# reporting a malformed name as a missing one.
_NAME = r"([^/]+(?:/[^/]+)?)"
_SCRIPT = re.compile(rf"^/api/scripts/{_NAME}/?$")
_CHUNK = re.compile(rf"^/api/scripts/{_NAME}/chunks/(\d+)/?$")
_AUDIO = re.compile(rf"^/api/scripts/{_NAME}/chunks/(\d+)/audio/?$")


def parse_path(raw_path):
    """(route, script name, chunk index) for a URL, or a null route.

    A traversing name is decoded and passed through rather than rejected here:
    recorder_scripts.resolve_script is the single guard, so there is one place
    to read when asking what the server considers a safe name.
    """
    path = urlparse(raw_path).path

    if _SCRIPTS.match(path):
        return ("scripts", None, None)

    for route, pattern in (("audio", _AUDIO), ("chunk", _CHUNK), ("script", _SCRIPT)):
        match = pattern.match(path)
        if match:
            name = unquote(match.group(1))
            index = int(match.group(2)) if match.lastindex > 1 else None
            return (route, name, index)

    return (None, None, None)


class RecorderHandler(SimpleHTTPRequestHandler):
    """Translates requests into the operations above and back into JSON.

    Inherits SimpleHTTPRequestHandler purely for its static-file serving, which
    covers the browser assets; every /api path is handled before that runs.
    """

    config = None

    # A class attribute, not set in __init__: end_headers can run while the base
    # class is still handling a malformed request line, before any per-instance
    # setup of ours has happened.
    _serving_static = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.config.static_dir), **kwargs)

    def do_GET(self):
        # Reset per request, not per connection: keep-alive reuses one handler
        # for many requests, so an /api call after a static one would otherwise
        # inherit the flag.
        self._serving_static = False
        route, name, index = parse_path(self.path)

        if route == "scripts":
            self._respond(lambda: scripts_payload(self.config))
        elif route == "script":
            self._respond(lambda: script_payload(self.config, name))
        elif route == "audio":
            self._respond_audio(name, index)
        elif self.path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, f"No such endpoint: {self.path}")
        else:
            self._serve_static()

    def do_HEAD(self):
        """A HEAD is a GET whose body is discarded.

        Inherited from SimpleHTTPRequestHandler it resolves against static_dir
        instead, so HEAD on an API path 404s while GET on the same path serves
        200. Media clients probe with HEAD as well as Range to learn a clip's
        size before playing, and a 404 to that probe stops playback for the
        same silent reason Accept-Ranges was added to prevent.
        """
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    # Set only for the duration of a HEAD, so the write below stays a plain
    # GET write on every other request.
    _head_only = False

    def _write_body(self, payload):
        if not self._head_only:
            self.wfile.write(payload)

    def do_POST(self):
        route, name, index = parse_path(self.path)
        if route != "chunk":
            self._error(HTTPStatus.NOT_FOUND, f"Cannot POST to {self.path}")
            return

        try:
            payload = audio_from_body(self._read_body(), self.headers.get("Content-Type"))
        except wp.PipelineError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return

        self._respond(lambda: save_chunk(self.config, name, index, payload))

    def do_DELETE(self):
        route, name, index = parse_path(self.path)
        if route != "chunk":
            self._error(HTTPStatus.NOT_FOUND, f"Cannot DELETE {self.path}")
            return
        self._respond(lambda: delete_chunk(self.config, name, index))

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise wp.PipelineError("The upload is too large to be a recording.")
        return self.rfile.read(length) if length else b""

    def _respond(self, operation):
        """Run an operation, turning its failure into a 4xx the client can show.

        PipelineError is the pipeline's 'you asked for something impossible',
        so it maps to 400 rather than 500: none of these are server faults.
        """
        try:
            self._send_json(HTTPStatus.OK, operation())
        except wp.PipelineError as error:
            self._error(self._status_for(error), str(error))

    @staticmethod
    def _status_for(error):
        """404 for something absent, 400 for something the client got wrong.

        Keyed on the exception type rather than its wording, so rewording an
        error message cannot quietly change the status it maps to.
        """
        return HTTPStatus.NOT_FOUND if isinstance(error, NotFound) \
            else HTTPStatus.BAD_REQUEST

    def _respond_audio(self, name, index):
        try:
            payload = clip_bytes(self.config, name, index)
        except wp.PipelineError as error:
            self._error(self._status_for(error), str(error))
            return

        total = len(payload)
        start, end = parse_range(self.headers.get("Range"), total)
        partial = start is not None
        if partial:
            payload = payload[start:end + 1]

        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        # Safari on iOS probes an <audio> source with `Range: bytes=0-1` and
        # refuses to play at all if the reply is a plain 200 with the whole
        # body - silently, with no error on the page. Advertising byte ranges
        # and honouring them is what makes playback work on a phone.
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        # A re-recorded take must not play from cache on the phone.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._write_body(payload)

    def end_headers(self):
        """Forbid caching of the assets before the body goes out.

        SimpleHTTPRequestHandler sends Last-Modified and no Cache-Control, which
        lets a browser heuristically cache the ES modules. Rebuilding the image
        then served a fresh index.html against app.js from the last build: the
        new markup has no #btn-prev, the old bindControls asked for it anyway,
        and the null it got threw out of boot() before the first repaint - the
        page sat on the literal "loading…" in the HTML with no script list and
        no error visible on screen. The assets are a few kB off the LAN, so
        never caching them costs nothing next to shipping a stale bundle.
        """
        if self._serving_static:
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def _serve_static(self):
        if not Path(self.config.static_dir).is_dir():
            self._error(HTTPStatus.NOT_FOUND, "No web assets are installed.")
            return

        self._serving_static = True
        if self.path.startswith(STATIC_PREFIX):
            # translate_path resolves against static_dir, and it also normalises
            # away any ../ in the request, so stripping the prefix cannot be
            # used to reach outside the asset directory.
            self.path = self.path[len(STATIC_PREFIX) - 1:]
        if not self._head_only:
            super().do_GET()
            return

        # The parent writes the body itself rather than through _write_body, so
        # a HEAD sends the headers from send_head() and stops there. Its own
        # do_HEAD is not reused: it re-resolves self.path, which the prefix
        # strip above has already rewritten.
        handle = self.send_head()
        if handle:
            handle.close()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _error(self, status, message):
        # 'message' rather than 'error': the client renders it verbatim, and
        # every failure here is already a sentence meant for a human.
        self._send_json(status, {"message": message})

    def log_message(self, fmt, *args):
        """One line per request on stderr, without the default's date noise."""
        sys.stderr.write(f"{self.command} {self.path} - {fmt % args}\n")


def parse_range(header, size):
    """(start, end) for a single `bytes=` range, or (None, None).

    Only the one-range form browsers actually send is honoured; anything else
    - multiple ranges, a suffix past the end, an unparseable header - falls
    back to serving the whole body, which is a valid response to any Range.
    """
    if not header:
        return (None, None)

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match:
        return (None, None)

    first, last = match.group(1), match.group(2)
    if first:
        start = int(first)
        end = int(last) if last else size - 1
    elif last:
        # A suffix range ("-500") counts back from the end.
        start = max(0, size - int(last))
        end = size - 1
    else:
        return (None, None)

    end = min(end, size - 1)
    if start > end or start >= size:
        return (None, None)
    return (start, end)


def build_server(config, host=DEFAULT_HOST, port=DEFAULT_PORT, certificate=None):
    """A configured, unstarted server, so tests can bind an ephemeral port.

    With a certificate the listening socket is wrapped in TLS. That is the
    whole of what HTTPS costs here: the browser withholds getUserMedia from a
    page served over plain HTTP to anything but localhost, so recording from a
    phone needs the scheme, not any change to how a request is handled.
    """
    handler = type("BoundRecorderHandler", (RecorderHandler,), {"config": config})
    if not certificate:
        return ThreadingHTTPServer((host, port), handler)

    server = TlsHTTPServer((host, port), handler)
    server.context = tls_context(certificate)
    return server


class TlsHTTPServer(ThreadingHTTPServer):
    """An HTTPS server that handshakes on the worker thread.

    Wrapping the *listening* socket instead puts the handshake inside
    get_request(), which socketserver runs on the single accept thread before
    it hands the connection to a worker - so one client that opens a TCP
    connection and then says nothing blocks every other request until it goes
    away. A port scanner or a phone that sleeps on Safari's certificate
    interstitial is enough to do it. Wrapping per connection keeps the
    handshake on the thread that already exists to be blocked, and the timeout
    bounds how long a silent peer can hold that one thread.
    """

    context = None

    def process_request_thread(self, request, client_address):
        """Handshake here, on the worker, never on the accept thread.

        get_request() is the wrong seam even though it is where the socket is
        born: socketserver calls it on the accept thread and spawns the worker
        only afterwards, so wrapping there serialises handshakes exactly as
        wrapping the listener does.
        """
        try:
            request.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
            request = self.context.wrap_socket(request, server_side=True)
        except OSError:
            # A peer that never completes a handshake - a scanner, or a phone
            # sitting on the certificate warning - is not an error worth
            # logging a traceback for; the thread simply ends.
            self.shutdown_request(request)
            return
        super().process_request_thread(request, client_address)


def tls_context(certificate):
    """An SSL context for a mkcert-issued certificate and its key.

    `certificate` names the .pem; the key is its `-key.pem` sibling, which is
    how mkcert names the pair. Both are checked here so a missing or misnamed
    file fails at startup naming the path, rather than as a handshake error on
    the phone with nothing in the server log.
    """
    certificate = Path(certificate)
    key = certificate.with_name(f"{certificate.stem}-key{certificate.suffix}")
    for path in (certificate, key):
        if not path.is_file():
            raise wp.PipelineError(f"TLS file not found: {path}")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    return context


def main():
    args = parse_arguments()
    config = Config(
        scripts_dir=args.scripts,
        csv_path=args.csv,
        audio_dir=args.out_dir,
        static_dir=args.static,
    )
    config.audio_dir.mkdir(parents=True, exist_ok=True)
    config.csv_path.parent.mkdir(parents=True, exist_ok=True)

    server = build_server(config, args.host, args.port, args.cert)
    scheme = "https" if args.cert else "http"
    print(f"Recorder on {scheme}://{args.host}:{args.port}  scripts={config.scripts_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("RECORDER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("RECORDER_PORT", DEFAULT_PORT)))
    parser.add_argument("--scripts", type=Path, default=SCRIPTS_DIR)
    parser.add_argument("--csv", type=Path, default=wp.DATASET_CSV)
    parser.add_argument("--out-dir", type=Path, default=wp.AUDIO_DIR)
    parser.add_argument("--static", type=Path, default=STATIC_DIR)
    # A phone will not open the microphone over plain HTTP; point this at a
    # mkcert-issued .pem to serve the page over TLS instead.
    parser.add_argument("--cert", type=Path,
                        default=os.environ.get("RECORDER_CERT") or None)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except wp.PipelineError as error:
        sys.exit(f"error: {error}")
