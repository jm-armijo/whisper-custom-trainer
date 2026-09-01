"""The browser modules, executed rather than read.

Every earlier check on static/*.js asserted on the *text* of the source, which
is how a completely dead UI stayed green: a test that greps for a string proves
the string is there, not that the code does anything. These run the real
modules under node against a real recorder_server on a real socket, so what is
asserted is behaviour.

node is not a dependency of this project - setup.sh installs a Python venv and
nothing else - so these skip with an explicit reason when it is absent, the way
the docker tests do. A silently-passing guard hid a real failure in this repo
before (see CLAUDE.md on ct2-transformers-converter).
"""

import io
import json
import shutil
import subprocess
import threading

import numpy as np
import pytest
import soundfile as sf

import recorder_server as srv
import recorder_state as rs
import whisper_pipeline as wp

pytestmark = pytest.mark.integration

STATIC = wp.PROJECT_ROOT / "static"

LINE = "this is a deliberately long sentence with plenty of words in it number {n}."

# api.js addresses the server with root-relative paths; the browser resolves
# those against the page's origin, and node has no page. The shim supplies one
# without api.js needing to know it is under test.
FETCH_SHIM = """
const origin = process.env.RECORDER_ORIGIN;
const realFetch = globalThis.fetch;
globalThis.fetch = (url, options) =>
  realFetch(typeof url === "string" && url.startsWith("/") ? origin + url : url, options);
"""


# app.js is a controller, not a library: it boots on import, binds buttons and
# talks to a screen. Driving it needs the browser globals it reaches for, so
# this stubs the few it uses - enough to click a button and observe what the
# controller did, not enough to prove anything about rendering.
DOM_STUB = """
const nodes = new Map();
const record = [];

function element(id) {
  const node = {
    id, textContent: "", disabled: false, hidden: false, dataset: {},
    children: [], handlers: {},
    classList: {
      _set: new Set(),
      add(name) { this._set.add(name); },
      toggle(name, on) { on ? this._set.add(name) : this._set.delete(name); },
      contains(name) { return this._set.has(name); },
    },
    setAttribute() {}, appendChild(child) { this.children.push(child); },
    append(...kids) { this.children.push(...kids); },
    replaceChildren() { this.children = []; },
    addEventListener(event, handler) { this.handlers[event] = handler; },
    querySelector() { return null; },
    getBoundingClientRect() { return { top: 0, bottom: 0 }; },
  };
  return node;
}

globalThis.document = {
  getElementById(id) {
    if (!nodes.has(id)) { nodes.set(id, element(id)); }
    return nodes.get(id);
  },
  createElement: () => element("created"),
};
globalThis.window = {
  matchMedia: () => ({ matches: false }),
  confirm: () => true,
};
globalThis.Audio = class { play() { return Promise.resolve(); } addEventListener() {} };

// MediaRecorder and getUserMedia, recording what the controller asked of them.
const tracks = [];
globalThis.MediaRecorder = class {
  static isTypeSupported() { return true; }
  constructor() { this.state = "recording"; this.mimeType = "audio/wav"; this.handlers = {}; }
  addEventListener(event, handler) { this.handlers[event] = handler; }
  start() {}
  stop() {
    this.state = "inactive";
    this.handlers.dataavailable?.({ data: WAV });
    this.handlers.stop?.();
  }
};
Object.defineProperty(globalThis, "navigator", {
  value: { mediaDevices: { getUserMedia: async () => {
    record.push("getUserMedia");
    return { getTracks: () => tracks };
  } } },
  configurable: true,
});

globalThis.__nodes = nodes;
globalThis.__record = record;
globalThis.__tracks = tracks;
const settle = () => new Promise((resolve) => setTimeout(resolve, 50));
globalThis.__settle = settle;
"""


def node_binary():
    """The node binary, or a skip naming what is missing."""
    binary = shutil.which("node")
    if binary is None:
        pytest.skip("node is not installed; cannot execute the browser modules")
    return binary


def run_node(body, origin="", preamble=""):
    """Run an ES module importing the real static/*.js, returning what it printed.

    The script is written beside static/ so its relative imports resolve to the
    shipped modules rather than to a copy that could drift from them.
    """
    binary = node_binary()
    script = STATIC / "_harness.mjs"
    script.write_text(FETCH_SHIM + preamble + body, encoding="utf8")
    try:
        result = subprocess.run(
            [binary, str(script)],
            capture_output=True, text=True, check=False, timeout=60,
            env={"RECORDER_ORIGIN": origin, "PATH": "/usr/bin:/bin"},
        )
    finally:
        script.unlink(missing_ok=True)

    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout)


def wav_bytes(seconds=1.0, sample_rate=16000):
    samples = np.sin(
        2 * np.pi * 440 * np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    ).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def blob_literal(payload):
    """The wav as a JS Blob expression, so the upload is the real bytes."""
    return f"new Blob([Uint8Array.from({list(payload)})])"


@pytest.fixture
def live_server(tmp_path):
    """A real recorder server the node process talks to over a real socket."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "es.txt").write_text(
        "\n\n".join(LINE.format(n=n) for n in range(4)), encoding="utf8"
    )

    audio = tmp_path / "data"
    audio.mkdir()

    config = srv.Config(
        scripts_dir=scripts,
        csv_path=tmp_path / "dataset.csv",
        audio_dir=audio,
        static_dir=STATIC,
    )
    httpd = srv.build_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class TestTheUploadingStateBlocksASecondTake:
    """Between stopping and the upload landing there is a window the controller
    must not treat as idle. On a phone that window is seconds long: the screen
    still read RECORDING with a frozen dot while a second tap started a fresh
    capture over the take still in flight."""

    def test_uploading_counts_as_busy(self):
        busy = run_node("""
        import { IDLE, RECORDING, UPLOADING, isBusy } from "./state.js";
        console.log(JSON.stringify({
          idle: isBusy(IDLE),
          recording: isBusy(RECORDING),
          uploading: isBusy(UPLOADING),
        }));
        """)
        assert busy == {"idle": False, "recording": True, "uploading": True}

    def test_uploading_is_not_the_idle_state(self):
        """Folding it into IDLE is exactly what let the second tap through."""
        states = run_node("""
        import { IDLE, UPLOADING } from "./state.js";
        console.log(JSON.stringify({ same: IDLE === UPLOADING }));
        """)
        assert states["same"] is False

    def test_the_screen_says_what_it_is_doing_while_uploading(self):
        view = run_node("""
        import { UPLOADING, buildView } from "./state.js";
        console.log(JSON.stringify(buildView({ session: null, scripts: [], state: UPLOADING })));
        """)
        assert view["state"] == "uploading"
        assert view["legend"] != "record · redo · play · prev · next"


class TestTheMicrophoneIsReleased:
    """close() had no caller anywhere, so the getUserMedia stream stayed open
    for the whole session and the phone's recording indicator stayed lit while
    the user was only reading."""

    def test_closing_stops_every_track(self):
        result = run_node("""
        import { Microphone } from "./microphone.js";
        const stopped = [];
        const microphone = new Microphone();
        microphone.stream = { getTracks: () => [{ stop: () => stopped.push("a") },
                                                 { stop: () => stopped.push("b") }] };
        microphone.close();
        console.log(JSON.stringify({ stopped, stream: microphone.stream }));
        """)
        assert result["stopped"] == ["a", "b"]
        assert result["stream"] is None

    def test_closing_twice_is_harmless(self):
        """stopRecording closes in a finally, which can run after an error that
        already released the stream."""
        result = run_node("""
        import { Microphone } from "./microphone.js";
        const microphone = new Microphone();
        microphone.stream = { getTracks: () => [{ stop: () => {} }] };
        microphone.close();
        microphone.close();
        console.log(JSON.stringify({ ok: true }));
        """)
        assert result["ok"] is True

    def test_the_stream_reopens_for_the_next_take(self):
        """Releasing must not make the mic unusable for the rest of the session:
        open() re-prompts only when the stream is genuinely gone."""
        result = run_node("""
        import { Microphone } from "./microphone.js";
        let opened = 0;
        // node defines navigator as a getter-only global, so it is replaced
        // rather than assigned.
        Object.defineProperty(globalThis, "navigator", {
          value: { mediaDevices: { getUserMedia: async () => {
            opened += 1;
            return { getTracks: () => [{ stop: () => {} }] };
          } } },
          configurable: true,
        });
        const microphone = new Microphone();
        await microphone.open();
        microphone.close();
        await microphone.open();
        console.log(JSON.stringify({ opened, hasStream: microphone.stream !== null }));
        """)
        assert result["opened"] == 2
        assert result["hasStream"] is True


class TestTheClipLimitsAreNotDuplicated:
    """MIN_CLIP_SECONDS and MAX_CLIP_SECONDS were hardcoded in state.js beside
    whisper_pipeline's copies with nothing pinning them together. The fix is
    that the browser no longer holds either: the server measures the clip it
    decoded and reports its own verdict."""

    def test_the_browser_holds_no_copy_of_the_limits(self):
        source = (STATIC / "state.js").read_text(encoding="utf8")
        for value in (str(wp.MIN_CLIP_SECONDS), str(wp.MAX_CLIP_SECONDS)):
            assert value not in source, f"state.js still hardcodes {value}"

    def test_the_saved_message_follows_the_servers_verdict(self):
        """Not a threshold of its own: tooLong is the server's answer."""
        messages = run_node("""
        import { savedMessage } from "./state.js";
        console.log(JSON.stringify({
          inside: savedMessage({ seconds: 2.0, tooLong: false }),
          over: savedMessage({ seconds: 31.0, tooLong: true }),
        }));
        """)
        assert "redo" not in messages["inside"]
        assert "redo" in messages["over"]

    def test_a_long_clip_the_server_accepts_is_not_warned_about(self):
        """The browser must not second-guess the flag from the duration."""
        message = run_node("""
        import { savedMessage } from "./state.js";
        console.log(JSON.stringify(savedMessage({ seconds: 45.0, tooLong: false })));
        """)
        assert "redo" not in message


class TestTheStatusVocabularyMatchesThePythonDomain:
    """state.js mirrors recorder_state.chunk_statuses. The two were in sync with
    nothing enforcing it, so this compares the real functions rather than the
    spelling of their constants."""

    def test_the_four_status_names_are_the_same_on_both_sides(self):
        names = run_node("""
        import { RECORDED, SELECTED, RECORDED_SELECTED, PENDING } from "./state.js";
        console.log(JSON.stringify([RECORDED, SELECTED, RECORDED_SELECTED, PENDING]));
        """)
        assert names == [rs.RECORDED, rs.SELECTED, rs.RECORDED_SELECTED, rs.PENDING]

    def test_chunk_statuses_agrees_line_for_line(self):
        total, cursor, recorded = 6, 2, [1, 2, 4]
        statuses = run_node(f"""
        import {{ chunkStatuses }} from "./state.js";
        console.log(JSON.stringify(
          chunkStatuses({total}, new Set({recorded}), {cursor})
        ));
        """)
        assert statuses == rs.chunk_statuses(total, set(recorded), cursor)

    def test_first_unrecorded_agrees(self):
        total, recorded = 5, [0, 1, 3]
        index = run_node(f"""
        import {{ firstUnrecorded }} from "./state.js";
        console.log(JSON.stringify(firstUnrecorded({total}, new Set({recorded}))));
        """)
        assert index == rs.first_unrecorded(total, set(recorded))

    def test_first_unrecorded_agrees_on_a_finished_script(self):
        """The end-of-script case, where both clamp rather than run off."""
        total, recorded = 4, [0, 1, 2, 3]
        index = run_node(f"""
        import {{ firstUnrecorded }} from "./state.js";
        console.log(JSON.stringify(firstUnrecorded({total}, new Set({recorded}))));
        """)
        assert index == rs.first_unrecorded(total, set(recorded))


def drive_app(body, origin, payload=None):
    """Boot the real app.js against a stubbed screen and a live server."""
    wav = blob_literal(payload if payload is not None else wav_bytes(1.0))
    return run_node(
        f"""
        await import("./app.js");
        await __settle();
        const button = (id) => __nodes.get(id);
        {body}
        """,
        origin,
        preamble=f"const WAV = {wav};\n{DOM_STUB}",
    )


class TestTheControllerReleasesTheMicrophone:
    """Microphone.close() had no caller anywhere in the frontend, so the
    getUserMedia stream stayed open for the whole session and the phone's
    recording indicator stayed lit while the user was only reading."""

    def test_a_finished_take_stops_the_capture_tracks(self, live_server):
        result = drive_app("""
        const stopped = [];
        __tracks.push({ stop: () => stopped.push("mic") });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        console.log(JSON.stringify({ stopped, state: button("status-state").textContent }));
        """, live_server)
        assert result["stopped"] == ["mic"], "the mic stream was never released"

    def test_the_screen_returns_to_idle_after_the_upload(self, live_server):
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        console.log(JSON.stringify({ state: button("status-state").textContent }));
        """, live_server)
        assert result["state"] == "IDLE"

    def test_the_take_actually_reached_the_dataset(self, live_server):
        """The whole drive is worthless if the controller never saved anything."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        console.log(JSON.stringify({ message: button("message").textContent }));
        """, live_server)
        assert result["message"].startswith("saved "), result["message"]


class TestASecondTapCannotRaceAnUpload:
    """The live half of the UPLOADING state. A phone upload takes seconds, and
    during it the screen read RECORDING with a frozen dot while the controller
    already held a state that let the next tap start capturing."""

    # Holds the POST open so the tap lands squarely inside the upload window,
    # which is the whole race; a fast local upload would close it before a test
    # could tap twice.
    SLOW_UPLOAD = """
    const beforeDelay = globalThis.fetch;
    globalThis.fetch = async (url, options) => {
      if (options && options.method === "POST") {
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
      return beforeDelay(url, options);
    };
    """

    def test_the_screen_stops_saying_recording_the_moment_capture_ends(
        self, live_server
    ):
        result = drive_app(self.SLOW_UPLOAD + """
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        button("btn-record").handlers.click();
        await __settle();
        console.log(JSON.stringify({ during: button("status-state").textContent }));
        """, live_server)
        assert result["during"] == "UPLOADING"

    def test_a_tap_during_the_upload_starts_no_second_capture(self, live_server):
        """getUserMedia is the tell: a second capture would open the mic again."""
        result = drive_app(self.SLOW_UPLOAD + """
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        button("btn-record").handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        const opens = __record.filter((entry) => entry === "getUserMedia").length;
        console.log(JSON.stringify({ opens }));
        """, live_server)
        assert result["opens"] == 1, "a second tap opened the mic mid-upload"

    def test_the_record_button_is_disabled_while_the_take_uploads(self, live_server):
        result = drive_app(self.SLOW_UPLOAD + """
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        button("btn-record").handlers.click();
        await __settle();
        console.log(JSON.stringify({ disabled: button("btn-record").disabled }));
        """, live_server)
        assert result["disabled"] is True


class TestTheClientReadsTheRealServerContract:
    """The server emits exactly one payload shape, so the client parses exactly
    one. These run against the live server rather than a fixture payload: a
    fixture would only prove the client agrees with the test's own idea of the
    contract, which is how the fallback branches survived unnoticed."""

    def test_the_picker_gets_a_name_language_and_counts(self, live_server):
        scripts = run_node("""
        import { listScripts } from "./api.js";
        console.log(JSON.stringify(await listScripts()));
        """, live_server)
        assert scripts == [
            {"name": "es.txt", "language": "es", "recorded": 0, "total": 4}
        ]

    def test_the_script_loads_its_chunk_text(self, live_server):
        loaded = run_node("""
        import { loadScript } from "./api.js";
        console.log(JSON.stringify(await loadScript("es.txt")));
        """, live_server)
        assert len(loaded["chunks"]) == 4
        assert all(chunk for chunk in loaded["chunks"]), loaded["chunks"]

    def test_a_recorded_line_comes_back_in_the_recorded_set(self, live_server):
        loaded = run_node(f"""
        import {{ loadScript, saveChunk }} from "./api.js";
        await saveChunk("es.txt", 2, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(await loadScript("es.txt")));
        """, live_server)
        assert loaded["recorded"] == [2]

    def test_the_counts_track_a_saved_take(self, live_server):
        scripts = run_node(f"""
        import {{ listScripts, saveChunk }} from "./api.js";
        await saveChunk("es.txt", 0, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(await listScripts()));
        """, live_server)
        assert scripts[0]["recorded"] == 1

    def test_a_deleted_take_reopens_the_line(self, live_server):
        loaded = run_node(f"""
        import {{ deleteChunk, loadScript, saveChunk }} from "./api.js";
        await saveChunk("es.txt", 1, {blob_literal(wav_bytes(1.0))});
        await deleteChunk("es.txt", 1);
        console.log(JSON.stringify(await loadScript("es.txt")));
        """, live_server)
        assert loaded["recorded"] == []


class TestTheClientUsesTheServersDuration:
    """The server decodes the upload and knows exactly how long it is; the
    browser knows only wall-clock elapsed, which includes MediaRecorder's
    startup and flush latency. Reporting "exceeds Whisper's 30s window" from
    the browser's number describes a clip the dataset does not contain."""

    def test_save_chunk_returns_the_servers_measurement(self, live_server):
        """saveChunk discarded the response body, so the decoded duration and
        the too_long flag never reached the caller at all."""
        saved = run_node(f"""
        import {{ saveChunk }} from "./api.js";
        const saved = await saveChunk("es.txt", 0, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(saved));
        """, live_server)
        assert saved["seconds"] == pytest.approx(1.0, abs=0.05)

    def test_the_servers_too_long_flag_reaches_the_caller(self, live_server):
        """The flag is the server's decision, not a second computation here."""
        payload = wav_bytes(wp.MAX_CLIP_SECONDS + 1)
        saved = run_node(f"""
        import {{ saveChunk }} from "./api.js";
        const saved = await saveChunk("es.txt", 0, {blob_literal(payload)});
        console.log(JSON.stringify(saved));
        """, live_server)
        assert saved["tooLong"] is True

    def test_a_clip_inside_the_window_is_not_flagged(self, live_server):
        saved = run_node(f"""
        import {{ saveChunk }} from "./api.js";
        const saved = await saveChunk("es.txt", 0, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(saved));
        """, live_server)
        assert saved["tooLong"] is False

    def test_a_rejected_take_still_raises(self, live_server):
        """The 400 path must stay an exception: returning a body would make a
        take the server refused look saved."""
        payload = wav_bytes(wp.MIN_CLIP_SECONDS / 2)
        result = run_node(f"""
        import {{ saveChunk }} from "./api.js";
        let message = null;
        try {{
          await saveChunk("es.txt", 0, {blob_literal(payload)});
        }} catch (error) {{ message = error.message; }}
        console.log(JSON.stringify({{ message }}));
        """, live_server)
        assert "too short" in (result["message"] or "")
