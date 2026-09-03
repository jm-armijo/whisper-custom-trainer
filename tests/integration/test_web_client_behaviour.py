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
import types

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
    // Kept rather than discarded: a glyph is the only visible label on the
    // transport keys, so aria-label is the accessible name, and a test that
    // could not read it back could not tell a mislabelled key from a correct one.
    attributes: {},
    // The confirmation is positioned from JS, so its top/left are the assert:
    // a stub that dropped style writes could not tell a dialog pinned beside
    // its line from one stuck at the origin.
    style: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
    appendChild(child) { child.parentNode = this; this.children.push(child); return child; },
    append(...kids) { for (const kid of kids) this.appendChild(kid); },
    replaceChildren() { this.children = []; },
    removeChild(child) { this.children = this.children.filter((kid) => kid !== child); },
    remove() { this.parentNode?.removeChild(this); this.parentNode = null; },
    addEventListener(event, handler) { this.handlers[event] = handler; },
    removeEventListener(event, handler) {
      if (this.handlers[event] === handler) delete this.handlers[event];
    },
    // Focus is the accessibility contract: a dialog nothing focuses strands a
    // keyboard user on the page behind it, so the stub records who took it.
    focus() { globalThis.__focused = this; },
    contains(other) {
      return other === this || this.children.some((kid) => kid.contains?.(other));
    },
    // Only the "#id" form render.js actually uses; a full selector engine here
    // would be a second implementation of the thing under test.
    querySelector(selector) {
      const wanted = selector.replace("#", "");
      const walk = (node) => {
        for (const kid of node.children) {
          if (kid.id === wanted) return kid;
          const found = walk(kid);
          if (found) return found;
        }
        return null;
      };
      return walk(this);
    },
    getBoundingClientRect() {
      return this.rect || { top: 0, left: 0, bottom: 56, right: 300, width: 300, height: 56 };
    },
  };
  return node;
}

// A 2D context that records the calls made against it, so a test can ask what
// the view drew rather than whether it holds the right string. Only the canvas
// element gets one: everything else must keep working without.
const drawn = [];
function canvasElement(id) {
  const node = element(id);
  node.width = 0;
  node.height = 0;
  node.clientWidth = 300;
  node.clientHeight = 56;
  node.getContext = () => ({
    clearRect: (...a) => drawn.push(["clearRect", ...a]),
    fillRect: (...a) => drawn.push(["fillRect", ...a]),
    strokeRect: () => {},
    beginPath: () => drawn.push(["beginPath"]),
    moveTo: () => {}, lineTo: () => {}, stroke: () => drawn.push(["stroke"]),
    set strokeStyle(v) {}, set fillStyle(v) {},
    set lineWidth(v) {}, set lineJoin(v) {},
  });
  return node;
}

// The re-record confirmation is appended to the body rather than looked up by
// id, so the stub needs one: it is a transient node, and adding it to
// index.html would put a permanently-present dialog in the markup contract.
const body = element("body");

globalThis.document = {
  body,
  getElementById(id) {
    if (!nodes.has(id)) {
      nodes.set(id, id === "waveform" ? canvasElement(id) : element(id));
    }
    return nodes.get(id);
  },
  // Chunk rows are built here rather than looked up by id, and selecting a
  // line is now a tap on its row - Prev/Next are gone, because a web page can
  // just be clicked. A test moving the cursor needs the row's own handler, so
  // the created nodes stay reachable through the list they are appended to.
  createElement: (tag) => {
    const node = element("created");
    node.tagName = String(tag).toUpperCase();
    return node;
  },
  addEventListener(event, handler) { this.handlers[event] = handler; },
  removeEventListener(event, handler) {
    if (this.handlers[event] === handler) delete this.handlers[event];
  },
  handlers: {},
};

// Web Audio, counting every context opened against every one closed. A leaked
// AudioContext is invisible until a long session hits the browser's per-page
// limit and the analyser silently stops attaching, so the count is the assert.
const audio = { opened: 0, closed: 0, disconnected: 0, resumed: 0, decoded: 0 };
globalThis.AudioContext = class {
  constructor() { audio.opened += 1; this.state = "suspended"; }
  resume() { audio.resumed += 1; this.state = "running"; return Promise.resolve(); }
  close() { audio.closed += 1; this.state = "closed"; return Promise.resolve(); }
  createMediaStreamSource() {
    return { connect: () => {}, disconnect: () => { audio.disconnected += 1; } };
  }
  createAnalyser() {
    return {
      fftSize: 1024,
      getByteTimeDomainData(target) {
        // A recognisable non-silent frame, so a trace that is drawn at all is
        // distinguishable from one drawn over an empty buffer.
        for (let i = 0; i < target.length; i += 1) {
          target[i] = 128 + Math.round(100 * Math.sin(i / 8));
        }
      },
    };
  }
  decodeAudioData(buffer, resolve) {
    audio.decoded += 1;
    const decoded = {
      numberOfChannels: 1,
      getChannelData: () => Float32Array.from({ length: 4096 }, (_, i) => Math.sin(i / 16)),
    };
    resolve(decoded);
    return Promise.resolve(decoded);
  }
};

// requestAnimationFrame, counting frames scheduled against frames cancelled.
// A loop left running after a take is a battery drain nobody sees on a desktop.
const frames = { scheduled: 0, cancelled: 0, live: new Set() };
globalThis.window = {
  matchMedia: () => ({ matches: false }),
  // A viewport with real numbers: the confirmation is clamped against these,
  // and a zero-sized window would let a broken clamp look correct.
  innerWidth: 400,
  innerHeight: 800,
  AudioContext: globalThis.AudioContext,
  devicePixelRatio: 1,
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  requestAnimationFrame(callback) {
    frames.scheduled += 1;
    const handle = setTimeout(() => { frames.live.delete(handle); callback(); }, 5);
    frames.live.add(handle);
    return handle;
  },
  cancelAnimationFrame(handle) {
    frames.cancelled += 1;
    frames.live.delete(handle);
    clearTimeout(handle);
  },
};

// An element with a clock, so the playhead can be driven the way a real one
// drives it: through timeupdate, never through Web Audio.
globalThis.Audio = class {
  constructor() {
    this.currentTime = 0; this.duration = NaN; this.handlers = {}; this.calls = [];
    // app.js keeps its one element private, which is the point of it; a test
    // asking what the playhead did needs a handle on that same instance.
    globalThis.__player = this;
  }
  addEventListener(event, handler) { this.handlers[event] = handler; }
  // The deck's three keys are only distinguishable by what they do to the
  // element: pause and resume both leave PLAYING/PAUSED behind, so a test
  // proving the clip was held rather than restarted has to see the calls.
  pause() { this.calls.push("pause"); }
  load() { this.calls.push("load"); }
  play() { this.calls.push("play"); return Promise.resolve(); }
};

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
globalThis.__body = body;
// The one dialog on screen, or null. Found by role rather than by a class the
// stylesheet owns: the class is a colour decision, the role is the contract.
globalThis.__dialog = () =>
  body.children.find((kid) => kid.attributes["role"] === "alertdialog") || null;
// Its two keys, told apart by the action they carry rather than by their
// order: a test that indexed into the children would pass on a dialog whose
// buttons were swapped, which is the exact mistake worth catching.
globalThis.__dialogButton = (action) => {
  const found = [];
  const walk = (node) => {
    for (const kid of node.children) {
      if (kid.dataset.action === action) found.push(kid);
      walk(kid);
    }
  };
  const dialog = globalThis.__dialog();
  if (dialog) walk(dialog);
  return found[0] || null;
};
// `target` defaults to the dialog itself, which is where a key pressed with
// nothing focused lands. Passing a button models the real event path instead:
// keydown fires on the focused button first and bubbles to the dialog, so a
// handler that cannot tell the two apart answers for a button it never saw.
globalThis.__press = (key, target) => {
  const dialog = globalThis.__dialog();
  dialog?.handlers.keydown?.({
    key,
    target: target ?? dialog,
    preventDefault: () => {},
    stopPropagation: () => {},
  });
};
globalThis.__record = record;
globalThis.__tracks = tracks;
globalThis.__audio = audio;
globalThis.__frames = frames;
globalThis.__drawn = drawn;
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
    # The directory names the language, so nothing is inferred from a filename.
    (scripts / "es").mkdir(parents=True)
    (scripts / "es" / "a.txt").write_text(
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


class TestTheNextUnrecordedRule:
    """What the stop-and-record-next key arms.

    Positional "cursor + 1" was the bug: it recorded over a line that already
    had a take. The rule is a forward search for a gap, and it deliberately
    does not wrap - this key is a read-record-read rhythm down the page, and
    jumping backwards mid-flow would arm a line the reader is not looking at.
    """

    @staticmethod
    def next_unrecorded(cursor, total, recorded):
        return run_node(f"""
        import {{ nextUnrecorded }} from "./state.js";
        console.log(JSON.stringify(
          nextUnrecorded({cursor}, {total}, new Set({recorded}))
        ));
        """)

    def test_the_immediate_next_line_when_it_is_free(self):
        assert self.next_unrecorded(0, 5, [0]) == 1

    def test_it_skips_over_lines_that_already_have_takes(self):
        """The reported bug: recording line 5 with 6 and 7 done must arm 8."""
        assert self.next_unrecorded(5, 10, [5, 6, 7]) == 8

    def test_it_is_null_when_every_later_line_is_recorded(self):
        """Nothing left to arm is an answer, not a position: the caller stops
        cleanly rather than clobbering a take."""
        assert self.next_unrecorded(2, 5, [2, 3, 4]) is None

    def test_it_is_null_on_the_last_line(self):
        assert self.next_unrecorded(4, 5, [4]) is None

    def test_it_never_looks_backwards(self):
        """Line 0 is the only gap, and it is behind the cursor."""
        assert self.next_unrecorded(3, 5, [1, 2, 3, 4]) is None

    def test_an_unrecorded_line_under_the_cursor_is_not_the_answer(self):
        """Strictly greater than the cursor: a failed take must not re-arm the
        line the caller has just left."""
        assert self.next_unrecorded(1, 4, []) == 2

    def test_an_empty_script_has_nothing_ahead(self):
        assert self.next_unrecorded(0, 0, []) is None


def drive_app(body, origin, payload=None):
    """Boot the real app.js against a stubbed screen and a live server."""
    wav = blob_literal(payload if payload is not None else wav_bytes(1.0))
    return run_node(
        f"""
        await import("./app.js");
        await __settle();
        const button = (id) => __nodes.get(id);
        // Selecting a line is a tap on its row: Prev/Next are gone, because a
        // web page can be clicked directly.
        const line = (index) => __nodes.get("chunk-list").children[index];
        // A record tap that is deliberately not awaited. On a finished line the
        // handler now waits on an in-page confirmation, so awaiting it here
        // would deadlock: the answer can only come from a test that has already
        // returned. Awaiting the tap is still right everywhere the line is
        // fresh, and those tests keep doing it.
        const tapRecord = () => {{ button("btn-record").handlers.click(); }};
        // The re-record in one call, for tests whose subject is something else
        // entirely. window.confirm used to answer itself in the stub; the
        // page's own dialog has to be pressed, so this is what replaces that.
        const reRecord = async () => {{
          tapRecord();
          await __settle();
          __dialogButton("confirm")?.handlers.click();
          await __settle();
        }};
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
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ stopped, state: button("status-state").textContent }));
        """, live_server)
        assert result["stopped"] == ["mic"], "the mic stream was never released"

    def test_the_screen_returns_to_idle_after_the_upload(self, live_server):
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
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
        await button("btn-stop").handlers.click();
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
        button("btn-stop").handlers.click();
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
        button("btn-stop").handlers.click();
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
        button("btn-stop").handlers.click();
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
            {"name": "es/a.txt", "language": "es", "recorded": 0, "total": 4}
        ]

    def test_the_script_loads_its_chunk_text(self, live_server):
        loaded = run_node("""
        import { loadScript } from "./api.js";
        console.log(JSON.stringify(await loadScript("es/a.txt")));
        """, live_server)
        assert len(loaded["chunks"]) == 4
        assert all(chunk for chunk in loaded["chunks"]), loaded["chunks"]

    def test_a_recorded_line_comes_back_in_the_recorded_set(self, live_server):
        loaded = run_node(f"""
        import {{ loadScript, saveChunk }} from "./api.js";
        await saveChunk("es/a.txt", 2, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(await loadScript("es/a.txt")));
        """, live_server)
        assert loaded["recorded"] == [2]

    def test_the_counts_track_a_saved_take(self, live_server):
        scripts = run_node(f"""
        import {{ listScripts, saveChunk }} from "./api.js";
        await saveChunk("es/a.txt", 0, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(await listScripts()));
        """, live_server)
        assert scripts[0]["recorded"] == 1

    def test_a_deleted_take_reopens_the_line(self, live_server):
        loaded = run_node(f"""
        import {{ deleteChunk, loadScript, saveChunk }} from "./api.js";
        await saveChunk("es/a.txt", 1, {blob_literal(wav_bytes(1.0))});
        await deleteChunk("es/a.txt", 1);
        console.log(JSON.stringify(await loadScript("es/a.txt")));
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
        const saved = await saveChunk("es/a.txt", 0, {blob_literal(wav_bytes(1.0))});
        console.log(JSON.stringify(saved));
        """, live_server)
        assert saved["seconds"] == pytest.approx(1.0, abs=0.05)

    def test_the_servers_too_long_flag_reaches_the_caller(self, live_server):
        """The flag is the server's decision, not a second computation here."""
        payload = wav_bytes(wp.MAX_CLIP_SECONDS + 1)
        saved = run_node(f"""
        import {{ saveChunk }} from "./api.js";
        const saved = await saveChunk("es/a.txt", 0, {blob_literal(payload)});
        console.log(JSON.stringify(saved));
        """, live_server)
        assert saved["tooLong"] is True

    def test_a_clip_inside_the_window_is_not_flagged(self, live_server):
        saved = run_node(f"""
        import {{ saveChunk }} from "./api.js";
        const saved = await saveChunk("es/a.txt", 0, {blob_literal(wav_bytes(1.0))});
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
          await saveChunk("es/a.txt", 0, {blob_literal(payload)});
        }} catch (error) {{ message = error.message; }}
        console.log(JSON.stringify({{ message }}));
        """, live_server)
        assert "too short" in (result["message"] or "")


class TestTheTitleNamesTheFile:
    """The picker and the title both identify a script; the language code was
    redundant there and told the reader nothing they had not just clicked."""

    def test_the_title_carries_the_filename(self):
        view = run_node("""
        import { ScriptSession, buildView, IDLE } from "./state.js";
        const session = new ScriptSession({
          name: "spanish_phonetic_training.txt", language: "es",
          chunks: ["one", "two"], recorded: new Set([0]),
        });
        console.log(JSON.stringify(buildView({ session, scripts: [], state: IDLE })));
        """)
        assert "spanish_phonetic_training.txt" in view["title"]

    def test_the_title_does_not_repeat_the_language_code(self):
        view = run_node("""
        import { ScriptSession, buildView, IDLE } from "./state.js";
        const session = new ScriptSession({
          name: "spanish_phonetic_training.txt", language: "es",
          chunks: ["one", "two"], recorded: new Set([0]),
        });
        console.log(JSON.stringify(buildView({ session, scripts: [], state: IDLE })));
        """)
        assert " · es · " not in view["title"]

    def test_the_title_still_reports_progress(self):
        view = run_node("""
        import { ScriptSession, buildView, IDLE } from "./state.js";
        const session = new ScriptSession({
          name: "en/a.txt", language: "en",
          chunks: ["one", "two", "three"], recorded: new Set([0, 1]),
        });
        console.log(JSON.stringify(buildView({ session, scripts: [], state: IDLE })));
        """)
        assert "2/3" in view["title"]


class TestTheWaveformArithmetic:
    """waveform.js holds no canvas and no AudioContext, so its rules are run
    here directly rather than inferred from what appeared on a screen."""

    def test_every_column_gets_a_peak(self):
        result = run_node("""
        import { peaksFromSamples } from "./waveform.js";
        const samples = Float32Array.from({ length: 8000 }, (_, i) => Math.sin(i / 20));
        const peaks = peaksFromSamples(samples, 200);
        console.log(JSON.stringify({
          columns: peaks.length,
          allFinite: peaks.every((p) => Number.isFinite(p)),
          max: Math.max(...peaks),
        }));
        """)
        assert result["columns"] == 200
        assert result["allFinite"] is True
        assert result["max"] == pytest.approx(1.0, abs=0.01)

    def test_a_column_reports_its_loudest_sample_not_its_average(self):
        """Averaging a few hundred samples of speech tends to zero and paints a
        flat line across a word that is plainly there."""
        result = run_node("""
        import { peaksFromSamples } from "./waveform.js";
        // Alternating +1/-1 averages to zero but peaks at one.
        const samples = Float32Array.from({ length: 1000 }, (_, i) => (i % 2 ? 1 : -1));
        console.log(JSON.stringify(peaksFromSamples(samples, 10)));
        """)
        assert result == [pytest.approx(1.0)] * 10

    def test_a_clip_shorter_than_the_column_count_still_spans_the_width(self):
        """Bunching a short take into the first few columns leaves most of the
        strip blank for a clip that filled it."""
        result = run_node("""
        import { peaksFromSamples } from "./waveform.js";
        const samples = Float32Array.from({ length: 7 }, () => 0.5);
        const peaks = peaksFromSamples(samples, 200);
        console.log(JSON.stringify({
          columns: peaks.length, empty: peaks.filter((p) => p === 0).length,
        }));
        """)
        assert result["columns"] == 200
        assert result["empty"] == 0

    def test_an_empty_clip_yields_no_peaks_rather_than_throwing(self):
        result = run_node("""
        import { peaksFromSamples } from "./waveform.js";
        console.log(JSON.stringify({
          empty: peaksFromSamples(new Float32Array(0), 200),
          missing: peaksFromSamples(null, 200),
        }));
        """)
        assert result == {"empty": [], "missing": []}

    def test_the_live_trace_is_centred_on_the_analysers_zero(self):
        """getByteTimeDomainData reports unsigned bytes centred on 128; drawing
        those raw puts silence at the top of the strip, not the middle."""
        result = run_node("""
        import { traceFromTimeDomain } from "./waveform.js";
        const silence = new Uint8Array(64).fill(128);
        const loud = new Uint8Array(64).fill(255);
        console.log(JSON.stringify({
          silence: traceFromTimeDomain(silence, 8),
          loud: traceFromTimeDomain(loud, 8),
        }));
        """)
        assert result["silence"] == [0] * 8
        assert all(value > 0.9 for value in result["loud"])

    def test_silence_reads_as_no_level_and_a_full_scale_tone_as_full(self):
        result = run_node("""
        import { levelFromTimeDomain } from "./waveform.js";
        const silence = new Uint8Array(128).fill(128);
        const full = Uint8Array.from({ length: 128 }, (_, i) => (i % 2 ? 255 : 1));
        console.log(JSON.stringify({
          silence: levelFromTimeDomain(silence),
          full: levelFromTimeDomain(full),
          none: levelFromTimeDomain(new Uint8Array(0)),
        }));
        """)
        assert result["silence"] == 0
        assert result["full"] == pytest.approx(1.0, abs=0.01)
        assert result["none"] == 0

    def test_the_playhead_stays_inside_the_strip(self):
        """currentTime overruns duration by a frame at the end of a clip, and
        duration is NaN until the element has metadata - both would otherwise
        put the playhead off the canvas."""
        result = run_node("""
        import { playheadFraction } from "./waveform.js";
        console.log(JSON.stringify({
          start: playheadFraction(0, 4),
          middle: playheadFraction(2, 4),
          overrun: playheadFraction(4.2, 4),
          noMetadata: playheadFraction(1, NaN),
          zero: playheadFraction(1, 0),
        }));
        """)
        assert result == {
            "start": 0, "middle": 0.5, "overrun": 1, "noMetadata": 0, "zero": 0,
        }


class TestTheLiveWaveformIsTornDown:
    """The two leaks this feature can cause, both invisible on a desktop: a rAF
    loop that keeps waking the phone's GPU after the take, and an AudioContext
    per take walking into the browser's per-page limit."""

    def test_recording_draws_the_incoming_signal(self, live_server):
        """The feature itself: without this the other three tests would pass on
        code that simply never starts."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        const strokes = __drawn.filter((call) => call[0] === "stroke").length;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ strokes, opened: __audio.opened }));
        """, live_server)
        assert result["opened"] == 1, "no AudioContext was ever opened"
        assert result["strokes"] > 0, "the live trace was never drawn"

    def test_every_context_opened_is_closed_when_the_take_ends(self, live_server):
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ ...__audio }));
        """, live_server)
        assert result["opened"] >= 1
        assert result["closed"] == result["opened"], "an AudioContext was leaked"
        assert result["disconnected"] == result["opened"]

    def test_the_animation_loop_stops_when_the_take_stops(self, live_server):
        """Not merely cancelled once: no frame may still be queued afterwards,
        which is what a loop that reschedules itself past the cancel leaves."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        const after = __frames.scheduled;
        await __settle();
        console.log(JSON.stringify({
          cancelled: __frames.cancelled, live: __frames.live.size,
          grewAfterStop: __frames.scheduled - after,
        }));
        """, live_server)
        assert result["cancelled"] >= 1, "the frame loop was never cancelled"
        assert result["live"] == 0, "a frame was still queued after the take"
        assert result["grewAfterStop"] == 0, "the loop kept rescheduling itself"

    def test_ten_takes_do_not_accumulate_contexts(self, live_server):
        """The limit is per page, so the leak only shows over a session - which
        is exactly the session a real recording is."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        // The script is four lines long, so from the fifth take on the cursor
        // is clamped to a line that already has audio and every tap is a
        // re-record. reRecord answers the dialog those raise.
        for (let take = 0; take < 10; take += 1) {
          await reRecord();
          await button("btn-stop").handlers.click();
          await __settle();
        }
        console.log(JSON.stringify({ ...__audio, live: __frames.live.size }));
        """, live_server)
        assert result["opened"] >= 10
        assert result["closed"] == result["opened"]
        assert result["live"] == 0

    def test_the_context_is_resumed_because_ios_starts_it_suspended(self, live_server):
        """An AudioContext created outside a gesture stays suspended and the
        analyser reports nothing but silence for the whole take."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ resumed: __audio.resumed }));
        """, live_server)
        assert result["resumed"] >= 1


class TestARefusedWaveformStillRecords:
    """A canvas or a Web Audio failure must cost the user nothing: the take is
    the work, the strip is a nicety."""

    def test_a_browser_without_web_audio_still_saves_the_take(self, live_server):
        result = drive_app("""
        delete globalThis.AudioContext;
        delete window.AudioContext;
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ message: button("message").textContent }));
        """, live_server)
        assert result["message"].startswith("saved "), result["message"]

    def test_a_canvas_that_refuses_a_context_still_saves_the_take(self, live_server):
        result = drive_app("""
        __nodes.get("waveform").getContext = () => { throw new Error("no 2d"); };
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ message: button("message").textContent }));
        """, live_server)
        assert result["message"].startswith("saved "), result["message"]

    def test_an_analyser_that_throws_leaves_no_context_behind(self, live_server):
        """The failure path is where a leak hides: start() opened the context
        before the node that threw."""
        result = drive_app("""
        AudioContext.prototype.createAnalyser = () => { throw new Error("nope"); };
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          message: button("message").textContent,
          opened: __audio.opened, closed: __audio.closed,
        }));
        """, live_server)
        assert result["message"].startswith("saved "), result["message"]
        assert result["closed"] == result["opened"], "a failed start leaked a context"


class TestThePlaybackWaveform:
    """Drawn from the clip fetched over the audio route and decoded on the
    spot. Nothing about it is stored: dataset.csv keeps the three columns
    train.py reads, and a cached peak file would need invalidating on every
    re-record."""

    # A saved take advances the cursor to the next line, the way the terminal
    # recorder's space-then-down rhythm does, so playing it back means stepping
    # the cursor up onto the line that now has audio.
    RECORD_ONE = """
    __tracks.push({ stop: () => {} });
    await button("btn-record").handlers.click();
    await __settle();
    await button("btn-stop").handlers.click();
    await __settle();
    line(0).handlers.click();
    await __settle();
    """

    def test_playing_a_take_draws_its_peaks(self, live_server):
        result = drive_app(self.RECORD_ONE + """
        __drawn.length = 0;
        await button("btn-play").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          bars: __drawn.filter((call) => call[0] === "fillRect").length,
          decoded: __audio.decoded,
        }));
        """, live_server)
        assert result["decoded"] == 1, "the clip was never decoded"
        assert result["bars"] > 1, "no peaks were drawn"

    def test_the_peaks_come_from_the_clip_the_server_stored(self, live_server):
        """A decode of zero bytes would still draw something; this pins that
        real audio was fetched over the audio route."""
        result = drive_app(self.RECORD_ONE + """
        const seen = [];
        const before = globalThis.fetch;
        globalThis.fetch = (url, options) => {
          seen.push(String(url));
          return before(url, options);
        };
        await button("btn-play").handlers.click();
        await __settle();
        console.log(JSON.stringify({ seen }));
        """, live_server)
        audio_gets = [url for url in result["seen"] if "/audio" in url]
        assert audio_gets, "the clip was never fetched for its waveform"
        assert all("%2F" not in url for url in audio_gets), audio_gets
        assert all("?v=" in url for url in audio_gets), "the fetch is not cache-busted"

    def test_the_playhead_advances_with_the_elements_own_clock(self, live_server):
        """Driven by timeupdate rather than Web Audio: routing the element
        through createMediaElementSource silences it unless the graph is also
        wired to destination, which on iOS is how playback gets lost."""
        result = drive_app(self.RECORD_ONE + """
        await button("btn-play").handlers.click();
        await __settle();
        const player = globalThis.__player;
        __drawn.length = 0;
        player.currentTime = 0.5;
        player.duration = 1.0;
        player.handlers.timeupdate();
        const half = __drawn.filter((c) => c[0] === "fillRect").length;
        console.log(JSON.stringify({ hasHandler: typeof player.handlers.timeupdate, half }));
        """, live_server)
        assert result["hasHandler"] == "function", "nothing listens for timeupdate"
        assert result["half"] > 1, "the playhead redraw drew nothing"

    def test_the_decoding_context_is_closed_after_every_playback(self, live_server):
        """Playback opens its own short-lived context; ten taps must not leave
        ten open."""
        result = drive_app(self.RECORD_ONE + """
        for (let tap = 0; tap < 10; tap += 1) {
          await button("btn-play").handlers.click();
          await __settle();
        }
        console.log(JSON.stringify({ opened: __audio.opened, closed: __audio.closed }));
        """, live_server)
        assert result["closed"] == result["opened"], "a decode leaked a context"

    def test_a_take_still_plays_when_it_cannot_be_decoded(self, live_server):
        """The <audio> element decodes the clip itself and may well manage what
        decodeAudioData refused; a waveform failure must not report one."""
        result = drive_app(self.RECORD_ONE + """
        AudioContext.prototype.decodeAudioData = () => { throw new Error("bad codec"); };
        await button("btn-play").handlers.click();
        await __settle();
        console.log(JSON.stringify({ message: button("message").textContent }));
        """, live_server)
        assert result["message"].startswith("played "), result["message"]

    def test_a_recording_replaces_the_playback_strip(self, live_server):
        """The strip shows one thing at a time; a stale set of peaks under a
        live trace says the mic is hearing a take that finished."""
        result = drive_app(self.RECORD_ONE + """
        await button("btn-play").handlers.click();
        await __settle();
        // The cursor is back on the line just recorded, so this tap is a
        // re-record and has to answer its confirmation first.
        await reRecord();
        const hiddenDuringRecord = __nodes.get("waveform").hidden;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          hiddenDuringRecord, hiddenAfter: __nodes.get("waveform").hidden,
        }));
        """, live_server)
        assert result["hiddenDuringRecord"] is False, "the live trace was hidden"
        assert result["hiddenAfter"] is True, "the strip stayed up after the take"


class TestTheTransportBehavesLikeACassetteDeck:
    """Play and pause share one key, Stop serves both transports, and Record
    only ever starts a take. The contract worth pinning is that a key's meaning
    does not change under the thumb between presses - the one deliberate
    exception being play/pause, which is the deck this is modelled on."""

    # Same rhythm as the playback class: a saved take advances the cursor, so
    # stepping back is what puts the cursor on the line that now has audio.
    RECORD_ONE = """
    __tracks.push({ stop: () => {} });
    await button("btn-record").handlers.click();
    await __settle();
    await button("btn-stop").handlers.click();
    await __settle();
    line(0).handlers.click();
    await __settle();
    """

    def test_pausing_holds_the_clip_rather_than_restarting_it(self, live_server):
        """The whole point of one key: the second press must pause what is
        playing, not start the take over."""
        result = drive_app(self.RECORD_ONE + """
        await button("btn-play").handlers.click();
        await __settle();
        globalThis.__player.calls.length = 0;
        await button("btn-play").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          calls: globalThis.__player.calls,
          state: button("status-state").textContent,
          legend: button("status-legend").textContent,
        }));
        """, live_server)
        assert result["calls"] == ["pause"], result["calls"]
        assert "paused" in result["legend"]

    def test_the_third_press_resumes_without_reloading_the_clip(self, live_server):
        """Resume is play() on the element as it stands. A load() here would
        mean the take was fetched again and restarted from zero."""
        result = drive_app(self.RECORD_ONE + """
        await button("btn-play").handlers.click();
        await __settle();
        await button("btn-play").handlers.click();
        await __settle();
        globalThis.__player.calls.length = 0;
        await button("btn-play").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          calls: globalThis.__player.calls,
          legend: button("status-legend").textContent,
        }));
        """, live_server)
        assert result["calls"] == ["play"], result["calls"]
        assert "playing" in result["legend"]

    def test_the_key_says_what_its_next_press_will_do(self, live_server):
        """A glyph is the only label on these keys, so it has to track the
        state; a pause glyph over a paused clip is a lie about the next tap."""
        result = drive_app(self.RECORD_ONE + """
        const face = () => ({
          glyph: button("btn-play").textContent,
          label: button("btn-play").attributes["aria-label"],
        });
        const stopped = face();
        await button("btn-play").handlers.click();
        await __settle();
        const playing = face();
        await button("btn-play").handlers.click();
        await __settle();
        console.log(JSON.stringify({ stopped, playing, paused: face() }));
        """, live_server)
        assert result["stopped"] == {"glyph": "⏯", "label": "Play"}
        assert result["playing"] == {"glyph": "⏸", "label": "Pause"}
        assert result["paused"] == {"glyph": "⏯", "label": "Resume"}

    def test_stop_rewinds_so_the_next_play_is_the_whole_take(self, live_server):
        result = drive_app(self.RECORD_ONE + """
        await button("btn-play").handlers.click();
        await __settle();
        globalThis.__player.currentTime = 0.5;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          at: globalThis.__player.currentTime,
          legend: button("status-legend").textContent,
        }));
        """, live_server)
        assert result["at"] == 0
        assert "record" in result["legend"], result["legend"]

    def test_stop_ends_a_take_as_well_as_a_clip(self, live_server):
        """One glyph, both transports. They can never run at once, so the key
        never has to choose between two live things."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          state: button("status-state").textContent,
          message: button("message").textContent,
        }));
        """, live_server)
        assert result["state"] == "IDLE"
        assert result["message"].startswith("saved "), result["message"]

    def test_record_never_stops_a_take(self, live_server):
        """Record used to toggle. Under a thumb that is a key whose meaning
        changes between presses, which is what the separate Stop is for: a
        second press must not end the take."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        const during = button("status-state").textContent;
        // Closed out deliberately: a take left running holds the waveform's
        // animation frame open, and node will not exit with one still armed.
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ during }));
        """, live_server)
        assert result["during"] == "RECORDING"

    def test_a_take_stops_whatever_was_playing(self, live_server):
        """Two live transports would leave Stop with two meanings, so the clip
        yields to the microphone."""
        result = drive_app(self.RECORD_ONE + """
        await button("btn-play").handlers.click();
        await __settle();
        // A re-record, because the cursor sits on the line just saved.
        await reRecord();
        const state = button("status-state").textContent;
        const glyph = button("btn-play").textContent;
        // See test_record_never_stops_a_take: the take has to be closed out or
        // its animation frame keeps node alive past the end of the script.
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ state, glyph }));
        """, live_server)
        assert result["state"] == "RECORDING"
        assert result["glyph"] == "⏯", "the play key still offered to pause"

    def test_stop_and_record_next_saves_then_arms_the_following_line(
        self, live_server
    ):
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        line(1).handlers.click();
        await __settle();
        const from = button("title").textContent;
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-next-take").handlers.click();
        await __settle();
        const armed = {
          from,
          title: button("title").textContent,
          state: button("status-state").textContent,
          opens: __record.filter((entry) => entry === "getUserMedia").length,
        };
        // See test_record_never_stops_a_take: node will not exit while the
        // newly armed take still holds an animation frame.
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify(armed));
        """, live_server)
        assert result["state"] == "RECORDING", "the next line was never armed"
        assert result["opens"] == 2, "the second take never opened the mic"
        assert "1/" in result["title"], result["title"]

    def test_a_failed_upload_leaves_the_cursor_on_the_line_to_redo(
        self, live_server
    ):
        """Advancing on a take that never landed would silently skip the line;
        the recorder would report progress the dataset does not have."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        const before = globalThis.fetch;
        globalThis.fetch = (url, options) => (
          options && options.method === "POST"
            ? Promise.reject(new Error("no network"))
            : before(url, options)
        );
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-next-take").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          state: button("status-state").textContent,
          title: button("title").textContent,
        }));
        """, live_server)
        assert result["state"] == "IDLE", "a failed save armed the next line anyway"
        assert "0/" in result["title"], result["title"]

    # The script is four lines. Seeding a take on line 2 by recording it
    # through the app rather than writing the CSV behind the server's back:
    # the point of this tier is that the real save path is what marks a line
    # recorded, and a hand-written row could be marked in a way the client
    # never sees.
    SEED_LINE_TWO = """
    __tracks.push({ stop: () => {} });
    const seed = async (index) => {
      line(index).handlers.click();
      await __settle();
      await button("btn-record").handlers.click();
      await __settle();
      await button("btn-stop").handlers.click();
      await __settle();
    };
    await seed(2);
    """

    def test_stop_and_record_next_skips_a_line_that_already_has_a_take(
        self, live_server
    ):
        """The reported bug. Arming the next line *by position* recorded over
        line 2's existing take; the key must find the next gap instead."""
        result = drive_app(self.SEED_LINE_TWO + """
        line(1).handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-next-take").handlers.click();
        await __settle();
        const armed = {
          state: button("status-state").textContent,
          onTwo: "aria-current" in line(2).attributes,
          onThree: line(3).attributes["aria-current"],
        };
        // See test_record_never_stops_a_take: the armed take holds an
        // animation frame that would keep node alive past the script's end.
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify(armed));
        """, live_server)
        assert result["state"] == "RECORDING", "the next free line was never armed"
        assert result["onTwo"] is False, "it armed the line that already had a take"
        assert result["onThree"] == "true", "line 3 was not the line armed"

    def test_it_stops_cleanly_when_every_later_line_is_recorded(
        self, live_server
    ):
        """Nothing ahead to arm. Falling through to a recording here would
        capture over line 2 or 3, whichever the cursor happened to land on."""
        result = drive_app(self.SEED_LINE_TWO + """
        // Line 3 too, so only line 0 - behind the cursor - is left open.
        await seed(3);
        line(1).handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-next-take").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          state: button("status-state").textContent,
          title: button("title").textContent,
        }));
        """, live_server)
        assert result["state"] == "IDLE", "it started a take with no free line ahead"
        assert "3/" in result["title"], result["title"]

    def test_the_key_is_dead_when_no_free_line_lies_ahead(self, live_server):
        """A key that lights up promising an arm it cannot perform is a lie.
        The old predicate only asked whether this was the last line."""
        result = drive_app(self.SEED_LINE_TWO + """
        await seed(3);
        line(1).handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        const midTake = button("btn-next-take").disabled;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ midTake }));
        """, live_server)
        assert result["midTake"] is True, "the key offered a next line that is not there"


class TestRecordAbsorbedTheRedoButton:
    """Redo was its own key for no reason a deck would recognise: recording
    over a line that already has audio *is* the re-record. What must survive
    the merge is the confirmation, because the old take is deleted.

    The confirmation is the page's own, not window.confirm: Chrome docks a
    native modal to the top of the viewport, half a screen away from both the
    line being asked about and the thumb that just pressed Record. The stub
    below defines no window.confirm at all, so a controller that still reached
    for one would throw rather than quietly pass."""

    RECORD_ONE = """
    __tracks.push({ stop: () => {} });
    await button("btn-record").handlers.click();
    await __settle();
    await button("btn-stop").handlers.click();
    await __settle();
    line(0).handlers.click();
    await __settle();
    """

    def test_recording_over_a_finished_line_asks_first(self, live_server):
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        const dialog = __dialog();
        console.log(JSON.stringify({
          asked: dialog !== null,
          // The accessible name, which is the question itself: a dialog whose
          // wording only exists in a child span says nothing to a screen reader.
          text: dialog?.attributes["aria-label"] || "",
          state: button("status-state").textContent,
        }));
        """, live_server)
        assert result["asked"] is True, "the old take would have been deleted unasked"
        assert "line 1" in result["text"], result["text"]
        assert "delete" in result["text"].lower(), (
            "the question does not say the take is destroyed"
        )
        assert result["state"] == "IDLE", "the take started before the answer"

    def test_the_dialog_is_the_pages_own_not_the_browsers(self, live_server):
        """A native confirm() is docked to the top of the viewport whatever the
        page does, which is the complaint this replaces: the mouse is down at
        the line, and the button to press is a screen away."""
        result = drive_app(self.RECORD_ONE + """
        let native = 0;
        window.confirm = () => { native += 1; return true; };
        tapRecord();
        await __settle();
        console.log(JSON.stringify({ native, dialog: __dialog() !== null }));
        """, live_server)
        assert result["native"] == 0, "the native confirm is still being used"
        assert result["dialog"] is True

    def test_the_dialog_sits_beside_the_line_it_is_asking_about(self, live_server):
        """Positioned from the row's own box, so it lands where the pointer
        already is rather than at a fixed corner of the page."""
        result = drive_app(self.RECORD_ONE + """
        line(0).rect =
          { top: 400, left: 20, bottom: 456, right: 320, width: 300, height: 56 };
        tapRecord();
        await __settle();
        console.log(JSON.stringify({ style: __dialog().style }));
        """, live_server)
        top = float(result["style"]["top"].removesuffix("px"))
        assert 300 < top < 520, f"the dialog is not near the line: {result['style']}"

    def test_declining_keeps_the_existing_take(self, live_server):
        """The clip must still be there afterwards: a confirm that deletes
        before asking is worse than no confirm at all."""
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        __dialogButton("cancel").handlers.click();
        await __settle();
        const response = await fetch("/api/scripts/es/a.txt");
        const script = await response.json();
        console.log(JSON.stringify({
          recorded: script.recorded,
          message: button("message").textContent,
          state: button("status-state").textContent,
          dialog: __dialog() !== null,
        }));
        """, live_server)
        assert result["recorded"] == [0], "the take was deleted after a refusal"
        assert result["message"] == "kept the existing take"
        assert result["state"] == "IDLE", "declining still started a take"
        assert result["dialog"] is False, "the dialog stayed up after an answer"

    def test_escape_cancels_and_never_confirms(self, live_server):
        """The key a user reaches for to back out. Wiring it to the same
        handler as Enter would delete the take it was pressed to protect."""
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        __press("Escape");
        await __settle();
        const response = await fetch("/api/scripts/es/a.txt");
        const script = await response.json();
        console.log(JSON.stringify({
          recorded: script.recorded,
          message: button("message").textContent,
          state: button("status-state").textContent,
          dialog: __dialog() !== null,
        }));
        """, live_server)
        assert result["recorded"] == [0], "Escape deleted the take"
        assert result["message"] == "kept the existing take"
        assert result["state"] == "IDLE"
        assert result["dialog"] is False

    def test_enter_confirms(self, live_server):
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        __press("Enter");
        await __settle();
        const during = button("status-state").textContent;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ during }));
        """, live_server)
        assert result["during"] == "RECORDING", "Enter did not confirm"

    def test_enter_on_the_focused_cancel_button_keeps_the_take(self, live_server):
        """Focus lands on Cancel precisely so a reflexive Enter is safe.

        The keydown listener sits on the dialog, and Enter bubbles to it from
        the button, so the dialog answered `true` for a press that visibly
        landed on "Keep" - and its preventDefault cancelled the synthetic click
        that would have answered `false`. A keyboard user who read the button
        under the cursor and pressed Enter deleted the take.
        """
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        __press("Enter", __dialogButton("cancel"));
        await __settle();
        // Only the dialog's own listener is dispatched here: the stub does not
        // synthesise the click a real Enter raises on a focused button, which
        // is what closes the dialog in a browser. What this can prove is the
        // half that mattered - the dialog did not answer `true` behind the
        // button's back, so no take was deleted.
        console.log(JSON.stringify({
          state: button("status-state").textContent,
        }));
        """, live_server)
        assert result["state"] != "RECORDING", "Enter on Keep deleted the take"

    def test_the_dialog_takes_focus_so_a_keyboard_user_is_not_stranded(
        self, live_server
    ):
        """Without this the keys above answer a dialog that never had focus,
        and a screen reader carries on reading the page behind it."""
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        console.log(JSON.stringify({
          focusedInside: __dialog().contains(globalThis.__focused),
          role: __dialog().attributes["role"],
          labelled: Boolean(__dialog().attributes["aria-label"]),
        }));
        """, live_server)
        assert result["focusedInside"] is True, "nothing inside the dialog took focus"
        assert result["role"] == "alertdialog"
        assert result["labelled"] is True

    def test_accepting_re_records_the_line(self, live_server):
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        __dialogButton("confirm").handlers.click();
        await __settle();
        const during = button("status-state").textContent;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ during, dialog: __dialog() !== null, opens:
          __record.filter((entry) => entry === "getUserMedia").length }));
        """, live_server)
        assert result["during"] == "RECORDING", "the re-record never started"
        assert result["opens"] == 2
        assert result["dialog"] is False

    def test_a_second_tap_while_the_dialog_is_open_opens_no_second_dialog(
        self, live_server
    ):
        """confirm() blocked the page; this does not. A key still live under
        the thumb could stack a second dialog over the first, and answering
        one would leave the other asking about a take already gone."""
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        tapRecord();
        await __settle();
        const dialogs = __body.children.filter(
          (kid) => kid.attributes["role"] === "alertdialog").length;
        __dialogButton("cancel").handlers.click();
        await __settle();
        console.log(JSON.stringify({
          dialogs,
          opens: __record.filter((entry) => entry === "getUserMedia").length,
        }));
        """, live_server)
        assert result["dialogs"] == 1, "a second tap stacked another dialog"
        assert result["opens"] == 1, "a second tap started a take behind the dialog"

    def test_tapping_another_line_while_the_dialog_is_open_moves_no_take(
        self, live_server
    ):
        """The dialog does not block the page, so the rows behind it stay live.

        record() captures the index before it asks, but deleted that index while
        recording into whatever the cursor had since become: a tap on line 3
        during the question threw line 1's take away and put the new one on
        line 3. window.confirm made this impossible by freezing the page; an
        in-page dialog has to refuse the move itself.
        """
        result = drive_app(self.RECORD_ONE + """
        tapRecord();
        await __settle();
        line(2).handlers.click();
        await __settle();
        __dialogButton("confirm").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        await __settle();
        // The recorded class is what the page shows about each line, so this
        // asks the screen rather than the session behind it.
        const isRecorded = (index) =>
          (line(index).className || "").includes("recorded");
        console.log(JSON.stringify({
          first: isRecorded(0),
          third: isRecorded(2),
        }));
        """, live_server)
        assert result["first"] is True, "line 1's take was deleted for a line nobody confirmed"
        assert result["third"] is False, "the new take landed on the line tapped mid-dialog"

    def test_a_fresh_line_records_without_asking(self, live_server):
        """The confirm belongs to the delete, not to the key: an unrecorded
        line has nothing to lose, and a prompt on every take would be noise."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        // Awaited, unlike every tap above: with nothing to confirm this one
        // settles by itself, and awaiting it is what proves it never waited.
        await button("btn-record").handlers.click();
        await __settle();
        const during = button("status-state").textContent;
        const asked = __dialog() !== null;
        await button("btn-stop").handlers.click();
        await __settle();
        console.log(JSON.stringify({ asked, during }));
        """, live_server)
        assert result["asked"] is False, "a first take asked for confirmation"
        assert result["during"] == "RECORDING"

    def test_the_key_announces_itself_as_a_re_record(self, live_server):
        """The glyph cannot say it, so the accessible name has to: without it
        a screen-reader user meets the confirm with no warning."""
        result = drive_app(self.RECORD_ONE + """
        const onRecorded = button("btn-record").attributes["aria-label"];
        line(1).handlers.click();
        await __settle();
        console.log(JSON.stringify({
          onRecorded, onFresh: button("btn-record").attributes["aria-label"],
        }));
        """, live_server)
        assert result["onRecorded"] == "Re-record"
        assert result["onFresh"] == "Record"


class TestTheConfirmationIsClampedIntoTheViewport:
    """The geometry, run as the pure function it is. It lives in state.js
    because deciding where a box fits is arithmetic, not painting: render.js
    measures the row and applies the answer, and this can be checked without a
    layout engine at all.

    A dialog anchored to a line near an edge is exactly where a naive
    "row.bottom + gap" puts half of it off screen - and on a phone, off the
    bottom is unreachable, so the take can be neither kept nor discarded."""

    CALL = """
    import {{ clampToViewport }} from "./state.js";
    console.log(JSON.stringify(clampToViewport({args})));
    """

    # A phone-shaped viewport and a dialog small enough to fit in it, so an
    # assertion that fails does so because of the clamp, not the numbers.
    VIEWPORT = types.MappingProxyType({"width": 400, "height": 800})
    BOX = types.MappingProxyType({"width": 260, "height": 120})

    def clamp(self, **args):
        return run_node(self.CALL.format(args=json.dumps(args, default=dict)))

    def test_it_sits_under_a_line_with_room_below_it(self):
        placed = self.clamp(
            anchor={"top": 300, "left": 20, "bottom": 356, "right": 320},
            box=self.BOX, viewport=self.VIEWPORT, gap=8,
        )
        assert placed["top"] == 364, placed

    def test_a_line_at_the_bottom_edge_puts_it_above_the_line(self):
        """Flipped rather than merely pushed up: sliding it up would cover the
        very line the question is about."""
        placed = self.clamp(
            anchor={"top": 740, "left": 20, "bottom": 796, "right": 320},
            box=self.BOX, viewport=self.VIEWPORT, gap=8,
        )
        assert placed["top"] + self.BOX["height"] <= self.VIEWPORT["height"]
        assert placed["top"] + self.BOX["height"] <= 740, placed

    def test_a_line_at_the_top_edge_keeps_the_whole_box_on_screen(self):
        """With no room either side the box is pinned rather than flipped off
        the top, because being reachable beats being beside the line."""
        placed = self.clamp(
            anchor={"top": 0, "left": 20, "bottom": 8, "right": 320},
            box={"width": 260, "height": 790}, viewport=self.VIEWPORT, gap=8,
        )
        assert placed["top"] >= 0, placed
        assert placed["top"] + 790 <= 800, placed

    def test_it_never_runs_off_the_right_edge(self):
        placed = self.clamp(
            anchor={"top": 300, "left": 380, "bottom": 356, "right": 399},
            box=self.BOX, viewport=self.VIEWPORT, gap=8,
        )
        assert placed["left"] + self.BOX["width"] <= self.VIEWPORT["width"], placed

    def test_it_never_runs_off_the_left_edge(self):
        placed = self.clamp(
            anchor={"top": 300, "left": -120, "bottom": 356, "right": 100},
            box=self.BOX, viewport=self.VIEWPORT, gap=8,
        )
        assert placed["left"] >= 0, placed

    def test_a_box_wider_than_the_viewport_is_pinned_to_the_left(self):
        """A phone in portrait with a long line: clamping both edges at once is
        impossible, so the left edge wins and the box is the one that scrolls."""
        placed = self.clamp(
            anchor={"top": 300, "left": 20, "bottom": 356, "right": 320},
            box={"width": 600, "height": 120}, viewport=self.VIEWPORT, gap=8,
        )
        assert placed["left"] == 0, placed


class TestTheConfirmationIsPlacedEvenWithNoLineToAnchorTo:
    """centreInViewport is the fallback half of the same arithmetic, and lives
    beside clampToViewport for the same reason.

    render.js positions the dialog against the row it is asking about. Every
    caller today passes the cursor's row, which is always drawn - but a dialog
    that skipped positioning kept `position: fixed` with no top or left, which
    does not mean "centred": it means the box falls to where it would have sat
    in the flow, below the whole script and off the bottom of a phone. The
    question would be unanswerable rather than merely misplaced."""

    CALL = """
    import {{ centreInViewport }} from "./state.js";
    console.log(JSON.stringify(centreInViewport({args})));
    """

    VIEWPORT = types.MappingProxyType({"width": 400, "height": 800})
    BOX = types.MappingProxyType({"width": 260, "height": 120})

    def centre(self, **args):
        return run_node(self.CALL.format(args=json.dumps(args, default=dict)))

    def test_it_centres_the_box_in_the_viewport(self):
        placed = self.centre(box=self.BOX, viewport=self.VIEWPORT)
        assert placed == {"top": 340, "left": 70}, placed

    def test_a_box_larger_than_the_viewport_is_pinned_to_the_top_left(self):
        """Same rule as the clamp: centring a box bigger than the screen puts
        its top edge off it, and the half you cannot reach holds the buttons."""
        placed = self.centre(
            box={"width": 600, "height": 900}, viewport=self.VIEWPORT,
        )
        assert placed == {"top": 0, "left": 0}, placed


class TestALineIsChosenByTappingIt:
    """Prev and Next are gone. They existed because the curses recorder has no
    pointer; a web page does, and every chunk row is already a click target."""

    def test_tapping_a_line_moves_the_cursor_there(self, live_server):
        result = drive_app("""
        line(2).handlers.click();
        await __settle();
        console.log(JSON.stringify({
          current: line(2).attributes["aria-current"],
          // Boolean rather than the value: an absent attribute is undefined,
          // which JSON.stringify drops from the object entirely.
          elsewhere: "aria-current" in line(0).attributes,
        }));
        """, live_server)
        assert result["current"] == "true", "the tapped line is not the cursor"
        assert result["elsewhere"] is False, "two lines claim to be the cursor"

    def test_the_tapped_line_is_the_one_that_gets_recorded(self, live_server):
        """The whole point of the tap: what is selected is what the mic writes."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        line(3).handlers.click();
        await __settle();
        await button("btn-record").handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        const response = await fetch("/api/scripts/es/a.txt");
        const script = await response.json();
        console.log(JSON.stringify({ recorded: script.recorded }));
        """, live_server)
        assert result["recorded"] == [3], "the take landed on the wrong line"

    def test_a_tap_mid_take_cannot_move_the_cursor(self, live_server):
        """Moving the line under a running take would save the audio against
        whichever row was tapped last."""
        result = drive_app("""
        __tracks.push({ stop: () => {} });
        await button("btn-record").handlers.click();
        await __settle();
        line(3).handlers.click();
        await __settle();
        await button("btn-stop").handlers.click();
        await __settle();
        const response = await fetch("/api/scripts/es/a.txt");
        const script = await response.json();
        console.log(JSON.stringify({ recorded: script.recorded }));
        """, live_server)
        assert result["recorded"] == [0], "a mid-take tap moved the recording"
