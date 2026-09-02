// Controller: joins the API client, the microphone and the view.
//
// The browser counterpart of record_data.py. Sequencing and plumbing only -
// every rule lives in state.js, every pixel in render.js, every URL in api.js.

import * as api from "./api.js";
import * as mic from "./microphone.js";
import * as render from "./render.js";
import {
  IDLE,
  RECORDING,
  UPLOADING,
  ScriptSession,
  buildView,
  isBusy,
  savedMessage,
} from "./state.js";

// The dot blinks by redrawing on a timer, matching recorder_theme's blink_ms;
// a CSS animation would drift from the elapsed counter beside it.
const BLINK_MS = 1000;

const dom = render.elements();
const microphone = new mic.Microphone();

const app = {
  scripts: [],
  session: null,
  state: IDLE,
  tick: 0,
  message: "",
  startedAt: 0,
  timer: null,
  // Bumped on every save so playback never serves a stale cached take.
  version: 0,
  // Docked open on a wide screen, closed on a phone where it overlays the text.
  menuOpen: window.matchMedia("(min-width: 40.0625rem)").matches,
};

function setMenu(open) {
  app.menuOpen = open;
  render.setSidebarOpen(dom, open);
}

function repaint({ scroll = false } = {}) {
  const view = buildView({
    session: app.session,
    scripts: app.scripts,
    state: app.state,
    tick: app.tick,
    elapsed: app.state === RECORDING ? (Date.now() - app.startedAt) / 1000 : 0,
    message: app.message,
  });
  render.draw(dom, view, {
    onSelectScript: openScript,
    onSelectChunk: selectChunk,
  });
  if (scroll && app.session) {
    render.scrollCursorIntoView(dom, app.session.cursor);
  }
}

function say(message) {
  app.message = message;
  repaint();
}

async function guard(work) {
  try {
    await work();
  } catch (error) {
    say(error.message || String(error));
  }
}

async function refreshScripts() {
  app.scripts = await api.listScripts();
}

async function openScript(name) {
  if (isBusy(app.state)) {
    return;
  }
  await guard(async () => {
    const payload = await api.loadScript(name);
    app.session = new ScriptSession(payload);
    app.message = "";
    // Get the drawer out of the way of the text it was opened to choose.
    if (window.matchMedia("(max-width: 40rem)").matches) {
      setMenu(false);
    }
    repaint({ scroll: true });
  });
}

function selectChunk(index) {
  if (isBusy(app.state) || !app.session) {
    return;
  }
  app.session.select(index);
  app.message = "";
  repaint({ scroll: true });
}

function move(action) {
  if (isBusy(app.state) || !app.session) {
    return;
  }
  app.session.move(action);
  app.message = "";
  repaint({ scroll: true });
}

async function toggleRecord() {
  if (app.state === RECORDING) {
    await stopRecording();
    return;
  }
  // A tap landing while the previous take is still uploading does nothing.
  // Falling through to startRecording here is what let a second tap capture
  // over an in-flight upload, so two takes raced for one line.
  if (app.state === UPLOADING) {
    return;
  }
  await startRecording();
}

async function startRecording() {
  if (!app.session) {
    return;
  }
  if (!mic.isSupported()) {
    say("this browser cannot capture audio - needs getUserMedia over https or localhost");
    return;
  }
  await guard(async () => {
    await microphone.start();
    app.state = RECORDING;
    app.tick = 0;
    app.startedAt = Date.now();
    app.message = "";
    app.timer = setInterval(() => {
      app.tick += 1;
      repaint();
    }, BLINK_MS);
    repaint();
  });
}

async function stopRecording() {
  clearInterval(app.timer);
  app.timer = null;
  const blob = await microphone.stop();

  // Paint UPLOADING before awaiting the POST, not after. Setting the state and
  // going straight into the await left the screen reading RECORDING with a
  // frozen dot for the whole of a slow phone upload, while toggleRecord already
  // saw a state that let a second tap start a new take over the in-flight one.
  const index = app.session.cursor;
  app.state = UPLOADING;
  app.message = "";
  repaint();

  try {
    await saveTake(index, blob);
  } finally {
    app.state = IDLE;
    // The mic is released only once the take is safely uploaded: closing it
    // before the POST would drop the stream while the blob was still in flight.
    microphone.close();
    repaint({ scroll: true });
  }
}

/** Upload one take and fold the server's verdict into the session. */
async function saveTake(index, blob) {
  await guard(async () => {
    const saved = await api.saveChunk(app.session.name, index, blob);
    app.session.markRecorded(index);
    app.version += 1;
    bumpScriptProgress();
    // Advance so a straight read needs one button per line, as the terminal
    // recorder's space-then-down rhythm does.
    app.session.move("down");
    // The duration is the server's: it decoded the clip that actually landed,
    // where this side could only time MediaRecorder's start-to-stop latency.
    app.message = savedMessage(saved);
  });
}

/** Keep the menu's count in step without a second round trip for the list. */
function bumpScriptProgress() {
  const entry = app.scripts.find((script) => script.name === app.session.name);
  if (entry) {
    entry.recorded = app.session.recorded.size;
  }
}

async function redo() {
  if (!app.session || isBusy(app.state)) {
    return;
  }
  const index = app.session.cursor;
  if (!window.confirm(`Re-record line ${index + 1}?`)) {
    say("kept the existing take");
    return;
  }
  await guard(async () => {
    await api.deleteChunk(app.session.name, index);
    app.session.clearRecorded(index);
    bumpScriptProgress();
    repaint();
    await startRecording();
  });
}

// One element reused for every take, created once and kept.
//
// iOS grants an audio element permission to play only when play() is reached
// inside a user gesture, and that permission belongs to the element, not the
// page: a fresh `new Audio(...)` per tap starts unpermitted, so its play()
// promise rejects with NotAllowedError and nothing is heard. Reusing one
// element that was unlocked on the first tap is what makes playback work on a
// phone at all.
let player = null;

function audioElement() {
  if (!player) {
    player = new Audio();
    player.addEventListener("error", () => say("could not play that take"));
  }
  return player;
}

function play() {
  if (!app.session || !app.session.isRecorded(app.session.cursor)) {
    say("nothing recorded on this line yet");
    return;
  }
  const index = app.session.cursor;
  const element = audioElement();
  element.pause();
  // Assigned synchronously inside the gesture, then played: setting src and
  // calling play() in the same turn is what keeps the gesture's permission.
  element.src = api.audioUrl(app.session.name, index, app.version);
  element.load();
  element.play().then(
    () => say(`played line ${index + 1}`),
    (error) => {
      // A second tap calls load(), which rejects the first tap's still-pending
      // play() with AbortError. That older promise settling must not report a
      // failure over the newer take that is playing correctly - with one
      // reused element both handlers run, and the last to settle wins.
      if (error.name === "AbortError") return;
      say(error.name === "NotAllowedError"
        ? "tap play again to allow audio"
        : `could not play that take: ${error.message}`);
    },
  );
}

function bindControls() {
  dom.recordButton.addEventListener("click", toggleRecord);
  dom.redoButton.addEventListener("click", redo);
  dom.playButton.addEventListener("click", play);
  dom.prevButton.addEventListener("click", () => move("up"));
  dom.nextButton.addEventListener("click", () => move("down"));
  dom.menuToggle.addEventListener("click", () => setMenu(!app.menuOpen));
}

async function boot() {
  bindControls();
  setMenu(app.menuOpen);
  repaint();
  await guard(async () => {
    await refreshScripts();
    repaint();
    if (app.scripts.length === 1) {
      await openScript(app.scripts[0].name);
    }
  });
}

boot();
