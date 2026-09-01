// Controller: joins the API client, the microphone and the view.
//
// The browser counterpart of record_data.py. Sequencing and plumbing only -
// every rule lives in state.js, every pixel in render.js, every URL in api.js.

import * as api from "./api.js";
import * as mic from "./microphone.js";
import * as render from "./render.js";
import { IDLE, RECORDING, ScriptSession, buildView, rejectClip, savedMessage } from "./state.js";

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
  if (app.state === RECORDING) {
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
  if (app.state === RECORDING || !app.session) {
    return;
  }
  app.session.select(index);
  app.message = "";
  repaint({ scroll: true });
}

function move(action) {
  if (app.state === RECORDING || !app.session) {
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
  const seconds = (Date.now() - app.startedAt) / 1000;
  const blob = await microphone.stop();
  app.state = IDLE;

  const index = app.session.cursor;
  const rejection = rejectClip(seconds);
  if (rejection) {
    const kept = app.session.isRecorded(index) ? " - previous take kept" : "";
    say(rejection + kept);
    return;
  }

  await guard(async () => {
    await api.saveChunk(app.session.name, index, blob);
    app.session.markRecorded(index);
    app.version += 1;
    bumpScriptProgress();
    // Advance so a straight read needs one button per line, as the terminal
    // recorder's space-then-down rhythm does.
    app.session.move("down");
    app.message = savedMessage(seconds);
    repaint({ scroll: true });
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
  if (!app.session || app.state === RECORDING) {
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

let player = null;

function play() {
  if (!app.session || !app.session.isRecorded(app.session.cursor)) {
    say("nothing recorded on this line yet");
    return;
  }
  const index = app.session.cursor;
  if (player) {
    player.pause();
  }
  player = new Audio(api.audioUrl(app.session.name, index, app.version));
  player.addEventListener("error", () => say("could not play that take"));
  player.play().catch((error) => say(error.message));
  say(`played line ${index + 1}`);
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
