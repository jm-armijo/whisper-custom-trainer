// Controller: joins the API client, the microphone and the view.
//
// The browser counterpart of record_data.py. Sequencing and plumbing only -
// every rule lives in state.js, every pixel in render.js, every URL in api.js.

import * as analysis from "./audio_analysis.js";
import * as api from "./api.js";
import * as mic from "./microphone.js";
import * as render from "./render.js";
import {
  levelFromTimeDomain,
  peaksFromSamples,
  playheadFraction,
  traceFromTimeDomain,
} from "./waveform.js";
import {
  IDLE,
  PAUSED,
  PLAYING,
  RECORDING,
  STOPPED,
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
  // Tracked separately from `state`: a clip playing does not commit the
  // microphone, so it must not read as busy.
  playback: STOPPED,
  tick: 0,
  message: "",
  startedAt: 0,
  timer: null,
  // Bumped on every save so playback never serves a stale cached take.
  version: 0,
  // Docked open on a wide screen, closed on a phone where it overlays the text.
  menuOpen: window.matchMedia("(min-width: 40.0625rem)").matches,
};

// ---------- the live waveform ----------
//
// One analyser and at most one animation frame, both owned here. The pair must
// be torn down together on every path out of a take: a rAF loop left running
// wakes the phone's GPU sixty times a second for a screen nobody is watching,
// and a leaked AudioContext counts against a per-page limit the browser will
// eventually refuse to raise.
const analyser = new analysis.LiveAnalyser();
let frame = null;

/** Attach the analyser to the stream microphone.js already opened.
 *
 * Called from inside the record tap's handler, never at module load: iOS
 * creates every AudioContext suspended and resumes one only within the gesture
 * that asked for it. */
async function startWaveform() {
  if (typeof window.requestAnimationFrame !== "function") {
    return;
  }
  const attached = await analyser.start(microphone.stream);
  if (!attached) {
    // No Web Audio, or it refused the stream. The take still records; there is
    // simply nothing to draw, and the strip stays hidden rather than blank.
    return;
  }
  const step = () => {
    const bytes = analyser.sample();
    if (!bytes) {
      // The analyser was closed between scheduling this frame and running it.
      frame = null;
      return;
    }
    render.drawTrace(dom, traceFromTimeDomain(bytes), levelFromTimeDomain(bytes));
    frame = window.requestAnimationFrame(step);
  };
  try {
    render.showWaveform(dom, true);
    frame = window.requestAnimationFrame(step);
  } catch {
    // Whatever the canvas did, the take is what matters: drop the waveform and
    // let the recording carry on rather than failing the whole tap.
    stopWaveform();
  }
}

/** Cancel the frame first, then release the graph: closing the context while a
 * frame is still queued leaves that frame sampling a dead analyser. */
function stopWaveform() {
  if (frame !== null && typeof window.cancelAnimationFrame === "function") {
    window.cancelAnimationFrame(frame);
  }
  frame = null;
  analyser.close();
  render.clearWaveform(dom);
  render.showWaveform(dom, false);
}

function setMenu(open) {
  app.menuOpen = open;
  render.setSidebarOpen(dom, open);
}

function repaint({ scroll = false } = {}) {
  const view = buildView({
    session: app.session,
    scripts: app.scripts,
    state: app.state,
    playback: app.playback,
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

/** Record only ever starts a take; Stop is its own button now.
 *
 * A tap landing while the previous take is still uploading does nothing.
 * Falling through to startRecording here is what let a second tap capture over
 * an in-flight upload, so two takes raced for one line.
 *
 * Recording over a line that already has a take is a re-record, which is what
 * the separate Redo button used to be. A deck has no redo key: the same key
 * records whatever line is selected, and the only thing a finished line needs
 * is the confirmation before its audio is thrown away.
 */
async function record() {
  if (isBusy(app.state) || !app.session) {
    return;
  }
  const index = app.session.cursor;
  if (!app.session.isRecorded(index)) {
    await startRecording();
    return;
  }
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

async function startRecording() {
  if (!app.session) {
    return;
  }
  if (!mic.isSupported()) {
    say("this browser cannot capture audio - needs getUserMedia over https or localhost");
    return;
  }
  // A take and a clip must never run at once: Stop serves both, so two live
  // transports would leave one button with two meanings. Playback yields.
  stopPlayback();
  // A new take replaces whatever the playback strip was showing.
  stopPlaybackWaveform();
  await guard(async () => {
    await microphone.start();
    // After the recorder is running, so a browser that refuses the analyser
    // costs the take nothing; awaited inside the tap's stack to keep iOS's
    // gesture, which is what lets resume() settle.
    await startWaveform();
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
  // Before awaiting the blob, not after: the analyser is watching a stream that
  // is about to end, and every frame between here and the upload landing paints
  // a trace of silence. Tearing down first is also what guarantees the loop
  // stops even if stop() rejects.
  stopWaveform();
  const blob = await microphone.stop();

  // Paint UPLOADING before awaiting the POST, not after. Setting the state and
  // going straight into the await left the screen reading RECORDING with a
  // frozen dot for the whole of a slow phone upload, while record() already
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
    // Bound once, to the one element. The playhead is driven from the element's
    // own clock rather than from Web Audio: routing it through
    // createMediaElementSource would silence it unless the graph were also
    // wired to destination, and on iOS that is the quickest way to lose the
    // playback the reused element exists to protect.
    player.addEventListener("timeupdate", drawPlayhead);
    player.addEventListener("durationchange", drawPlayhead);
    player.addEventListener("ended", () => {
      drawPlayhead();
      // The clip ran to its end on its own; the transport is idle again and
      // the buttons must stop offering Pause and Stop for it.
      app.playback = STOPPED;
      repaint();
    });
  }
  return player;
}

// The peaks currently on screen, and the take they belong to.
//
// The token is the guard against a slow decode: taps land faster than a fetch
// and decode complete, so an earlier take's peaks can arrive after a later tap
// has already started drawing. Only the newest tap's token may paint.
let playbackPeaks = null;
let playbackToken = 0;

function stopPlaybackWaveform() {
  playbackToken += 1;
  playbackPeaks = null;
  render.clearWaveform(dom);
  render.showWaveform(dom, false);
}

function drawPlayhead() {
  if (!playbackPeaks || !player) {
    return;
  }
  render.drawPeaks(dom, playbackPeaks, playheadFraction(player.currentTime, player.duration));
}

/** Fetch the clip, decode it, and draw its peaks.
 *
 * Deliberately not awaited by play(): the waveform must never delay the play()
 * call, which has to reach the element in the same turn as the tap to keep
 * iOS's permission. It paints whenever it is ready, a beat behind the audio.
 *
 * The peaks are computed from the wav on every playback and never stored -
 * dataset.csv keeps the three columns train.py reads, and a cached peak file
 * would be one more artifact to invalidate on a re-record.
 */
async function drawPlaybackWaveform(name, index, token) {
  if (!analysis.isSupported()) {
    return;
  }
  try {
    const buffer = await api.fetchAudio(name, index, app.version);
    const samples = await analysis.decodeChannel(buffer);
    // A newer tap owns the strip now; this decode is for a take already gone.
    if (token !== playbackToken || !samples) {
      return;
    }
    playbackPeaks = peaksFromSamples(samples);
    render.showWaveform(dom, true);
    drawPlayhead();
  } catch {
    // The take plays regardless: the <audio> element fetches the clip itself
    // and does not care that this second fetch or decode failed.
  }
}

/** The play/pause key: plays, pauses what is playing, resumes what is paused.
 *
 * One key for the three because that is the deck it is modelled on. Resume is
 * not a re-play: the clip is still loaded in the element, so play() continues
 * from where pause left it rather than fetching the take again.
 */
function playPause() {
  if (app.playback === PLAYING) {
    pause();
    return;
  }
  if (app.playback === PAUSED && player) {
    player.play().then(
      () => {
        app.playback = PLAYING;
        repaint();
      },
      () => say("could not resume playback"),
    );
    return;
  }
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

  // Invalidates any in-flight decode, then claims the strip for this tap. The
  // fetch is started without awaiting so play() below still runs in this turn.
  stopPlaybackWaveform();
  drawPlaybackWaveform(app.session.name, index, playbackToken);

  element.play().then(
    () => {
      app.playback = PLAYING;
      repaint();
      say(`played line ${index + 1}`);
    },
    (error) => {
      // A second tap calls load(), which rejects the first tap's still-pending
      // play() with AbortError. That older promise settling must not report a
      // failure over the newer take that is playing correctly - with one
      // reused element both handlers run, and the last to settle wins.
      if (error.name === "AbortError") return;
      // A rejected play must not leave the transport reading PLAYING: the
      // buttons would offer Pause and Stop for a clip that never started.
      app.playback = STOPPED;
      repaint();
      say(error.name === "NotAllowedError"
        ? "tap play again to allow audio"
        : `could not play that take: ${error.message}`);
    },
  );
}

/** Hold the clip where it is, keeping it loaded so the same key resumes it. */
function pause() {
  if (app.playback !== PLAYING || !player) {
    return;
  }
  player.pause();
  app.playback = PAUSED;
  repaint();
  say("paused");
}

/** Stop whichever transport is running.
 *
 * One button for both because that is what the glyph means on a deck, and
 * because the two can never run at once: recording is refused while a clip
 * plays and play is disabled while a take records.
 */
async function stop() {
  if (app.state === RECORDING) {
    await stopRecording();
    return;
  }
  stopPlayback();
}

function stopPlayback() {
  if (app.playback === STOPPED || !player) {
    return;
  }
  player.pause();
  // Rewound rather than left mid-clip: Stop on a deck returns to the start,
  // and the next Play on this line should be the whole take again.
  player.currentTime = 0;
  app.playback = STOPPED;
  stopPlaybackWaveform();
  repaint();
  say("stopped");
}

/** Save this take and arm the next line in one press.
 *
 * The read-record-read rhythm otherwise costs two taps per line with a look
 * down at the screen between them. Awaiting stopRecording is what makes this
 * safe: it returns only once the upload has landed, so the cursor never moves
 * off a line whose take is still in flight.
 */
async function stopAndRecordNext() {
  if (app.state !== RECORDING || !app.session) {
    return;
  }
  // Captured before the stop, because a successful save advances the cursor
  // itself: asking isRecorded(cursor) afterwards asks about the *next* line,
  // which is unrecorded either way, so the button armed nothing on success.
  const index = app.session.cursor;
  await stopRecording();
  // Only go on if the take actually saved. A failed upload leaves the cursor
  // put, so the line can be redone rather than silently skipped.
  if (!app.session.isRecorded(index)) {
    return;
  }
  // No move() here: saveTake already stepped down. Moving again would skip the
  // line this button exists to start recording.
  if (index >= app.session.chunks.length - 1) {
    return;
  }
  repaint({ scroll: true });
  await startRecording();
}

function bindControls() {
  dom.recordButton.addEventListener("click", record);
  dom.playButton.addEventListener("click", playPause);
  dom.stopButton.addEventListener("click", stop);
  dom.nextTakeButton.addEventListener("click", stopAndRecordNext);
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
