// Presentation layer: paints a view object into the DOM.
//
// The browser counterpart of recorder_ui.py. Knows elements, classes and
// geometry; knows nothing of fetch, MediaRecorder, or what makes a chunk
// count as recorded. Every function here takes data and patches DOM.

import {
  PAUSED,
  PLAYING,
  RECORDING,
  blinkGlyph,
  clampToViewport,
  elapsedLabel,
  isBusy,
  nextUnrecorded,
} from "./state.js";

export function elements() {
  return {
    waveform: document.getElementById("waveform"),
    title: document.getElementById("title"),
    menu: document.getElementById("script-menu"),
    menuToggle: document.getElementById("menu-toggle"),
    sidebar: document.getElementById("sidebar"),
    chunks: document.getElementById("chunk-list"),
    status: document.getElementById("status-bar"),
    statusState: document.getElementById("status-state"),
    statusLegend: document.getElementById("status-legend"),
    dot: document.getElementById("record-dot"),
    timer: document.getElementById("record-timer"),
    message: document.getElementById("message"),
    recordButton: document.getElementById("btn-record"),
    playButton: document.getElementById("btn-play"),
    stopButton: document.getElementById("btn-stop"),
    nextTakeButton: document.getElementById("btn-next-take"),
  };
}

/** Paint the whole screen. `handlers.onSelectScript` is bound to menu rows,
 * which are rebuilt on every draw; every other control is bound once by the
 * controller against a stable element. */
export function draw(dom, view, handlers = {}) {
  drawTitle(dom, view);
  drawMenu(dom, view, handlers.onSelectScript);
  drawChunks(dom, view, handlers.onSelectChunk);
  drawStatus(dom, view);
  drawControls(dom, view);
}

function drawTitle(dom, view) {
  dom.title.textContent = view.title;
}

function drawMenu(dom, view, onSelect) {
  const list = dom.menu;
  list.replaceChildren();
  for (const entry of view.scripts) {
    list.appendChild(scriptRow(entry, onSelect));
  }
}

function scriptRow(entry, onSelect) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "script";
  button.dataset.script = entry.name;
  button.setAttribute("aria-current", entry.active ? "true" : "false");
  if (entry.active) {
    button.classList.add("is-active");
  }
  if (entry.complete) {
    button.classList.add("is-complete");
  }

  const label = document.createElement("span");
  label.className = "script-name";
  label.textContent = entry.name;

  const progress = document.createElement("span");
  progress.className = "script-progress";
  progress.textContent = entry.complete ? `✓ ${entry.progress}` : entry.progress;

  button.append(label, progress);
  if (onSelect) {
    button.addEventListener("click", () => onSelect(entry.name));
  }
  item.appendChild(button);
  return item;
}

function drawChunks(dom, view, onSelect) {
  const list = dom.chunks;
  list.replaceChildren();
  view.chunks.forEach((text, index) => {
    list.appendChild(chunkRow(text, index, view, onSelect));
  });
}

function chunkRow(text, index, view, onSelect) {
  const item = document.createElement("li");
  // The status class is the whole colour decision: the domain says
  // "recorded_selected", the stylesheet decides what that looks like.
  item.className = `chunk chunk--${view.statuses[index]}`;
  item.dataset.index = String(index);
  item.id = `chunk-${index}`;
  if (index === view.cursor) {
    item.setAttribute("aria-current", "true");
  }

  const gutter = document.createElement("span");
  gutter.className = "chunk-gutter";
  gutter.textContent = `${index === view.cursor ? "▸" : " "}${String(index + 1).padStart(3, " ")} ${
    view.recorded.has(index) ? "✓" : " "
  }`;

  const body = document.createElement("span");
  body.className = "chunk-text";
  body.textContent = text;

  item.append(gutter, body);
  if (onSelect) {
    item.addEventListener("click", () => onSelect(index));
  }
  return item;
}

/** Scroll the cursor line into view, holding still when it is already visible. */
export function scrollCursorIntoView(dom, cursor) {
  const row = dom.chunks.querySelector(`#chunk-${cursor}`);
  if (!row) {
    return;
  }
  const box = dom.chunks.getBoundingClientRect();
  const rowBox = row.getBoundingClientRect();
  if (rowBox.top < box.top || rowBox.bottom > box.bottom) {
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

function drawStatus(dom, view) {
  const recording = view.state === RECORDING;
  dom.status.classList.toggle("is-recording", recording);
  dom.statusState.textContent = recording ? "RECORDING" : view.state.toUpperCase();
  dom.statusLegend.textContent = view.legend;
  dom.dot.textContent = recording ? blinkGlyph(view.tick) : "";
  dom.dot.hidden = !recording;
  dom.timer.textContent = recording ? elapsedLabel(view.elapsed) : "";
  dom.message.textContent = view.message || "";
}

function drawControls(dom, view) {
  const recording = view.state === RECORDING;
  const busy = isBusy(view.state);
  const hasScript = view.chunks.length > 0;
  const onRecorded = hasScript && view.recorded.has(view.cursor);
  const playing = view.playback === PLAYING;
  const paused = view.playback === PAUSED;

  // Record only ever starts a take and Stop only ever ends one, so the key
  // under the thumb keeps its meaning between presses. Play/pause is the one
  // deliberate exception: a deck pairs those on a single key.
  //
  // On a line that already has audio this key is the re-record that the Redo
  // button used to be, so its accessible name says so - the glyph cannot, and
  // a blind user would otherwise get no warning before the confirm appears.
  dom.recordButton.classList.toggle("is-recording", recording);
  dom.recordButton.setAttribute("aria-label", onRecorded ? "Re-record" : "Record");
  dom.recordButton.disabled = !hasScript || busy;

  // One key plays, pauses and resumes, so its face has to say which of those
  // the next press does. Pausing shows play again because that is what resumes.
  const playPauseLabel = playing ? "Pause" : paused ? "Resume" : "Play";
  // Written on the button rather than a child span: the glyph is the button's
  // only content, and addressing it through the element already bound here
  // keeps every lookup in elements() where the view's DOM contract lives.
  // The deck's pair glyph at rest, because one key does both; a lone pause bar
  // while a clip runs, because then the next press can only be pause.
  dom.playButton.textContent = playing ? "⏸" : "⏯";
  dom.playButton.setAttribute("aria-label", playPauseLabel);
  dom.playButton.classList.toggle("is-playing", playing);
  // Enabled while paused even though the cursor may sit on an unrecorded line:
  // the clip is still loaded, and this key is what resumes it.
  dom.playButton.disabled = busy || (!onRecorded && !paused);
  // Stop ends whichever transport is running - a take being captured, or a
  // clip being played - and is dead only when neither is.
  dom.stopButton.disabled = !recording && !playing && !paused;

  // The one control that spans two lines: it saves this take and immediately
  // arms the next, so it means nothing unless a take is actually running and
  // there is a line left for it to arm.
  //
  // The same rule the controller obeys rather than a second guess at it: "not
  // the last line" used to stand in for it, which lit the key on a mid-script
  // line with every later line already recorded - promising an arm the
  // controller would then decline to perform.
  dom.nextTakeButton.disabled =
    !recording
    || !hasScript
    || nextUnrecorded(view.cursor, view.chunks.length, view.recorded) === null;
}

// ---------- waveform canvas ----------
//
// Pixels only. The numbers drawn here are computed by waveform.js and fetched
// by api.js; this decides bar widths and colours from them and nothing else.
//
// Every entry point tolerates a missing canvas or a missing 2D context. On a
// browser that cannot give one, the recorder keeps working with a blank strip
// where the trace would be - a canvas failure must never cost a take.

/** The drawing surface, sized to its CSS box in device pixels, or null.
 *
 * Resolved on every draw rather than cached: the canvas is laid out by CSS and
 * a phone rotating changes its box without any event this module sees. */
function surface(canvas) {
  if (!canvas || typeof canvas.getContext !== "function") {
    return null;
  }
  let context = null;
  try {
    context = canvas.getContext("2d");
  } catch {
    return null;
  }
  if (!context) {
    return null;
  }

  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const box = canvas.getBoundingClientRect();
  const width = Math.max(Math.round((box.width || canvas.clientWidth || 0) * ratio), 1);
  const height = Math.max(Math.round((box.height || canvas.clientHeight || 0) * ratio), 1);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { context, width, height };
}

/** The colours the strip is painted in, read from the stylesheet rather than
 * spelled here, so the palette stays one table in style.css. */
function ink(canvas, name, fallback) {
  const styles = window.getComputedStyle ? window.getComputedStyle(canvas) : null;
  const value = styles ? styles.getPropertyValue(name).trim() : "";
  return value || fallback;
}

export function showWaveform(dom, visible) {
  if (dom.waveform) {
    dom.waveform.hidden = !visible;
  }
}

export function clearWaveform(dom) {
  const target = surface(dom.waveform);
  if (target) {
    target.context.clearRect(0, 0, target.width, target.height);
  }
}

/** One live analyser frame: a trace of -1..1 offsets about the centre line. */
export function drawTrace(dom, trace, level = 0) {
  const target = surface(dom.waveform);
  if (!target) {
    return;
  }
  const { context, width, height } = target;
  const middle = height / 2;
  context.clearRect(0, 0, width, height);

  // A quiet mic is the failure this whole feature exists to make visible, so
  // the centre line is drawn even when there is no signal at all: a blank
  // canvas reads as "broken", a flat line reads as "hearing nothing".
  context.strokeStyle = ink(dom.waveform, "--waveform-axis", "#2b3039");
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, middle);
  context.lineTo(width, middle);
  context.stroke();

  if (!trace || trace.length === 0) {
    return;
  }

  context.strokeStyle = ink(dom.waveform, "--waveform-live", "#e01b24");
  context.lineWidth = Math.max(Math.round(height / 40), 1);
  context.lineJoin = "round";
  context.beginPath();
  trace.forEach((offset, column) => {
    const x = (column / Math.max(trace.length - 1, 1)) * width;
    const y = middle - offset * middle * 0.92;
    if (column === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();

  drawLevelBar(context, dom.waveform, width, height, level);
}

/** A thin loudness bar along the foot of the strip. The trace alone is hard to
 * read at a glance on a phone held at reading distance; the bar answers "is it
 * hearing me" from the corner of an eye. */
function drawLevelBar(context, canvas, width, height, level) {
  const bar = Math.max(Math.round(height / 16), 2);
  context.fillStyle = ink(canvas, "--waveform-level", "#33d17a");
  context.fillRect(0, height - bar, width * Math.min(Math.max(level, 0), 1), bar);
}

/** A saved clip's peaks, with the playhead at `progress` (0..1).
 *
 * Peaks are mirrored about the centre line: the pair of bars is the shape a
 * reader recognises as a waveform, where the upper half alone reads as a chart.
 */
export function drawPeaks(dom, peaks, progress = 0) {
  const target = surface(dom.waveform);
  if (!target) {
    return;
  }
  const { context, width, height } = target;
  const middle = height / 2;
  context.clearRect(0, 0, width, height);

  if (!peaks || peaks.length === 0) {
    return;
  }

  const played = ink(dom.waveform, "--waveform-played", "#33d17a");
  const ahead = ink(dom.waveform, "--waveform-peaks", "#62a0ea");
  const step = width / peaks.length;
  const barWidth = Math.max(step * 0.7, 1);
  const head = progress * width;

  peaks.forEach((peak, column) => {
    const x = column * step;
    // Coloured by position rather than redrawn per frame: the playhead moving
    // is the whole animation, and repainting only its colour keeps a timeupdate
    // (four a second on most browsers) cheap.
    context.fillStyle = x + step / 2 <= head ? played : ahead;
    const half = Math.max(peak * middle * 0.92, 0.5);
    context.fillRect(x, middle - half, barWidth, half * 2);
  });

  context.fillStyle = ink(dom.waveform, "--waveform-playhead", "#e5a50a");
  context.fillRect(Math.min(head, width - 1), 0, Math.max(Math.round(width / 400), 1), height);
}

/** Two classes because the two layouts disagree about the resting state: on a
 * phone the drawer is closed until `is-open`, on a desktop it is docked open
 * until `is-collapsed`. The controller holds one boolean either way. */
export function setSidebarOpen(dom, open) {
  dom.sidebar.classList.toggle("is-open", open);
  dom.sidebar.classList.toggle("is-collapsed", !open);
  dom.menuToggle.setAttribute("aria-expanded", open ? "true" : "false");
}

// ---------- the re-record confirmation ----------
//
// window.confirm was doing this job, and Chrome docks a native modal to the top
// of the viewport whatever the page is showing. Re-recording is started from a
// line the user has just tapped, usually far down a long script, so the answer
// button landed half a screen from both the line and the thumb that asked.
//
// Built here rather than declared in index.html: a dialog that exists in the
// markup is a permanent part of the view's contract (every id render.js looks
// up must be in the page), and this one is transient - it exists only between
// the question and its answer.

/** Ask about `index`, resolving true to go ahead and false to leave it alone.
 *
 * The promise is the whole seam: confirm() blocked the page, this does not, so
 * the controller awaits an answer instead of reading one. Only one dialog is
 * ever open, because a second question about a take the first has already
 * discarded can only be answered wrongly. */
export function confirmNearChunk(dom, { index, question, confirmLabel, cancelLabel }) {
  if (openDialog) {
    // Already asking. Resolving false here would be a silent "no" the user
    // never gave; the standing dialog is the one they can see, so it answers.
    return Promise.resolve(false);
  }
  return new Promise((resolve) => {
    const dialog = buildDialog({ question, confirmLabel, cancelLabel });
    openDialog = dialog.root;
    document.body.appendChild(dialog.root);
    // Positioned after it is in the document: an unattached node measures zero
    // in every browser, which would clamp every dialog to the top-left corner.
    positionNearChunk(dom, dialog.root, index);

    const answer = (verdict) => {
      dialog.root.remove();
      openDialog = null;
      resolve(verdict);
    };
    dialog.confirm.addEventListener("click", () => answer(true));
    dialog.cancel.addEventListener("click", () => answer(false));
    dialog.root.addEventListener("keydown", (event) => {
      // Escape only ever cancels. Folding it into one handler with Enter is
      // how a key pressed to back out ends up deleting the take.
      if (event.key === "Escape") {
        event.preventDefault();
        answer(false);
      } else if (event.key === "Enter") {
        event.preventDefault();
        answer(true);
      }
    });
    // Focus lands on Cancel, not Confirm: a stray Space or Enter on a dialog
    // that just appeared must not be the press that throws a take away.
    dialog.cancel.focus();
  });
}

// The dialog currently on screen, or null. Module-scoped rather than passed in
// because "is one already open" is a fact about the screen, and the controller
// owning a copy of it would be a second answer to the same question.
let openDialog = null;

function buildDialog({ question, confirmLabel, cancelLabel }) {
  const root = document.createElement("div");
  root.className = "confirm";
  // alertdialog rather than dialog: it interrupts to ask about something
  // destructive, which is exactly the distinction the role draws.
  root.setAttribute("role", "alertdialog");
  root.setAttribute("aria-modal", "true");
  root.setAttribute("aria-label", question);
  // Focusable as a container so keydown lands on the dialog rather than on
  // whichever button happens to hold focus; -1 keeps it out of the tab order.
  root.setAttribute("tabindex", "-1");

  const text = document.createElement("p");
  text.className = "confirm-question";
  text.textContent = question;

  const actions = document.createElement("div");
  actions.className = "confirm-actions";

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "confirm-button confirm-button--cancel";
  cancel.dataset.action = "cancel";
  cancel.textContent = cancelLabel;

  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.className = "confirm-button confirm-button--confirm";
  confirm.dataset.action = "confirm";
  confirm.textContent = confirmLabel;

  // Cancel first, so the key nearest the thumb on a phone is the harmless one.
  actions.append(cancel, confirm);
  root.append(text, actions);
  return { root, cancel, confirm };
}

/** Pin the dialog beside its line, or as near as the viewport allows.
 *
 * Fixed rather than absolute: the chunk list scrolls under it, and an
 * absolutely-positioned box would drift off with the row it was measured
 * against while the question was still being read. */
function positionNearChunk(dom, root, index) {
  const row = dom.chunks.querySelector(`#chunk-${index}`);
  if (!row) {
    return;
  }
  const anchor = row.getBoundingClientRect();
  const box = root.getBoundingClientRect();
  const { top, left } = clampToViewport({
    anchor,
    box: { width: box.width, height: box.height },
    viewport: { width: window.innerWidth, height: window.innerHeight },
  });
  root.style.top = `${top}px`;
  root.style.left = `${left}px`;
}
