// Presentation layer: paints a view object into the DOM.
//
// The browser counterpart of recorder_ui.py. Knows elements, classes and
// geometry; knows nothing of fetch, MediaRecorder, or what makes a chunk
// count as recorded. Every function here takes data and patches DOM.

import { RECORDING, UPLOADING, blinkGlyph, elapsedLabel, isBusy } from "./state.js";

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
    redoButton: document.getElementById("btn-redo"),
    playButton: document.getElementById("btn-play"),
    prevButton: document.getElementById("btn-prev"),
    nextButton: document.getElementById("btn-next"),
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

  dom.recordButton.textContent = recording ? "Stop" : "Record";
  dom.recordButton.classList.toggle("is-recording", recording);
  // Disabled while a take uploads, so a second tap cannot start capturing over
  // one still in flight; the controller refuses it too, but a live-looking
  // button that does nothing reads as a dropped tap.
  dom.recordButton.disabled = !hasScript || view.state === UPLOADING;

  // Redo and play act on the take under the cursor, so they are meaningless
  // both mid-take and on a line with no audio yet.
  dom.redoButton.disabled = busy || !onRecorded;
  dom.playButton.disabled = busy || !onRecorded;
  dom.prevButton.disabled = busy || !hasScript || view.cursor === 0;
  dom.nextButton.disabled =
    busy || !hasScript || view.cursor >= view.chunks.length - 1;
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
