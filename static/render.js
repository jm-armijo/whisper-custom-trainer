// Presentation layer: paints a view object into the DOM.
//
// The browser counterpart of recorder_ui.py. Knows elements, classes and
// geometry; knows nothing of fetch, MediaRecorder, or what makes a chunk
// count as recorded. Every function here takes data and patches DOM.

import { RECORDING, UPLOADING, blinkGlyph, elapsedLabel, isBusy } from "./state.js";

export function elements() {
  return {
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

/** Two classes because the two layouts disagree about the resting state: on a
 * phone the drawer is closed until `is-open`, on a desktop it is docked open
 * until `is-collapsed`. The controller holds one boolean either way. */
export function setSidebarOpen(dom, open) {
  dom.sidebar.classList.toggle("is-open", open);
  dom.sidebar.classList.toggle("is-collapsed", !open);
  dom.menuToggle.setAttribute("aria-expanded", open ? "true" : "false");
}
