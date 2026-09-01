// Domain layer: chunk bookkeeping and derived view state.
//
// The browser mirror of recorder_state.py. Pure data in, pure data out - no
// DOM, no fetch, no colours. Everything here must be reasonable about without
// a screen, which is what lets the view be replaced without touching rules.

export const RECORDED = "recorded";
export const SELECTED = "selected";
export const RECORDED_SELECTED = "recorded_selected";
export const PENDING = "pending";

export const IDLE = "idle";
export const RECORDING = "recording";

// A take shorter than this carries no speech; whisper_pipeline.MIN_CLIP_SECONDS
// rejects it server-side, so the UI refuses to upload it in the first place.
export const MIN_CLIP_SECONDS = 0.4;
// Whisper's encoder window. Longer takes still save, but are worth redoing.
export const MAX_CLIP_SECONDS = 29.0;

/** Per-line status, mirroring recorder_state.chunk_statuses.
 *
 * A finished line under the cursor is its own status rather than plain
 * SELECTED: collapsing the two leaves a line marked "read this next" about
 * work already done.
 */
export function chunkStatuses(total, recorded, cursor) {
  const statuses = [];
  for (let index = 0; index < total; index += 1) {
    statuses.push(status(recorded.has(index), index === cursor));
  }
  return statuses;
}

function status(isRecorded, isCursor) {
  if (isRecorded) {
    return isCursor ? RECORDED_SELECTED : RECORDED;
  }
  return isCursor ? SELECTED : PENDING;
}

/** The lowest gap, so a skipped chunk is revisited rather than lost. */
export function firstUnrecorded(total, recorded) {
  for (let index = 0; index < total; index += 1) {
    if (!recorded.has(index)) {
      return index;
    }
  }
  return Math.max(total - 1, 0);
}

/** Clamp at both ends so the cursor never leaves the script. */
export function moveCursor(cursor, action, total) {
  if (total <= 0) {
    return 0;
  }
  if (action === "up") {
    return Math.max(cursor - 1, 0);
  }
  if (action === "down") {
    return Math.min(cursor + 1, total - 1);
  }
  if (action === "top") {
    return 0;
  }
  return total - 1;
}

/** mm:ss for the recording timer, matching recorder_ui.elapsed_label. */
export function elapsedLabel(seconds) {
  const whole = Math.max(Math.floor(seconds), 0);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Filled on even ticks so a fresh recording starts visibly lit. */
export function blinkGlyph(tick) {
  return tick % 2 === 0 ? "●" : "○";
}

/** Why a clip cannot be kept, or null when it is usable. */
export function rejectClip(seconds) {
  if (seconds < MIN_CLIP_SECONDS) {
    return `discarded: ${seconds.toFixed(2)}s is too short to use`;
  }
  return null;
}

export function savedMessage(seconds) {
  if (seconds > MAX_CLIP_SECONDS) {
    return `saved ${seconds.toFixed(1)}s - exceeds Whisper's 30s window, consider redo`;
  }
  return `saved ${seconds.toFixed(1)}s`;
}

/** Session state for one open script. Holds no DOM and performs no I/O. */
export class ScriptSession {
  constructor({ name, language, chunks, recorded }) {
    this.name = name;
    this.language = language;
    this.chunks = chunks;
    this.recorded = new Set(recorded);
    this.cursor = firstUnrecorded(chunks.length, this.recorded);
  }

  markRecorded(index) {
    this.recorded.add(index);
  }

  clearRecorded(index) {
    this.recorded.delete(index);
  }

  isRecorded(index) {
    return this.recorded.has(index);
  }

  move(action) {
    this.cursor = moveCursor(this.cursor, action, this.chunks.length);
  }

  select(index) {
    if (index >= 0 && index < this.chunks.length) {
      this.cursor = index;
    }
  }
}

/** The whole screen as plain data, mirroring record_data.build_view.
 *
 * The view decides pixels from this; the title text is content rather than
 * presentation, so it is composed here where it can be asserted headlessly.
 */
export function buildView({ session, scripts, state, tick = 0, elapsed = 0, message = "" }) {
  if (!session) {
    return {
      title: " no script selected ",
      chunks: [],
      statuses: [],
      recorded: new Set(),
      cursor: 0,
      state,
      tick,
      elapsed,
      message,
      scripts: scriptEntries(scripts, null),
      legend: legendFor(state),
    };
  }

  return {
    title:
      ` ${session.name} · ${session.language} · ` +
      `${session.recorded.size}/${session.chunks.length} recorded `,
    chunks: session.chunks,
    statuses: chunkStatuses(session.chunks.length, session.recorded, session.cursor),
    recorded: session.recorded,
    cursor: session.cursor,
    state,
    tick,
    elapsed,
    message,
    scripts: scriptEntries(scripts, session.name),
    legend: legendFor(state),
  };
}

const LEGEND_IDLE = "record · redo · play · prev · next";
const LEGEND_RECORDING = "stop to save the take";

function legendFor(state) {
  return state === RECORDING ? LEGEND_RECORDING : LEGEND_IDLE;
}

/** One menu row per script: how far along it is, and whether it is finished. */
export function scriptEntries(scripts, activeName) {
  return (scripts || []).map((script) => ({
    name: script.name,
    language: script.language,
    recorded: script.recorded,
    total: script.total,
    complete: script.total > 0 && script.recorded >= script.total,
    active: script.name === activeName,
    progress: `${script.recorded}/${script.total}`,
  }));
}
