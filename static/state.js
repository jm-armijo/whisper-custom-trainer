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
// A take that has stopped but is still in flight to the server. Its own state
// rather than IDLE: on a phone the upload can take seconds, and while the
// controller read IDLE a second tap started a fresh recording on top of the
// one still uploading.
export const UPLOADING = "uploading";

// Playback state is tracked apart from the capture states above, not as a
// fourth value of `state`. The two are orthogonal - isBusy() asks whether the
// microphone is committed, and playing a take back does not commit it - so
// folding PLAYING into that enum would have made every isBusy() caller wrong
// about a line being played back.
export const STOPPED = "stopped";
export const PLAYING = "playing";
export const PAUSED = "paused";

// Deliberately absent: MIN_CLIP_SECONDS and MAX_CLIP_SECONDS.
//
// Both used to be hardcoded here alongside whisper_pipeline's copies, with
// nothing to catch the two drifting apart. Neither number is the browser's to
// know: the server decodes the upload and is the only side that can measure
// what actually landed in the dataset, so it reports `seconds` and `too_long`
// on every save and rejects a short take with a 400. rejectClip and
// savedMessage below take that verdict as an argument rather than deciding it.

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

/** What to say about a saved take, from what the server measured.
 *
 * `seconds` and `tooLong` are the server's, not a stopwatch here: the browser
 * can only time wall-clock elapsed, which includes MediaRecorder's startup and
 * flush latency, so it would report a duration the stored clip does not have.
 */
export function savedMessage({ seconds, tooLong }) {
  const saved = `saved ${seconds.toFixed(1)}s`;
  return tooLong ? `${saved} - exceeds Whisper's 30s window, consider redo` : saved;
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
export function buildView({
  session,
  scripts,
  state,
  playback = STOPPED,
  tick = 0,
  elapsed = 0,
  message = "",
}) {
  if (!session) {
    return {
      title: " no script selected ",
      chunks: [],
      statuses: [],
      recorded: new Set(),
      cursor: 0,
      state,
      playback,
      tick,
      elapsed,
      message,
      scripts: scriptEntries(scripts, null),
      legend: legendFor(state, playback),
    };
  }

  return {
    title:
      ` ${session.name} · ` +
      `${session.recorded.size}/${session.chunks.length} recorded `,
    chunks: session.chunks,
    statuses: chunkStatuses(session.chunks.length, session.recorded, session.cursor),
    recorded: session.recorded,
    cursor: session.cursor,
    state,
    playback,
    tick,
    elapsed,
    message,
    scripts: scriptEntries(scripts, session.name),
    legend: legendFor(state, playback),
  };
}

const LEGENDS = {
  [IDLE]: "record · play · redo · prev · next",
  [RECORDING]: "stop to save the take",
  [UPLOADING]: "saving the take…",
};

// Playback runs alongside IDLE, so what the user is doing right now is the
// clip, not the cursor - the transport legend wins while it is not stopped.
const PLAYBACK_LEGENDS = {
  [PLAYING]: "playing · pause · stop",
  [PAUSED]: "paused · play to resume · stop",
};

function legendFor(state, playback = STOPPED) {
  if (state === IDLE && PLAYBACK_LEGENDS[playback]) {
    return PLAYBACK_LEGENDS[playback];
  }
  return LEGENDS[state] || LEGENDS[IDLE];
}

/** Whether a take is in progress, capturing or uploading.
 *
 * The single rule the controller and the view both ask, so "can this button do
 * anything right now" cannot answer differently in the two places. Folding
 * UPLOADING into this is what stops a second tap from starting a recording
 * over an upload that has not landed yet.
 */
export function isBusy(state) {
  return state === RECORDING || state === UPLOADING;
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
