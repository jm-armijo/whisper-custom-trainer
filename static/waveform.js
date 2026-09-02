// Domain layer: the arithmetic that turns samples into a drawable shape.
//
// Pure data in, pure data out - no canvas, no AudioContext, no DOM. The view
// decides pixels and colours from what this returns, so every rule here is
// reasonable about headlessly, the way state.js is.

// Roughly one peak per pixel-ish column on a phone-width canvas. More columns
// than the canvas has pixels only costs work nobody can see.
export const PEAK_COUNT = 200;

/** Peaks from a decoded clip's channel data, as a bar height per column.
 *
 * Each column reports the largest absolute sample in its slice rather than the
 * mean: an average over a few hundred samples of speech tends to zero and
 * paints a flat line where there is clearly a word.
 */
export function peaksFromSamples(samples, columns = PEAK_COUNT) {
  const total = samples ? samples.length : 0;
  if (total === 0 || columns <= 0) {
    return [];
  }
  const peaks = new Array(columns);
  // Float division of the boundaries, so a clip shorter than `columns` still
  // spreads across the full width instead of bunching into the first few.
  for (let column = 0; column < columns; column += 1) {
    const start = Math.floor((column * total) / columns);
    const end = Math.max(Math.floor(((column + 1) * total) / columns), start + 1);
    peaks[column] = maxAbs(samples, start, Math.min(end, total));
  }
  return peaks;
}

function maxAbs(samples, start, end) {
  let peak = 0;
  for (let index = start; index < end; index += 1) {
    const value = Math.abs(samples[index]);
    if (value > peak) {
      peak = value;
    }
  }
  return peak;
}

/** A live analyser frame as a signed -1..1 trace.
 *
 * getByteTimeDomainData reports unsigned bytes centred on 128; the view wants
 * an offset from the centre line, and converting here keeps that byte encoding
 * out of the drawing code.
 */
export function traceFromTimeDomain(bytes, columns = PEAK_COUNT) {
  const total = bytes ? bytes.length : 0;
  if (total === 0 || columns <= 0) {
    return [];
  }
  const trace = new Array(columns);
  for (let column = 0; column < columns; column += 1) {
    // Nearest sample rather than an average: this is a scope trace of one
    // frame, and averaging adjacent samples flattens the waveform's own shape.
    const index = Math.min(Math.floor((column * total) / columns), total - 1);
    trace[column] = (bytes[index] - 128) / 128;
  }
  return trace;
}

/** Loudness of one analyser frame, 0..1, for the level meter beside the trace. */
export function levelFromTimeDomain(bytes) {
  const total = bytes ? bytes.length : 0;
  if (total === 0) {
    return 0;
  }
  let sum = 0;
  for (let index = 0; index < total; index += 1) {
    const offset = (bytes[index] - 128) / 128;
    sum += offset * offset;
  }
  // RMS, not peak: a single clipped sample should not park the meter at full.
  return Math.min(Math.sqrt(sum / total), 1);
}

/** Where the playhead sits, 0..1. Clamped because `currentTime` can exceed
 * `duration` by a frame at the end of a clip, and duration is NaN until the
 * element has metadata. */
export function playheadFraction(currentTime, duration) {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(currentTime)) {
    return 0;
  }
  return Math.min(Math.max(currentTime / duration, 0), 1);
}
