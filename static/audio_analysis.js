// Device adapter: Web Audio, and nothing else.
//
// The analyser that watches the live mic and the decoder that turns a saved
// clip into samples. Isolated the way microphone.js isolates MediaRecorder, so
// the controller never holds an AudioContext and the view never sees one.
//
// Every entry point degrades to "no waveform" rather than throwing: a browser
// without Web Audio, or one that refuses to decode a container it cannot play,
// must still record and play takes. A canvas is a nicety; a take is the work.

function audioContextClass() {
  if (typeof window === "undefined") {
    return null;
  }
  return window.AudioContext || window.webkitAudioContext || null;
}

export function isSupported() {
  return audioContextClass() !== null;
}

/** A live view of the mic signal, driven by whoever owns the animation frame.
 *
 * Holds the AudioContext, the source node and the analyser as one unit so
 * `close()` can release all three. A leaked context is not a tidiness point:
 * browsers cap how many a page may hold, and a page that opens one per take
 * stops being able to open any.
 */
export class LiveAnalyser {
  constructor(fftSize = 1024) {
    this.fftSize = fftSize;
    this.context = null;
    this.source = null;
    this.analyser = null;
    this.bytes = null;
  }

  /** Attach to an already-open MediaStream. Returns false when the browser has
   * no Web Audio, which is the caller's cue to record without a waveform.
   *
   * The stream is the one microphone.js already opened: a second getUserMedia
   * would raise the phone's permission sheet in the middle of a session.
   */
  async start(stream) {
    const Context = audioContextClass();
    if (!Context || !stream) {
      return false;
    }
    try {
      this.context = new Context();
      // iOS starts every context suspended and only honours resume() inside the
      // gesture that created it - this runs in the record tap's call stack, so
      // the promise settles rather than hanging until the next tap.
      if (this.context.state === "suspended") {
        await this.context.resume();
      }
      this.source = this.context.createMediaStreamSource(stream);
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = this.fftSize;
      // The analyser is a leaf: it is never connected to destination, so the
      // mic is not echoed back out of the phone's speaker into itself.
      this.source.connect(this.analyser);
      this.bytes = new Uint8Array(this.analyser.fftSize);
      return true;
    } catch {
      this.close();
      return false;
    }
  }

  get active() {
    return this.analyser !== null;
  }

  /** The current frame's raw bytes, or null when there is nothing attached. */
  sample() {
    if (!this.analyser || !this.bytes) {
      return null;
    }
    this.analyser.getByteTimeDomainData(this.bytes);
    return this.bytes;
  }

  /** Release the graph and the context. Safe to call twice: stopRecording
   * closes in a finally that can run after a failed start already did. */
  close() {
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // A node the context already tore down; nothing left to release.
      }
    }
    if (this.context && typeof this.context.close === "function") {
      try {
        this.context.close();
      } catch {
        // Closing an already-closed context throws in some browsers.
      }
    }
    this.source = null;
    this.analyser = null;
    this.context = null;
    this.bytes = null;
  }
}

/** Decode an ArrayBuffer into one channel of samples, or null.
 *
 * Its own short-lived context, closed before returning: decoding happens once
 * per playback and keeping a context open between takes is what exhausts the
 * browser's limit over a long session.
 */
export async function decodeChannel(buffer) {
  const Context = audioContextClass();
  if (!Context || !buffer || buffer.byteLength === 0) {
    return null;
  }
  let context = null;
  try {
    context = new Context();
    const decoded = await decode(context, buffer);
    return decoded.numberOfChannels > 0 ? decoded.getChannelData(0) : null;
  } catch {
    // An undecodable clip loses its waveform, never its playback: the <audio>
    // element decodes independently and may well manage what this could not.
    return null;
  } finally {
    if (context && typeof context.close === "function") {
      try {
        context.close();
      } catch {
        // Already closed.
      }
    }
  }
}

/** Safari implements only the callback form of decodeAudioData, and returns
 * undefined rather than a promise, so awaiting the call itself hangs forever. */
function decode(context, buffer) {
  return new Promise((resolve, reject) => {
    const pending = context.decodeAudioData(buffer, resolve, reject);
    if (pending && typeof pending.then === "function") {
      pending.then(resolve, reject);
    }
  });
}
