// Device adapter: getUserMedia + MediaRecorder, and nothing else.
//
// Isolated the way whisper_pipeline.py isolates third-party quirks, so the
// browser's codec negotiation stays out of the controller.

// Android Chrome and desktop Chrome record webm/opus; Safari only offers mp4.
// The first supported type wins, and the blob carries its own MIME type to the
// server, which re-encodes to the 16 kHz wav the dataset needs.
const PREFERRED_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];

export function pickMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }
  return PREFERRED_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

export function isSupported() {
  return Boolean(
    navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia &&
      typeof MediaRecorder !== "undefined",
  );
}

export class Microphone {
  constructor() {
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
  }

  /** Open the stream once and keep it: re-prompting on every take makes the
   * phone permission sheet appear mid-session. */
  async open() {
    if (this.stream) {
      return this.stream;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
    });
    return this.stream;
  }

  async start() {
    await this.open();
    const mimeType = pickMimeType();
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        this.chunks.push(event.data);
      }
    });
    this.recorder.start();
  }

  get recording() {
    return Boolean(this.recorder && this.recorder.state === "recording");
  }

  /** Resolve with the finished blob. `stop` is asynchronous in every browser,
   * so the last dataavailable arrives after the call returns. */
  stop() {
    return new Promise((resolve) => {
      if (!this.recorder || this.recorder.state === "inactive") {
        resolve(new Blob(this.chunks, { type: pickMimeType() || "audio/webm" }));
        return;
      }
      this.recorder.addEventListener(
        "stop",
        () => {
          const type = this.recorder.mimeType || pickMimeType() || "audio/webm";
          resolve(new Blob(this.chunks, { type }));
        },
        { once: true },
      );
      this.recorder.stop();
    });
  }

  close() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.recorder = null;
  }
}
