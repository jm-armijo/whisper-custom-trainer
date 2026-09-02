// The only module that talks to the server.
//
// Every URL and every JSON field name the server contract uses lives here, so
// reconciling with recorder_server.py is a one-file edit rather than a hunt
// through the view. Nothing else in the frontend may call fetch.

// A script name is qualified by its language directory ("es/general.txt"), so
// each segment is escaped on its own: encodeURIComponent over the whole name
// turns the separator into %2F, which the server's routes - matched before
// unquote - do not accept. Escaping per segment keeps every other character
// protected while leaving the one structural slash literal.
const scriptPath = (name) => name.split("/").map(encodeURIComponent).join("/");

export const ENDPOINTS = {
  scripts: "/api/scripts",
  script: (name) => `/api/scripts/${scriptPath(name)}`,
  chunk: (name, index) => `/api/scripts/${scriptPath(name)}/chunks/${index}`,
  chunkAudio: (name, index) =>
    `/api/scripts/${scriptPath(name)}/chunks/${index}/audio`,
};

// The server's field names, matching recorder_scripts.script_progress and
// chunk_view. Changing how the server names something is a one-table edit here;
// nothing downstream reads a raw payload.
export const FIELDS = {
  scripts: "scripts",
  name: "name",
  language: "language",
  total: "total",
  // script_progress reports the count separately from the index set, because
  // "recorded" there is a set of indices rather than a number.
  recordedCount: "recorded_count",
  recorded: "recorded",
  chunks: "chunks",
  chunkText: "text",
  index: "index",
  // The decoded length of the take the server actually stored, and its verdict
  // on whether that exceeds Whisper's encoder window.
  seconds: "seconds",
  tooLong: "too_long",
  message: "message",
};

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await errorText(response));
  }
  return response;
}

async function errorText(response) {
  try {
    const payload = await response.json();
    return payload[FIELDS.message] || payload.detail || `request failed (${response.status})`;
  } catch {
    return `request failed (${response.status})`;
  }
}

/** Every script with its progress: [{name, language, recorded, total}]. */
export async function listScripts() {
  const response = await request(ENDPOINTS.scripts);
  const payload = await response.json();
  return (payload[FIELDS.scripts] || []).map(readScriptSummary);
}

function readScriptSummary(row) {
  return {
    name: row[FIELDS.name],
    // A script whose language cannot be inferred is listed with a null
    // language so the picker can show why it is unrecordable.
    language: row[FIELDS.language] || "",
    recorded: Number(row[FIELDS.recordedCount]),
    total: Number(row[FIELDS.total]),
  };
}

/** One script's chunks plus the indices already recorded. */
export async function loadScript(name) {
  const response = await request(ENDPOINTS.script(name));
  const payload = await response.json();

  return {
    name: payload[FIELDS.name],
    language: payload[FIELDS.language],
    chunks: (payload[FIELDS.chunks] || []).map((row) => row[FIELDS.chunkText]),
    // script_payload sends the index set at the top level, already sorted.
    recorded: (payload[FIELDS.recorded] || []).map(Number),
  };
}

/** Upload a take, returning what the server measured: {seconds, tooLong}.
 *
 * The duration comes back from the server rather than being timed here. The
 * browser can only measure wall-clock elapsed, which includes MediaRecorder's
 * startup and flush latency; the server decodes the clip and knows exactly how
 * many samples landed in the dataset. Warning about Whisper's window from the
 * browser's number describes a clip that was never written.
 */
export async function saveChunk(name, index, blob) {
  const body = new FormData();
  body.append("audio", blob, `chunk-${index}.webm`);
  const response = await request(ENDPOINTS.chunk(name, index), { method: "POST", body });
  return readSavedChunk(await response.json());
}

function readSavedChunk(payload) {
  return {
    index: Number(payload[FIELDS.index]),
    seconds: Number(payload[FIELDS.seconds]),
    tooLong: payload[FIELDS.tooLong] === true,
  };
}

export async function deleteChunk(name, index) {
  await request(ENDPOINTS.chunk(name, index), { method: "DELETE" });
}

/** URL for playback. Cache-busted so a re-record is not played from cache. */
export function audioUrl(name, index, version = 0) {
  return `${ENDPOINTS.chunkAudio(name, index)}?v=${version}`;
}

/** The stored clip's bytes, for drawing its waveform.
 *
 * The waveform is computed from the wav on every playback and never stored:
 * dataset.csv holds the three columns train.py reads and nothing else, and a
 * sidecar of peaks would be one more thing to keep in step with a re-record.
 * The same URL the <audio> element plays, so both sides see the same take.
 */
export async function fetchAudio(name, index, version = 0) {
  const response = await request(audioUrl(name, index, version));
  return response.arrayBuffer();
}
