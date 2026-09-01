// The only module that talks to the server.
//
// Every URL and every JSON field name the server contract uses lives here, so
// reconciling with recorder_server.py is a one-file edit rather than a hunt
// through the view. Nothing else in the frontend may call fetch.

export const ENDPOINTS = {
  scripts: "/api/scripts",
  script: (name) => `/api/scripts/${encodeURIComponent(name)}`,
  chunk: (name, index) => `/api/scripts/${encodeURIComponent(name)}/chunks/${index}`,
  chunkAudio: (name, index) =>
    `/api/scripts/${encodeURIComponent(name)}/chunks/${index}/audio`,
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
  chunkRecorded: "recorded",
  chunkStatus: "status",
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
  const rows = Array.isArray(payload) ? payload : payload[FIELDS.scripts] || [];
  return rows.map(readScriptSummary);
}

function readScriptSummary(row) {
  return {
    name: row[FIELDS.name],
    language: row[FIELDS.language] || "",
    recorded: countRecorded(row),
    total: Number(row[FIELDS.total] || 0),
  };
}

// recorded_count when the server sends script_progress verbatim; otherwise a
// plain "recorded" number, or the length of an index list.
function countRecorded(row) {
  if (row[FIELDS.recordedCount] !== undefined) {
    return Number(row[FIELDS.recordedCount]);
  }
  const value = row[FIELDS.recorded];
  return Array.isArray(value) ? value.length : Number(value || 0);
}

/** One script's chunks plus the indices already recorded. */
export async function loadScript(name) {
  const response = await request(ENDPOINTS.script(name));
  const payload = await response.json();
  const rows = payload[FIELDS.chunks] || [];

  return {
    name: payload[FIELDS.name] || name,
    language: payload[FIELDS.language] || "",
    chunks: rows.map(readChunkText),
    recorded: recordedIndices(payload, rows),
  };
}

// A chunk may arrive as a bare string or as a chunk_view object.
function readChunkText(row) {
  return typeof row === "string" ? row : row[FIELDS.chunkText] || "";
}

/** Which indices already have audio: a top-level list if the server sends one,
 * otherwise derived from the per-chunk flag chunk_view carries. */
function recordedIndices(payload, rows) {
  const listed = payload[FIELDS.recorded];
  if (Array.isArray(listed)) {
    return listed.map(Number);
  }
  // chunk_view carries its own index; trusting it rather than array position
  // keeps the mapping right if rows ever arrive out of order.
  return rows
    .map((row, position) => (isRecorded(row) ? chunkIndex(row, position) : -1))
    .filter((index) => index >= 0);
}

function chunkIndex(row, position) {
  return typeof row === "object" && row.index !== undefined ? Number(row.index) : position;
}

function isRecorded(row) {
  if (typeof row === "string") {
    return false;
  }
  if (row[FIELDS.chunkRecorded] === true) {
    return true;
  }
  // A four-status vocabulary is accepted too, though the cursor-dependent half
  // of a status is the client's business, not the server's.
  const status = row[FIELDS.chunkStatus];
  return status === "recorded" || status === "recorded_selected";
}

/** Upload a take. The blob's own MIME type is preserved for the server. */
export async function saveChunk(name, index, blob) {
  const body = new FormData();
  body.append("audio", blob, `chunk-${index}.webm`);
  await request(ENDPOINTS.chunk(name, index), { method: "POST", body });
}

export async function deleteChunk(name, index) {
  await request(ENDPOINTS.chunk(name, index), { method: "DELETE" });
}

/** URL for playback. Cache-busted so a re-record is not played from cache. */
export function audioUrl(name, index, version = 0) {
  return `${ENDPOINTS.chunkAudio(name, index)}?v=${version}`;
}
