"""The recorder API over a real socket.

The unit tier proves the rules; this tier proves they survive the wire - status
codes, JSON bodies, multipart uploads and the static route the browser assets
are served from. A real ThreadingHTTPServer on an ephemeral port is cheap
enough to be worth more than a mocked handler.
"""

import csv
import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

import recorder_server as srv
import recorder_state as rs
import whisper_pipeline as wp

LINE = "this is a deliberately long sentence with plenty of words in it number {n}."


def script_of(sentences):
    return "\n\n".join(LINE.format(n=n) for n in range(sentences))


def wav_bytes(seconds=1.0, sample_rate=16000):
    samples = np.sin(
        2 * np.pi * 440 * np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    ).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def server(tmp_path):
    """A live server on an ephemeral port, torn down with the test."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "es.txt").write_text(script_of(4), encoding="utf8")
    (scripts / "en.txt").write_text(script_of(3), encoding="utf8")

    audio = tmp_path / "data"
    audio.mkdir()
    static = tmp_path / "static"

    config = srv.Config(
        scripts_dir=scripts,
        csv_path=tmp_path / "dataset.csv",
        audio_dir=audio,
        static_dir=static,
    )
    # Port 0 lets the OS pick a free port, so parallel test runs cannot collide.
    httpd = srv.build_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    class Live:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def __init__(self):
            self.config = config
            self.static = static

    try:
        yield Live()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def call(url, method="GET", body=None, content_type=None):
    """(status, parsed body) for a request, treating a 4xx as a normal result."""
    request = urllib.request.Request(url, data=body, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, _read(response)
    except urllib.error.HTTPError as error:
        return error.code, _read(error)


def _read(response):
    payload = response.read()
    if "json" in (response.headers.get("Content-Type") or ""):
        return json.loads(payload)
    return payload


def multipart(payload, field="audio"):
    boundary = "----recorderboundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="c.wav"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


class TestListScripts:
    def test_returns_every_script(self, server):
        status, payload = call(f"{server.base}/api/scripts")
        assert status == 200
        assert [row["name"] for row in payload["scripts"]] == ["en.txt", "es.txt"]

    def test_reports_progress_per_script(self, server):
        rows = call(f"{server.base}/api/scripts")[1]["scripts"]
        assert {row["name"]: row["recorded_count"] for row in rows} == {
            "en.txt": 0, "es.txt": 0,
        }

    def test_responds_as_json(self, server):
        with urllib.request.urlopen(f"{server.base}/api/scripts", timeout=10) as response:
            assert "application/json" in response.headers["Content-Type"]


class TestOneScript:
    def test_returns_the_chunks(self, server):
        status, payload = call(f"{server.base}/api/scripts/es.txt")
        assert status == 200 and len(payload["chunks"]) == 4

    def test_each_chunk_carries_text_index_and_state(self, server):
        chunk = call(f"{server.base}/api/scripts/es.txt")[1]["chunks"][0]
        assert set(chunk) == {"index", "text", "recorded"}

    def test_an_unknown_script_is_a_404(self, server):
        status, payload = call(f"{server.base}/api/scripts/absent.txt")
        assert status == 404 and "message" in payload

    def test_a_traversing_name_is_rejected(self, server):
        """The server binds to the LAN; ..%2F must not reach the filesystem."""
        status, _ = call(f"{server.base}/api/scripts/..%2F..%2Fetc%2Fpasswd")
        assert status == 400

    def test_a_traversal_does_not_leak_file_contents(self, server, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("classified", encoding="utf8")

        _, payload = call(f"{server.base}/api/scripts/..%2Fsecret.txt")
        assert b"classified" not in json.dumps(payload).encode("utf8")


class TestRecordingOverHttp:
    def test_a_posted_take_reaches_the_dataset(self, server):
        status, payload = call(
            f"{server.base}/api/scripts/es.txt/chunks/0",
            method="POST", body=wav_bytes(), content_type="audio/wav",
        )
        assert status == 200 and payload["recorded"] is True

    def test_the_wav_lands_where_the_terminal_recorder_puts_it(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/2",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        assert rs.clip_path(server.config.audio_dir, "es", 2).exists()

    def test_the_row_carries_the_chunk_text(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/1",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        chunks = call(f"{server.base}/api/scripts/es.txt")[1]["chunks"]
        rows = list(csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        ))
        assert rows[0]["text"] == chunks[1]["text"]

    def test_a_multipart_upload_works(self, server):
        """This is the shape the browser's FormData actually sends."""
        body, content_type = multipart(wav_bytes())
        status, payload = call(
            f"{server.base}/api/scripts/es.txt/chunks/0",
            method="POST", body=body, content_type=content_type,
        )
        assert status == 200 and payload["recorded"] is True

    def test_the_script_then_reports_the_line_recorded(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        chunks = call(f"{server.base}/api/scripts/es.txt")[1]["chunks"]
        assert [chunk["recorded"] for chunk in chunks] == [True, False, False, False]

    def test_a_short_clip_is_refused_with_a_reason(self, server):
        status, payload = call(
            f"{server.base}/api/scripts/es.txt/chunks/0",
            method="POST", body=wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2),
            content_type="audio/wav",
        )
        assert status == 400 and "too short" in payload["message"]

    def test_a_refused_clip_leaves_no_dataset_row(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2),
             content_type="audio/wav")

        assert not server.config.csv_path.exists()

    def test_undecodable_bytes_are_refused_rather_than_crashing(self, server):
        status, payload = call(
            f"{server.base}/api/scripts/es.txt/chunks/0",
            method="POST", body=b"this is not audio", content_type="audio/wav",
        )
        assert status == 400 and "message" in payload

    def test_an_index_past_the_end_is_refused(self, server):
        status, _ = call(
            f"{server.base}/api/scripts/es.txt/chunks/99",
            method="POST", body=wav_bytes(), content_type="audio/wav",
        )
        assert status == 400

    def test_posting_to_a_traversing_name_is_refused(self, server):
        status, _ = call(
            f"{server.base}/api/scripts/..%2Fes.txt/chunks/0",
            method="POST", body=wav_bytes(), content_type="audio/wav",
        )
        assert status == 400


class TestPlayback:
    def test_serves_the_stored_take_as_wav(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        status, payload = call(f"{server.base}/api/scripts/es.txt/chunks/0/audio")
        assert status == 200 and payload.startswith(b"RIFF")

    def test_the_served_clip_is_decodable_at_the_pipeline_rate(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        payload = call(f"{server.base}/api/scripts/es.txt/chunks/0/audio")[1]
        info = sf.info(io.BytesIO(payload))
        assert (info.samplerate, info.channels, info.subtype) == (
            wp.SAMPLE_RATE, 1, "PCM_16",
        )

    def test_playback_is_not_cached(self, server):
        """A re-record must not play the previous take from the phone's cache."""
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        url = f"{server.base}/api/scripts/es.txt/chunks/0/audio"
        with urllib.request.urlopen(url, timeout=10) as response:
            assert response.headers["Cache-Control"] == "no-store"

    def test_an_unrecorded_line_is_a_404(self, server):
        status, _ = call(f"{server.base}/api/scripts/es.txt/chunks/0/audio")
        assert status == 404


class TestDeletingATake:
    def test_removes_the_clip(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        status, payload = call(
            f"{server.base}/api/scripts/es.txt/chunks/0", method="DELETE"
        )
        assert status == 200 and payload["recorded"] is False

    def test_the_line_reopens_for_recording(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/es.txt/chunks/0", method="DELETE")

        chunks = call(f"{server.base}/api/scripts/es.txt")[1]["chunks"]
        assert chunks[0]["recorded"] is False

    def test_the_dataset_row_goes_with_it(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/es.txt/chunks/0", method="DELETE")

        rows = list(csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        ))
        assert rows == []

    def test_a_repeated_delete_is_not_an_error(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0", method="DELETE")
        status, _ = call(f"{server.base}/api/scripts/es.txt/chunks/0", method="DELETE")
        assert status == 200


class TestBothLanguagesShareOneDataset:
    """The whole point of the web recorder: one dataset.csv, as the terminal
    recorder writes it, with the per-row language the bilingual adapter needs."""

    def test_takes_in_both_languages_land_in_one_csv(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/en.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        rows = list(csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        ))
        assert sorted(row["language"] for row in rows) == ["en", "es"]

    def test_the_columns_match_the_pipeline_contract(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        reader = csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        )
        assert tuple(reader.fieldnames) == wp.CSV_COLUMNS

    def test_the_same_index_in_each_language_is_a_separate_clip(self, server):
        call(f"{server.base}/api/scripts/es.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/en.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        assert (
            rs.clip_path(server.config.audio_dir, "es", 0).exists()
            and rs.clip_path(server.config.audio_dir, "en", 0).exists()
        )


class TestStaticAssets:
    def test_serves_index_html_at_the_root(self, server):
        server.static.mkdir()
        (server.static / "index.html").write_text("<h1>recorder</h1>", encoding="utf8")

        status, payload = call(f"{server.base}/")
        assert status == 200 and b"recorder" in payload

    def test_serves_a_named_asset(self, server):
        server.static.mkdir()
        (server.static / "app.js").write_text("export const x = 1;", encoding="utf8")

        status, payload = call(f"{server.base}/app.js")
        assert status == 200 and b"export const x" in payload

    def test_an_absent_static_directory_is_a_404_not_a_crash(self, server):
        """The frontend is deployed separately; a missing static/ must not stop
        the API from serving."""
        status, _ = call(f"{server.base}/")
        assert status == 404

    def test_the_api_still_answers_without_static_assets(self, server):
        assert call(f"{server.base}/api/scripts")[0] == 200

    def test_an_unknown_api_endpoint_is_a_404(self, server):
        status, _ = call(f"{server.base}/api/nothing")
        assert status == 404
