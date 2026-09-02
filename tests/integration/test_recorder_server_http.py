"""The recorder API over a real socket.

The unit tier proves the rules; this tier proves they survive the wire - status
codes, JSON bodies, multipart uploads and the static route the browser assets
are served from. A real ThreadingHTTPServer on an ephemeral port is cheap
enough to be worth more than a mocked handler.
"""

import csv
import io
import json
import re
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
    # The directory names the language, so nothing is inferred from a filename.
    (scripts / "es").mkdir(parents=True)
    (scripts / "en").mkdir(parents=True)
    (scripts / "es" / "a.txt").write_text(script_of(4), encoding="utf8")
    (scripts / "en" / "a.txt").write_text(script_of(3), encoding="utf8")

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


@pytest.fixture
def real_assets_server(tmp_path):
    """A live server serving the assets that actually ship in static/.

    The fixture that writes its own index.html proves only that the fixture is
    self-consistent; serving the real page is what catches an asset URL the
    page requests and the server does not answer.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()

    config = srv.Config(
        scripts_dir=scripts,
        csv_path=tmp_path / "dataset.csv",
        audio_dir=tmp_path / "data",
        static_dir=srv.STATIC_DIR,
    )
    httpd = srv.build_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
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
        assert [row["name"] for row in payload["scripts"]] == ["en/a.txt", "es/a.txt"]

    def test_reports_progress_per_script(self, server):
        rows = call(f"{server.base}/api/scripts")[1]["scripts"]
        assert {row["name"]: row["recorded_count"] for row in rows} == {
            "en/a.txt": 0, "es/a.txt": 0,
        }

    def test_responds_as_json(self, server):
        with urllib.request.urlopen(f"{server.base}/api/scripts", timeout=10) as response:
            assert "application/json" in response.headers["Content-Type"]


class TestOneScript:
    def test_returns_the_chunks(self, server):
        status, payload = call(f"{server.base}/api/scripts/es/a.txt")
        assert status == 200 and len(payload["chunks"]) == 4

    def test_each_chunk_carries_text_index_and_state(self, server):
        chunk = call(f"{server.base}/api/scripts/es/a.txt")[1]["chunks"][0]
        assert set(chunk) == {"index", "text", "recorded"}

    def test_an_unknown_script_is_a_404(self, server):
        status, payload = call(f"{server.base}/api/scripts/es/absent.txt")
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
            f"{server.base}/api/scripts/es/a.txt/chunks/0",
            method="POST", body=wav_bytes(), content_type="audio/wav",
        )
        assert status == 200 and payload["recorded"] is True

    def test_the_wav_lands_where_the_terminal_recorder_puts_it(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/2",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        assert rs.clip_path(server.config.audio_dir, "es", 2).exists()

    def test_the_row_carries_the_chunk_text(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/1",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        chunks = call(f"{server.base}/api/scripts/es/a.txt")[1]["chunks"]
        rows = list(csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        ))
        assert rows[0]["text"] == chunks[1]["text"]

    def test_a_multipart_upload_works(self, server):
        """This is the shape the browser's FormData actually sends."""
        body, content_type = multipart(wav_bytes())
        status, payload = call(
            f"{server.base}/api/scripts/es/a.txt/chunks/0",
            method="POST", body=body, content_type=content_type,
        )
        assert status == 200 and payload["recorded"] is True

    def test_the_script_then_reports_the_line_recorded(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        chunks = call(f"{server.base}/api/scripts/es/a.txt")[1]["chunks"]
        assert [chunk["recorded"] for chunk in chunks] == [True, False, False, False]

    def test_a_short_clip_is_refused_with_a_reason(self, server):
        status, payload = call(
            f"{server.base}/api/scripts/es/a.txt/chunks/0",
            method="POST", body=wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2),
            content_type="audio/wav",
        )
        assert status == 400 and "too short" in payload["message"]

    def test_a_refused_clip_leaves_no_dataset_row(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2),
             content_type="audio/wav")

        assert not server.config.csv_path.exists()

    def test_undecodable_bytes_are_refused_rather_than_crashing(self, server):
        status, payload = call(
            f"{server.base}/api/scripts/es/a.txt/chunks/0",
            method="POST", body=b"this is not audio", content_type="audio/wav",
        )
        assert status == 400 and "message" in payload

    def test_an_index_past_the_end_is_refused(self, server):
        status, _ = call(
            f"{server.base}/api/scripts/es/a.txt/chunks/99",
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
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        status, payload = call(f"{server.base}/api/scripts/es/a.txt/chunks/0/audio")
        assert status == 200 and payload.startswith(b"RIFF")

    def test_the_served_clip_is_decodable_at_the_pipeline_rate(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        payload = call(f"{server.base}/api/scripts/es/a.txt/chunks/0/audio")[1]
        info = sf.info(io.BytesIO(payload))
        assert (info.samplerate, info.channels, info.subtype) == (
            wp.SAMPLE_RATE, 1, "PCM_16",
        )

    def test_playback_is_not_cached(self, server):
        """A re-record must not play the previous take from the phone's cache."""
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        url = f"{server.base}/api/scripts/es/a.txt/chunks/0/audio"
        with urllib.request.urlopen(url, timeout=10) as response:
            assert response.headers["Cache-Control"] == "no-store"

    def test_an_unrecorded_line_is_a_404(self, server):
        status, _ = call(f"{server.base}/api/scripts/es/a.txt/chunks/0/audio")
        assert status == 404


class TestDeletingATake:
    def test_removes_the_clip(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        status, payload = call(
            f"{server.base}/api/scripts/es/a.txt/chunks/0", method="DELETE"
        )
        assert status == 200 and payload["recorded"] is False

    def test_the_line_reopens_for_recording(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0", method="DELETE")

        chunks = call(f"{server.base}/api/scripts/es/a.txt")[1]["chunks"]
        assert chunks[0]["recorded"] is False

    def test_the_dataset_row_goes_with_it(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0", method="DELETE")

        rows = list(csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        ))
        assert rows == []

    def test_a_repeated_delete_is_not_an_error(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0", method="DELETE")
        status, _ = call(f"{server.base}/api/scripts/es/a.txt/chunks/0", method="DELETE")
        assert status == 200


class TestBothLanguagesShareOneDataset:
    """The whole point of the web recorder: one dataset.csv, as the terminal
    recorder writes it, with the per-row language the bilingual adapter needs."""

    def test_takes_in_both_languages_land_in_one_csv(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/en/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        rows = list(csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        ))
        assert sorted(row["language"] for row in rows) == ["en", "es"]

    def test_the_columns_match_the_pipeline_contract(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")

        reader = csv.DictReader(
            server.config.csv_path.open(newline="", encoding="utf8")
        )
        assert tuple(reader.fieldnames) == wp.CSV_COLUMNS

    def test_the_same_index_in_each_language_is_a_separate_clip(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(), content_type="audio/wav")
        call(f"{server.base}/api/scripts/en/a.txt/chunks/0",
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


# href="..." / src="..." on the served page, ignoring anything off-site.
ASSET_REFERENCE = re.compile(r'(?:href|src)="([^"]+)"')


def page_asset_urls(markup):
    return [
        reference
        for reference in ASSET_REFERENCE.findall(markup)
        if not reference.startswith(("http://", "https://", "//", "data:", "#"))
    ]


class TestTheRealPageLoads:
    """Every URL the shipped index.html requests must actually be served.

    Asserting the spelling of an asset URL inside index.html proves only that
    the file says what it says. The page and the server disagreed about the
    /static/ prefix for exactly that reason: the page 200'd while every script
    and stylesheet on it 404'd, so the recorder was blank in a browser while
    the suite was green.
    """

    def test_the_page_is_served(self, real_assets_server):
        status, payload = call(f"{real_assets_server}/")
        assert status == 200 and b"<title>" in payload

    def test_the_page_references_assets_at_all(self, real_assets_server):
        markup = call(f"{real_assets_server}/")[1].decode("utf8")
        assert page_asset_urls(markup), "index.html should load a stylesheet and a module"

    def test_every_asset_the_page_requests_is_served(self, real_assets_server):
        markup = call(f"{real_assets_server}/")[1].decode("utf8")

        broken = [
            url for url in page_asset_urls(markup)
            if call(f"{real_assets_server}{url}")[0] != 200
        ]
        assert broken == []

    def test_the_static_prefix_cannot_be_walked_out_of(self, real_assets_server):
        """Stripping /static/ must not turn ../ into a way out of the asset dir."""
        status, _ = call(f"{real_assets_server}/static/../recorder_server.py")
        assert status == 404

    def test_every_module_the_entry_imports_is_served(self, real_assets_server):
        """The page names only app.js; its own imports are fetched next."""
        entry = [url for url in page_asset_urls(
            call(f"{real_assets_server}/")[1].decode("utf8")
        ) if url.endswith(".js")]

        broken = []
        for url in entry:
            source = call(f"{real_assets_server}{url}")[1].decode("utf8")
            base = url.rsplit("/", 1)[0]
            for imported in re.findall(r'from "\./([^"]+)"', source):
                if call(f"{real_assets_server}{base}/{imported}")[0] != 200:
                    broken.append(imported)
        assert broken == []


class TestTheAssetsAreNeverCached:
    """A rebuilt page must not be run against the previous build's modules.

    SimpleHTTPRequestHandler sends Last-Modified and nothing else, so a browser
    is free to heuristically reuse a module it fetched before. That shipped a
    fresh index.html against a stale app.js: the new markup had dropped
    #btn-prev, the cached bindControls still asked for it, and the null threw
    out of boot() before the first repaint - a page stuck on the placeholder
    title with no script list and nothing on screen to say why.
    """

    def test_a_module_is_not_cached(self, real_assets_server):
        with urllib.request.urlopen(f"{real_assets_server}/static/app.js") as response:
            assert "no-store" in response.headers["Cache-Control"]

    def test_the_page_itself_is_not_cached(self, real_assets_server):
        with urllib.request.urlopen(f"{real_assets_server}/") as response:
            assert "no-store" in response.headers["Cache-Control"]

    def test_the_stylesheet_is_not_cached(self, real_assets_server):
        with urllib.request.urlopen(f"{real_assets_server}/static/style.css") as response:
            assert "no-store" in response.headers["Cache-Control"]

    def test_a_json_reply_is_untouched(self, real_assets_server):
        """Only the static route sets this; the API must not acquire it by
        sharing one handler across a keep-alive connection."""
        with urllib.request.urlopen(f"{real_assets_server}/api/scripts") as response:
            assert response.headers.get("Cache-Control") is None


# document.getElementById("...") in a module, and id="..." in the markup.
ELEMENT_LOOKUP = re.compile(r'getElementById\(\s*"([^"]+)"\s*\)')
ELEMENT_ID = re.compile(r'\bid="([^"]+)"')


class TestThePageCarriesEveryElementTheModulesLookUp:
    """The failure that caching merely delayed: markup and modules disagreeing.

    render.elements() is the view's whole DOM contract, and a getElementById
    that misses returns null rather than raising - the crash lands later, at
    the first addEventListener, out of boot(), where nothing reaches the
    screen. Removing a button from index.html and leaving its lookup behind is
    silent until the page is opened in a browser, so it is asserted here.
    """

    def _sources(self, base):
        markup = call(f"{base}/")[1].decode("utf8")
        modules = [
            call(f"{base}/static/{name}")[1].decode("utf8")
            for name in ("app.js", "render.js")
        ]
        return markup, modules

    def test_every_looked_up_id_exists_in_the_page(self, real_assets_server):
        markup, modules = self._sources(real_assets_server)
        present = set(ELEMENT_ID.findall(markup))

        missing = sorted(
            {i for source in modules for i in ELEMENT_LOOKUP.findall(source)} - present
        )
        assert missing == []

    def test_the_modules_look_up_anything_at_all(self, real_assets_server):
        """Guards the regex above: a lookup spelled differently would make the
        assertion pass over an empty set."""
        _, modules = self._sources(real_assets_server)
        assert any(ELEMENT_LOOKUP.findall(source) for source in modules)


class TestAudioPlaysOnAPhone:
    """The clip route as Safari drives it.

    iOS sends `Range: bytes=0-1` before playing an <audio> source and refuses
    to play when the answer is a plain 200 carrying the whole file. That
    failure is silent on the page, so it is asserted here on the wire.
    """

    def _record(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(seconds=1.0), content_type="audio/wav")
        return f"{server.base}/api/scripts/es/a.txt/chunks/0/audio"

    def test_a_range_request_is_answered_partial(self, server):
        url = self._record(server)
        request = urllib.request.Request(url, headers={"Range": "bytes=0-1"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert len(response.read()) == 2

    def test_the_partial_reply_names_the_range_and_total(self, server):
        url = self._record(server)
        request = urllib.request.Request(url, headers={"Range": "bytes=0-1"})
        with urllib.request.urlopen(request) as response:
            total = response.headers["Content-Range"].split("/")[1]
            assert response.headers["Content-Range"].startswith("bytes 0-1/")
            assert int(total) > 2

    def test_byte_ranges_are_advertised(self, server):
        """Without this header Safari does not bother asking."""
        url = self._record(server)
        with urllib.request.urlopen(url) as response:
            assert response.headers["Accept-Ranges"] == "bytes"

    def test_a_plain_request_still_returns_the_whole_clip(self, server):
        url = self._record(server)
        with urllib.request.urlopen(url) as response:
            assert response.status == 200
            assert response.read().startswith(b"RIFF")

    def test_a_tail_range_returns_the_end_of_the_clip(self, server):
        """What a seek to the end of the scrubber asks for."""
        url = self._record(server)
        whole = urllib.request.urlopen(url).read()
        request = urllib.request.Request(url, headers={"Range": "bytes=-100"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.read() == whole[-100:]

    def test_a_mid_range_matches_that_slice_of_the_file(self, server):
        url = self._record(server)
        whole = urllib.request.urlopen(url).read()
        request = urllib.request.Request(url, headers={"Range": "bytes=100-199"})
        with urllib.request.urlopen(request) as response:
            assert response.read() == whole[100:200]


class TestHeadIsAnsweredLikeGet:
    """Media clients probe with HEAD as well as Range to learn a clip's size.

    Inherited from SimpleHTTPRequestHandler, HEAD resolves against static_dir,
    so an API path 404s while GET on that same path serves 200 - and a client
    that probes with HEAD never plays.
    """

    def _record(self, server):
        call(f"{server.base}/api/scripts/es/a.txt/chunks/0",
             method="POST", body=wav_bytes(seconds=1.0), content_type="audio/wav")
        return f"{server.base}/api/scripts/es/a.txt/chunks/0/audio"

    def test_head_on_a_clip_is_not_a_not_found(self, server):
        url = self._record(server)
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert response.status == 200

    def test_head_reports_the_size_without_the_body(self, server):
        url = self._record(server)
        length = len(urllib.request.urlopen(url).read())

        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert int(response.headers["Content-Length"]) == length
            assert response.read() == b""

    def test_head_advertises_byte_ranges(self, server):
        url = self._record(server)
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert response.headers["Accept-Ranges"] == "bytes"

    def test_head_on_the_script_list_matches_get(self, server):
        request = urllib.request.Request(f"{server.base}/api/scripts", method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
            assert response.read() == b""

    def test_head_on_a_static_asset_still_works(self, server, tmp_path):
        """The static route writes its own body, so it takes the other path."""
        static = tmp_path / "static"
        static.mkdir(exist_ok=True)
        (static / "probe.js").write_text("export const x = 1;\n", encoding="utf8")

        request = urllib.request.Request(f"{server.base}/static/probe.js", method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
            assert response.read() == b""
            assert int(response.headers["Content-Length"]) > 0
