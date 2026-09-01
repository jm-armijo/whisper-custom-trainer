"""The web recorder's request handling, exercised without a socket.

The routing table, the audio decoding and the save/delete rules are plain
functions taking plain arguments, so they are asserted here directly. Only the
integration tier binds a port; anything provable without one is proved here.
"""

import csv
import subprocess

import numpy as np
import pytest
import soundfile as sf

import recorder_server as srv
import recorder_state as rs
import whisper_pipeline as wp

LINE = "this is a deliberately long sentence with plenty of words in it number {n}."


def script_of(sentences):
    return "\n\n".join(LINE.format(n=n) for n in range(sentences))


@pytest.fixture
def paths(tmp_path):
    """The three directories the server is configured with."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "es.txt").write_text(script_of(4), encoding="utf8")
    (scripts / "en.txt").write_text(script_of(3), encoding="utf8")

    audio = tmp_path / "data"
    audio.mkdir()
    return srv.Config(
        scripts_dir=scripts,
        csv_path=tmp_path / "dataset.csv",
        audio_dir=audio,
        static_dir=tmp_path / "static",
    )


def wav_bytes(seconds=1.0, sample_rate=16000, channels=1):
    """A real wav blob, as a browser recording WAV would send."""
    import io

    samples = np.sin(
        2 * np.pi * 440 * np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    ).astype("float32")
    if channels == 2:
        samples = np.stack([samples, samples], axis=1)

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class TestDecodeAudio:
    def test_decodes_a_wav_to_mono_float32(self):
        samples = srv.decode_audio(wav_bytes(seconds=0.5))
        assert samples.dtype == np.float32 and samples.ndim == 1

    def test_resamples_to_the_pipeline_rate(self):
        """The dataset is 16 kHz; a browser may hand over 48 kHz."""
        samples = srv.decode_audio(wav_bytes(seconds=1.0, sample_rate=48000))
        assert abs(len(samples) - wp.SAMPLE_RATE) < wp.SAMPLE_RATE * 0.02

    def test_downmixes_stereo(self):
        samples = srv.decode_audio(wav_bytes(seconds=0.5, channels=2))
        assert samples.ndim == 1

    def test_rejects_bytes_that_are_not_audio(self):
        with pytest.raises(wp.PipelineError, match="decode"):
            srv.decode_audio(b"not audio at all")

    def test_rejects_an_empty_body(self):
        with pytest.raises(wp.PipelineError, match="empty"):
            srv.decode_audio(b"")


class TestClipDuration:
    def test_reports_seconds_from_the_sample_count(self):
        assert srv.clip_seconds(np.zeros(wp.SAMPLE_RATE, dtype="float32")) == 1.0


class TestSaveChunk:
    """The web path must produce the same dataset rows as the terminal one."""

    def test_writes_the_wav_where_recorder_state_expects_it(self, paths):
        srv.save_chunk(paths, "es.txt", 1, wav_bytes(seconds=1.0))
        assert rs.clip_path(paths.audio_dir, "es", 1).exists()

    def test_writes_pcm_16_at_the_pipeline_rate(self, paths):
        """Byte-identical to record_data.write_clip, or train.py sees two formats."""
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))

        info = sf.info(str(rs.clip_path(paths.audio_dir, "es", 0)))
        assert (info.samplerate, info.channels, info.subtype) == (
            wp.SAMPLE_RATE, 1, "PCM_16",
        )

    def test_upserts_the_row_with_the_chunk_text(self, paths):
        srv.save_chunk(paths, "es.txt", 2, wav_bytes(seconds=1.0))

        chunks = wp.chunk_text((paths.scripts_dir / "es.txt").read_text(encoding="utf8"))
        rows = list(csv.DictReader(paths.csv_path.open(newline="", encoding="utf8")))
        assert rows[0]["text"] == chunks[2]

    def test_labels_the_row_with_the_inferred_language(self, paths):
        srv.save_chunk(paths, "en.txt", 0, wav_bytes(seconds=1.0))

        rows = list(csv.DictReader(paths.csv_path.open(newline="", encoding="utf8")))
        assert rows[0]["language"] == "en"

    def test_re_recording_replaces_rather_than_appends(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.5))

        rows = list(csv.DictReader(paths.csv_path.open(newline="", encoding="utf8")))
        assert len(rows) == 1

    def test_reports_the_saved_duration(self, paths):
        result = srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        assert result["seconds"] == pytest.approx(1.0, abs=0.05)

    def test_reports_the_index_as_recorded(self, paths):
        result = srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        assert result["recorded"] is True

    def test_rejects_a_clip_below_the_minimum(self, paths):
        """is_unusable in the terminal recorder; the dataset must not gain junk
        rows just because the take arrived over HTTP."""
        too_short = wp.MIN_CLIP_SECONDS / 2
        with pytest.raises(wp.PipelineError, match="too short"):
            srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=too_short))

    def test_a_rejected_clip_writes_no_wav(self, paths):
        with pytest.raises(wp.PipelineError):
            srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2))
        assert not rs.clip_path(paths.audio_dir, "es", 0).exists()

    def test_a_rejected_clip_leaves_an_earlier_take_intact(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        with pytest.raises(wp.PipelineError):
            srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2))

        assert rs.clip_path(paths.audio_dir, "es", 0).exists()

    def test_rejects_an_index_outside_the_script(self, paths):
        with pytest.raises(wp.PipelineError, match="index"):
            srv.save_chunk(paths, "es.txt", 999, wav_bytes(seconds=1.0))

    def test_rejects_a_negative_index(self, paths):
        with pytest.raises(wp.PipelineError, match="index"):
            srv.save_chunk(paths, "es.txt", -1, wav_bytes(seconds=1.0))

    def test_rejects_a_traversing_script_name(self, paths):
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            srv.save_chunk(paths, "../es.txt", 0, wav_bytes(seconds=1.0))

    def test_creates_the_audio_directory_if_it_is_absent(self, paths, tmp_path):
        config = paths._replace(audio_dir=tmp_path / "fresh")
        srv.save_chunk(config, "es.txt", 0, wav_bytes(seconds=1.0))
        assert rs.clip_path(config.audio_dir, "es", 0).exists()


class TestDeleteChunk:
    def test_removes_the_wav_so_the_line_reopens(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        srv.delete_chunk(paths, "es.txt", 0)

        assert rs.recorded_indices(paths.csv_path, paths.audio_dir, "es") == set()

    def test_drops_the_dataset_row_too(self, paths):
        """A row pointing at a missing file is what train.py cannot load."""
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        srv.delete_chunk(paths, "es.txt", 0)

        rows = list(csv.DictReader(paths.csv_path.open(newline="", encoding="utf8")))
        assert rows == []

    def test_keeps_other_takes(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        srv.save_chunk(paths, "es.txt", 1, wav_bytes(seconds=1.0))
        srv.delete_chunk(paths, "es.txt", 0)

        assert rs.recorded_indices(paths.csv_path, paths.audio_dir, "es") == {1}

    def test_deleting_an_unrecorded_line_is_not_an_error(self, paths):
        """The browser may retry a delete; a second one must not 500."""
        assert srv.delete_chunk(paths, "es.txt", 0)["recorded"] is False

    def test_rejects_a_traversing_script_name(self, paths):
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            srv.delete_chunk(paths, "../../es.txt", 0)


class TestErrorClassification:
    """Which failures are 'gone' and which are 'you asked wrong'.

    Keyed on the exception type rather than the message, so rewording an error
    cannot quietly turn a 404 into a 400.
    """

    def test_an_absent_script_is_a_not_found(self, paths):
        with pytest.raises(srv.NotFound):
            srv.script_payload(paths, "absent.txt")

    def test_an_absent_take_is_a_not_found(self, paths):
        with pytest.raises(srv.NotFound):
            srv.clip_bytes(paths, "es.txt", 0)

    def test_a_traversing_name_is_not_a_not_found(self, paths):
        """A rejected name must not read as 'no such file' - that would tell a
        prober which paths exist."""
        with pytest.raises(wp.PipelineError) as raised:
            srv.script_payload(paths, "../secrets.txt")
        assert not isinstance(raised.value, srv.NotFound)

    def test_a_short_clip_is_not_a_not_found(self, paths):
        with pytest.raises(wp.PipelineError) as raised:
            srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=wp.MIN_CLIP_SECONDS / 2))
        assert not isinstance(raised.value, srv.NotFound)


class TestClipBytes:
    def test_returns_the_stored_wav(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        assert srv.clip_bytes(paths, "es.txt", 0).startswith(b"RIFF")

    def test_rejects_a_line_with_no_take(self, paths):
        with pytest.raises(wp.PipelineError, match="No recording"):
            srv.clip_bytes(paths, "es.txt", 0)

    def test_rejects_a_traversing_script_name(self, paths):
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            srv.clip_bytes(paths, "../es.txt", 0)


class TestScriptsPayload:
    def test_lists_every_script_with_its_progress(self, paths):
        payload = srv.scripts_payload(paths)
        assert [row["name"] for row in payload["scripts"]] == ["en.txt", "es.txt"]

    def test_carries_counts_the_picker_can_render(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))

        rows = {row["name"]: row for row in srv.scripts_payload(paths)["scripts"]}
        assert (rows["es.txt"]["recorded_count"], rows["es.txt"]["total"]) == (1, 4)

    def test_omits_the_chunk_text_from_the_list(self, paths):
        """The picker only needs counts; sending every script's prose would
        make the list page grow with the corpus."""
        row = srv.scripts_payload(paths)["scripts"][0]
        assert "chunks" not in row

    def test_serialises_recorded_as_a_sorted_list(self, paths):
        """A set is not JSON; the order must be stable for the client."""
        srv.save_chunk(paths, "es.txt", 2, wav_bytes(seconds=1.0))
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))

        rows = {row["name"]: row for row in srv.scripts_payload(paths)["scripts"]}
        assert rows["es.txt"]["recorded"] == [0, 2]

    def test_an_empty_scripts_directory_lists_nothing(self, paths, tmp_path):
        empty = tmp_path / "none"
        empty.mkdir()
        assert srv.scripts_payload(paths._replace(scripts_dir=empty)) == {"scripts": []}

    def test_a_script_with_no_inferable_language_is_still_listed(self, paths):
        """The picker must show it, greyed out or otherwise, rather than hide it."""
        (paths.scripts_dir / "misc.txt").write_text(script_of(2), encoding="utf8")

        rows = {row["name"]: row for row in srv.scripts_payload(paths)["scripts"]}
        assert rows["misc.txt"]["language"] is None


class TestScriptPayload:
    def test_carries_the_chunks_with_their_status(self, paths):
        srv.save_chunk(paths, "es.txt", 1, wav_bytes(seconds=1.0))

        payload = srv.script_payload(paths, "es.txt")
        assert [row["recorded"] for row in payload["chunks"]] == [
            False, True, False, False,
        ]

    def test_each_chunk_carries_its_text_and_index(self, paths):
        payload = srv.script_payload(paths, "es.txt")
        assert payload["chunks"][0]["index"] == 0 and payload["chunks"][0]["text"]

    def test_carries_the_language_the_takes_will_be_labelled_with(self, paths):
        assert srv.script_payload(paths, "es.txt")["language"] == "es"

    def test_carries_the_next_line_to_read(self, paths):
        srv.save_chunk(paths, "es.txt", 0, wav_bytes(seconds=1.0))
        assert srv.script_payload(paths, "es.txt")["next_index"] == 1

    def test_rejects_a_traversing_name(self, paths):
        with pytest.raises(wp.PipelineError, match="Invalid script"):
            srv.script_payload(paths, "../../etc/passwd")

    def test_rejects_an_unknown_script(self, paths):
        with pytest.raises(wp.PipelineError, match="not found"):
            srv.script_payload(paths, "absent.txt")


class TestRouteParsing:
    """URL shape, decided without a socket."""

    def test_matches_the_scripts_collection(self):
        assert srv.parse_path("/api/scripts") == ("scripts", None, None)

    def test_ignores_a_trailing_slash(self):
        assert srv.parse_path("/api/scripts/") == ("scripts", None, None)

    def test_matches_one_script(self):
        assert srv.parse_path("/api/scripts/es.txt") == ("script", "es.txt", None)

    def test_matches_a_chunk(self):
        assert srv.parse_path("/api/scripts/es.txt/chunks/3") == ("chunk", "es.txt", 3)

    def test_matches_a_chunks_audio(self):
        assert srv.parse_path("/api/scripts/es.txt/chunks/3/audio") == (
            "audio", "es.txt", 3,
        )

    def test_percent_decodes_the_script_name(self):
        """The client encodes the name, so a space must survive the round trip."""
        assert srv.parse_path("/api/scripts/my%20script.txt")[1] == "my script.txt"

    def test_a_traversing_name_survives_decoding_for_the_domain_to_reject(self):
        """Rejecting here as well as in the domain would hide which layer guards."""
        assert srv.parse_path("/api/scripts/..%2Fsecrets.txt")[1] == "../secrets.txt"

    def test_a_non_numeric_index_does_not_match(self):
        assert srv.parse_path("/api/scripts/es.txt/chunks/abc")[0] is None

    def test_an_unknown_api_path_does_not_match(self):
        assert srv.parse_path("/api/nothing")[0] is None

    def test_a_non_api_path_does_not_match(self):
        assert srv.parse_path("/index.html")[0] is None


class TestBodyExtraction:
    """The browser posts multipart/form-data; a raw blob must work too."""

    def test_reads_a_raw_body_unchanged(self):
        assert srv.audio_from_body(b"RIFFdata", "audio/wav") == b"RIFFdata"

    def test_extracts_the_audio_part_of_a_multipart_body(self):
        body, content_type = multipart(b"RIFFdata")
        assert srv.audio_from_body(body, content_type) == b"RIFFdata"

    def test_keeps_binary_bytes_intact_through_multipart(self):
        """A text-mode parse would mangle the 0x0d 0x0a runs inside audio."""
        payload = bytes(range(256)) * 4
        body, content_type = multipart(payload)
        assert srv.audio_from_body(body, content_type) == payload

    def test_rejects_a_multipart_body_without_an_audio_part(self):
        body, content_type = multipart(b"x", field="notaudio")
        with pytest.raises(wp.PipelineError, match="audio"):
            srv.audio_from_body(body, content_type)


def multipart(payload, field="audio", boundary="----testboundary"):
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="c.webm"\r\n'.encode(),
        b"Content-Type: audio/webm\r\n\r\n",
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


class TestWebmDecoding:
    """The browser records WebM/Opus, which libsndfile cannot open.

    librosa 1.x dropped the audioread fallback, so soundfile raising
    'Format not recognised' is the whole story without an ffmpeg hop.
    """

    @pytest.fixture
    def webm(self, tmp_path):
        if not srv.ffmpeg_command():
            pytest.skip("ffmpeg is not installed")

        destination = tmp_path / "take.webm"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:a", "libopus", "-f", "webm", str(destination), "-y"],
            check=True,
        )
        return destination.read_bytes()

    def test_soundfile_alone_cannot_open_it(self, webm):
        """Guards the reason the ffmpeg hop exists: if libsndfile ever gains
        WebM support this test fails and the fallback can be reconsidered."""
        import io

        with pytest.raises(sf.LibsndfileError):
            sf.read(io.BytesIO(webm))

    def test_decode_audio_handles_it_anyway(self, webm):
        samples = srv.decode_audio(webm)
        assert len(samples) == pytest.approx(wp.SAMPLE_RATE, rel=0.05)

    def test_a_webm_take_reaches_the_dataset(self, paths, webm):
        srv.save_chunk(paths, "es.txt", 0, webm)
        assert rs.recorded_indices(paths.csv_path, paths.audio_dir, "es") == {0}
