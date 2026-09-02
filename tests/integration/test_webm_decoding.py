"""Decoding a real browser take: the one path that needs ffmpeg on PATH.

The rest of the server's decode logic is unit-tested against in-memory bytes.
These cases shell out to ffmpeg to produce genuine WebM/Opus, which is what
the phone actually uploads, so they belong to the integration tier.
"""

import io
import subprocess

import pytest
import soundfile as sf

import recorder_server as srv
import recorder_state as rs
import whisper_pipeline as wp


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

        with pytest.raises(sf.LibsndfileError):
            sf.read(io.BytesIO(webm))

    def test_decode_audio_handles_it_anyway(self, webm):
        samples = srv.decode_audio(webm)
        assert len(samples) == pytest.approx(wp.SAMPLE_RATE, rel=0.05)

    def test_a_webm_take_reaches_the_dataset(self, paths, webm):
        srv.save_chunk(paths, "es.txt", 0, webm)
        assert rs.recorded_indices(paths.csv_path, paths.audio_dir, "es") == {0}
