"""Audio must reach the model as 16 kHz mono regardless of the source file."""

import pytest

import whisper_pipeline as wp


class TestLoadAudio:
    def test_returns_a_one_dimensional_array(self, wav_factory):
        assert wp.load_audio(wav_factory()).ndim == 1

    def test_resamples_a_higher_rate_file_to_16khz(self, wav_factory):
        clip = wav_factory(sample_rate=44100, seconds=1.0)
        assert len(wp.load_audio(clip)) == pytest.approx(wp.SAMPLE_RATE, rel=0.01)

    def test_resamples_a_lower_rate_file_to_16khz(self, wav_factory):
        clip = wav_factory(sample_rate=8000, seconds=1.0)
        assert len(wp.load_audio(clip)) == pytest.approx(wp.SAMPLE_RATE, rel=0.01)

    def test_downmixes_stereo_to_mono(self, wav_factory):
        assert wp.load_audio(wav_factory(channels=2)).ndim == 1

    def test_preserves_duration_when_resampling(self, wav_factory):
        clip = wav_factory(sample_rate=22050, seconds=2.0)
        seconds = len(wp.load_audio(clip)) / wp.SAMPLE_RATE
        assert seconds == pytest.approx(2.0, abs=0.05)

    def test_raises_for_a_missing_file(self, tmp_path):
        # soundfile raises LibsndfileError, a RuntimeError subclass; asserting the
        # public base keeps the test off a library-private type.
        with pytest.raises(RuntimeError):
            wp.load_audio(tmp_path / "missing.wav")


class TestClipLengthWarnings:
    """Bad takes poison training, so the recorder must flag them.

    Under curses there is no stdout to print to; the warning reaches the user
    as the status-bar message returned when the clip is saved.
    """

    def build_clip(self, seconds):
        import numpy as np

        return np.zeros(int(wp.SAMPLE_RATE * seconds), dtype="float32")

    def test_warns_when_clip_exceeds_the_whisper_window(self):
        from record_data import saved_message

        assert "exceeds" in saved_message(self.build_clip(31.0))

    def test_stays_quiet_for_a_normal_clip(self):
        from record_data import saved_message

        assert "exceeds" not in saved_message(self.build_clip(4.0))

    def test_reports_the_duration_that_was_saved(self):
        from record_data import saved_message

        assert "4.0s" in saved_message(self.build_clip(4.0))

    def test_a_clip_too_short_is_rejected_before_saving(self):
        """The short-clip case is a rejection, not a warning."""
        from record_data import is_unusable

        assert is_unusable(self.build_clip(0.1))
