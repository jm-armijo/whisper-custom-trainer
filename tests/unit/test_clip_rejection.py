"""Regression test: a clip too short to hold speech never reaches the dataset."""

import numpy as np

import record_data
import whisper_pipeline as wp


def clip_of(seconds):
    return np.zeros(int(wp.SAMPLE_RATE * seconds), dtype="float32")


class TestIsUnusable:
    """Guards the shortest clip the recorder will keep."""

    def test_rejects_an_empty_take(self):
        assert record_data.is_unusable(clip_of(0))

    def test_rejects_a_clip_under_the_minimum(self):
        assert record_data.is_unusable(clip_of(wp.MIN_CLIP_SECONDS / 2))

    def test_keeps_a_clip_of_normal_length(self):
        assert not record_data.is_unusable(clip_of(2.0))
