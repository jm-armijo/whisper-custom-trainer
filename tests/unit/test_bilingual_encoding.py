"""One adapter serves both languages only if each row carries its own prefix."""

import pytest

import whisper_pipeline as wp


class FakeFeatureExtractor:
    def __call__(self, samples, sampling_rate):
        assert sampling_rate == wp.SAMPLE_RATE
        return type("Features", (), {"input_features": [[0.0, 1.0]]})()


class FakeTokenizer:
    """Records prefix changes so tests can assert per-row language tagging."""

    def __init__(self):
        self.prefix_calls = []
        self.language = None

    def set_prefix_tokens(self, language, task):
        self.prefix_calls.append((language, task))
        self.language = language

    def __call__(self, text):
        token = f"<|{self.language}|>"
        return type("Encoded", (), {"input_ids": [token, text]})()


class FakeProcessor:
    def __init__(self):
        self.feature_extractor = FakeFeatureExtractor()
        self.tokenizer = FakeTokenizer()


@pytest.fixture
def processor():
    return FakeProcessor()


@pytest.fixture
def encode(monkeypatch):
    monkeypatch.setattr(wp, "load_audio", lambda path: [0.0] * wp.SAMPLE_RATE)
    from train import encode_example

    return encode_example


class TestEncodeExample:
    def test_tags_spanish_rows_with_the_spanish_prefix(self, encode, processor):
        row = {"audio_path": "a.wav", "text": "hola", "language": "es"}
        assert encode(row, processor)["labels"][0] == "<|es|>"

    def test_tags_english_rows_with_the_english_prefix(self, encode, processor):
        row = {"audio_path": "a.wav", "text": "hello", "language": "en"}
        assert encode(row, processor)["labels"][0] == "<|en|>"

    def test_switches_prefix_between_consecutive_rows(self, encode, processor):
        encode({"audio_path": "a.wav", "text": "hola", "language": "es"}, processor)
        encode({"audio_path": "b.wav", "text": "hello", "language": "en"}, processor)

        assert processor.tokenizer.prefix_calls == [("es", "transcribe"), ("en", "transcribe")]

    def test_always_requests_the_transcribe_task(self, encode, processor):
        """Translate would train the model to output the wrong language."""
        encode({"audio_path": "a.wav", "text": "hola", "language": "es"}, processor)
        assert processor.tokenizer.prefix_calls[0][1] == "transcribe"

    def test_produces_input_features(self, encode, processor):
        row = {"audio_path": "a.wav", "text": "hola", "language": "es"}
        assert encode(row, processor)["input_features"] == [0.0, 1.0]

    def test_decodes_the_row_audio_path(self, monkeypatch, processor):
        seen = {}
        monkeypatch.setattr(wp, "load_audio", lambda path: seen.setdefault("path", path) or [0.0])
        from train import encode_example

        encode_example({"audio_path": "clip.wav", "text": "hi", "language": "en"}, processor)
        assert seen["path"] == "clip.wav"
