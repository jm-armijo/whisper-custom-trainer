"""Exercises the real Whisper processor: the unit tier fakes it, so the
assumptions about its API are only proven here."""

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def processor():
    return wp.build_processor()


class TestRealProcessor:
    def test_feature_extractor_produces_the_expected_mel_shape(self, processor, wav_factory):
        samples = wp.load_audio(wav_factory(seconds=2.0))
        features = processor.feature_extractor(samples, sampling_rate=wp.SAMPLE_RATE)
        # Whisper always pads to 30s: 80 mel bins x 3000 frames.
        assert features.input_features[0].shape == (80, 3000)

    def test_set_prefix_tokens_exists_on_this_transformers_version(self, processor):
        """transformers 5.x renamed much of this API; the pipeline depends on it."""
        assert hasattr(processor.tokenizer, "set_prefix_tokens")

    def test_spanish_prefix_produces_the_spanish_language_token(self, processor):
        processor.tokenizer.set_prefix_tokens(language="es", task="transcribe")
        decoded = processor.tokenizer.decode(
            processor.tokenizer("hola").input_ids, skip_special_tokens=False
        )
        assert "<|es|>" in decoded

    def test_english_prefix_produces_the_english_language_token(self, processor):
        processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
        decoded = processor.tokenizer.decode(
            processor.tokenizer("hello").input_ids, skip_special_tokens=False
        )
        assert "<|en|>" in decoded

    def test_prefix_switches_between_languages_on_one_tokenizer(self, processor):
        """A single adapter trains on both languages through the same tokenizer."""
        processor.tokenizer.set_prefix_tokens(language="es", task="transcribe")
        spanish = processor.tokenizer("hola").input_ids
        processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
        english = processor.tokenizer("hello").input_ids

        assert spanish[1] != english[1]

    def test_transcribe_task_token_is_used(self, processor):
        processor.tokenizer.set_prefix_tokens(language="es", task="transcribe")
        decoded = processor.tokenizer.decode(
            processor.tokenizer("hola").input_ids, skip_special_tokens=False
        )
        assert "<|transcribe|>" in decoded

    def test_round_trips_accented_spanish_text(self, processor):
        text = "¿Cómo estás? Añejo señor"
        encoded = processor.tokenizer(text).input_ids
        assert processor.tokenizer.decode(encoded, skip_special_tokens=True).strip() == text
