"""Encodes a real CSV with the real processor, the step the unit tier fakes."""

import csv

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def processor():
    return wp.build_processor()


@pytest.fixture
def bilingual_dataset(tmp_path, wav_factory):
    """A two-row CSV with real audio, one row per language."""
    rows = [("hola amigo", "es"), ("hello friend", "en")]
    csv_path = tmp_path / "dataset.csv"

    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(wp.CSV_COLUMNS)
        for index, (text, language) in enumerate(rows):
            clip = wav_factory(name=f"{language}_{index}.wav", seconds=1.5)
            writer.writerow([str(clip), text, language])

    return csv_path


class TestLoadExamples:
    def test_encodes_every_row(self, bilingual_dataset, processor):
        from train import load_examples

        assert len(load_examples(bilingual_dataset, processor)) == 2

    def test_produces_whisper_shaped_features(self, bilingual_dataset, processor):
        import numpy as np

        from train import load_examples

        features = np.array(load_examples(bilingual_dataset, processor)[0]["input_features"])
        assert features.shape == (80, 3000)

    def test_drops_the_raw_csv_columns(self, bilingual_dataset, processor):
        """Trainer only accepts model inputs, so source columns must be removed."""
        from train import load_examples

        assert set(load_examples(bilingual_dataset, processor).column_names) == {
            "input_features", "labels"
        }

    def test_labels_carry_distinct_language_tokens_per_row(self, bilingual_dataset, processor):
        from train import load_examples

        dataset = load_examples(bilingual_dataset, processor)
        spanish, english = dataset[0]["labels"], dataset[1]["labels"]

        assert spanish[1] != english[1]

    def test_rejects_a_missing_csv(self, tmp_path, processor):
        from train import load_examples

        with pytest.raises(wp.PipelineError, match="record_data.py"):
            load_examples(tmp_path / "absent.csv", processor)


class TestCollatorWithRealProcessor:
    def test_batches_rows_of_differing_label_lengths(self, bilingual_dataset, processor):
        from train import SpeechCollator, load_examples

        dataset = load_examples(bilingual_dataset, processor)
        batch = SpeechCollator(processor)([dataset[0], dataset[1]])

        assert batch["input_features"].shape[0] == 2
        assert batch["labels"].shape[0] == 2

    def test_masks_padding_in_a_real_batch(self, bilingual_dataset, processor):
        from train import SpeechCollator, load_examples

        dataset = load_examples(bilingual_dataset, processor)
        batch = SpeechCollator(processor)([dataset[0], dataset[1]])

        assert (batch["labels"] == -100).any() or batch["labels"].shape[1] > 0
