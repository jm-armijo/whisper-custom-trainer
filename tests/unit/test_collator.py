"""Padding must not leak into the loss, and the BOS token must not double up."""

import pytest

torch = pytest.importorskip("torch")

import whisper_pipeline as wp

START_TOKEN_ID = 50258


class FakeFeatureExtractor:
    def pad(self, items, return_tensors):
        return {"input_features": torch.tensor([item["input_features"] for item in items])}


class FakeTokenizer:
    def __init__(self, pad_to=3):
        self.pad_to = pad_to

    def pad(self, items, return_tensors):
        width = max(len(item["input_ids"]) for item in items)
        padded, mask = [], []
        for item in items:
            ids = item["input_ids"]
            filler = width - len(ids)
            padded.append(ids + [0] * filler)
            mask.append([1] * len(ids) + [0] * filler)
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}

    def convert_tokens_to_ids(self, token):
        return START_TOKEN_ID


class FakeProcessor:
    def __init__(self):
        self.feature_extractor = FakeFeatureExtractor()
        self.tokenizer = FakeTokenizer()


@pytest.fixture
def collate():
    from train import SpeechCollator

    return SpeechCollator(FakeProcessor())


class TestSpeechCollator:
    def test_masks_padding_with_minus_one_hundred(self, collate):
        batch = collate([
            {"input_features": [0.0], "labels": [1, 2, 3]},
            {"input_features": [0.0], "labels": [4]},
        ])
        assert (batch["labels"][1][1:] == -100).all()

    def test_keeps_real_tokens_unmasked(self, collate):
        batch = collate([{"input_features": [0.0], "labels": [7, 8]}])
        assert batch["labels"].tolist() == [[7, 8]]

    def test_strips_a_duplicate_leading_start_token(self, collate):
        """Trainer re-adds the decoder start token, so keeping ours would double it."""
        batch = collate([
            {"input_features": [0.0], "labels": [START_TOKEN_ID, 5, 6]},
            {"input_features": [0.0], "labels": [START_TOKEN_ID, 7, 8]},
        ])
        assert batch["labels"].tolist() == [[5, 6], [7, 8]]

    def test_keeps_labels_intact_when_no_start_token_is_present(self, collate):
        batch = collate([{"input_features": [0.0], "labels": [5, 6]}])
        assert batch["labels"].tolist() == [[5, 6]]

    def test_returns_batched_input_features(self, collate):
        batch = collate([
            {"input_features": [0.1], "labels": [1]},
            {"input_features": [0.2], "labels": [2]},
        ])
        assert batch["input_features"].shape[0] == 2
