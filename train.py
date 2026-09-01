"""Trains a single bilingual LoRA adapter on the recorded dataset."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import whisper_pipeline as wp


def main():
    args = parse_arguments()
    processor = wp.build_processor()

    dataset = load_examples(args.csv, processor)
    model = build_lora_model()
    trainer = build_trainer(model, dataset, processor, args)

    trainer.train()

    model.save_pretrained(str(wp.ADAPTER_DIR))
    processor.save_pretrained(str(wp.ADAPTER_DIR))
    print(f"\nAdapter saved to {wp.ADAPTER_DIR}\nNext: python merge.py")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(wp.DATASET_CSV))
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def load_examples(csv_path, processor):
    """Read the CSV and encode each row into features and labels."""
    from datasets import load_dataset

    if not Path(csv_path).exists():
        raise wp.PipelineError(f"{csv_path} not found. Run record_data.py first.")

    # The audio column stays a plain path: datasets 5.x needs torchcodec to
    # decode an Audio column, so wp.load_audio handles decoding instead.
    dataset = load_dataset("csv", data_files=str(csv_path), split="train")
    return dataset.map(
        lambda row: encode_example(row, processor),
        remove_columns=dataset.column_names,
        desc="Encoding audio",
    )


def encode_example(row, processor):
    """Encode one row, tagging labels with that row's own language token."""
    samples = wp.load_audio(row["audio_path"])
    features = processor.feature_extractor(samples, sampling_rate=wp.SAMPLE_RATE)

    # Set per row so a single adapter learns both languages with the correct
    # <|en|> / <|es|> prefix rather than collapsing them into one.
    processor.tokenizer.set_prefix_tokens(language=row["language"], task="transcribe")

    return {
        "input_features": features.input_features[0],
        "labels": processor.tokenizer(row["text"]).input_ids,
    }


def build_lora_model():
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(wp.BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    ))
    model.print_trainable_parameters()
    return model


def build_trainer(model, dataset, processor, args):
    from transformers import Seq2SeqTrainer

    return Seq2SeqTrainer(
        model=model,
        args=training_arguments(args),
        train_dataset=dataset,
        data_collator=SpeechCollator(processor),
    )


def training_arguments(args):
    from transformers import Seq2SeqTrainingArguments

    return Seq2SeqTrainingArguments(
        output_dir=str(wp.PROJECT_ROOT / "training-output"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        # MPS half precision is unreliable for training; fp32 is the safe default.
        fp16=False,
        bf16=False,
        # PEFT wraps the model, so Trainer cannot infer these two on its own.
        remove_unused_columns=False,
        label_names=["labels"],
        # Forked dataloader workers are flaky alongside the MPS backend.
        dataloader_num_workers=0,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )


@dataclass
class SpeechCollator:
    """Pads audio features and text labels independently into one batch."""

    processor: object

    def __call__(self, examples):

        batch = self.processor.feature_extractor.pad(
            [{"input_features": item["input_features"]} for item in examples],
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in examples],
            return_tensors="pt",
        )

        # -100 keeps padding out of the loss.
        masked = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)

        # Trainer re-adds the decoder start token, so drop a duplicate leading one.
        # Tested per row rather than across the batch: an all-or-nothing check would
        # either keep a duplicate on every row or strip a real token from rows that
        # never carried the prefix.
        start_token = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        if (masked[:, 0] == start_token).all():
            masked = masked[:, 1:]
        elif (masked[:, 0] == start_token).any():
            raise wp.PipelineError(
                "Batch mixes rows with and without a leading start token; "
                "label encoding is inconsistent."
            )

        batch["labels"] = masked
        return batch


if __name__ == "__main__":
    try:
        main()
    except wp.PipelineError as error:
        sys.exit(f"error: {error}")
