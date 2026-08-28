"""Folds the LoRA adapter into the base model to produce the portable master.

The merged directory is the artifact every other format converts from, so it is
made self-contained here rather than in each exporter.
"""

import sys

import whisper_pipeline as wp


def main():
    if not wp.ADAPTER_DIR.is_dir():
        raise wp.PipelineError(f"{wp.ADAPTER_DIR} not found. Run train.py first.")

    model = merge_adapter()
    model.save_pretrained(str(wp.MERGED_MODEL_DIR))
    wp.build_processor().save_pretrained(str(wp.MERGED_MODEL_DIR))

    wp.restore_legacy_tokenizer_files(wp.MERGED_MODEL_DIR)

    print(f"\nMerged model saved to {wp.MERGED_MODEL_DIR}")
    print("Next: python export.py --format all")


def merge_adapter():
    """Merge on CPU in fp32; MPS offers no benefit here and risks dtype drift."""
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    base = WhisperForConditionalGeneration.from_pretrained(
        wp.BASE_MODEL, dtype="float32"
    )
    adapted = PeftModel.from_pretrained(base, str(wp.ADAPTER_DIR))
    return adapted.merge_and_unload()


if __name__ == "__main__":
    try:
        main()
    except wp.PipelineError as error:
        sys.exit(f"error: {error}")
