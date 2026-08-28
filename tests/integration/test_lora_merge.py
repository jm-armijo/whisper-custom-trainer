"""Verifies LoRA wrapping and merging against the real model and PEFT."""

import pytest

import whisper_pipeline as wp

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def lora_model():
    from train import build_lora_model

    return build_lora_model()


class TestBuildLoraModel:
    def test_trains_far_fewer_parameters_than_the_full_model(self, lora_model):
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        assert trainable / total < 0.02

    def test_adapts_the_targeted_projections(self, lora_model):
        adapted = {
            name.split(".lora_A")[0].rsplit(".", 1)[-1]
            for name, _ in lora_model.named_parameters()
            if ".lora_A" in name
        }
        assert adapted == {"q_proj", "v_proj"}

    def test_clears_forced_decoder_ids(self, lora_model):
        """Left set, they would pin generation to one language."""
        assert lora_model.base_model.model.config.forced_decoder_ids is None

    def test_disables_cache_for_training(self, lora_model):
        assert lora_model.base_model.model.config.use_cache is False


class TestMergeRoundTrip:
    """Merging must fold the adapter in without altering the architecture."""

    @pytest.fixture(scope="class")
    @classmethod
    def merged_dir(cls, tmp_path_factory):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import WhisperForConditionalGeneration

        from merge import merge_adapter

        adapter_dir = tmp_path_factory.mktemp("adapter")
        base = WhisperForConditionalGeneration.from_pretrained(wp.BASE_MODEL, dtype="float32")
        peft_model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none",
        ))

        # lora_B initialises to zero, so both factors must be perturbed for the
        # B@A product - and therefore the merge - to change any base weight.
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.add_(0.01)

        peft_model.save_pretrained(str(adapter_dir))

        merged_out = tmp_path_factory.mktemp("merged")
        import whisper_pipeline
        original = whisper_pipeline.ADAPTER_DIR
        whisper_pipeline.ADAPTER_DIR = adapter_dir
        try:
            merged = merge_adapter()
            merged.save_pretrained(str(merged_out))
            wp.build_processor().save_pretrained(str(merged_out))
        finally:
            whisper_pipeline.ADAPTER_DIR = original
        return merged_out

    def test_merged_model_has_no_adapter_layers_left(self, merged_dir):
        from transformers import WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(merged_dir)
        assert not any("lora" in name for name, _ in model.named_parameters())

    def test_merged_weights_differ_from_the_base(self, merged_dir):
        import torch
        from transformers import WhisperForConditionalGeneration

        base = WhisperForConditionalGeneration.from_pretrained(wp.BASE_MODEL, dtype="float32")
        merged = WhisperForConditionalGeneration.from_pretrained(merged_dir)

        base_q = base.model.encoder.layers[0].self_attn.q_proj.weight
        merged_q = merged.model.encoder.layers[0].self_attn.q_proj.weight
        assert not torch.allclose(base_q, merged_q)

    def test_merged_model_still_generates(self, merged_dir, wav_factory):
        import torch
        from transformers import WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(merged_dir)
        processor = wp.build_processor()
        samples = wp.load_audio(wav_factory(seconds=1.0))
        features = processor.feature_extractor(
            samples, sampling_rate=wp.SAMPLE_RATE, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            tokens = model.generate(features, max_new_tokens=8, language="es", task="transcribe")
        assert tokens.shape[0] == 1

    def test_restores_the_files_downstream_converters_need(self, merged_dir):
        wp.restore_legacy_tokenizer_files(merged_dir)
        wp.verify_converter_inputs(merged_dir)
