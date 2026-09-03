"""Exports the merged model into the formats other applications consume.

Every format derives from ./merged-whisper-model, so supporting a new target
means adding one exporter here - never retraining.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import whisper_pipeline as wp

GGML_BINARY_NAME = "ggml-custom-whisper-small.bin"
CT2_DIR_NAME = "ct2"


def main():
    args = parse_arguments()
    wp.verify_converter_inputs(wp.MERGED_MODEL_DIR)
    wp.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    exporters = {"ct2": export_ctranslate2, "ggml": export_ggml}
    selected = exporters if args.format == "all" else {args.format: exporters[args.format]}

    for name, export in selected.items():
        print(f"\n=== Exporting {name} ===")
        print(f"  -> {export()}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["ct2", "ggml", "all"], default="all")
    return parser.parse_args()


def export_ctranslate2():
    """Produce the CTranslate2 model used by faster-whisper and WhisperX."""
    destination = wp.EXPORTS_DIR / CT2_DIR_NAME
    if destination.exists():
        shutil.rmtree(destination)

    run([
        converter_command(),
        "--model", str(wp.MERGED_MODEL_DIR),
        "--output_dir", str(destination),
        "--quantization", "float16",
        # CTranslate2 builds its own vocabulary; these are copied only so the
        # exported directory carries a complete tokenizer for its consumers.
        "--copy_files", "tokenizer.json", "tokenizer_config.json",
    ])
    return destination


def export_ggml():
    """Produce the ggml binary used by whisper.cpp and its many front-ends."""
    converter = wp.WHISPER_CPP_REPO / "models" / "convert-h5-to-ggml.py"
    if not converter.exists():
        raise wp.PipelineError(f"{converter} not found. Run setup.sh first.")

    # The converter reads mel filters from the openai/whisper checkout and always
    # writes ggml-model.bin into the output directory.
    run([
        sys.executable, str(converter),
        f"{wp.MERGED_MODEL_DIR}/", str(wp.WHISPER_REPO), str(wp.EXPORTS_DIR),
    ])

    produced = wp.EXPORTS_DIR / "ggml-model.bin"
    if not produced.exists():
        raise wp.PipelineError(
            f"{converter.name} exited successfully but wrote no {produced.name}."
        )

    destination = wp.EXPORTS_DIR / GGML_BINARY_NAME
    produced.replace(destination)
    return destination


def converter_command():
    """Locate ct2-transformers-converter, preferring this interpreter's venv.

    It is a console script installed beside the running python, so resolving it
    that way means export.py works when invoked as venv/bin/python without the
    venv activated. Bare-name lookup fails there with a raw FileNotFoundError.
    """
    name = "ct2-transformers-converter"
    local = Path(sys.executable).parent / name
    if local.exists():
        return str(local)

    found = shutil.which(name)
    if found is None:
        raise wp.PipelineError(
            f"{name} not found. Run setup.sh, or activate the venv."
        )
    return found


def run(command):
    result = subprocess.run(command, cwd=str(wp.PROJECT_ROOT))
    if result.returncode != 0:
        raise wp.PipelineError(f"Command failed: {' '.join(command)}")


if __name__ == "__main__":
    try:
        main()
    except wp.PipelineError as error:
        sys.exit(f"error: {error}")
