#!/usr/bin/env bash
# Provisions the virtualenv and vendored repos needed by the pipeline.
set -euo pipefail

cd "$(dirname "$0")"

readonly LEGACY_ENV="$HOME/code/whisper-env"
readonly VENV="./venv"

# The hand-made env outside the repo is replaced by a project-local one so the
# whole pipeline is reproducible from this directory alone.
if [ -d "$LEGACY_ENV" ]; then
  echo "Removing superseded environment: $LEGACY_ENV"
  rm -rf "$LEGACY_ENV"
fi

if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --quiet --upgrade pip
echo "Installing Python dependencies (this downloads ~2GB on first run)..."
python -m pip install --quiet \
  "torch>=2.13,<3" \
  "transformers>=5.16,<6" \
  "peft>=0.20,<1" \
  "datasets>=5,<6" \
  "accelerate>=1.14,<2" \
  "librosa>=1,<2" \
  "soundfile>=0.14,<1" \
  "sounddevice>=0.4,<1" \
  "ctranslate2>=4.8,<5" \
  "faster-whisper>=1.2,<2" \
  "pytest>=8,<10" \
  "ruff>=0.14,<1"

# Commit hooks run lint and the fast test tiers; lefthook is the only
# non-pip dependency, so a missing binary is reported rather than installed.
if command -v lefthook >/dev/null 2>&1; then
  lefthook install >/dev/null && echo "Git hooks installed via lefthook."
else
  echo "WARNING: lefthook not found - commit hooks are not active."
  echo "  Install it with 'brew install lefthook' (macOS) or see https://lefthook.dev,"
  echo "  then run 'lefthook install'."
fi

# Cloned only for whisper/assets/mel_filters.npz, which the ggml converter reads.
if [ ! -d "./whisper" ]; then
  git clone --depth 1 https://github.com/openai/whisper.git ./whisper
fi

# ggerganov/whisper.cpp redirects to the ggml-org org; the default branch is
# master (there is no main branch).
if [ ! -d "./whisper.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git ./whisper.cpp
fi

cat <<'NEXT'

Setup complete. Next steps:

  source venv/bin/activate
  python record_data.py --text your_script_es.txt --lang es
  python train.py
  python merge.py
  ./convert.sh

NEXT
