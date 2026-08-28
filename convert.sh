#!/usr/bin/env bash
# Exports every format and installs the ggml build into OpenWhispr.
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source ./venv/bin/activate

python export.py --format all

echo
echo "=== Installing tools ==="
brew install --cask openwhispr || echo "openwhispr already installed"
brew install whisper-cpp || echo "whisper-cpp already installed"

# OpenWhispr resolves models through a fixed registry (tiny/base/small/medium/
# large/turbo) and rejects any other name as a path-traversal guard, so the
# custom build must be installed under an accepted filename.
readonly MODELS_DIR="$HOME/.cache/openwhispr/whisper-models"
readonly INSTALLED="$MODELS_DIR/ggml-small.bin"
readonly CUSTOM_MODEL="./exports/ggml-custom-whisper-small.bin"

mkdir -p "$MODELS_DIR"
# Compare against our own export first: re-running this script (or restoring by
# deleting the backup) must never overwrite the stock backup with a custom model.
if [ -f "$INSTALLED" ] && [ ! -f "$INSTALLED.orig" ] &&
   ! cmp -s "$INSTALLED" "$CUSTOM_MODEL"; then
  echo "Backing up stock model to ggml-small.bin.orig"
  cp "$INSTALLED" "$INSTALLED.orig"
fi
cp "$CUSTOM_MODEL" "$INSTALLED"

cat <<'NEXT'

Installed. To use it:

  OpenWhispr  -> Settings > local model > "small" (now your fine-tuned build)
  whisper-cli -> whisper-cli -m exports/ggml-custom-whisper-small.bin -f clip.wav -l es
  Python      -> WhisperModel("exports/ct2")   # faster-whisper

Restore the stock model with:
  mv ~/.cache/openwhispr/whisper-models/ggml-small.bin.orig \
     ~/.cache/openwhispr/whisper-models/ggml-small.bin

NEXT
