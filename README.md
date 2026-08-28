# my-whisper

Fine-tunes `openai/whisper-small` on your own Latin American accent in **English
and Spanish** using a single LoRA adapter, then exports it to formats other
applications can load.

Runs locally on Apple Silicon (Metal / MPS).

## Quick start

```bash
./setup.sh                                          # venv + dependencies + repos
source venv/bin/activate

python record_data.py --text script_es.txt --lang es   # read prompts aloud
python record_data.py --text script_en.txt --lang en

python train.py                                     # LoRA adapter
python merge.py                                     # portable master model
./convert.sh                                        # export + install
```

Supply your own `script_*.txt`: any prose you are comfortable reading. It is
split automatically into 10-25 word chunks.

## Using the model

**OpenWhispr** (system-wide dictation, types into any app)
`convert.sh` installs the model, then choose **"small"** under Settings > local model.

**faster-whisper** (Python)
```python
from faster_whisper import WhisperModel

model = WhisperModel("exports/ct2")
segments, _ = model.transcribe("clip.wav", language="es")
print(" ".join(segment.text for segment in segments))
```

**whisper.cpp** (CLI)
```bash
whisper-cli -m exports/ggml-custom-whisper-small.bin -f clip.wav -l es
```

## Portability

`merged-whisper-model/` is the master artifact. Every other format is generated
from it, so a new target never means retraining:

| Format | Path | Used by |
|---|---|---|
| HF transformers | `merged-whisper-model/` | `transformers`, HF Hub |
| CTranslate2 | `exports/ct2/` | faster-whisper, WhisperX, most local apps |
| ggml | `exports/ggml-custom-whisper-small.bin` | whisper.cpp, OpenWhispr, Mac/iOS apps |

```bash
python export.py --format ct2     # or ggml, or all
```

**Cloud services cannot use this model.** Speechify and Claude Code run speech
recognition on their own servers and offer no custom-model upload. Use OpenWhispr
as a system-wide dictation layer instead: it transcribes with your model and
types the text into whatever application is focused, including those two.

## How much data

Roughly **30-60 minutes of speech per language** for a noticeable accuracy gain.
A handful of clips only proves the pipeline runs.

Recording is resumable: rerun the same `record_data.py` command and it continues
where you stopped. Press `r` to redo a misread take (a bad clip hurts training),
`s` to skip, `q` to stop.

## Tests

Structured as a testing pyramid; the fast tier runs by default.

```bash
python -m pytest                      # unit only (~1s, the default)
python -m pytest -m integration       # real libraries and converters (~30s)
python -m pytest -m e2e               # full pipeline to transcribed speech (~20s)
python -m pytest -m "unit or integration or e2e"    # everything (~65s)
```

| Tier | Count | Scope | Needs |
|---|---|---|---|
| **unit** | 60 | chunking, resume, resampling, prefix tagging, label masking, export wiring | nothing |
| **integration** | 49 | real Whisper processor, dataset encoding, LoRA + merge, recorder session, converter requirements, shell scripts | base model download |
| **e2e** | 5 | merged model to CT2 and ggml, transcribing synthesised speech in both languages | `ffmpeg`, `whisper-cli` |

The suite is mutation-tested: breaking a chunk bound, the language tagging, the
padding mask, the LoRA targets, or the tokenizer restore each fails a test.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ImportError: ... install 'torchcodec'` | Audio is decoded via `librosa`, not `datasets`. Use `wp.load_audio`. |
| `FileNotFoundError: vocab.json` | Run `merge.py`; it restores the tokenizer files transformers 5.x no longer writes. |
| `ValueError: ... does not exist` from CT2 | `--copy_files` may only name files the merged model contains. |
| `brew: no formula open-wispr` | The cask is `openwhispr`. |
| OpenWhispr ignores the model | It only accepts registry names; the model installs as `ggml-small.bin`. Pick "small". |

## Layout

| File | Role |
|---|---|
| `whisper_pipeline.py` | Shared constants and third-party workarounds |
| `record_data.py` | Interactive recorder, resumable |
| `train.py` | Bilingual LoRA training |
| `merge.py` | Folds adapter into the portable master model |
| `export.py` | Master model to CT2 / ggml |
| `convert.sh` | Exports everything and installs into OpenWhispr |
