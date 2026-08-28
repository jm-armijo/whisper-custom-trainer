# whisper-custom-trainer

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

Recording runs in a full-screen view showing the whole script: lines already
recorded are green, the selected line yellow, and the rest light grey. Move with
the arrow keys to any line — including one already recorded, to re-read it — and
press space to record. A blinking red dot and a timer show while the mic is live.

Resuming needs no bookkeeping: the cursor opens on the first line without a clip,
so rerunning the same command continues where you stopped. Deleting a `.wav`
re-opens that line.

| Key | Action |
|---|---|
| `↑` `↓` | move between lines |
| `space` | start / stop recording (confirms before overwriting a take) |
| `p` | play the selected line back |
| `s` | skip to the next line |
| `q` | quit |

Colours and the blink interval live in `recorder_theme.json`:

```json
{
  "blink_ms": 1000,
  "recorded": {"fg": "green",  "bold": false},
  "selected": {"fg": "yellow", "bold": true},
  "pending":  {"fg": "white",  "bold": false}
}
```

Any of the eight terminal colour names, `"default"`, or `"color:N"` for a
256-colour index. `--theme other.json` selects a different file.

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
| `record_data.py` | Recorder controller: keys, microphone, dataset rows |
| `recorder_ui.py` | Full-screen curses view (no audio or file knowledge) |
| `recorder_state.py` | Which chunks are recorded, and where the cursor sits |
| `recorder_theme.py` | Loads and validates `recorder_theme.json` |
| `train.py` | Bilingual LoRA training |
| `merge.py` | Folds adapter into the portable master model |
| `export.py` | Master model to CT2 / ggml |
| `convert.sh` | Exports everything and installs into OpenWhispr |
