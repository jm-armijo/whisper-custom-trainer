# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tunes `openai/whisper-small` on the user's Latin American accent in **English and
Spanish** using one bilingual LoRA adapter, then exports it to formats other applications
can load. Runs locally on Apple Silicon (MPS).

## Commands

```bash
./setup.sh                                    # venv + deps + vendored repos (idempotent)
source venv/bin/activate

python record_data.py --text scripts/es.txt --lang es   # record; resumable
python train.py                               # LoRA adapter -> ./custom-lora-adapter
python merge.py                               # portable master -> ./merged-whisper-model
python export.py --format all                 # ct2 | ggml | all -> ./exports
./convert.sh                                  # export everything + install into OpenWhispr
```

### Tests

```bash
python -m pytest                              # unit only (~1s) - the default tier
python -m pytest -m integration               # real libs/converters (~30s)
python -m pytest -m e2e                       # full pipeline to speech (~20s)
python -m pytest -m "unit or integration or e2e"        # every tier (~4min)

python -m pytest tests/unit/test_chunking.py -q         # single file
python -m pytest -k test_masks_padding -m "unit or integration"   # single test
```

`pytest.ini` sets `addopts = -m "not integration and not e2e"`, so a bare `pytest` runs
only the fast tier. Tier markers are applied by path in `tests/conftest.py` — a test is
tagged by which of `tests/{unit,integration,e2e}/` it lives in, not by a decorator.

### Lint and commit hooks

`ruff` is the linter; its configuration lives in `pyproject.toml`.

```bash
ruff check .                                  # what the commit hook runs
ruff check . --fix                            # apply the safe fixes
```

Hooks are managed by `lefthook` (`lefthook.yml`), installed by `setup.sh`:

- **pre-commit** — `ruff`, the unit tier, the integration tier, and one e2e
  happy case (`TestRecordingLifecycle`, ~70s total).
- **pre-push** — the full suite (~7min).

Two things are deliberately kept off pre-commit. `tests/e2e/test_pipeline.py`
downloads `whisper-small` and trains an adapter, so a commit would block on a
model download. The rest of `test_recorder_app.py` drives a pty in real time,
including blink-interval timing that cannot be shortened without testing
something other than what ships — the full file costs ~3min against ~20s for
the single happy case that proves a take reaches disk.

A few ruff rules are switched off per file rather than obeyed, because obeying
them would break working code: `recorder_state.py` must hold its temp file open
across the `fsync`/`os.replace` pair that makes the dataset write atomic
(`SIM115`), and `export.run` inspects `returncode` itself to raise
`PipelineError` naming the failed command (`PLW1510`). The reasons are recorded
beside each ignore in `pyproject.toml`.

## Architecture

**The merged model is the product.** `./merged-whisper-model` is the durable artifact;
every distribution format is a cheap re-export from it. Adding a new target means adding
one function to `export.py` — never retraining.

```
dataset.csv ──train.py──> custom-lora-adapter ──merge.py──> merged-whisper-model
                                                                   │
                                              export.py ───────────┼──> exports/ct2  (faster-whisper, WhisperX)
                                                                   └──> exports/*.bin (whisper.cpp, OpenWhispr)
```

### Chunking

`chunk_text` splits on three things, in order of preference: sentence ends, blank
lines, and — only when a single sentence exceeds the maximum — the last natural
pause (`,;:` or a dash) within range, falling back to a plain word cut. Two
details are load-bearing: a **blank line ends a sentence** even without terminal
punctuation (collapsing all whitespace merged a heading ending in `:` into the
paragraph below, which was then cut mid-clause), and `_break_point` ignores a
pause in the first `MIN_WORDS_BEFORE_BREAK` words so a cut cannot leave a stub
line. Whatever changes, every word must survive: `chunk_text` is lossless.

### Recorder UI boundary

The recorder is split so the screen can change without touching the dataset
logic, mirroring how `whisper_pipeline.py` isolates third-party quirks:

- `recorder_ui.py` — curses only. Draws a view dict and maps keys to actions;
  knows nothing of audio, CSV or paths. Wrapping/scrolling are pure functions.
- `recorder_state.py` — pure chunk bookkeeping. A chunk counts as recorded only
  when its CSV row **and** its `.wav` both exist, so deleting a clip re-opens
  that line and no sidecar state file is needed.
  `chunk_statuses` returns four statuses, not three: a recorded line under the
  cursor is `recorded_selected`, because folding it into `selected` left a line
  yellow after its take was saved — the screen said "read this next" about work
  already done.
- `recorder_theme.py` + `recorder_theme.json` — colours and `blink_ms`, merged
  over defaults and validated at startup.
- `record_data.py` — the controller joining those to the microphone.

Two constraints worth keeping: the record dot blinks by **redrawing on a timer**
(`curses.A_BLINK` is ignored by most modern terminals), which is why the input
loop is non-blocking; and `read_key` decodes **raw `ESC [ A` sequences** as well
as `KEY_*` constants, because `keypad()` does not always fold them — arrows
silently stopped working without this.

Rendering is tested through a stub screen, never a real `initscr()`: pytest
replaces `sys.stdout` while curses drives the terminal fd, which corrupts the
run. The stub proves the UI calls curses as intended, not that curses paints
correctly — verify real rendering by running the recorder.

`whisper_pipeline.py` is the boundary layer: constants, paths, and every third-party
workaround live there so a library upgrade is a one-function edit. Prefer extending it
over scattering fixes across scripts.

### Bilingual training

One adapter serves both languages because `train.encode_example` calls
`tokenizer.set_prefix_tokens(language=row["language"], ...)` **per row**, so each sample
carries its own `<|en|>`/`<|es|>` token. Setting the language once globally would collapse
the two languages — this is the load-bearing detail of the whole approach.

## Version-specific constraints

This stack is transformers 5.x / datasets 5.x. Tutorials written for 4.x break here:

- **`datasets` cannot decode audio** without `torchcodec`, and returns a decoder object
  rather than an array. The CSV keeps plain paths; `wp.load_audio()` decodes via librosa.
  Do not `cast_column("audio", Audio())`.
- **`save_pretrained` no longer writes** `vocab.json` / `added_tokens.json` / `merges.txt`.
  **whisper.cpp's converter reads them directly and fails without them; CTranslate2 does
  not need them.** `merge.py` calls `restore_legacy_tokenizer_files()` to copy the
  canonical files from the base-model snapshot — fine-tuning never alters the tokenizer,
  so copying is correct, not a workaround.
- **`ct2-transformers-converter --copy_files` may only name files that exist** in the
  model dir, or it aborts. transformers 5.x writes `processor_config.json`, *not*
  `preprocessor_config.json`.
- **`ct2-transformers-converter` is a console script**, resolved via
  `export.converter_command()` rather than a bare name. `shutil.which` only finds it
  when the venv is on `PATH`, so `venv/bin/python export.py` without activating hit a
  raw `FileNotFoundError` — and tests guarding on `which` skipped silently, hiding
  real failures. Look it up beside `sys.executable` first.

- `Seq2SeqTrainingArguments`: `use_mps_device` was removed (accelerate detects MPS on
  its own), and `evaluation_strategy` is now `eval_strategy` — relevant if an eval split
  is added, since `train.py` currently trains without one.
- Training keeps `fp16=False, bf16=False` (MPS half precision is unreliable),
  `dataloader_num_workers=0` (forked workers are flaky on MPS), and
  `remove_unused_columns=False` + `label_names=["labels"]` (required under PEFT).

## OpenWhispr integration

OpenWhispr resolves models through a fixed registry (`tiny|base|small|medium|large|turbo`)
and rejects any other name as a path-traversal guard. There is no custom-path setting, so
`convert.sh` installs the build as `~/.cache/openwhispr/whisper-models/ggml-small.bin`
(backing up the original to `.orig`) and the user selects "small". The Homebrew cask is
`openwhispr` — `open-wispr` does not exist.

Cloud services (Speechify, Claude Code) run recognition server-side and cannot load this
model at all; OpenWhispr covers them by typing into whatever app is focused.

## Conventions

- Reading material for `record_data.py --text` goes in `scripts/` (gitignored, user-managed).
- `data/`, `dataset.csv`, and all model directories are gitignored — audio is personal,
  models are regenerable.
- Tests mock only what is slow or external. Unit tests fake the processor; the integration
  tier asserts those fakes match the real API. When changing a fake, update its integration
  counterpart in `tests/integration/test_processor.py`.
- Comments explain *why* (a version quirk, a downstream requirement), not what the line does.
