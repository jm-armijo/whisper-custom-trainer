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
split automatically into 10-25 word chunks at sentence ends, blank lines, and —
for a sentence too long to fit — the last comma or similar pause in range, so a
line rarely ends mid-clause. Leave a blank line between paragraphs and they will
not be run together.

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
recorded are green, the selected line yellow, and the rest light grey. A line
that is both — selected and already recorded — stays green but turns bold, so a
take that has just been saved no longer reads as still waiting. Move with
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
  "recorded_selected": {"fg": "green", "bold": true},
  "pending":  {"fg": "white",  "bold": false}
}
```

Any of the eight terminal colour names, `"default"`, or `"color:N"` for a
256-colour index. `--theme other.json` selects a different file.

## Running on DietPi / Docker

The web recorder runs in a container on a home server (DietPi, Raspberry Pi, any
Debian-ish box) so you can record from a phone browser on the LAN. The container
**records only** — training stays on the laptop, so torch, transformers and
datasets are not installed and the image is roughly **400-500 MB**, not 2 GB.
The base is `python:3.12-slim`, a multi-arch manifest, so it builds on arm64
unchanged.

```bash
touch dataset.csv          # bind-mounted as a file; Docker would otherwise
mkdir -p data scripts      # create a directory here and every CSV write fails

docker compose up -d --build
docker compose logs -f     # watch requests arrive
```

Then open `http://<box-ip>:8080` on the phone — but read the next section first,
because the microphone will not work over plain HTTP.

| Host path | In container | Why |
|---|---|---|
| `./scripts` | `/data/scripts` (read-only) | reading material you supply |
| `./data` | `/data/audio` | recorded `.wav` files |
| `./dataset.csv` | `/data/dataset.csv` | the dataset rows |

All three are bind mounts, so takes survive `docker compose down` and a rebuild.
Copy `data/` and `dataset.csv` back to the laptop (`rsync -a`) when it is time to
train. Override the published port with `RECORDER_PORT`, and the file ownership
with `RECORDER_UID` / `RECORDER_GID` if your DietPi user is not `1000:1000`.

### The microphone needs a secure context

**This is the one thing that will bite you.** `getUserMedia` — the API the page
uses to reach the microphone — is gated behind a *secure context*. Browsers grant
it over **HTTPS** or on **`localhost`**, and nowhere else. Loading
`http://192.168.1.50:8080` on your phone will render the page fine and then fail
to record: Chrome reports `navigator.mediaDevices` as `undefined`, Safari and
Firefox refuse the permission prompt. Nothing is wrong with the server.

Three practical ways out, cheapest first:

**1. Tell Chrome to trust the one origin** (fastest; desktop and Android Chrome)

Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enable it, and
add the exact origin — scheme, IP and port, no trailing slash:

```
http://192.168.1.50:8080
```

Relaunch the browser. This is per-device and per-origin, so it survives until the
box changes IP — give it a DHCP reservation. iOS Safari has no equivalent flag.

**2. A hostname with real HTTPS** (best if you already run Tailscale)

Tailscale issues a genuine certificate for a machine on your tailnet:

```bash
tailscale cert "$(tailscale status --json | jq -r .Self.DNSName | sed 's/\.$//')"
tailscale serve --bg 8080     # https://<host>.<tailnet>.ts.net -> the recorder
```

The phone reaches it over the tailnet from anywhere, and the mic works with no
per-device configuration. Any reverse proxy holding a Let's Encrypt certificate
(Caddy, nginx) does the same job on the LAN.

**3. A self-signed certificate**

Terminate TLS in a proxy in front of the container. Every browser will show an
interstitial you must accept per device, and iOS additionally requires the CA to
be installed *and* trusted under Settings > General > About > Certificate Trust
Settings. It works, but option 1 or 2 is less trouble.

The container itself always speaks plain HTTP — TLS belongs in front of it, not
in a stdlib `http.server`.

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
