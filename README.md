# whisper-custom-trainer

Fine-tunes `openai/whisper-small` on your own Latin American accent in **English
and Spanish** using a single LoRA adapter, then exports it to formats other
applications can load.

Runs locally on Apple Silicon (Metal / MPS).

## Quick start

```bash
./setup.sh                                          # venv + dependencies + repos
source venv/bin/activate

python record_data.py --text scripts/es.txt --lang es   # read prompts aloud
python record_data.py --text scripts/en.txt --lang en

python train.py                                     # LoRA adapter
python merge.py                                     # portable master model
./convert.sh                                        # export + install
```

Supply your own `scripts/*.txt`: any prose you are comfortable reading. Name the
file for its language — `es.txt`, `en.txt` — and the web recorder can infer it
too. It is
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

## The web recorder

A second front end onto the same dataset: a browser page instead of curses, so a
phone can record while the box sits in a cupboard. It is stdlib-only — no web
framework — and shares every rule with the terminal recorder.

> **New, and not yet verified in a real browser.** The server, its API and the
> dataset writes are tested; the page itself has not been driven on real
> hardware. Do one manual `docker compose up` and one browser load before
> trusting it with a recording session.

### Running it

**On the laptop**, against the working tree:

```bash
venv/bin/python recorder_server.py                    # http://0.0.0.0:8080
venv/bin/python recorder_server.py --port 9000        # anything is overridable
```

With no flags it reads `scripts/`, writes clips to `data/` and rows to
`dataset.csv` — the same three paths `record_data.py` uses, which is what lets
the two front ends share one dataset.

**On the DietPi box**, in Docker. The container **records only** — training stays
on the laptop, so torch, transformers and datasets are not installed and the
image is roughly **400-500 MB**, not 2 GB. The base is `python:3.12-slim`, a
multi-arch manifest, so it builds on arm64 unchanged.

```bash
touch dataset.csv          # bind-mounted as a file; Docker would otherwise
mkdir -p data scripts      # create a directory here and every CSV write fails

docker compose up -d --build
docker compose logs -f     # one line per request
```

That first line is not optional and not cosmetic: a bind mount whose host side
is missing is created by Docker as a **directory**, and every subsequent CSV
write then fails with `EISDIR`.

Then open `http://<box-ip>:8080` on the phone — but read
[the secure-context section](#the-microphone-needs-a-secure-context) first,
because the microphone will not work over plain HTTP.

| Host path | In container | Why |
|---|---|---|
| `./scripts` | `/data/scripts` (read-only) | reading material you supply |
| `./data` | `/data/audio` | recorded `.wav` files |
| `./dataset.csv` | `/data/dataset.csv` | the dataset rows |

All three are bind mounts, so takes survive `docker compose down` and a rebuild.

### Configuration

Every knob, and there are only six. The server takes flags; two of them fall
back to an environment variable, and the rest are set by the container's `CMD`:

| Flag | Env var | Default | What it decides |
|---|---|---|---|
| `--host` | `RECORDER_HOST` | `0.0.0.0` | which interfaces to listen on — see below |
| `--port` | `RECORDER_PORT` | `8080` | the listening port |
| `--scripts` | — | `./scripts` | directory of `*.txt` reading material |
| `--csv` | — | `./dataset.csv` | the dataset rows |
| `--out-dir` | — | `./data` | where `.wav` takes are written |
| `--static` | — | `./static` | the browser assets |

Those four defaults resolve against the **project root** — where the modules
live, not the directory you happen to launch from — so running the server from
anywhere still finds the same dataset.

Only `--host` and `--port` read the environment directly. `RECORDER_SCRIPTS_DIR`,
`RECORDER_CSV` and `RECORDER_OUT_DIR` exist in the **Dockerfile** only, where the
`CMD` expands them into the matching flags — so they configure the container, not
the script. (The flag is `--scripts`, not `--scripts-dir`; the Dockerfile passing
the latter is what made an early image exit at startup.)

Two more variables belong to `docker-compose.yml` rather than the server:
`RECORDER_PORT` also picks the **published** port on the host, and
`RECORDER_UID` / `RECORDER_GID` (default `1000:1000`) set the ownership of the
files it writes. Match them to your DietPi user, or the takes come back owned by
someone the laptop cannot read.

A script's **language comes from its filename**: `scripts/es.txt` records as
Spanish, `en.txt` as English. Any other stem is listed in the picker but cannot
be recorded, because a mislabelled row poisons the bilingual adapter.

### Reaching the box from the phone

`--host` is the whole story. The default `0.0.0.0` listens on every interface,
which is what makes the box reachable at its LAN address. Binding it to
`127.0.0.1` accepts only connections from the box itself — the page will be
unreachable from the phone, and inside Docker the published port would answer
nothing at all. Use loopback only when you are browsing from the same machine.

Find the address to type:

```bash
hostname -I | awk '{print $1}'      # on the DietPi box
ipconfig getifaddr en0              # on macOS
```

Give the box a **DHCP reservation** while you are in the router. The address ends
up in a browser flag below, and a lease change would silently break recording.

To record away from home, put the box on a [Tailscale](https://tailscale.com)
tailnet and reach it by its `*.ts.net` name — which also solves the certificate
problem in the next section outright.

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

Relaunch the browser. This is per-device and per-origin, so it lasts exactly as
long as the box keeps that address — hence the DHCP reservation above. iOS
Safari has no equivalent flag.

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

### Recording from two devices

The phone and the laptop write into **one dataset**, and nothing synchronises
them, because there is nothing to synchronise. A chunk's clip filename is
derived, not allocated: line `n` of a Spanish script is always
`{lang}_{index:05d}.wav` — `es_00042.wav` — whichever front end recorded it. So
`dataset.csv` is a projection of the audio directory over the script rather than
a database of its own, and re-reading a line simply overwrites its one file.
That is also why deleting a `.wav` re-opens the line in both UIs.

The practical loop: record on the phone against the container, then bring the
takes home before training.

```bash
rsync -a dietpi:my-whisper/data/ data/       # the clips
rsync -a dietpi:my-whisper/dataset.csv .     # the rows
python train.py
```

Record on the laptop with either front end — `record_data.py` for the curses TUI,
or `recorder_server.py` if you prefer the browser. Both land on the same files.
Keep one machine authoritative for a given script while a session is in progress:
two people reading the same line at once is the one case the derived filename
cannot arbitrate, since the second take simply wins.

### The API

Six endpoints, all JSON except the audio. Useful for scripting a session or
checking progress without a browser:

| Endpoint | Does |
|---|---|
| `GET /api/scripts` | every script with its recorded / total counts |
| `GET /api/scripts/{name}` | one script's chunks, each flagged recorded or not |
| `POST /api/scripts/{name}/chunks/{index}` | upload a take for one line |
| `DELETE /api/scripts/{name}/chunks/{index}` | drop the take, re-opening the line |
| `GET /api/scripts/{name}/chunks/{index}/audio` | the stored `.wav`, for playback |

`POST` accepts the blob raw or as a multipart `audio` field, so `curl` and the
browser's `FormData` both work. Anything `libsndfile` cannot open is decoded
through `ffmpeg`, which is what covers the WebM/Opus that `MediaRecorder`
actually produces. A take shorter than the minimum is rejected **before** the
dataset is touched, so a bad upload cannot clobber a good take on that line.

```bash
curl -s localhost:8080/api/scripts
curl -s -X POST --data-binary @take.wav \
    localhost:8080/api/scripts/es.txt/chunks/0
```

Errors come back as `{"message": "..."}` — 404 when something does not exist,
400 when the request was wrong.

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
| `recorder_server.py` | Web recorder: routing, uploads, audio decoding |
| `recorder_scripts.py` | Script discovery, language inference, per-script progress |
| `static/` | Browser page for the web recorder |
| `Dockerfile`, `docker-compose.yml` | Recording-only container for the DietPi box |
| `train.py` | Bilingual LoRA training |
| `merge.py` | Folds adapter into the portable master model |
| `export.py` | Master model to CT2 / ggml |
| `convert.sh` | Exports everything and installs into OpenWhispr |
