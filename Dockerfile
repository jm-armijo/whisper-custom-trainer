# Recording-only image for the web recorder.
#
# The container records; it never trains. Training pulls torch + transformers +
# datasets (~2GB) and runs on the user's laptop against MPS, which a DietPi box
# does not have, so none of that stack is installed here. The image needs only
# enough to serve HTTP, decode what the browser uploads, and write wav + CSV.
#
# python:3.12-slim is a multi-arch manifest, so this builds unchanged on the
# arm64 DietPi targets (Raspberry Pi) as well as amd64.
FROM python:3.12-slim

# ffmpeg decodes the WebM/Opus that MediaRecorder produces in Chrome and
# Firefox; libsndfile is what soundfile binds to for writing the wav.
# librosa reaches for ffmpeg through audioread/soundfile for anything
# soundfile cannot open itself, so it is a runtime dependency, not a build one.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Bounds match setup.sh for the packages they share, so the container decodes
# audio exactly the way the training host does. sounddevice is deliberately
# absent: the browser captures the microphone and POSTs the clip, so the
# container needs neither PortAudio nor a sound device.
RUN pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "librosa>=1,<2" \
        "soundfile>=0.14,<1"

WORKDIR /app

# Only the modules the server actually imports. Copying the whole tree would
# drag in train.py/merge.py/export.py, whose imports are not installed here.
COPY recorder_server.py recorder_scripts.py recorder_state.py whisper_pipeline.py ./
COPY static/ ./static/

# Defaults match the compose mounts; every one is overridable on the command
# line. 0.0.0.0 is required for the published port to reach a phone on the LAN:
# bound to 127.0.0.1 the server would only answer inside the container.
# RECORDER_CSV sits *inside* RECORDER_OUT_DIR rather than beside it: that
# directory is the one bind mount, and the dataset must be an ordinary file
# within a mounted directory. Mounted as a file of its own, os.replace onto it
# fails with EBUSY - see the volumes comment in docker-compose.yml.
ENV RECORDER_HOST=0.0.0.0 \
    RECORDER_PORT=8080 \
    RECORDER_SCRIPTS_DIR=/data/scripts \
    RECORDER_OUT_DIR=/data/audio \
    RECORDER_CSV=/data/audio/dataset.csv

EXPOSE 8080

# Unbuffered so `docker compose logs -f` shows requests as they arrive rather
# than in 8KB bursts.
ENV PYTHONUNBUFFERED=1

# --cert is appended only when RECORDER_CERT is set, so an unset variable
# leaves the server on plain HTTP rather than passing an empty --cert that
# argparse would take as the literal path "". A phone will not open its
# microphone over http, so the LAN deployment sets it; the localhost dev loop
# does not need to.
CMD ["sh", "-c", "exec python recorder_server.py \
    --host \"$RECORDER_HOST\" \
    --port \"$RECORDER_PORT\" \
    --scripts \"$RECORDER_SCRIPTS_DIR\" \
    --out-dir \"$RECORDER_OUT_DIR\" \
    --csv \"$RECORDER_CSV\" \
    ${RECORDER_CERT:+--cert \"$RECORDER_CERT\"}"]
