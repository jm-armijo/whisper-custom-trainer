"""Interactive recorder that turns a text script into a labelled audio dataset.

The screen lives in recorder_ui, the chunk bookkeeping in recorder_state; this
module is the controller that joins them to the microphone and the dataset.
"""

import argparse
import sys
import time
from pathlib import Path

import recorder_state as rs
import recorder_theme as rt
import recorder_ui as ui
import whisper_pipeline as wp


def main():
    args = parse_arguments()
    chunks = wp.chunk_text(read_script(args.text))
    if not chunks:
        raise wp.PipelineError(f"No usable text found in {args.text}")

    theme = rt.load_theme(args.theme)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    ui.start(run, chunks, args, theme)
    print(f"Dataset: {args.csv}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path,
                        help="UTF-8 text file to read aloud")
    parser.add_argument("--lang", required=True, choices=wp.SUPPORTED_LANGUAGES)
    parser.add_argument("--out-dir", type=Path, default=wp.AUDIO_DIR)
    parser.add_argument("--csv", type=Path, default=wp.DATASET_CSV)
    parser.add_argument("--theme", type=Path, default=rt.DEFAULT_THEME_PATH,
                        help="JSON colour and blink configuration")
    return parser.parse_args()


def read_script(path):
    if not path.exists():
        raise wp.PipelineError(f"Script not found: {path}")
    return path.read_text(encoding="utf8")


def run(stdscr, chunks, args, theme):
    """Event loop: draw, read a key, act, repeat until quit."""
    view = ui.RecorderUI(stdscr, theme)
    recorded = rs.recorded_indices(args.csv, args.out_dir, args.lang)
    cursor = rs.first_unrecorded(len(chunks), recorded)
    message = ""

    while True:
        view.draw(build_view(chunks, recorded, cursor, args, message, ui.IDLE))
        action = view.read_key()

        if action == "quit":
            return
        if action in ("up", "down", "top", "bottom"):
            cursor = move_cursor(cursor, action, len(chunks))
            message = ""
        elif action in ("record", "redo"):
            recorded, message = handle_record(
                view, chunks, recorded, cursor, args, theme
            )
        elif action == "play":
            message = play_clip(args, cursor, recorded)
        elif action == "skip":
            cursor = move_cursor(cursor, "down", len(chunks))
            message = ""


def move_cursor(cursor, action, total):
    """Clamp at both ends so the cursor never leaves the script."""
    if action == "up":
        return max(cursor - 1, 0)
    if action == "down":
        return min(cursor + 1, total - 1)
    if action == "top":
        return 0
    return total - 1


def build_view(chunks, recorded, cursor, args, message, state, tick=0, elapsed=0):
    return {
        "title": f" {args.text.name if hasattr(args, 'text') else 'script'} · "
                 f"{args.lang} · {len(recorded)}/{len(chunks)} recorded ",
        "chunks": chunks,
        "statuses": rs.chunk_statuses(len(chunks), recorded, cursor),
        "recorded": recorded,
        "cursor": cursor,
        "state": state,
        "tick": tick,
        "elapsed": elapsed,
        "message": message,
    }


def handle_record(view, chunks, recorded, cursor, args, theme):
    """Record over the cursor line, confirming first if it already has audio."""
    if cursor in recorded and not view.confirm(
        f"Re-record line {cursor + 1}? (y/n)"
    ):
        return recorded, "kept the existing take"

    def redraw(tick, elapsed):
        view.draw(build_view(
            chunks, recorded, cursor, args, "", ui.RECORDING, tick, elapsed
        ))

    clip = capture_clip(view, redraw)

    if is_unusable(clip):
        seconds = len(clip) / wp.SAMPLE_RATE
        return recorded, f"discarded: {seconds:.2f}s is too short to use"

    destination = rs.clip_path(args.out_dir, args.lang, cursor)
    write_clip(destination, clip)
    rs.upsert_row(args.csv, destination, chunks[cursor], args.lang)

    return recorded | {cursor}, saved_message(clip)


def saved_message(clip):
    seconds = len(clip) / wp.SAMPLE_RATE
    if seconds > wp.MAX_CLIP_SECONDS:
        return f"saved {seconds:.1f}s - exceeds Whisper's 30s window, consider redo"
    return f"saved {seconds:.1f}s"


def capture_clip(view, on_tick):
    """Capture microphone input until the stop key, ticking for the blink.

    The sounddevice callback fills frames on its own thread while curses polls
    for a keypress, so the dot keeps blinking throughout the take.
    """
    import numpy as np
    import sounddevice as sd

    frames = []
    started = time.monotonic()
    tick = 0

    with sd.InputStream(
        samplerate=wp.SAMPLE_RATE, channels=1, dtype="float32",
        callback=lambda data, *_: frames.append(data.copy()),
    ):
        on_tick(tick, 0)
        while True:
            action = view.read_key(timeout_ms=view.theme.blink_ms)
            if action in ("record", "redo", "quit"):
                break
            tick += 1
            on_tick(tick, time.monotonic() - started)

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def play_clip(args, cursor, recorded):
    """Play a take back so it can be checked without leaving the recorder."""
    if cursor not in recorded:
        return "nothing recorded on this line yet"

    import sounddevice as sd

    samples = wp.load_audio(rs.clip_path(args.out_dir, args.lang, cursor))
    sd.play(samples, wp.SAMPLE_RATE)
    sd.wait()
    return f"played line {cursor + 1}"


def is_unusable(clip):
    """A clip this short carries no speech; keeping it would mislabel the dataset."""
    return len(clip) / wp.SAMPLE_RATE < wp.MIN_CLIP_SECONDS


def write_clip(destination, clip):
    import soundfile as sf

    sf.write(str(destination), clip, wp.SAMPLE_RATE, subtype="PCM_16")


if __name__ == "__main__":
    try:
        main()
    except wp.PipelineError as error:
        sys.exit(f"error: {error}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Re-run to resume.")
