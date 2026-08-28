"""Interactive recorder that turns a text script into a labelled audio dataset."""

import argparse
import csv
import sys
from pathlib import Path

import whisper_pipeline as wp


def main():
    args = parse_arguments()
    chunks = wp.chunk_text(read_script(args.text))
    if not chunks:
        raise wp.PipelineError(f"No usable text found in {args.text}")

    resume_index = wp.next_chunk_index(args.csv, args.lang)
    if resume_index >= len(chunks):
        print(f"All {len(chunks)} chunks for '{args.lang}' are already recorded.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if resume_index:
        print(f"Resuming at chunk {resume_index + 1} of {len(chunks)}.\n")

    record_session(chunks, resume_index, args)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path,
                        help="UTF-8 text file to read aloud")
    parser.add_argument("--lang", required=True, choices=wp.SUPPORTED_LANGUAGES)
    parser.add_argument("--out-dir", type=Path, default=wp.AUDIO_DIR)
    parser.add_argument("--csv", type=Path, default=wp.DATASET_CSV)
    return parser.parse_args()


def read_script(path):
    if not path.exists():
        raise wp.PipelineError(f"Script not found: {path}")
    return path.read_text(encoding="utf8")


def record_session(chunks, start_index, args):
    print_instructions()
    for index in range(start_index, len(chunks)):
        text = chunks[index]
        print(f"\n[{index + 1}/{len(chunks)}] {text}")

        clip = prompt_until_recorded(text)
        if clip is None:
            print("  skipped")
            continue
        if clip is QUIT:
            print("\nStopped. Re-run the same command to resume here.")
            return

        destination = args.out_dir / f"{args.lang}_{index:05d}.wav"
        write_clip(destination, clip)
        append_row(args.csv, destination, text, args.lang)
        print(f"  saved {destination.name} ({len(clip) / wp.SAMPLE_RATE:.1f}s)")

    print(f"\nDone. Dataset: {args.csv}")


def print_instructions():
    print(
        "\nControls: ENTER start/stop recording | r re-record | s skip | q quit\n"
        "Read each line at your natural pace.\n" + "-" * 60
    )


QUIT = object()


def prompt_until_recorded(text):
    """Record one chunk, honouring re-record requests until the take is kept."""
    while True:
        command = input("  ENTER to record (r/s/q): ").strip().lower()
        if command == "q":
            return QUIT
        if command == "s":
            return None

        clip = record_clip()
        if is_unusable(clip):
            print(f"  discarded: {len(clip) / wp.SAMPLE_RATE:.2f}s is too short to use")
            continue

        warn_if_unusual_length(clip)

        decision = input("  ENTER to keep, 'r' to redo: ").strip().lower()
        if decision != "r":
            return clip


def record_clip():
    """Capture microphone input between two ENTER presses."""
    import numpy as np
    import sounddevice as sd

    frames = []
    with sd.InputStream(
        samplerate=wp.SAMPLE_RATE, channels=1, dtype="float32",
        callback=lambda data, *_: frames.append(data.copy()),
    ):
        input("  recording... ENTER to stop ")

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def is_unusable(clip):
    """A clip this short carries no speech; keeping it would mislabel the dataset."""
    return len(clip) / wp.SAMPLE_RATE < wp.MIN_CLIP_SECONDS


def warn_if_unusual_length(clip):
    seconds = len(clip) / wp.SAMPLE_RATE
    if seconds < wp.MIN_CLIP_SECONDS:
        print(f"  WARNING: only {seconds:.2f}s - likely cut off, consider 'r'")
    elif seconds > wp.MAX_CLIP_SECONDS:
        print(f"  WARNING: {seconds:.1f}s exceeds Whisper's 30s window, consider 'r'")


def write_clip(destination, clip):
    import soundfile as sf

    sf.write(str(destination), clip, wp.SAMPLE_RATE, subtype="PCM_16")


def append_row(csv_path, audio_path, text, language):
    """Append and flush per row so an interrupted session keeps prior work."""
    is_new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        if is_new_file:
            writer.writerow(wp.CSV_COLUMNS)
        writer.writerow([str(audio_path), text, language])
        handle.flush()


if __name__ == "__main__":
    try:
        main()
    except wp.PipelineError as error:
        sys.exit(f"error: {error}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Re-run to resume.")
