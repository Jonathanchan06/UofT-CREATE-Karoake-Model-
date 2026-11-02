"""Run Demucs on every cover in covers_data/ and save the vocals and backing track.

Output: separated/htdemucs/<song>/<video id>/vocals.wav and no_vocals.wav
Log:    separated/htdemucs/run_log.csv (one row per track, including failures)

Safe to stop and re-run: tracks that are already finished are skipped. Originals are never
changed. Recordings listed in covers_data/exclusions.csv (song, id, reason) are skipped.

Usage:
    .venv/Scripts/python.exe separate_covers.py [--limit 1] [--threads 10]
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parent
COVERS_DIR = ROOT / "covers_data"
OUT_DIR = ROOT / "separated" / "htdemucs"
MODEL_NAME = "htdemucs"

# Folders that hold recordings of the same song as another folder
SONG_ALIASES = {"EdSheeran - Perfect": "Ed Sheeran - Perfect"}


def list_tracks():
    """(song, video id, wav path) for every recording in the manifests, minus the exclusions."""
    with open(COVERS_DIR / "exclusions.csv", encoding="utf-8", newline="") as f:
        excluded = {(row["song"], row["id"]) for row in csv.DictReader(f)}

    tracks = []
    for folder in sorted(p for p in COVERS_DIR.iterdir() if p.is_dir()):
        with open(folder / "manifest.csv", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if (folder.name, row["id"]) not in excluded:
                    song = SONG_ALIASES.get(folder.name, folder.name)
                    tracks.append((song, row["id"], folder / row["file"]))
    return tracks


def separate(model, wav_path, out_dir, device):
    """Split one file into vocals and everything else. Returns the audio length in seconds."""
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if sample_rate != model.samplerate or audio.shape[1] != model.audio_channels:
        raise ValueError(f"expected {model.samplerate} Hz stereo, got {sample_rate} Hz, "
                         f"{audio.shape[1]} channels")
    mix = torch.from_numpy(audio.T)

    # Same normalization as the demucs command line
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std()
    if std == 0:
        raise ValueError("track is completely silent")
    with torch.no_grad():
        sources = apply_model(model, ((mix - mean) / std)[None], shifts=1, overlap=0.25, device=device)[0]
    sources = sources * std + mean

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]
    backing = sources.sum(0) - vocals

    # Write to temporary names first, so a stopped run never leaves a half-written "finished" file
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, stem in [("no_vocals", backing), ("vocals", vocals)]:
        temp_path = out_dir / f"{name}.tmp.wav"
        sf.write(temp_path, np.clip(stem.numpy().T, -1, 1), sample_rate, subtype="PCM_16")
        os.replace(temp_path, out_dir / f"{name}.wav")
    return len(audio) / sample_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="stop after this many tracks (for timing a test)")
    parser.add_argument("--threads", type=int, default=10, help="CPU threads; lower keeps the laptop usable")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}", flush=True)
    tracks = list_tracks()
    todo = [t for t in tracks if not (OUT_DIR / t[0] / t[1] / "vocals.wav").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(tracks)} tracks, {len(tracks) - len(todo)} skipped or finished, {len(todo)} to run", flush=True)

    model = get_model(MODEL_NAME)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run_log.csv"
    is_new_log = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        log = csv.writer(log_file)
        if is_new_log:
            log.writerow(["song", "id", "status", "audio_s", "elapsed_s", "error"])

        for i, (song, video_id, wav_path) in enumerate(todo, start=1):
            start = time.time()
            try:
                audio_s = separate(model, wav_path, OUT_DIR / song / video_id, device)
                row = [song, video_id, "ok", round(audio_s, 1), round(time.time() - start, 1), ""]
            except Exception as error:  # one bad file must not stop a multi-day run
                row = [song, video_id, "failed", "", round(time.time() - start, 1), repr(error)]
            log.writerow(row)
            log_file.flush()
            print(f"[{i}/{len(todo)}] {row[2]} {row[0]} {row[1]}: {row[3]} s of audio in {row[4]} s", flush=True)


if __name__ == "__main__":
    main()
