"""Run Demucs on the original songs in originals_data/ (the source of the target notes).

Output: separated/originals/htdemucs/<song folder>/vocals.wav and no_vocals.wav
Uses the same separate() as separate_covers.py. Finished songs are skipped, so it is safe to re-run.

Usage:
    python labeling/separate_originals.py            # the pilot songs
    python labeling/separate_originals.py --all      # every song in originals_data/manifest.csv
"""

import argparse
import csv
import sys
import time

import torch
from demucs.pretrained import get_model

import config

sys.path.insert(0, str(config.ROOT))
from separate_covers import MODEL_NAME, separate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="every song, not just the pilot")
    args = parser.parse_args()

    with open(config.ORIGINALS_DIR / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        manifest = {row["song_folder"]: row for row in csv.DictReader(f)}
    folders = sorted(manifest) if args.all else config.PILOT_SONGS
    folders = [s for s in folders if s not in config.LEFT_OUT_SONGS]
    todo = [s for s in folders if not (config.ORIGINAL_STEMS_DIR / s / "vocals.wav").exists()]
    print(f"{len(folders)} songs, {len(folders) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(MODEL_NAME)
    model.eval()
    for i, folder in enumerate(todo, start=1):
        start = time.time()
        audio_s = separate(model, config.ORIGINALS_DIR / manifest[folder]["file"],
                           config.ORIGINAL_STEMS_DIR / folder, device)
        print(f"[{i}/{len(todo)}] {folder}: {audio_s:.0f} s of audio in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
