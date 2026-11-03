"""Target notes for each original song: vocal stem -> PESTO -> notes -> phrases.

Writes (in ml/data/real/, not in Git):
    melodies/<song_id>.json   the contract format, notes in the original key, times from song start
    songs.csv                 song_id, tempo_bpm, original_key, language, genre, tuning_cents,
                              plus folder and note counts (extra columns are ignored by ml/)
Pitch frames are cached in derived/pitch/originals/<song_id>.parquet.

Run separate_originals.py first. Usage:
    python labeling/make_targets.py [--all]
"""

import argparse
import json

import librosa
import numpy as np
import pandas as pd

import config
from pitch import load_mono, loudness_db, track_file
from targets import frames_to_notes, melody_json, notes_to_phrases, song_id_for, tuning_cents

SONG_COLUMNS = ["song_id", "tempo_bpm", "original_key", "language", "genre", "tuning_cents",
                "folder", "num_phrases", "num_notes", "voiced_share"]


def make_song_targets(folder):
    """(melody dict, songs.csv row) for one song folder."""
    song_id = song_id_for(folder)
    stem = config.ORIGINAL_STEMS_DIR / folder / "vocals.wav"
    if not stem.exists():
        raise FileNotFoundError(f"no vocal stem at {stem}; run separate_originals.py first")

    frames = track_file(stem, config.DERIVED_DIR / "pitch" / "originals" / f"{song_id}.parquet")
    samples, sample_rate = load_mono(stem)
    tuning = tuning_cents(frames["midi"])
    loudness = loudness_db(samples, sample_rate, frames["time_s"])
    notes = frames_to_notes(frames["midi"] - tuning / 100, frames["time_s"], loudness)
    phrases = notes_to_phrases(notes)

    # Tempo from the original's backing track; key, language and genre are filled in later
    tempo = estimate_tempo(config.ORIGINAL_STEMS_DIR / folder / "no_vocals.wav")
    row = {"song_id": song_id, "tempo_bpm": tempo, "original_key": None, "language": None,
           "genre": None, "tuning_cents": round(tuning, 1), "folder": folder,
           "num_phrases": len(phrases), "num_notes": len(notes),
           "voiced_share": round(float(frames["voiced"].mean()), 3)}
    return melody_json(song_id, phrases, tempo), row


def estimate_tempo(backing_path):
    """Beats per minute of a backing track (librosa's beat tracker), rounded to 0.1."""
    y, sr = librosa.load(backing_path, sr=config.ALIGN_SAMPLE_RATE, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(np.atleast_1d(tempo)[0]), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="every separated original, not just the pilot")
    args = parser.parse_args()

    folders = (sorted(p.name for p in config.ORIGINAL_STEMS_DIR.iterdir() if p.is_dir())
               if args.all else config.PILOT_SONGS)
    melody_dir = config.REAL_DATA_DIR / "melodies"
    melody_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in folders:
        melody, row = make_song_targets(folder)
        (melody_dir / f"{row['song_id']}.json").write_text(json.dumps(melody, indent=1))
        rows.append(row)
        lengths = [len(p) for p in melody["phrases"]]
        print(f"{folder}: {row['num_notes']} notes in {row['num_phrases']} phrases "
              f"(notes per phrase median {np.median(lengths):.0f}, max {max(lengths)}), "
              f"tuning {row['tuning_cents']:+.1f} cents, voiced {row['voiced_share']:.0%}")

    # Keep rows for songs made in earlier runs
    songs_path = config.REAL_DATA_DIR / "songs.csv"
    songs = pd.DataFrame(rows, columns=SONG_COLUMNS)
    if songs_path.exists():
        old = pd.read_csv(songs_path)
        songs = pd.concat([old[~old["song_id"].isin(songs["song_id"])], songs], ignore_index=True)
    songs.sort_values("song_id").to_csv(songs_path, index=False)
    print(f"Wrote {len(rows)} melodies to {melody_dir} and {songs_path}")


if __name__ == "__main__":
    main()
