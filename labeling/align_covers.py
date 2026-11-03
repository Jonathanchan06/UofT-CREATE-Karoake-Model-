"""Run the two-pass alignment on every cover of the chosen songs and summarise how it went.

Writes (derived/, not in Git):
    alignment/<song_id>/<video id>.json      pass 1 result: key shift, octave, cost, time map
    alignment/<song_id>/<video id>.parquet   pass 2 result: one row per target note with onsets
    alignment/summary.csv                    one row per cover
    beep_check/covers/<song_id>__<video id>.wav   the cover with target notes beeped at the
                                             times and key pass 1 found (with --beeps)
Cover vocal pitch is cached in derived/pitch/covers/<song_id>/<video id>.parquet.

Run make_targets.py first. Usage:
    python labeling/align_covers.py [--all] [--beeps 2]
"""

import argparse
import csv
import json

import numpy as np
import pandas as pd
import soundfile as sf

import config
from align import align_backing, backing_chroma, find_onsets, note_table, recording_octave
from beep_check import render_beeps
from pitch import track_file


def covers_of(folder):
    """[(video id, cover wav)] for a song, minus the exclusions, only those with stems."""
    with open(config.ROOT / "covers_data" / "exclusions.csv", encoding="utf-8", newline="") as f:
        excluded = {row["id"] for row in csv.DictReader(f)}
    with open(config.ROOT / "covers_data" / folder / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r["id"], config.ROOT / "covers_data" / folder / r["file"]) for r in rows
            if r["id"] not in excluded and (config.COVER_STEMS_DIR / folder / r["id"] / "vocals.wav").exists()]


def align_cover(song, melody, song_chroma, video_id):
    """Both passes for one cover. Returns (summary row, alignment dict, notes DataFrame)."""
    stems = config.COVER_STEMS_DIR / song.folder / video_id
    cover_chroma, cover_loudness = backing_chroma(stems / "no_vocals.wav")
    alignment = align_backing(cover_chroma, cover_loudness, song_chroma)

    frames = track_file(stems / "vocals.wav",
                        config.DERIVED_DIR / "pitch" / "covers" / song.song_id / f"{video_id}.parquet")
    notes = note_table(melody, alignment)
    octave = recording_octave(notes, frames, song.tuning_cents)
    notes["target_midi"] += octave
    notes = find_onsets(notes, frames, song.tuning_cents)
    alignment.update({"octave": octave, "transpose": alignment["key_shift"] + octave})

    covered = notes[notes["expected_cover_s"].notna()]
    found = covered[covered["onset_found"]]
    errors = found["onset_error_ms"]
    row = {
        "song_id": song.song_id, "video_id": video_id,
        "aligned": (alignment["cost"] <= config.MAX_ALIGN_COST
                    and abs(alignment["slope"] - 1) <= config.MAX_SLOPE_ERROR),
        "cost": round(alignment["cost"], 3), "slope": round(alignment["slope"], 3),
        "key_shift": alignment["key_shift"], "octave": octave, "transpose": alignment["transpose"],
        "song_start_s": round(alignment["song_start_s"], 1), "song_end_s": round(alignment["song_end_s"], 1),
        "notes_covered": len(covered), "onsets_found": round(len(found) / max(len(covered), 1), 2),
        "median_onset_error_ms": round(float(errors.median()), 0) if len(errors) else None,
        "median_abs_onset_error_ms": round(float(errors.abs().median()), 0) if len(errors) else None,
        "within_100ms": round(float((errors.abs() <= 100).mean()), 2) if len(errors) else None,
        "vocals_voiced": round(float(frames["voiced"].mean()), 2),
    }
    return row, alignment, notes


def write_cover_beeps(song, notes, cover_wav, path):
    """The cover's full mix with each target note beeped where pass 1 says it should be."""
    audio, sample_rate = sf.read(cover_wav, dtype="float32", always_2d=True)
    placed = notes[notes["expected_cover_s"].notna()]
    phrase = [{"midi_pitch": n.target_midi, "start_s": n.expected_cover_s, "duration_s": n.duration_s}
              for n in placed.itertuples(index=False)]
    beeps = render_beeps([phrase], song.tuning_cents, len(audio), sample_rate)
    mix = np.clip(config.SONG_LEVEL * audio + config.BEEP_LEVEL * beeps[:, None], -1, 1)
    sf.write(path, mix, sample_rate, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="every song in songs.csv, not just the pilot")
    parser.add_argument("--beeps", type=int, default=0, help="beep files for this many covers per song")
    args = parser.parse_args()

    songs = pd.read_csv(config.REAL_DATA_DIR / "songs.csv")
    if not args.all:
        songs = songs[songs["folder"].isin(config.PILOT_SONGS)]
    out_dir = config.DERIVED_DIR / "alignment"
    beep_dir = config.DERIVED_DIR / "beep_check" / "covers"
    beep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for song in songs.itertuples(index=False):
        melody = json.loads((config.REAL_DATA_DIR / "melodies" / f"{song.song_id}.json").read_text())
        song_chroma, _ = backing_chroma(config.ORIGINAL_STEMS_DIR / song.folder / "no_vocals.wav")
        (out_dir / song.song_id).mkdir(parents=True, exist_ok=True)

        for i, (video_id, cover_wav) in enumerate(covers_of(song.folder)):
            row, alignment, notes = align_cover(song, melody, song_chroma, video_id)
            rows.append(row)
            notes.to_parquet(out_dir / song.song_id / f"{video_id}.parquet", index=False)
            saved = {k: (v.round(3).tolist() if isinstance(v, np.ndarray) else v) for k, v in alignment.items()}
            (out_dir / song.song_id / f"{video_id}.json").write_text(json.dumps(saved))
            if i < args.beeps:
                write_cover_beeps(song, notes, cover_wav, beep_dir / f"{song.song_id}__{video_id}.wav")
            print(f"{song.folder} {video_id}: cost {row['cost']:.3f}, key {row['key_shift']:+d}, "
                  f"octave {row['octave']:+d}, onsets found {row['onsets_found']:.0%}, "
                  f"median |onset error| {row['median_abs_onset_error_ms']} ms", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n" + summary.drop(columns=["song_start_s", "song_end_s"]).to_string(index=False))


if __name__ == "__main__":
    main()
