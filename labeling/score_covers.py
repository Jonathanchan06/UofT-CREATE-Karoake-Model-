"""Score every aligned cover with scorer/score.py, plus two label checks (CLAUDE.md step 3).

For each cover that pass 1 accepted: its PESTO frames are moved onto the song's timeline with the
pass-1b map (align.to_song_timeline), then scored with the cover's key and the song's tuning.

    --self-score  each original's vocal stem scored against its own target notes. Circular (the
                  notes come from that stem), so it only catches broken answer keys: songs below
                  SELF_SCORE_MIN_ACCURACY are flagged.
    --clips N     the N best and N worst scored phrases as WAV clips (the cover's own audio with the
                  target notes as beeps), to check by ear that high scores sound in tune.

Phrases that nearly every cover fails (CONSENSUS_* in config.py) are taken to have a wrong answer
key and are dropped from every cover; they are listed in derived/scores/dropped_phrases.csv.

Writes derived/scores/cover_scores.csv (one row per phrase; singer_id is still the video id, not
the dataset) and derived/scores/clips/. Run align_covers.py first. Usage:
    python labeling/score_covers.py [--all] [--self-score] [--clips 5]
"""

import argparse
import csv
import json
import sys

import numpy as np
import pandas as pd
import soundfile as sf

import config
from align import song_to_cover_time, to_song_timeline
from pitch import midi_to_hz

sys.path.insert(0, str(config.ROOT / "scorer"))
import score  # noqa: E402

OUT_DIR = config.DERIVED_DIR / "scores"
CLIP_PAD_S = 0.5
CLIP_BEEP_TO_MIX = 0.8          # beep loudness relative to the clip's (RMS)
CLIP_MIX_IN_BEEP_EAR = 0.4      # how much of the cover is left in the beep (right) ear


def self_scores(songs):
    """One row per original: median phrase labels against its own target notes, and a flag."""
    rows = []
    for song in songs.itertuples(index=False):
        frames = pd.read_parquet(config.DERIVED_DIR / "pitch" / "originals" / f"{song.song_id}.parquet")
        df = score.score_take(frames, config.REAL_DATA_DIR / "melodies" / f"{song.song_id}.json",
                              "original", transpose=0, tuning_cents=song.tuning_cents)
        rows.append({"song_id": song.song_id, "phrases": len(df),
                     "accuracy": df["accuracy"].median(), "timing_accuracy": df["timing_accuracy"].median(),
                     "missed_note_rate": df["missed_note_rate"].median(),
                     "flag": df["accuracy"].median() < config.SELF_SCORE_MIN_ACCURACY})
    return pd.DataFrame(rows)


def failed_by_every_cover(scores):
    """Phrases (song_id, phrase_idx) that nearly every cover fails: at least CONSENSUS_MIN_COVERS
    covers and their median accuracy below CONSENSUS_MAX_MEDIAN_ACCURACY. One row each, with why."""
    by_phrase = scores.groupby(["song_id", "phrase_idx"]).agg(
        covers=("accuracy", "size"), median_accuracy=("accuracy", "median"),
        median_octave_error_rate=("octave_error_rate", "median"),
        median_cents_error=("mean_cents_error", "median"))
    failed = by_phrase[(by_phrase["covers"] >= config.CONSENSUS_MIN_COVERS)
                       & (by_phrase["median_accuracy"] < config.CONSENSUS_MAX_MEDIAN_ACCURACY)]
    return failed.reset_index()


def score_cover(song, video_id):
    """Phrase rows for one aligned cover, plus the alignment dict."""
    alignment = json.loads((config.DERIVED_DIR / "alignment" / song.song_id / f"{video_id}.json").read_text())
    alignment = {k: np.array(v) if isinstance(v, list) else v for k, v in alignment.items()}
    frames = pd.read_parquet(config.DERIVED_DIR / "pitch" / "covers" / song.song_id / f"{video_id}.parquet")
    df = score.score_take(to_song_timeline(frames, alignment),
                          config.REAL_DATA_DIR / "melodies" / f"{song.song_id}.json", video_id,
                          transpose=int(alignment["key_shift"]), tuning_cents=song.tuning_cents)
    return df, alignment


def cover_wav(folder, video_id):
    with open(config.ROOT / "covers_data" / folder / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        files = {r["id"]: r["file"] for r in csv.DictReader(f)}
    return config.ROOT / "covers_data" / folder / files[video_id]


def write_clip(path, folder, tuning_cents, row, alignment):
    """The cover's audio around one phrase with its target notes as beeps. Left ear: the cover
    alone. Right ear: the cover quieter plus the beeps, scaled to the clip's loudness, so they stay
    audible in loud mixes and when the singer is exactly on the beep's pitch."""
    notes = row["target_notes"]
    start, end = notes[0]["start_s"] - CLIP_PAD_S, notes[-1]["start_s"] + notes[-1]["duration_s"] + CLIP_PAD_S
    a_s, b_s = song_to_cover_time(alignment, [max(start, alignment["song_start_s"]),
                                              min(end, alignment["song_end_s"])])
    audio, sr = sf.read(cover_wav(folder, row["singer_id"]), dtype="float32", always_2d=True)
    clip = audio[int(a_s * sr):int(b_s * sr)].mean(axis=1)
    beeps = np.zeros(len(clip), dtype=np.float32)
    for n in notes:
        s = song_to_cover_time(alignment, [n["start_s"]])[0] - a_s
        if np.isnan(s):
            continue
        t = np.arange(int(n["duration_s"] * sr)) / sr
        hz = float(midi_to_hz(n["midi_pitch"] + tuning_cents / 100))
        tone = (np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(4 * np.pi * hz * t)) / 1.3
        tone *= np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        i = int(s * sr)
        if 0 <= i < len(beeps):
            beeps[i:i + len(tone)] += tone[:len(beeps) - i]
    clip_rms = np.sqrt(np.mean(clip ** 2)) + 1e-9
    beep_rms = np.sqrt(np.mean(beeps[beeps != 0] ** 2)) + 1e-9
    beeps *= CLIP_BEEP_TO_MIX * clip_rms / beep_rms
    left, right = clip, CLIP_MIX_IN_BEEP_EAR * clip + beeps
    peak = max(np.abs(left).max(), np.abs(right).max(), 1e-9)
    sf.write(path, np.column_stack([left, right]) * min(1.0, 0.9 / peak), sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="every song in songs.csv, not just the pilot")
    parser.add_argument("--self-score", action="store_true")
    parser.add_argument("--clips", type=int, default=0, help="best and worst phrases to export as clips")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    songs = pd.read_csv(config.REAL_DATA_DIR / "songs.csv")
    if not args.all:
        songs = songs[songs["folder"].isin(config.PILOT_SONGS)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.self_score:
        table = self_scores(songs)
        print("Originals scored against their own target notes (flag = below "
              f"{config.SELF_SCORE_MIN_ACCURACY}):")
        print(table.round(2).to_string(index=False))

    summary = pd.read_csv(config.DERIVED_DIR / "alignment" / "summary.csv")
    rows, alignments = [], {}
    for song in songs.itertuples(index=False):
        for video_id in summary.loc[(summary.song_id == song.song_id) & summary.aligned, "video_id"]:
            df, alignments[video_id] = score_cover(song, video_id)
            rows.append(df.assign(folder=song.folder))
    scores = pd.concat(rows, ignore_index=True)
    labels = ["accuracy", "accuracy_strict", "octave_error_rate", "mean_cents_error",
              "timing_accuracy", "mean_onset_error_ms", "missed_note_rate"]
    before = scores.groupby("song_id")["accuracy"].median()

    failed = failed_by_every_cover(scores)
    failed.to_csv(OUT_DIR / "dropped_phrases.csv", index=False)
    keys = scores.set_index(["song_id", "phrase_idx"]).index
    scores = scores[~keys.isin(failed.set_index(["song_id", "phrase_idx"]).index)]
    scores.drop(columns="target_notes").to_csv(OUT_DIR / "cover_scores.csv", index=False)
    scores.to_parquet(OUT_DIR / "cover_scores.parquet", index=False)        # with target_notes
    print(f"\nDropped {len(failed)} phrases that nearly every cover fails -> {OUT_DIR / 'dropped_phrases.csv'}")
    print(failed.groupby("song_id").size().reindex(before.index, fill_value=0).rename("dropped").to_string())
    unchecked = scores.groupby("song_id")["singer_id"].nunique() < config.CONSENSUS_MIN_COVERS
    if unchecked.any():
        print(f"  (fewer than {config.CONSENSUS_MIN_COVERS} covers, cannot be checked: "
              f"{', '.join(unchecked[unchecked].index)})")
    print(f"\n{scores.singer_id.nunique()} covers, {len(scores)} phrases kept -> {OUT_DIR / 'cover_scores.csv'}")
    table = scores.groupby("song_id")[labels].median()
    table.insert(0, "accuracy_before_drop", before)
    print(table.round(2).to_string())

    if args.clips:
        clip_dir = OUT_DIR / "clips"
        clip_dir.mkdir(exist_ok=True)
        ranked = scores[scores["n_scored"] >= 100].sort_values("accuracy")
        picks = [("worst", ranked.head(args.clips)), ("best", ranked.tail(args.clips)[::-1])]
        song_by_id = songs.set_index("song_id")
        print("\nClips to check by ear:")
        for kind, part in picks:
            for i, row in enumerate(part.to_dict("records"), start=1):
                song = song_by_id.loc[row["song_id"]]
                name = f"{kind}{i}_acc{row['accuracy']:.2f}_{row['song_id']}_p{row['phrase_idx']}_{row['singer_id']}.wav"
                write_clip(clip_dir / name, song["folder"], song["tuning_cents"], row,
                           alignments[row["singer_id"]])
                print(f"  {name}")


if __name__ == "__main__":
    main()
