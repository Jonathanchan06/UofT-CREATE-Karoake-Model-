"""Build the labeled dataset in the contract format (CLAUDE.md "Data format"), so ml/ can load it.

Reads derived/scores/cover_scores.parquet (score_covers.py) and writes ml/data/real/:
    phrases_<DATASET_VERSION>.parquet   one row per phrase per cover; contract columns typed like
                                        the fake data, plus extra label and quality columns
    singers.csv                         singer_id, range_low_midi, range_high_midi, range_source
                                        (every cover singer: config.COVER_RANGE_MIDI)
    songs.csv                           tempo folded into one band, original_key, language added
    melodies/                           already there (make_targets.py)

Singers are anonymous: each uploader gets a stable random id (c0001, ...). The uploader -> id map and
the recording -> video map live in derived/private/, outside the dataset folder, and are never shared.

Usage:
    python labeling/build_dataset.py
"""

import csv
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from align import to_song_timeline

sys.path.insert(0, str(config.ROOT / "melody"))
from extract_melody import Note, guess_key  # noqa: E402

CONTRACT = [("song_id", pa.string()), ("singer_id", pa.string()), ("take_id", pa.int64()),
            ("transpose", pa.int64()), ("phrase_idx", pa.int64()),
            ("target_notes", pa.list_(pa.struct([("midi_pitch", pa.int32()), ("start_s", pa.float64()),
                                                 ("duration_s", pa.float64())]))),
            ("accuracy", pa.float64()), ("mean_cents_error", pa.float64()),
            ("timing_accuracy", pa.float64()), ("mean_onset_error_ms", pa.float64()),
            ("missed_note_rate", pa.float64())]
EXTRA = ["accuracy_strict", "octave_error_rate", "phrase_octave", "n_scored", "bias_cents",
         "duet_flag", "pitch_correction_flag"]


# ---------- singers ----------

def uploaders():
    """{video_id: uploader} from every covers_data manifest."""
    out = {}
    for manifest in sorted((config.ROOT / "covers_data").glob("*/manifest.csv")):
        with open(manifest, encoding="utf-8-sig", newline="") as f:
            out.update({row["id"]: row["uploader"] for row in csv.DictReader(f)})
    return out


def singer_ids(names, map_path, seed=0):
    """{uploader: singer_id}, stable across runs: ids already in map_path are kept, new uploaders get
    the next numbers in a random order (so ids do not follow the alphabet of the names)."""
    known = {}
    if map_path.exists():
        known = dict(pd.read_csv(map_path, dtype=str)[["uploader", "singer_id"]].values)
    new = sorted(set(names) - set(known))
    np.random.default_rng(seed + len(known)).shuffle(new)
    start = len(known) + 1
    known.update({name: f"c{start + i:04d}" for i, name in enumerate(new)})
    map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(known.items(), key=lambda kv: kv[1]), columns=["uploader", "singer_id"]).to_csv(map_path, index=False)
    return known


def cover_singers(singer_ids, cover_range=config.COVER_RANGE_MIDI):
    """singers.csv rows for cover singers: all get the same default range (no sweeps exist)."""
    low, high = cover_range
    return pd.DataFrame({"singer_id": sorted(set(singer_ids)), "range_low_midi": low,
                         "range_high_midi": high, "range_source": "default_cover"})


# ---------- songs ----------

def fold_tempo(bpm, band_min=config.TEMPO_BAND_MIN_BPM):
    """Halve or double into [band_min, 2*band_min): 136 -> 68, 172 -> 86, 99 -> 99."""
    if not bpm or np.isnan(bpm):
        return np.nan
    while bpm >= 2 * band_min:
        bpm /= 2
    while bpm < band_min:
        bpm *= 2
    return round(bpm, 1)


def original_key(melody):
    """Krumhansl key guess from the target notes (melody/extract_melody.guess_key), e.g. 'A major'."""
    notes = [Note(n["midi_pitch"], n["start_s"], n["start_s"] + n["duration_s"])
             for phrase in melody["phrases"] for n in phrase]
    key, mode = guess_key(notes)
    return f"{key} {mode}" if key else None


# ---------- quality flags ----------

def cover_flags(frames_on_song, melody, tuning_cents):
    """(share of singing where no target note is due, share of singing within PITCH_CORRECTION_CENTS
    of an exact semitone) for one cover, on the song timeline."""
    t = frames_on_song["time_s"].to_numpy()
    voiced = (frames_on_song["confidence"].to_numpy() >= 0.5) & frames_on_song["midi"].notna().to_numpy()
    due = np.zeros(len(t), dtype=bool)
    for phrase in melody["phrases"]:
        for n in phrase:
            due |= (t >= n["start_s"] - 0.15) & (t < n["start_s"] + n["duration_s"] + 0.15)
    outside = float((voiced & ~due).sum() / max(voiced.sum(), 1))
    cents = (frames_on_song["midi"].to_numpy()[voiced] - tuning_cents / 100) * 100
    on_semitone = float(np.mean(np.abs((cents + 50) % 100 - 50) <= config.PITCH_CORRECTION_CENTS)) if len(cents) else 0.0
    return outside, on_semitone


def write_phrases(df, path):
    """Parquet with the contract columns typed exactly like the fake data, then the extra columns."""
    table = pa.Table.from_pandas(df[[c for c, _ in CONTRACT]], schema=pa.schema(CONTRACT), preserve_index=False)
    for column in EXTRA:
        table = table.append_column(column, pa.array(df[column].to_numpy()))
    pq.write_table(table, path)


def main():
    real = config.REAL_DATA_DIR
    scores = pd.read_parquet(config.DERIVED_DIR / "scores" / "cover_scores.parquet")
    songs = pd.read_csv(real / "songs.csv")
    tuning = songs.set_index("song_id")["tuning_cents"]
    melodies = {s: json.loads((real / "melodies" / f"{s}.json").read_text()) for s in songs["song_id"]}

    # singers: anonymous ids, takes numbered per singer and song
    uploader_of = uploaders()
    ids = singer_ids(uploader_of.values(), config.PRIVATE_DIR / "singer_map.csv")
    scores = scores.rename(columns={"singer_id": "video_id"}).drop(columns="take_id")   # scorer's is always 1
    scores["singer_id"] = scores["video_id"].map(uploader_of).map(ids)
    recordings = scores[["song_id", "singer_id", "video_id"]].drop_duplicates().sort_values(["singer_id", "song_id", "video_id"])
    recordings["take_id"] = recordings.groupby(["singer_id", "song_id"]).cumcount() + 1
    recordings.to_csv(config.PRIVATE_DIR / "recordings.csv", index=False)
    scores = scores.merge(recordings, on=["song_id", "singer_id", "video_id"])

    # per cover: median sung pitch (kept privately, for reference) and the two quality flags
    rows = []
    for r in recordings.itertuples(index=False):
        alignment = json.loads((config.DERIVED_DIR / "alignment" / r.song_id / f"{r.video_id}.json").read_text())
        alignment = {k: np.array(v) if isinstance(v, list) else v for k, v in alignment.items()}
        frames = pd.read_parquet(config.DERIVED_DIR / "pitch" / "covers" / r.song_id / f"{r.video_id}.parquet")
        outside, on_semitone = cover_flags(to_song_timeline(frames, alignment), melodies[r.song_id], tuning[r.song_id])
        rows.append({"video_id": r.video_id, "singer_id": r.singer_id, "median_midi": frames["midi"].median(),
                     "sung_outside_targets": outside, "on_semitone_share": on_semitone,
                     "duet_flag": outside > config.DUET_FLAG_SHARE,
                     "pitch_correction_flag": on_semitone > config.PITCH_CORRECTION_FLAG_SHARE})
    covers = pd.DataFrame(rows)
    scores = scores.merge(covers[["video_id", "duet_flag", "pitch_correction_flag"]], on="video_id")

    singers = cover_singers(covers["singer_id"])
    singers.to_csv(real / "singers.csv", index=False)

    # songs: consistent tempo, key guess, language
    songs["tempo_bpm"] = [fold_tempo(t) for t in songs["tempo_bpm"]]
    songs["original_key"] = [original_key(melodies[s]) for s in songs["song_id"]]
    songs["language"] = songs["language"].fillna(config.DEFAULT_LANGUAGE)
    songs.to_csv(real / "songs.csv", index=False)
    for song_id, melody in melodies.items():             # melody files carry the same tempo
        melody["tempo_bpm"] = float(songs.set_index("song_id").loc[song_id, "tempo_bpm"])
        (real / "melodies" / f"{song_id}.json").write_text(json.dumps(melody, indent=1))

    path = real / f"phrases_{config.DATASET_VERSION}.parquet"
    write_phrases(scores.sort_values(["song_id", "singer_id", "take_id", "phrase_idx"]), path)

    print(f"{path}: {len(scores)} phrase rows, {scores.song_id.nunique()} songs, "
          f"{scores.singer_id.nunique()} singers, {len(recordings)} recordings")
    print(f"singers.csv: {len(singers)} cover singers, all on the default range {config.COVER_RANGE_MIDI}")
    print(f"songs.csv:\n{songs[['song_id', 'tempo_bpm', 'original_key', 'tuning_cents']].to_string(index=False)}")
    print(f"\nflags per cover: duet {covers.duet_flag.sum()} of {len(covers)}, "
          f"pitch correction {covers.pitch_correction_flag.sum()} of {len(covers)}")
    print(f"  sung outside target notes: median {covers.sung_outside_targets.median():.2f}, max {covers.sung_outside_targets.max():.2f}")
    print(f"  on-semitone share: median {covers.on_semitone_share.median():.2f}, max {covers.on_semitone_share.max():.2f}")
    covers.drop(columns="video_id").round(3).to_csv(config.PRIVATE_DIR / "cover_flags.csv", index=False)


if __name__ == "__main__":
    main()
