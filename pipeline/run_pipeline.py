"""One command from a folder of songs and recordings to the dataset parquet.

    python pipeline/run_pipeline.py --data data

    data/songs/<song_id>.mp3                                   the original songs
    data/recordings/<singer_id>/<song_id>_t<take>_k<key>.wav   one file per take
                                                               (_t and _k are optional)
    A recording can also be a pitch CSV (already tracked), which skips the PESTO step.

Stages, each saved under --work so a re-run only does what is missing:
    song     -> Demucs -> work/separated/<song_id>/vocals.wav
             -> melody extractor -> work/melodies/<song_id>.json   (the answer key)
    take     -> PESTO -> work/pitch/<singer>__<name>.csv
    pitch+key -> scorer -> rows in the output parquet

Demucs, basic-pitch and PESTO are heavy, so they sit behind small functions in `Tools`. Tests swap in
fakes; the real ones are the defaults.
"""

import argparse
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scorer"))
sys.path.insert(0, str(ROOT / "melody"))

AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
NAME = re.compile(r"^(?P<song>.+?)(?:_t(?P<take>\d+))?(?:_k(?P<key>-?\d+))?$")


# ---- the two heavy tools, behind small swappable functions ---------------------------------
def demucs_separate(song: Path, out_dir: Path) -> Path:
    """Run Demucs, return the path of the vocals-only file."""
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(out_dir), str(song)],
                   check=True)
    found = list(out_dir.glob(f"*/{song.stem}/vocals.wav"))
    if not found:
        raise FileNotFoundError("Demucs ran but no vocals.wav was found")
    return found[0]


def pesto_track(recording: Path, out_csv: Path) -> None:
    """Pitch-track a recording with PESTO. Same output as program.py."""
    import numpy as np
    import pesto
    import torchaudio

    x, sr = torchaudio.load(str(recording))
    timesteps, pitch, confidence, _ = pesto.predict(x.mean(dim=0), sr)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, np.column_stack((timesteps, pitch, confidence)), delimiter=",",
               header="timestamp,frequency,confidence", comments="", fmt=["%.4f", "%.3f", "%.3f"])


def basic_pitch_melody(song_id: str, vocals: Path, song: Path, out_dir: Path, check_dir: Path) -> Path:
    """Turn a vocals file into the answer-key JSON (melody/extract_melody.py)."""
    import extract_melody

    path, _ = extract_melody.extract(
        song_id, out_dir, audio=vocals, tempo_audio=song, check_dir=check_dir)
    return Path(path)


@dataclass
class Tools:
    # (song_audio, out_dir) -> vocals_path
    separate: Callable = demucs_separate
    # (song_id, vocals, song, out_dir, check_dir) -> json path
    make_melody: Callable = basic_pitch_melody
    track_pitch: Callable = pesto_track    # (recording, out_csv) -> None


def is_fresh(output: Path, *inputs: Path) -> bool:
    """True if `output` exists and is at least as new as every input (like a build tool)."""
    return output.exists() and all(output.stat().st_mtime >= i.stat().st_mtime for i in inputs)


def parse_name(stem: str):
    """'perfect_t2_k-2' -> ('perfect', 2, -2). Missing parts give take 1 and key None."""
    m = NAME.match(stem)
    return m["song"], int(m["take"] or 1), (int(m["key"]) if m["key"] is not None else None)


def short(err: Exception) -> str:
    return f"{type(err).__name__}: {err}"[:200]


def build_melody(song: Path, work: Path, tools: Tools, force: bool) -> Path:
    song_id = song.stem
    melody = work / "melodies" / f"{song_id}.json"
    if not force and is_fresh(melody, song):
        return melody
    vocals = work / "separated" / song_id / "vocals.wav"
    if force or not is_fresh(vocals, song):
        made = Path(tools.separate(song, work / "separated" / "_demucs"))
        vocals.parent.mkdir(parents=True, exist_ok=True)
        vocals.write_bytes(made.read_bytes())
    tools.make_melody(song_id, vocals, song, work /
                      "melodies", work / "melody_checks")
    return melody


def pitch_for(recording: Path, singer: str, work: Path, tools: Tools, force: bool) -> Path:
    if recording.suffix.lower() == ".csv":
        return recording
    out = work / "pitch" / f"{singer}__{recording.stem}.csv"
    if force or not is_fresh(out, recording):
        tools.track_pitch(recording, out)
    return out


def run(data, work="work", out="phrases.parquet", extra_melodies=None, singers_csv=None,
        songs_csv=None, force=False, tools=None):
    import score

    data, work, out, tools = Path(data), Path(
        work), Path(out), tools or Tools()
    report, frames = [], []

    # songs -> answer keys
    melodies = {}
    if extra_melodies:
        melodies = {p.stem: p for p in sorted(
            Path(extra_melodies).glob("*.json"))}
    for song in sorted(p for p in (data / "songs").glob("*") if p.suffix.lower() in AUDIO):
        try:
            melodies[song.stem] = build_melody(song, work, tools, force)
            report.append({"kind": "song", "item": song.stem, "status": "ok"})
        except Exception as e:  # one bad song must not stop the batch
            report.append({"kind": "song", "item": song.stem,
                          "status": "failed", "note": short(e)})

    # recordings -> labeled rows
    files = sorted(p for p in (data / "recordings").glob("*/*")
                   if p.suffix.lower() in AUDIO | {".csv"})
    for rec in files:
        singer = rec.parent.name
        song_id, take, key = parse_name(rec.stem)
        item = f"{singer}/{rec.name}"
        if song_id not in melodies:
            report.append({"kind": "take", "item": item, "status": "skipped",
                           "note": f"no answer key for song '{song_id}'"})
            continue
        try:
            pitch = pitch_for(rec, singer, work, tools, force)
            df = score.score_take(
                pitch, melodies[song_id], singer, take_id=take, transpose=key)
            kept, dropped = len(df), df.attrs.get("dropped_phrases", 0)
            if kept == 0:
                report.append({"kind": "take", "item": item, "status": "failed", "phrases_dropped": dropped,
                               "note": "no phrase could be scored (silent take or wrong song?)"})
                continue
            frames.append(df)
            report.append({"kind": "take", "item": item, "status": "ok", "phrases_kept": kept,
                           "phrases_dropped": dropped, "transpose": int(df["transpose"].iloc[0]),
                           "mean_accuracy": round(float(df["accuracy"].mean()), 3),
                           "qc_flag": df["qc_flag"].iloc[0]})
        except Exception as e:
            report.append({"kind": "take", "item": item,
                          "status": "failed", "note": short(e)})
            if "--debug" in sys.argv:
                traceback.print_exc()

    if frames:
        dataset = pd.concat(frames, ignore_index=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(out)

    # the ML code also needs these two tables, so say what is missing
    for path, column, ids, label in ((singers_csv, "singer_id", {r.parent.name for r in files}, "singers.csv"),
                                     (songs_csv, "song_id", set(melodies), "songs.csv")):
        if path and Path(path).exists():
            missing = sorted(ids - set(pd.read_csv(path)[column].astype(str)))
            for m in missing:
                report.append({"kind": "missing_metadata", "item": m, "status": "warning",
                               "note": f"not in {label}"})

    rep = pd.DataFrame(report).convert_dtypes()
    work.mkdir(parents=True, exist_ok=True)
    rep.to_csv(work / "pipeline_report.csv", index=False)
    return rep, (pd.concat(frames, ignore_index=True) if frames else None)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="folder with songs/ and recordings/")
    ap.add_argument("--work", default="work",
                    help="where intermediate files go (default: work)")
    ap.add_argument("--out", default="phrases.parquet",
                    help="dataset to write")
    ap.add_argument(
        "--melodies", help="folder of ready-made melody JSON files (skips Demucs for those songs)")
    ap.add_argument(
        "--singers", help="singers.csv, to warn about missing singers")
    ap.add_argument("--songs", help="songs.csv, to warn about missing songs")
    ap.add_argument("--force", action="store_true", help="redo every stage")
    ap.add_argument("--debug", action="store_true",
                    help="print the full error for failed takes")
    a = ap.parse_args()
    rep, dataset = run(a.data, a.work, a.out, a.melodies,
                       a.singers, a.songs, a.force)
    print(rep.fillna("").to_string(index=False))
    print(f"\n{(rep.status == 'ok').sum()} ok, {(rep.status == 'failed').sum()} failed, "
          f"{(rep.status == 'skipped').sum()} skipped, {(rep.status == 'warning').sum()} warnings")
    print(
        f"dataset: {len(dataset)} phrase rows -> {a.out}" if dataset is not None else "no dataset written")
    print(f"report:  {Path(a.work) / 'pipeline_report.csv'}")


if __name__ == "__main__":
    main()
