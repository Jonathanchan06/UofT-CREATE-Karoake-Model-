# Pipeline (glue script)

One command from a folder of songs and recordings to the dataset parquet.

```
python pipeline/run_pipeline.py --data data --melodies ml/data/fake/melodies ^
    --singers ml/data/singers.csv --songs ml/data/songs.csv
```

## Folder layout

```
data/
  songs/perfect_ed_sheeran.mp3
  recordings/
    arshiya/perfect_ed_sheeran_t1_k0.wav     take 1, sung 0 semitones from the original key
    arshiya/perfect_ed_sheeran_t2.wav        no _k, so the scorer estimates the key
    arshiya/perfect_ed_sheeran_t1_k0.csv     a pitch CSV instead of audio: skips PESTO
```

## What it does

| Step | Tool | Saved to `work/` |
|---|---|---|
| song to vocals | Demucs | `separated/<song>/vocals.wav` |
| vocals to answer key | melody/extract_melody.py | `melodies/<song>.json`, `melody_checks/` |
| recording to pitch | PESTO | `pitch/<singer>__<file>.csv` |
| pitch + answer key to labels | scorer/score.py | `phrases.parquet` (the `--out` file) |

Every run also writes `work/pipeline_report.csv`: one row per song and per take, with status,
phrases kept and dropped, accuracy, QC flags, and a reason for any failure or skip. Singers and
songs missing from `singers.csv` / `songs.csv` show up as warnings.

## Rules

- A step is skipped if its output already exists and is newer than its input. `--force` redoes all.
- One bad file never stops the batch. It is logged with its reason and the rest carry on.
- `_t<take>` and `_k<key>` in a file name are optional. Without `_k` the scorer estimates the key.
- `--melodies DIR` uses ready-made answer keys (such as `ml/data/fake/melodies`) and skips Demucs.
- The dataset is rebuilt from every good take on each run. Scoring is fast, so there is no cache.

## Try it with no audio

`fake_singer.py` turns any answer key into a fake PESTO CSV (late start, wobble, wrong notes), so
the whole flow runs on the fake data already in the repo. The tests do exactly this, with fake
Demucs, basic-pitch and PESTO swapped in through the `Tools` class.

```
python -m pytest pipeline
```

## Known limits

- The real Demucs and PESTO calls have not been run end to end yet (they need audio and a GPU or
  patience). `pesto_track` copies `program.py`; check the first CSV it writes.
- Demucs is run with `--two-stems=vocals`; the vocals file is copied into `work/`.