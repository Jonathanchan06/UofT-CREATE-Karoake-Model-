# Karaoke ML Coach: ML model context

This file gives Claude Code the context for the ML side of a UofT CREATE team project.
Read it before writing code. Ask before changing anything listed under "Fixed decisions".

## Project in one paragraph

A karaoke app that predicts how accurately a specific person will sing each phrase of a song
**before they sing it**, then recommends the best key (transposition from -5 to +5 semitones)
and flags the hardest phrases. The app is meant for **any user**: a shared model learns what makes
phrases hard for people in general, and a per-user offset is learned as each user sings more (model
ladder step 4). A new user starts with a quick range sweep and an offset of 0. Online cover
recordings are the candidate data for the shared model; the six team members are the first users
and the test group for personalization. The ML team (this code) builds the predictor. The data team
builds the audio pipeline that produces labeled training data.

## Where this repo sits in the pipeline

1. **Data team (separate code, already cloned here for reference):** Demucs separates vocals from
   the original song; basic-pitch (or a MIDI file) turns them into **target notes**, the answer key.
2. **Data team:** each recording (team or cover) is pitch-tracked with **PESTO** (not CREPE), aligned
   to the target notes, and scored per phrase by a rule-based script. Those scores are the **labels**.
   Covers need Demucs first to remove the backing track.
3. **ML team (this code):** load labeled phrases, compute features from the target notes, train
   and evaluate models, and expose a `predict` function for the app team.

Do not modify the data team's Demucs code. The ML code should depend only on the data format below.

## Current status (Nov 2026)

- Cover recordings have been collected (see "Cover recordings" below), but **no real labels exist
  yet**: there are no target notes and no scoring script. Keep building on fake data, in the exact
  real format, so swapping in the real file later needs no code changes.
- Team recordings do not exist yet.
- Baseline, linear regression and XGBoost are built and tested on fake data. The hierarchical model
  (ladder step 4) and the leave-one-singer-out test are **not built**.
- No large public amateur karaoke dataset was found (DAMP is not available), so plan for **small
  data**. Do not design around pretraining.

## Cover recordings (`covers_data/`)

Online covers, mostly Smule, downloaded from YouTube. One folder per song (`<artist> - <song>/`), each
with WAVs and a `manifest.csv` (`file, id, url, title, uploader, subscribers, views, duration`), plus
`batch_summary.csv` at the top. All WAVs are 44.1 kHz, stereo, 16-bit PCM: about 1,017 files, 101
songs, 62.5 hours, 40 GB.

**Handling rules:**
- Never commit or publish this data. It is copyrighted audio of identifiable private people.
- Use anonymous `singer_id`s in anything derived from it, not uploader names.
- Do not train on it until the team confirms it is allowed (supervisor or ethics contact). **Open.**

**Clean up before Demucs** (do not delete originals; list exclusions in a file):
- Merge `EdSheeran - Perfect` into `Ed Sheeran - Perfect` (they share video `9g_fTJwjJUc`).
- Exclude `Mr. Rabbit - I Will Survive` (99% silent).
- Check by ear: `Daiwey1` (Sk8er Boi), `MINIMALISMO MUSICAL` (Girls Just Want to Have Fun), and the
  `Sorry - Justin Bieber dance cover`.

**Problems the scoring script must handle before these are valid labels:**
- **Target notes:** none exist, and the originals or MIDI files are not in `covers_data/`.
- **Key:** covers may be in a different key and the manifest has no transpose, so estimate it per cover.
- **Alignment:** 205 recordings differ by over 20% from their song's median length (excerpts or
  rearrangements), so one fixed time offset is not enough. Reject recordings that do not align.
- **Multiple voices:** Smule duets are common and often unlabelled. PESTO tracks one pitch and Demucs
  puts both singers in one stem, so duets must be detected and rejected, not trusted from titles.
- **Pitch correction:** Smule offers it, so accuracy may be too high. Check before trusting these labels.
- **Singer range:** no sweeps exist; estimate from the notes each singer sings accurately.
- **Singer identity:** an uploader is not always one person. Only 75 of 892 uploaders appear on 2+
  songs and 11 on 4+, so this data can train the shared model but cannot test personalization.

Start with a pilot of about 5 songs and check the scores by ear before processing the full batch.

## Data format (the contract with the data team)

One row = one phrase, from one recording, by one singer. Stored as Parquet (`phrases_vX.Y.parquet`).

| Column | Type | Meaning |
|---|---|---|
| `song_id` | str | Song identifier. **Train/test splits are grouped by this.** |
| `singer_id` | str | Who sang it. Also used as a feature / singer offset key. |
| `take_id` | int | Attempt number for that singer and song. |
| `transpose` | int | Semitones shifted from the original key (e.g. -2). Estimated for covers. |
| `phrase_idx` | int | Position of the phrase in the song, from 0. |
| `target_notes` | list of (midi_pitch: int, start_s: float, duration_s: float) | Notes that should be sung, already shifted by `transpose`. |
| `accuracy` | float 0-1 | **Main label.** Share of voiced frames within +/-50 cents of the target. |
| `mean_cents_error` | float | Backup label. Mean absolute pitch error in cents (100 cents = 1 semitone). |

Planned timing labels (the team wants timing to count too; support them as optional columns):
`timing_accuracy` (share of notes started within ~100 ms), `mean_onset_error_ms` (signed),
`missed_note_rate`. Code should work whether or not these columns exist.

Supporting files:
- `singers.csv`: `singer_id`, `range_low_midi`, `range_high_midi` (comfortable range from a sweep;
  for covers, estimated from the notes the singer sings accurately)
- `songs.csv`: `song_id`, `tempo_bpm`, `original_key`, `language`, `genre`
- `melodies/<song_id>.json`: `{"song_id", "tempo_bpm", "phrases": [[{"midi_pitch", "start_s",
  "duration_s"}, ...], ...]}` in the original key, for every song including ones nobody has recorded

Useful conversions: `midi = 69 + 12 * log2(hz / 440)`; MIDI 60 = middle C.

## Fixed decisions

- **Features come only from `target_notes` (and singer range / song metadata), never from the
  recording.** The model must predict songs nobody has sung yet.
- **Split by song.** Use `GroupKFold(groups=song_id)`. Never split phrases of the same song across
  train and test. Hold out ~15-20% of songs as a final test set touched once at the end.
- **Every model must beat the baseline** (predict each singer's mean accuracy from training data).
- Log every run (dataset version, settings, metrics) with Weights & Biases or MLflow.

## Features (draft, computed per phrase after applying transpose)

Range (relative to the singer's comfortable range):
`max_above_range`, `max_below_range`, `pct_outside_range`, `mean_pitch_rel`, `pitch_span`

Intervals: `biggest_leap_up`, `biggest_leap_down`, `mean_abs_interval`, `num_leaps` (> 5 semitones)

Rhythm: `notes_per_sec`, `shortest_note`, `longest_note`, `tempo`, `phrase_duration`, `num_notes`

Context: `phrase_position` (phrase_idx / total phrases), `singer_id` (categorical)

## Model ladder (each step must beat the previous on unseen songs)

1. **Baseline:** singer's mean accuracy.
2. **Linear regression** on features (sanity check, interpretable).
3. **XGBoost** (`objective="reg:logistic"`, `enable_categorical=True`), tuned with Optuna inside
   GroupKFold. Explain with SHAP.
4. **Shared model + singer offsets (hierarchical / partial pooling):** shared model for melody
   difficulty plus a shrunk per-singer offset (statsmodels MixedLM random intercept, or Ridge on
   singer dummies). Evaluate with a **leave-one-singer-out** test, giving the held-out singer
   0, 1, 2, 4, 8 songs. This is how the app serves new users: 0 songs = a brand-new user with only a
   range sweep, and the curve shows how fast predictions improve as they sing.
5. Optional later: small PyTorch LSTM over note sequences with singer embeddings. Only keep it if it
   beats steps 3-4; it is data hungry and low priority.

## Metrics

- MAE of predicted accuracy (main)
- Spearman correlation (does the model rank phrases/songs correctly)
- Key recommendation: is the predicted best key within +/-1 semitone of the actual best key
  (needs transposed takes)
- Learning curve: test MAE vs number of training songs (tells the data team how much to record)

## Key recommendation

For a song and singer, shift all target notes by k for k in -5..+5, recompute features, predict
per phrase, average (optionally penalize the worst phrase), return the best k.

## Interface for the app team

```python
predict(song_notes: list[list[tuple[int, float, float]]], singer_id: str, transpose: int) -> list[float]
# song_notes: one list of (midi, start_s, duration_s) per phrase, in the original key
# returns predicted accuracy per phrase
recommend_key(song_id, singer_id) -> dict  # best transpose + predicted accuracy per key
# looks up notes and tempo in melodies/<song_id>.json, so the app only passes ids
load_song(song_id) -> dict  # {"song_id", "tempo_bpm", "phrases"} (phrases as above)
recommend_key_from_notes(song_notes, singer_id, tempo_bpm=None) -> dict  # same, from raw notes
```

Save trained models with joblib.

## Fake data generator (build this first)

Generate data in the exact format above with **planted patterns** the model should recover:
- 6 singers with different comfortable ranges and different skill offsets (e.g. +/-0.1 accuracy)
- ~40 songs, ~15-20 phrases each, 4-10 notes per phrase, MIDI pitches roughly 50-80
- Some songs sung by every singer, others by only one or two
- Some takes at transpose -2 and +2
- Accuracy drops ~4% per semitone above the singer's range, ~2% per semitone below it,
  ~1.5% per semitone of biggest upward leap beyond 4, and with notes_per_sec above ~3;
  add Gaussian noise (sd ~0.05); clip to [0, 1]
- `mean_cents_error` roughly `(1 - accuracy) * 100` plus noise
- Also write matching `singers.csv` and `songs.csv`

If a model cannot recover planted patterns (e.g. SHAP does not show `max_above_range` as
important), there is a bug.

## Suggested repo layout

```
ml/
  data/fake/            # generated fake data
  src/
    make_fake_data.py
    features.py         # target_notes -> feature row
    evaluate.py         # GroupKFold, holdout, metrics, learning curve
    models/
      baseline.py
      xgb.py
      hierarchical.py
    recommend.py        # predict() and recommend_key()
  notebooks/            # exploration only
  tests/                # unit tests for features and splits
  requirements.txt
```

## Tech stack

Python, pandas, pyarrow, NumPy, scikit-learn, XGBoost, statsmodels, Optuna, SHAP, SciPy,
matplotlib, joblib, Weights & Biases or MLflow. PyTorch only for the optional step 5.

## Working style

- Keep code simple and readable; the team must be able to explain every line in interviews.
- Write unit tests for feature extraction and for the song-level split (check no song appears in
  both train and test).
- Be suspicious of results that look too good; check for leakage first.

## First tasks

1. `make_fake_data.py` producing `phrases_v0.0.parquet`, `singers.csv`, `songs.csv`
2. `features.py` with unit tests on a few hand-checked phrases
3. `evaluate.py` with GroupKFold by song, holdout set, MAE and Spearman
4. Baseline, then linear regression, then XGBoost; confirm XGBoost recovers the planted patterns
5. Learning curve plot
6. `recommend.py` with `predict` and `recommend_key`
