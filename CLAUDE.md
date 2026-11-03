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
   the original song; PESTO plus note segmentation turns them into **target notes**, the answer key.
   (basic-pitch may be added later as an alternative note maker; see "Labeling decisions".)
2. **Data team:** each recording (team or cover) is pitch-tracked with **PESTO** (not CREPE), aligned
   to the target notes, and scored per phrase by a rule-based script. Those scores are the **labels**.
   Covers need Demucs first to remove the backing track.
3. **ML team (this code):** load labeled phrases, compute features from the target notes, train
   and evaluate models, and expose a `predict` function for the app team.

Do not modify the data team's Demucs code. The ML code should depend only on the data format below.

## Current status (Nov 2026)

- Cover recordings have been collected and separated with Demucs (`separated/htdemucs/`; originals
  not yet separated), but **no real labels exist yet**: there are no target notes and no scoring
  script. Labeling decisions are below. Target notes exist for the 5 pilot songs (beep check passed by
  ear). Two-pass alignment is built, trialled on the 50 pilot covers and **approved** (2026-11-03);
  it needs a fine-tuning step next. The scorer is not built yet. See "Next steps" below. Keep building on fake data, in the exact
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
- Training on it was **approved** by the team (supervisor / ethics contact) on 2026-11-03. The rules
  above (no commits, anonymous ids) still apply.

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

## Original songs (`originals_data/`)

One `original.wav` per song folder plus `manifest.csv` (YouTube Music sources). These are the
source of the target notes: Demucs vocal stem -> PESTO -> notes. Same handling rules as the covers.
`Whitney Houston - I Will Always Love You` is listed but its WAV is missing: **left out** for now.

## Labeling decisions (agreed 2026-11-03)

- **Target notes:** PESTO on the original's vocal stem plus note segmentation (split at pitch changes
  and at loudness bumps, so repeated notes on one pitch stay separate). The note maker sits behind
  one function so it can be swapped.
- **basic-pitch: possible later feature, not needed now.** It finds note onsets directly, gives a
  second tracker independent of PESTO, shows backing-vocal lines, and exports editable MIDI. It does
  not install on the venv's Python 3.13 (needs a separate 3.10/3.11 env). Revisit if the pilot's
  beep check shows PESTO's notes are poor.
- **PESTO resolution:** its output snaps to a 33.3-cent grid (3 bins per semitone, very peaked
  softmax). Averaging 3 copies pitch-shifted by 0 / 11.1 / 22.2 cents (by passing a slightly higher
  sample rate) halves the error on known tones (mean 8.7 -> 4.5 cents); softmax temperature barely
  helps. Costs 3x PESTO time. PESTO timestamps are in **milliseconds**.
- **Tests are split:** scorer-only tests on synthetic pitch arrays must recover known errors exactly;
  tests through PESTO use tolerances (about +/-17 cents without the shifted copies, +/-20 ms).
- **Tuning:** each song gets `tuning_cents` (offset of the original from A440, measured from its
  vocal stem) as an extra column in `songs.csv`, applied when scoring. The "own vocal stem scores in
  the 90s" check is circular (targets come from that stem), so it is a per-song flag (threshold in
  config), and pilot songs also get a **beep check**: target notes rendered as tones over the original,
  checked by ear.
- **Octaves:** the recording-level octave goes into `transpose` (e.g. -12 for a man singing a song an
  octave down). Within a phrase, short octave jumps (tracker glitches, about <= 150 ms, configurable) are
  folded back and count as correct; held notes in the wrong octave count as wrong. Both go into
  `octave_error_rate`. Also write `accuracy_strict` (no folding) next to `accuracy`.
- **Singer range:** estimating range from notes sung accurately leaks the label into the range
  features (most cover singers have one song). Use **voice-type ranges** for covers (a few standard
  ranges picked from the singer's median sung pitch) and **sweeps** for the team. Before trusting any
  real result, run the **shuffle test**: retrain with ranges shuffled between singers; if scores
  barely drop, range was not what worked, and if they collapse, look for leakage.
- **phrase_position** uses the phrase count from `melodies/<song_id>.json` (`load_dataset` reads it),
  so rows rejected at the end of a song do not change it.
- **Alignment (two-pass, approved 2026-11-03):** a single DTW on the singer's pitch would bend to
  hide wrong notes and mix singer timing with recording differences, so:
  - *Pass 1* aligns the cover's `no_vocals` to the original's `no_vocals` (chroma, 100 ms frames,
    subsequence DTW in all 12 keys). Gives a time map and the key, independent of the singer.
    Rejected when path cost / random-pair cost > 0.5 or the map's tempo ratio is outside 1 +/- 0.1.
  - *Pass 2* looks within +/-300 ms of each mapped note for where the singer started it (octave-folded,
    within 1 semitone); the offset is the timing error.
  - Pilot (50 covers): 34 aligned (Adele 10, ABBA 9, Bruno Mars 7, a-ha 6, Coldplay 2); the rest are
    different arrangements (acoustic, piano-only) and need the single-DTW fallback or are dropped.
    Pass 2 on the originals' own stems finds 96-98% of onsets at 10-40 ms median error; on aligned
    covers 60% at 90 ms, part singer, part pass 1's 100 ms map, not yet separated. Check timing labels
    before trusting them.
  - librosa's subsequence DTW returns a wrong path when the query is longer than the reference, so
    `align._dtw` always puts the shorter recording first.
  - Cover beep check (user, by ear): aligned covers sound "generally right but not 1:1". Measured with
    backing-track onsets in 4 s windows, pass 1's map is off by a median 30-190 ms locally (90% within
    180-430 ms; noisy measure, beat-period ambiguity). Fine for pitch where pass 2 finds the onset,
    **not good enough for timing labels** (error as big as the +/-100 ms window).
  - Missing beeps: 6-26% of a cover's sung time has no target note. Mostly (56-89%) the original's
    lead is silent there (ad-libs, harmonies, duet partner, humming), which is correctly unscored; the
    rest (11-44%) are notes the note maker dropped (< 80 ms, low confidence, ABBA's stacked harmonies).
- **Tempo** comes from librosa's beat tracker on the original's backing. It doubles slow ballads
  (Someone Like You 136, Fix You 144 instead of ~70); not fixed yet.

## Labeling code (`labeling/`, run from the project folder)

| File | Purpose |
|---|---|
| `config.py` | every rule and threshold, units in the names; `PILOT_SONGS`, `LEFT_OUT_SONGS` |
| `pitch.py` | PESTO, every frame kept (`time_s`, `hz`, `midi`, `confidence`, `voiced`), 3 shifted copies |
| `targets.py` | pitch -> `tuning_cents` -> notes -> phrases; `song_id_for()` slugs folder names |
| `align.py` | pass 1 (`backing_chroma`, `align_backing`) and pass 2 (`find_onsets`, `recording_octave`) |
| `separate_originals.py` | Demucs on `originals_data/` -> `separated/originals/htdemucs/<song>/` |
| `make_targets.py` | writes `ml/data/real/melodies/*.json` and `ml/data/real/songs.csv` |
| `beep_check.py` | original + target beeps -> `derived/beep_check/` |
| `align_covers.py` | both passes on every cover -> `derived/alignment/` (+ `summary.csv`, `--beeps N`) |
| `tests/` | `py -m pytest labeling` (22 tests); `py -m pytest ml` (24 tests) |

Order: `separate_originals.py` -> `make_targets.py` -> `beep_check.py` -> `align_covers.py`
(each takes `--all`; default is the 5 pilot songs). `derived/` and `ml/data/real/` are gitignored.
Caches: `derived/pitch/{originals,covers}/` (PESTO), so re-runs are fast.

## Next steps (agreed 2026-11-03, in order)

1. **Fine-tuning step for pass 1.** Every few seconds, adjust the time map by cross-correlating the
   backing tracks' onset envelopes (librosa `onset_strength`, 10 ms frames) within about +/-300 ms;
   smooth the corrections; keep it singer-independent. Done when: a synthetic test with a known
   offset passes, the pass-2 onset error on aligned covers drops well below ~90 ms, and re-rendered
   cover beep files (same 7 aligned covers in `derived/beep_check/covers/`) sound in sync to the user.
2. **Scorer** (`labeling/score.py`): per phrase `accuracy`, `accuracy_strict`, `mean_cents_error`,
   `octave_error_rate`, `timing_accuracy`, `mean_onset_error_ms`, `missed_note_rate`, using the rules
   in "Labeling decisions" (`tuning_cents`, octave rules, skip first 50 ms of each note, ~100 ms
   smoothing, confidence threshold), all in `config.py`. Exact unit tests on synthetic pitch tracks.
   Phrases with too few voiced frames are dropped and logged (NaN labels break training).
3. **Labeling accuracy checks** (need the scorer):
   - *Synthetic singer:* render a voice from a pilot song's target notes with planted errors (notes
     +30 / +60 cents, notes 80 ms late, skipped notes, one phrase an octave down), mix over the
     original's backing, run the whole pipeline as if it were a cover; labels must match the plants
     within PESTO tolerances. This is the main test.
   - *Self-score:* each original's vocal stem vs its own targets; flag songs below 0.85.
   - *By ear:* the user listens to a few real cover phrases with very high and very low scores.
4. **Pilot dataset:** `phrases_v1.0.parquet` in the contract format, `songs.csv`, `singers.csv` with
   anonymous `singer_id`s (hash of uploader, mapping kept out of Git) and voice-type ranges. Fix the
   tempo doubling. Add simple duet and pitch-correction flags (see "Cover recordings").
5. **Test run through `ml/`** on the pilot data: only checks the plumbing (5 songs = 1 song per fold,
   metrics are meaningless).
6. **Scale up:** `--all` for every step (about 25 min Demucs, 4-5 h PESTO, 1.5 h alignment), spot-check
   more beep files, then the real model run with the **shuffle test** before trusting any result.

Optional, only if step 6 is short of songs: single-DTW fallback for covers whose backing does not match
(16 of 50 in the pilot), marked in a column.

Housekeeping still open: fix `program.py` (empty paths crash, outputs overwrite each other,
timestamps are ms but unlabelled, it deletes low-confidence frames; make it a thin wrapper around
`labeling/pitch.py`), update README.md with the labeling pipeline, consider converting
`requirements.txt` from UTF-16 to UTF-8 (Git shows it as binary). Nothing from 2026-11-03 is committed yet.

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
