# Karaoke ML Coach: ML model context

This file gives Claude Code the context for the ML side of a UofT CREATE team project: rules,
decisions and the data format. **Progress, results and next steps are in `roadmap.md`**: read it at
the start of every session and update it when a step finishes. Read both before writing code. Ask
before changing anything listed under "Fixed decisions".

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
   the original song; PESTO plus note segmentation (`labeling/targets.py`) turns them into **target
   notes**, the answer key. `melody/extract_melody.py` (Arshiya) does the same with basic-pitch or a
   MIDI file; which one to keep is **open** (see "Teammate code on main").
2. **Data team:** each recording (team or cover) is pitch-tracked with **PESTO** (not CREPE), aligned
   to the target notes, and scored per phrase by a rule-based script (`scorer/score.py`). Those scores
   are the **labels**. Covers need Demucs first, then the two-pass alignment (`labeling/align.py`).
3. **ML team (this code):** load labeled phrases, compute features from the target notes, train
   and evaluate models, and expose a `predict` function for the app team.

Do not modify the data team's Demucs code. The ML code should depend only on the data format below.

## Current status (snapshot 2026-11-17; details in `roadmap.md`)

- Labeling runs end to end on 5 pilot songs: `ml/data/real/phrases_v1.0.parquet` (1,362 phrases,
  34 cover singers). Next: a plumbing run of `ml/` on it (roadmap step 5), then scale up to every song.
- The ML code (baseline, linear, XGBoost, `recommend.py`) is tested on fake data only. The
  hierarchical model and the leave-one-singer-out test are not built. Team recordings do not exist yet.
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
- Exclude video `8cJQc8llWLg` (I Will Survive, 99% silent).
- Check by ear: videos `oasL_TDbe4g` (Sk8er Boi), `i4xE1K6brA0` (Girls Just Want to Have Fun) and
  `po94Wr7Lev8` (Sorry, a dance cover).
- In anything committed (this file included), refer to recordings by video id, never uploader name.

**Problems with covers as labels, and how they are handled:**
- **Target notes:** not in `covers_data/`; made from the originals in `originals_data/`.
- **Key:** covers may be in a different key and the manifest has no transpose; pass 1 of the
  alignment estimates it per cover (all 12 keys tried).
- **Alignment:** 205 recordings differ by over 20% from their song's median length (excerpts or
  rearrangements), so one fixed time offset is not enough. Two-pass alignment; recordings that do
  not align are rejected (16 of 50 in the pilot).
- **Multiple voices:** Smule duets are common and often unlabelled. PESTO tracks one pitch and Demucs
  puts both singers in one stem, so duets must be detected, not trusted from titles. `duet_flag`
  marks them (threshold untested on a real duet).
- **Pitch correction:** Smule offers it, so accuracy may be too high. `pitch_correction_flag` marks
  covers with too much singing exactly on a semitone (none in the pilot).
- **Singer range:** no sweeps exist. Every cover singer gets one default range (see "Labeling
  decisions"); never estimate it from the notes sung accurately (leaks the label).
- **Singer identity:** an uploader is not always one person. Only 75 of 892 uploaders appear on 2+
  songs and 11 on 4+, so this data can train the shared model but cannot test personalization.

Pilot first (5 songs, `PILOT_SONGS`), checked by ear, before processing the full batch.

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
  octave down). **A whole phrase sung one octave up or down** (>= 70% of its readings in that
  octave, `phrase_octave_share`; e.g. a line too low for the singer) **counts as right** in `accuracy`
  and is reported in `phrase_octave` (+1/-1); two octaves away never does (user, 2026-11-16, after
  hearing such a phrase sung well but scored 0). Within the phrase's octave, short jumps (tracker
  glitches, <= 150 ms, configurable) are folded back; held notes in another octave count as wrong.
  `accuracy_strict` never forgives octaves and `octave_error_rate` counts every reading in another
  octave than the take's, so range avoidance stays visible.
- **Phrases every cover fails are dropped** (user, 2026-11-16): >= 3 scored covers and median
  accuracy < 0.15 (`CONSENSUS_*` in `labeling/config.py`) means the answer key is wrong there (the
  note maker followed another voice); `labeling/score_covers.py` drops the phrase from every cover
  and lists it in `derived/scores/dropped_phrases.csv`. Songs with < 3 aligned covers cannot be
  checked. The melody JSON keeps the phrase, so `phrase_position` is unchanged.
- **Singer range:** estimating range from notes sung accurately leaks the label into the range
  features (most cover singers have one song). **Every cover singer gets one default range**,
  MIDI 52-72 (E3-C5, `COVER_RANGE_MIDI`; user, 2026-11-17), and the team gets **sweeps**. Voice types
  from the median sung pitch were tried and fail: covers keep the song's key, so the median measures
  the song (30 of 34 pilot covers within 2.2 semitones of the song's median). `singers.csv` has
  `range_source` (`default_cover` or `sweep`). Before trusting any real result, run the **shuffle test**: retrain with ranges shuffled between singers; if scores
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
  - *Pass 1b (fine-tuning, built 2026-11-16)* sits between them: every 1 s, a 4 s stretch of the
    cover's backing onsets (drum hits, chord attacks; 10 ms frames) is lined up again with the song's
    within +/-300 ms, against a straight line through pass 1's map; corrections are median-smoothed
    over 5 windows and the map is rebuilt through one anchor per window. Still singer-independent.
    On a real backing with a known cut, 2% slowdown and an inserted 150 ms gap, map error drops
    from 22 ms median (p90 53) to 3 ms (p90 9). On the pilot covers only about half the windows find
    a clear match (Smule backings are re-recordings); pass-2 onset error on aligned covers went
    90 -> 84 ms median, within-100 ms 0.53 -> 0.58. In matched windows notes are still ~76 ms off,
    so most of the remaining timing error is the singers (plus onset-detection noise on Demucs
    vocals); unmatched windows are ~93 ms. Before/after beep files: `derived/beep_check/covers/`
    vs `covers_before_refine/`; summaries `derived/alignment/summary.csv` vs `summary_before_refine.csv`.
  - librosa's subsequence DTW returns a wrong path when the query is longer than the reference, so
    `align._dtw` always puts the shorter recording first.
  - Units trap: a "10 ms" hop at 22,050 Hz is 220 samples = 9.977 ms. `onset_envelope` resamples onto
    an exact 10 ms grid; without that the error grows along the song (~30 ms by 45 s).
  - Test-audio trap: librosa's `time_stretch`/`pitch_shift` (phase vocoder) smear drum hits ~50 ms
    late, so pass 1b tests build fake covers by resampling and inserting silence instead.
  - Cover beep check (user, by ear): aligned covers sound "generally right but not 1:1". Measured with
    backing-track onsets in 4 s windows, pass 1's map is off by a median 30-190 ms locally (90% within
    180-430 ms; noisy measure, beat-period ambiguity). Fine for pitch where pass 2 finds the onset,
    **not good enough for timing labels** (error as big as the +/-100 ms window).
  - Missing beeps: 6-26% of a cover's sung time has no target note. Mostly (56-89%) the original's
    lead is silent there (ad-libs, harmonies, duet partner, humming), which is correctly unscored; the
    rest (11-44%) are notes the note maker dropped (< 80 ms, low confidence, ABBA's stacked harmonies).
- **Tempo** comes from librosa's beat tracker on the original's backing. It doubles slow ballads
  (Someone Like You 136, Fix You 144 instead of ~70), so `build_dataset.py` folds every tempo into
  [60, 120) BPM (`TEMPO_BAND_MIN_BPM`): the feature only needs to be consistent.
- **Quality flags are marked, not dropped:** `duet_flag` (> 35% of the singing where no target note
  is due) and `pitch_correction_flag` (> 55% of the singing within 15 cents of a semitone).
- **Singers are anonymous:** stable random ids `c0001...`; the uploader -> id and recording -> video
  maps are in `derived/private/` (never share, never in the dataset folder).

## Labeling code (`labeling/`, run from the project folder)

| File | Purpose |
|---|---|
| `config.py` | every rule and threshold, units in the names; `PILOT_SONGS`, `LEFT_OUT_SONGS` |
| `pitch.py` | PESTO, every frame kept (`time_s`, `hz`, `midi`, `confidence`, `voiced`), 3 shifted copies |
| `targets.py` | pitch -> `tuning_cents` -> notes -> phrases; `song_id_for()` slugs folder names |
| `align.py` | pass 1 (`backing_chroma`, `align_backing`), pass 1b (`onset_envelope`, `refine_alignment`) and pass 2 (`find_onsets`, `recording_octave`) |
| `separate_originals.py` | Demucs on `originals_data/` -> `separated/originals/htdemucs/<song>/` |
| `make_targets.py` | writes `ml/data/real/melodies/*.json` and `ml/data/real/songs.csv` |
| `beep_check.py` | original + target beeps -> `derived/beep_check/` |
| `align_covers.py` | both passes on every cover -> `derived/alignment/` (+ `summary.csv`, `--beeps N`) |
| `score_covers.py` | `scorer/` on every aligned cover -> `derived/scores/cover_scores.csv`; `--self-score`, `--clips N` |
| `synthetic_singer.py` | step-3 checks with planted errors (test A synthetic voice, test B real voice) |
| `build_dataset.py` | `ml/data/real/phrases_v1.0.parquet`, `singers.csv`, `songs.csv`; private maps in `derived/private/` |
| `tests/` | `py -m pytest labeling` (31 tests); `py -m pytest ml` (24); `py -m pytest scorer` (26) |

Order: `separate_originals.py` -> `make_targets.py` -> `beep_check.py` -> `align_covers.py` -> `score_covers.py` -> `build_dataset.py`
(each takes `--all`; default is the 5 pilot songs). `derived/` and `ml/data/real/` are gitignored.
Caches: `derived/pitch/{originals,covers}/` (PESTO), so re-runs are fast.

## Teammate code on main (Arshiya, PRs #3-#5, merged 2026-11-05; reviewed 2026-11-16)

| Folder | What it does |
|---|---|
| `scorer/score.py` | pitch CSV (`program.py` format) + melody JSON -> phrase rows in the contract format. Global lag search (+/-0.6 s), key estimated from the singing (or `transpose` given), per-phrase DTW with a 100 ms band, `qc_flag`. All rules in its `Config`. |
| `melody/extract_melody.py` | answer key from basic-pitch (or a karaoke MIDI): one melody line, merge/drop rules, phrases at 0.6 s rests (max 10 s), preview WAV, QC report, Krumhansl key guess. Its README has a basic-pitch install workaround (`pip install --no-deps basic-pitch`). |
| `pipeline/run_pipeline.py` | team recordings: `data/songs/` + `data/recordings/<singer>/<song>_t<take>_k<key>.wav` -> parquet + report. |
| `.github/workflows/tests.yml` | CI on Python 3.12: `pytest ml scorer melody pipeline` (not `labeling`). |

What the review found (tested on the real pilot data, 2026-11-16):
- **`scorer/` works on clean, in-sync pitch:** originals vs their own targets give median phrase
  accuracy 0.79-0.92 (ABBA lowest: no `tuning_cents` support, ABBA is -24 cents).
- **It cannot take covers raw** (median accuracy 0.20, wrong key on 8 of 34): their timelines differ
  by more than its +/-0.6 s lag search. Fed with cover pitch moved onto the song timeline by our pass-1
  map and the key passed in, it gives 0.51 median, close to our own estimate. So the plan is: covers go
  through `labeling/align.py` first, then `scorer/`.
- **Its timing labels are wrong even on the originals** (`timing_accuracy` 0.50-0.82 where ~1 is
  right; covers 70-150 ms early): `onset_stats` takes the first in-tune run from 300 ms before the
  note, which is often the end of the previous same-pitch note. `missed_note_rate` counts off-pitch
  notes as missed (the contract says: no voiced frames).
- **Changes made to `scorer/score.py` on 2026-11-16** (the user said go ahead; tell Arshiya, it is
  their code; their 13 tests still pass, 13 exact tests added):
  - `score_take(pitch, ..., tuning_cents=0.0)`: `pitch` is a CSV or a DataFrame of frames
    (`time_s`, `midi`/`hz`, `confidence`); tuning removed before scoring; `--tuning-cents` on the CLI.
  - Recording-level octave added to `transpose` (given or estimated). Octave rule: other-octave runs
    <= 150 ms are glitches (count correct), longer runs count wrong. New columns `accuracy_strict`,
    `octave_error_rate`. `mean_cents_error`/`bias_cents` stay pitch-class (folded).
  - Smoothing is cut at jumps > 0.5 semitone between readings (it used to average across note
    changes, costing in-tune singers accuracy at every note end and smearing octave glitches).
  - Onsets: moves onto the note's pitch (0.75 semitone, octave-folded, 30 ms) nearest the note start
    within +/-300 ms, from unsmoothed pitch; notes without one are left out of the timing labels.
    `missed_note_rate` = no voiced reading while the note is due (the contract).
  - Later the same day: a whole phrase one octave away counts as right (`phrase_octave`); octave runs
    are not cut at gaps between notes; an onset needs the singer to stay on the note for 30% of it
    (`onset_min_run_frac`).
- Results with the updated scorer (before the phrase drops and the phrase-octave rule; current
  numbers in `roadmap.md`): originals vs their own notes give accuracy 0.90-0.94 (was
  0.79-0.92) and timing_accuracy 0.96-1.00 (was 0.50-0.82), onset error -25 to -1 ms. The 34 aligned
  pilot covers (moved onto the song timeline by the pass-1b map, key passed in): median phrase
  accuracy 0.49, accuracy_strict 0.47, octave_error_rate 0.02, timing_accuracy 0.60 (was 0.1-0.2),
  mean onset error -6 ms (was -70 to -150), missed_note_rate 0. Per song accuracy: Fix You 0.72,
  Bruno Mars 0.58, a-ha 0.50, Adele 0.49, ABBA 0.33. The planted-error tests (roadmap step 3) show
  the pipeline reads accuracy within ~0.01-0.05, so ~0.5 is mostly the singers.
- `labeling/align.find_onsets` (pass 2, now only used for diagnostics and beep files) still uses a
  1-semitone onset tolerance, which misses onsets a semitone from the previous note; the scorer uses 0.75.
- `pipeline/test_pipeline.py` is an exact copy of `fake_singer.py`: the pipeline has **no tests**,
  though its README says it does. `melody` tests need `pretty_midi`, which no requirements file lists
  (CI installs it ad hoc).
- **Open decision:** which answer-key maker to keep (PESTO `labeling/targets.py` vs basic-pitch
  `melody/`). Compare both on the 5 pilot songs by ear, then keep one; they also split phrases
  differently (0.4 s rests vs 0.6 s rests with a 10 s cap).

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
  for covers, one default range for everyone, since no sweeps exist), optional `range_source`
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

## Fake data generator (`ml/src/make_fake_data.py`, built)

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

## Repo layout

```
CLAUDE.md               # this file: rules, decisions, data format
roadmap.md              # progress, results, next steps
labeling/               # cover labeling pipeline (see "Labeling code")
scorer/ melody/ pipeline/ program.py   # teammates' code (see "Teammate code on main")
ml/
  data/fake/            # generated fake data (v0.0)
  data/real/            # pilot dataset from labeling/ (gitignored)
  src/
    make_fake_data.py
    features.py         # target_notes -> feature row
    evaluate.py         # load_dataset, GroupKFold, holdout, metrics, learning curve
    train.py            # trains and scores every model, logs to MLflow, saves the shipped model
    learning_curve.py
    models/
      baseline.py
      linear.py
      xgb.py
      hierarchical.py   # not built yet (ladder step 4)
    recommend.py        # predict() and recommend_key()
  tests/                # unit tests for features, splits, recommend
  requirements.txt
derived/                # caches, alignment, scores, clips, private maps (gitignored)
```

## Tech stack

Python, pandas, pyarrow, NumPy, scikit-learn, XGBoost, statsmodels, Optuna, SHAP, SciPy,
matplotlib, joblib, Weights & Biases or MLflow. PyTorch only for the optional step 5.

## Working style

- Keep code simple and readable; the team must be able to explain every line in interviews.
- Write unit tests for feature extraction and for the song-level split (check no song appears in
  both train and test).
- Be suspicious of results that look too good; check for leakage first.
