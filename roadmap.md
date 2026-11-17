# Roadmap: Karaoke ML Coach

Where the project stands and what comes next. **Update this file whenever a step finishes or a plan
changes** (with the date). The rules, decisions and data format live in `CLAUDE.md`; this file only
tracks progress and results.

Status key: **done**, **next**, **later**, **blocked** (with what it waits for).

## Where we are (2026-11-17)

- The labeling pipeline runs end to end on the **5 pilot songs**: Demucs, target notes, two-pass
  alignment with fine-tuning, scoring, label checks, dataset. Output:
  `ml/data/real/phrases_v1.0.parquet` (1,362 phrases, 5 songs, 34 cover singers), loadable by `ml/`.
- The ML code (baseline, linear regression, XGBoost, `predict` / `recommend_key`) works on **fake data**
  and recovers the planted patterns. It has never been run on real labels.
- **Next session starts at labeling step 5**: run `ml/` on the pilot dataset to check the plumbing.
  This runs the training script, but it is a **connection test, not real training**: with 5 songs
  every metric is meaningless. Real training is step 6, after all ~100 songs are labeled.
- Nothing since PR #2 (commit `ec08a80`) is committed. Steps 1-4 sit uncommitted on the local branch
  `jonathan`. Commit only when the user asks.
- Arshiya has **not yet been told** about the changes to their `scorer/` (see "People" below).

## Part A: ML code (`ml/`)

| # | Task | Status |
|---|---|---|
| A1 | `make_fake_data.py` -> `phrases_v0.0.parquet`, `singers.csv`, `songs.csv` | done |
| A2 | `features.py` with hand-checked unit tests | done |
| A3 | `evaluate.py`: GroupKFold by song, holdout, MAE, Spearman | done |
| A4 | Baseline, linear regression, XGBoost (Optuna, SHAP); XGBoost recovers the planted patterns | done |
| A5 | Learning curve plot (`learning_curve.py`) | done |
| A6 | `recommend.py`: `predict`, `recommend_key`, `load_song`, `recommend_key_from_notes` | done |
| A7 | `phrase_position` from the melody JSON phrase count (`count_phrases`) | done 2026-11-03 |
| A8 | Hierarchical model (ladder step 4: shared model + shrunk singer offsets) | later |
| A9 | Leave-one-singer-out test (0, 1, 2, 4, 8 songs) | blocked: needs team recordings (cover singers rarely sing 2+ songs) |
| A10 | Shuffle test (retrain with singer ranges shuffled) | later, before trusting step 6 results |
| A11 | Optional LSTM (ladder step 5) | later, low priority |

Tests: `py -m pytest ml` (24 pass).

## Part B: Labeling pipeline (`labeling/`, `scorer/`), plan agreed 2026-11-03

### 1. Fine-tune the alignment (pass 1b): done 2026-11-16

- Backing-track onsets are lined up again in 4 s windows every 1 s (see CLAUDE.md "Alignment").
- Test with a known cut, 2% slowdown and a 150 ms gap: map error 22 ms -> 3 ms median.
- User confirmed by ear that the re-rendered cover beep files sync.
- Pilot covers: only about half the windows find a clear match (Smule backings are re-recordings).
  Onset error 90 -> 84 ms median; in matched windows ~76 ms, so most of what remains is the singers.
- Possible later gain: longer windows or a lower `REFINE_MIN_CORR` for the ~93 ms unmatched stretches.

### 2. Scorer: done 2026-11-16

- Adapted Arshiya's `scorer/score.py` instead of writing a second scorer (changes listed in CLAUDE.md
  "Teammate code on main"). 26 scorer tests pass (their 13 plus 13 exact ones).
- Covers reach it through the pass-1b map (`labeling/score_covers.py`).

### 3. Check that the labels are right: done 2026-11-16

Planted errors on Someone Like You (48 phrases, 8 error types cycling: in tune, +30 c, +60 c, one
note 80 ms late + one 50 ms early, two notes skipped, octave down, one wrong note +2 semitones,
whole phrase 120 ms late):

- **Test A, synthetic voice** (PESTO + scorer vs the exact rendered pitch): accuracy within 0.054,
  onsets within 15 ms. Demucs puts the synthetic "ah" in the backing, so this test skips Demucs.
- **Test B, the real singer's vocal with errors planted**, mixed over the backing with a 3.2 s cut,
  full pipeline: map error 1 ms, accuracy within 0.011, onsets within 8 ms. +30 c reads 0.76,
  +60 c 0.37, one wrong note 0.83, 120 ms late +100 to +107 ms, octave down 0.78-0.88 (forgiven
  by design). Skipped notes are under-read after Demucs (0.07 missed vs 0.15 planted).
  Limit: this uses the original's own backing, the easy case.
- **Self-score**: all 5 originals pass (accuracy 0.90-0.94, timing 0.94-1.00).
- **Cover agreement**: phrases that nearly every cover fails mean a wrong answer key. 19 dropped
  (ABBA 12, a-ha 4, Bruno Mars 2, Adele 1; Coldplay has only 2 covers, unchecked).
- **By ear** (user): the worst clip sounded right. It was a phrase sung a whole octave up, which led
  to the "one octave is forgiven" rule. Clips in `derived/scores/clips/` were regenerated after the
  drops; the user has not yet listened to the new set.
- Median cover accuracy after both changes: a-ha 0.59, ABBA 0.46, Adele 0.50, Bruno Mars 0.58,
  Coldplay 0.75.

### 4. Pilot dataset: done 2026-11-17

- `labeling/build_dataset.py` -> `ml/data/real/phrases_v1.0.parquet`, `singers.csv`, `songs.csv`.
  1,362 rows, 5 songs, 34 singers / recordings. Loads through `load_dataset` with no missing features
  (`timing_accuracy` is empty for 90 rows where no onset was found).
- Anonymous singer ids `c0001...`; the private maps are in `derived/private/` (never share).
- Tempo folded into 60-120 BPM; key guessed from the target notes.
- Duet and pitch-correction flags: none of the 34 covers flagged (thresholds untested on a real duet).
- Singer range: every cover singer gets MIDI 52-72 (user, 2026-11-17), after voice types from the
  median sung pitch turned out to measure the song. 25% of phrases have a note outside that range.
- Seen in the features: one phrase spans 29 semitones (probably an octave slip in an answer key).

### 5. Test run through `ml/` on the pilot data: **next**

Purpose: check that real data flows through training, evaluation and `recommend.py` without errors.
Do not read anything into the metrics.

```
cd ml
python src/train.py --data-dir data/real --version v1.0 --no-log
```

Expected problems to fix:
- The holdout takes 1 of the 5 songs, leaving 4 for a 5-fold `GroupKFold`, which will fail. Use
  fewer folds when there are few songs.
- `train.py` saves the shipped model at the end. Make sure this run does not overwrite the
  fake-data model the app team may be using.
- Check that `singer_id` categories, the 90 empty `timing_accuracy` rows and the extra columns
  (`range_source`, `accuracy_strict`, ...) cause no trouble.

### 6. Scale up and real training: later

1. Run every labeling step with `--all`: about 25 min Demucs, 4-5 h PESTO, 1.5 h alignment.
2. Spot-check more beep files and clips by ear.
3. Build `phrases_v1.1` (or v2.0) from all songs.
4. Real model run: baseline, linear, XGBoost, then the **shuffle test** before trusting any result.
5. Learning curve on real data (tells the data team whether more songs would help).

Optional, only if short of songs: a single-DTW fallback for covers whose backing does not match pass 1
(16 of 50 in the pilot), marked in a column.

## Open decisions

- **Which answer-key maker to keep:** PESTO (`labeling/targets.py`) or basic-pitch (`melody/`).
  Compare both on the 5 pilot songs by ear. They also split phrases differently (0.4 s rests vs
  0.6 s rests with a 10 s cap).

## People

- **Arshiya:** tell them about the `scorer/score.py` changes: tuning, octave rules and
  `accuracy_strict`, smoothing cut at note changes, the new onset rule, `missed_note_rate` per the
  contract, the phrase-octave rule, and 13 new tests.
- **Team recordings:** none yet. They are needed for range sweeps, personalization (A8, A9) and key
  recommendation checks (transposed takes).

## Housekeeping

- Fix `program.py`: empty paths crash, outputs overwrite each other, timestamps are ms but unlabelled,
  it deletes low-confidence frames. Make it a thin wrapper around `labeling/pitch.py`.
- Update `README.md` with the labeling pipeline.
- Convert `requirements.txt` from UTF-16 to UTF-8 (Git shows it as binary).
- `pipeline/` has no real tests (`test_pipeline.py` is a copy of `fake_singer.py`); `pretty_midi` is
  in no requirements file.
- `labeling/align.find_onsets` still uses a 1-semitone tolerance (diagnostics only; the scorer uses 0.75).
- CI does not run the `labeling` tests.
- Tempo: the beat tracker doubles slow ballads; folding into 60-120 BPM hides it, but a real 130 BPM
  song becomes 65.
