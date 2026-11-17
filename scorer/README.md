# Scorer

Turns one recording's pitch track into labeled phrase rows, in the shared dataset format. It is
stage 3 of the pipeline: pitch CSV (from `program.py`) plus melody JSON in, parquet rows out.

## Run it

```
python scorer/score.py --pitch my_file_ORIGINAL.csv --melody ml/data/fake/melodies/song_00.json \
    --singer-id s1 --take 1 --out phrases_v0.1.parquet
python -m pytest scorer
```

Leave out `--transpose` to have the key estimated. Pass it when you know it (your own team's
recordings, where you chose the key).

## The design in six ideas

1. **One file in, one file out.** The scorer never touches audio. It reads the pitch CSV and the
   melody JSON, so you can re-score old recordings without running PESTO again.
2. **Pitch becomes MIDI, errors become cents.** `midi = 69 + 12 * log2(hz / 440)`, then
   `error = 100 * (sung - target)`. A reading is a hit if the error is within 50 cents.
3. **Fix the whole take's timing first.** Singers start late, and headphones add latency. We find
   one offset for the take using only when there is singing, never the pitch, so it cannot hide a
   wrong note.
4. **DTW gets only a small band (100 ms).** DTW stretches time to line two sequences up. With too
   much freedom it pairs a sharp note with a neighbouring target note and the error disappears.
   Small band plus a penalty for stretching keeps it honest.
5. **The key is estimated, with a confidence.** Strangers sing in any key. We try shifts from -6 to
   +6 semitones and keep the one with the most hits. If two keys score almost the same, the row
   gets the flag `unsure_key`.
6. **Bad takes are flagged, not hidden.** `qc_flag` can hold `mostly_silent`, `unsure_key` or
   `lag_at_limit`. Phrases with fewer than 10 scored readings are dropped.
7. **Octaves (agreed rule).** The octave the singer used for the whole take is found and added to
   `transpose` (e.g. -12 for a man singing a woman's song an octave down). Inside a phrase, a jump
   to another octave lasting up to 150 ms is a tracker glitch and still counts as correct; a held
   note in the other octave counts as wrong. `accuracy_strict` never forgives octaves.
8. **Smoothing never crosses a note change.** The 100 ms average that removes vibrato is cut
   wherever the pitch jumps more than half a semitone between two readings.
9. **Onsets are moves onto the note.** A note starts where the singer moves onto its pitch (within
   0.75 semitone, any octave) and stays 30 ms, closest to the note's start within +/-300 ms. The
   end of a previous note on the same pitch is never taken as this note's start.

Covers reach the scorer through `labeling/align.py`, which moves their pitch onto the song's
timeline and finds the key. `score_take` accepts those pitch frames as a DataFrame (`time_s`,
`midi` or `hz`, `confidence`) as well as a CSV, plus `tuning_cents` from `songs.csv`.

## Output columns

The first eleven match the contract in `CLAUDE.md`. `target_notes` is already shifted by
`transpose`, which is what `ml/src/features.py` expects. Extra columns:

| Column | Meaning |
|---|---|
| `accuracy_strict` | like `accuracy`, but a reading in the wrong octave is always a miss |
| `octave_error_rate` | share of readings in another octave (glitches and held notes) |
| `n_scored` | readings the accuracy is based on |
| `bias_cents` | positive means tends sharp, negative means tends flat |
| `lag_s` | whole-take timing offset that was removed |
| `transpose_conf` | gap between the best key and the runner up (empty if the key was given) |
| `transpose_source` | `given` or `estimated` |
| `qc_flag` | quality warnings for the take |

## Settings

All the rules the two teams need to agree on live in the `Config` class at the top of
`score.py`: confidence threshold, tolerance, octave rule, vibrato smoothing, and so on.

## Known limits

- Timing labels are measured after the whole-take offset is removed, so a take that is late
  everywhere shows lag in `lag_s`, not in the onset errors.
- Notes with no clear onset (legato into the same pitch, or not sung) are left out of
  `timing_accuracy` and `mean_onset_error_ms`. `missed_note_rate` is the share of notes with no
  voiced reading at all while they are due; an off-pitch note is not missed.
- The key estimate assumes one key for the whole take. A singer who changes key midway will get a
  low confidence and the `unsure_key` flag.
- Tested on the 5 pilot songs (2026-11-16): the originals scored against their own notes give
  accuracy 0.90-0.94 and timing_accuracy 0.96-1.00. See CLAUDE.md for the cover results.