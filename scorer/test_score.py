"""Tests for score.py. They fake PESTO's output (a pitch CSV), so no audio or PESTO is needed."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

MELODY = [
    [(60, 0.5, 0.5), (62, 1.0, 0.5), (64, 1.5, 0.5), (65, 2.0, 0.5)],
    [(67, 3.5, 0.5), (69, 4.0, 0.5), (67, 4.5, 0.5), (64, 5.0, 0.5)],
]


def write_melody(path, phrases=MELODY, song_id="demo"):
    notes = [[{"midi_pitch": int(p), "start_s": s, "duration_s": d} for p, s, d in phrase] for phrase in phrases]
    path.write_text(json.dumps({"song_id": song_id, "tempo_bpm": 100, "phrases": notes}))
    return path


def write_pitch(path, phrases=MELODY, shift_s=0.0, semitones=0.0, per_note=None, note_shift_s=None,
                total_s=7.0, vibrato_cents=15.0, seed=0, milliseconds=True, silent=False):
    """Fake PESTO CSV: a reading every 10 ms. Notes become pitch with a little noise and vibrato;
    between notes the tracker reports a low confidence reading, like PESTO does."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, total_s, score.HOP_S)
    hz = np.full(len(t), 200.0)
    conf = np.full(len(t), 0.15)
    k = 0
    for phrase in phrases:
        for pitch, start, dur in phrase:
            extra = semitones + (per_note[k] if per_note else 0.0)
            begin = start + shift_s + (note_shift_s[k] if note_shift_s else 0.0)
            on = (t >= begin) & (t < begin + dur)
            cents = rng.normal(0, 8, on.sum()) + vibrato_cents * np.sin(2 * np.pi * 5.5 * t[on])
            hz[on] = 440.0 * 2 ** ((pitch + extra - 69) / 12 + cents / 1200)
            conf[on] = 0.95
            k += 1
    if silent:
        conf[:] = 0.1
    stamp = t * 1000 if milliseconds else t
    pd.DataFrame({"timestamp": stamp, "frequency": hz, "confidence": conf}).to_csv(path, index=False)
    return path


@pytest.fixture
def melody(tmp_path):
    return write_melody(tmp_path / "melody.json")


def run(tmp_path, melody, transpose=None, **pitch_args):
    pitch = write_pitch(tmp_path / "pitch.csv", **pitch_args)
    return score.score_take(pitch, melody, "s1", 1, transpose)


def test_worked_example_from_doc():
    """Target E4 (MIDI 64): 3 of 5 readings within 50 cents, mean error about 29 cents."""
    sung = score.hz_to_midi([329.6, 335.0, 320.0, 327.0, 340.0])
    result = score.summarize_errors(sung, np.full(5, 64.0))
    assert result["accuracy"] == pytest.approx(0.6)
    assert result["mean_cents_error"] == pytest.approx(29.0, abs=1.5)


def test_hz_to_midi_and_no_pitch():
    assert score.hz_to_midi(440.0) == pytest.approx(69.0)
    assert score.hz_to_midi(880.0) == pytest.approx(81.0)
    assert score.hz_to_midi(220.0) == pytest.approx(57.0)
    assert np.isnan(score.hz_to_midi(0.0))


@pytest.mark.parametrize("late", [0.12, 0.35])
def test_in_tune_but_late_is_accurate(tmp_path, melody, late):
    df = run(tmp_path, melody, transpose=0, shift_s=late)
    assert len(df) == 2
    assert (df["accuracy"] > 0.9).all(), df[["phrase_idx", "accuracy"]]
    assert df["lag_s"].iloc[0] == pytest.approx(late, abs=0.05)


def test_one_semitone_sharp_phrase_is_caught(tmp_path, melody):
    df = run(tmp_path, melody, transpose=0, per_note=[0] * 4 + [1.0] * 4)
    accuracy = dict(zip(df["phrase_idx"], df["accuracy"]))
    assert accuracy[0] > 0.9
    assert accuracy[1] < 0.2
    assert df.loc[df.phrase_idx == 1, "bias_cents"].iloc[0] > 50  # reported as sharp


def test_octave_lower_counts_as_correct(tmp_path, melody):
    df = run(tmp_path, melody, transpose=0, semitones=-12.0)
    assert (df["accuracy"] > 0.9).all()


def test_key_is_estimated_and_notes_are_shifted(tmp_path, melody):
    df = run(tmp_path, melody, semitones=-2.0)  # transpose not given
    assert (df["transpose"] == -2).all()
    assert (df["transpose_source"] == "estimated").all()
    assert (df["accuracy"] > 0.9).all()
    first = df.iloc[0]["target_notes"]
    assert first[0]["midi_pitch"] == 60 - 2  # the contract: notes already shifted by transpose


def test_wrong_given_transpose_scores_low(tmp_path, melody):
    right = run(tmp_path, melody, transpose=-2, semitones=-2.0)
    wrong = run(tmp_path, melody, transpose=0, semitones=-2.0)
    assert (right["accuracy"] > 0.9).all()
    assert (wrong["accuracy"] < 0.2).all()


def test_late_note_shows_in_timing_labels(tmp_path, melody):
    late = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    late[1] = 0.25  # second note of phrase 0 starts 250 ms late
    df = run(tmp_path, melody, transpose=0, note_shift_s=late)
    phrase0, phrase1 = df.iloc[0], df.iloc[1]
    assert phrase0["timing_accuracy"] < phrase1["timing_accuracy"]
    assert phrase0["mean_onset_error_ms"] > 40


def test_silent_take_is_dropped_and_flagged(tmp_path, melody):
    df = run(tmp_path, melody, transpose=0, silent=True)
    assert len(df) == 0 and df.attrs["dropped_phrases"] == 2


def test_seconds_and_milliseconds_timestamps_agree(tmp_path, melody):
    a = run(tmp_path, melody, transpose=0, milliseconds=True)
    b = run(tmp_path, melody, transpose=0, milliseconds=False)
    assert a["accuracy"].round(6).tolist() == b["accuracy"].round(6).tolist()


def test_append_replaces_the_same_take(tmp_path, melody):
    df = run(tmp_path, melody, transpose=0)
    out = tmp_path / "phrases.parquet"
    score.append_to_dataset(df, out)
    score.append_to_dataset(df, out)  # running it twice must not duplicate rows
    assert len(pd.read_parquet(out)) == len(df)


@pytest.mark.skipif(not (REPO / "ml" / "src").exists(), reason="ml/ folder not present")
def test_output_works_with_the_ml_feature_code(tmp_path):
    """Score a perfect singer of one of the fake songs, then build features exactly like train.py."""
    sys.path.insert(0, str(REPO / "ml" / "src"))
    from features import build_feature_table

    fake = REPO / "ml" / "data" / "fake"
    melody_path = fake / "melodies" / "song_00.json"
    _, _, phrases = score.load_melody(melody_path)
    end = max(s + d for phrase in phrases for _, s, d in phrase) + 1
    pitch = write_pitch(tmp_path / "pitch.csv", phrases=phrases, total_s=end)

    df = score.score_take(pitch, melody_path, "s1", 1, transpose=0)
    assert len(df) > 5 and df["accuracy"].mean() > 0.9

    singers, songs = pd.read_csv(fake / "singers.csv"), pd.read_csv(fake / "songs.csv")
    table = build_feature_table(df, singers, songs)
    assert len(table) == len(df)
    assert not table.drop(columns=["mean_onset_error_ms", "transpose_conf"]).select_dtypes("number").isna().any().any()

# ---- Exact tests on clean synthetic pitch frames (agreed 2026-11-03, CLAUDE.md "Tests are split") ----
# No noise or vibrato, so every planted error must come back exactly.

def clean_frames(segments, total_s=7.0):
    """Pitch frames (labeling/pitch.py format) from (midi or None, start_s, end_s) segments."""
    t = np.round(np.arange(0, total_s, score.HOP_S), 6)
    midi = np.full(len(t), np.nan)
    for m, a, b in segments:
        midi[(t >= a - 1e-9) & (t < b - 1e-9)] = np.nan if m is None else m
    return pd.DataFrame({"time_s": t, "midi": midi, "confidence": np.where(np.isnan(midi), 0.1, 0.95)})


def sung_as_written(phrases=MELODY, cents=0.0):
    return [(p + cents / 100, s, s + d) for phrase in phrases for p, s, d in phrase]


def test_exact_cents_errors_per_phrase(melody):
    """Phrase 0 sung +30 cents (inside +/-50), phrase 1 sung +60 cents (outside)."""
    segs = sung_as_written(MELODY[:1], 30) + sung_as_written(MELODY[1:], 60)
    df = score.score_take(clean_frames(segs), melody, "s1", transpose=0)
    assert df["accuracy"].tolist() == [1.0, 0.0]
    assert df["mean_cents_error"].round(6).tolist() == [30.0, 60.0]
    assert df["bias_cents"].round(6).tolist() == [30.0, 60.0]


def test_tuning_cents_is_removed(melody):
    """A song 24 cents flat of A440, sung exactly in its own tuning: perfect once tuning is passed."""
    frames = clean_frames(sung_as_written(cents=-24))
    without = score.score_take(frames, melody, "s1", transpose=0)
    with_tuning = score.score_take(frames, melody, "s1", transpose=0, tuning_cents=-24)
    assert without["mean_cents_error"].round(6).tolist() == [24.0, 24.0]
    assert with_tuning["mean_cents_error"].round(6).tolist() == [0.0, 0.0]
    assert (with_tuning["accuracy"] == 1.0).all()


def test_known_onset_errors_exactly():
    """onset_stats alone: note 2 sung 80 ms late, note 3 50 ms early, notes 1 and 4 on time."""
    notes = MELODY[0]                       # 60 @0.5, 62 @1.0, 64 @1.5, 65 @2.0, 0.5 s each
    sung = np.full(400, np.nan)
    for (p, s, d), shift in zip(notes, [0.0, 0.08, -0.05, 0.0]):
        sung[int(round((s + shift) / score.HOP_S)): int(round((s + d) / score.HOP_S))] = p
    timing = score.onset_stats(sung, notes, score.Config())
    assert timing["mean_onset_error_ms"] == pytest.approx((0 + 80 - 50 + 0) / 4)
    assert timing["timing_accuracy"] == 1.0          # all within 100 ms
    assert timing["missed_note_rate"] == 0.0


def test_same_pitch_repeat_is_not_taken_from_the_previous_note():
    """Two notes on the same pitch with a 60 ms breath between them, both on time. The old rule
    took the end of the first note as the start of the second (an error of about -300 ms)."""
    notes = [(60, 0.5, 0.44), (60, 1.0, 0.5)]
    sung = np.full(200, np.nan)
    sung[50:94] = 60
    sung[100:150] = 60
    timing = score.onset_stats(sung, notes, score.Config())
    assert timing["mean_onset_error_ms"] == pytest.approx(0.0)
    assert timing["timing_accuracy"] == 1.0


def test_missed_means_not_sung_not_off_pitch():
    """Note 2 sung 3 semitones wrong (not missed), note 3 silent (missed): 1 of 4 missed."""
    notes = MELODY[0]
    sung = np.full(400, np.nan)
    for (p, s, d), extra in zip(notes, [0, 3, None, 0]):
        if extra is not None:
            sung[int(round(s / score.HOP_S)): int(round((s + d) / score.HOP_S))] = p + extra
    assert score.onset_stats(sung, notes, score.Config())["missed_note_rate"] == 0.25


def test_octave_glitch_is_forgiven_but_a_held_wrong_octave_is_not(melody):
    """Phrase 0: a 100 ms jump up an octave inside a note (a tracker glitch). Phrase 1: its whole
    third note (500 ms) sung an octave up."""
    segs = sung_as_written()
    segs += [(72, 0.70, 0.80)]               # glitch inside the 60 at 0.5-1.0
    segs += [(79, 4.5, 5.0)]                 # 67 held an octave up
    df = score.score_take(clean_frames(segs), melody, "s1", transpose=0)
    glitch, held = df.iloc[0], df.iloc[1]
    assert glitch["accuracy"] == 1.0 and glitch["accuracy_strict"] < 1.0
    assert glitch["octave_error_rate"] > 0
    assert held["accuracy"] == pytest.approx(held["accuracy_strict"])
    assert held["accuracy"] == pytest.approx(1 - held["octave_error_rate"])
    assert 0.6 < held["accuracy"] < 0.8      # one of four notes wrong (its first 50 ms are skipped)
    assert held["phrase_octave"] == 0        # one note of four is not the whole phrase


def test_whole_take_an_octave_down_goes_into_transpose(melody):
    df = score.score_take(clean_frames(sung_as_written(cents=-1200)), melody, "s1", transpose=0)
    assert (df["transpose"] == -12).all()
    assert (df["accuracy"] == 1.0).all() and (df["octave_error_rate"] == 0).all()
    assert df.iloc[0]["target_notes"][0]["midi_pitch"] == 60 - 12


def test_frames_and_csv_give_the_same_labels(tmp_path, melody):
    frames = clean_frames(sung_as_written(cents=20))
    csv = tmp_path / "pitch.csv"
    pd.DataFrame({"timestamp": frames["time_s"] * 1000,
                  "frequency": 440 * 2 ** ((frames["midi"].fillna(60) - 69) / 12),
                  "confidence": frames["confidence"]}).to_csv(csv, index=False)
    a = score.score_take(frames, melody, "s1", transpose=0)
    b = score.score_take(csv, melody, "s1", transpose=0)
    assert a["accuracy"].round(6).tolist() == b["accuracy"].round(6).tolist()
    assert a["mean_cents_error"].round(3).tolist() == b["mean_cents_error"].round(3).tolist()


def test_short_notes_all_an_octave_down_are_not_forgiven_one_by_one(tmp_path):
    """Four 120 ms notes with gaps, all an octave down in a take otherwise in the right octave.
    Each note alone is shorter than the glitch limit, but together they are a held wrong octave."""
    phrases = [[(60, 0.5, 0.5), (62, 1.0, 0.5), (64, 1.5, 0.5), (65, 2.0, 0.5)],
               [(67, 3.5, 0.12), (69, 3.7, 0.12), (67, 3.9, 0.12), (64, 4.1, 0.12)],
               [(60, 5.0, 0.5), (62, 5.5, 0.5), (64, 6.0, 0.5), (65, 6.5, 0.5)]]
    melody = write_melody(tmp_path / "m.json", phrases)
    segs = sung_as_written(phrases[:1]) + [(p - 12, s, s + d) for p, s, d in phrases[1]] + sung_as_written(phrases[2:])
    # with whole-phrase octave moves switched off, this tests the glitch rule on its own
    df = score.score_take(clean_frames(segs, total_s=8.0), melody, "s1", transpose=0,
                          cfg=score.Config(min_scored_frames=5, phrase_octave_share=None))
    assert (df["transpose"] == 0).all()
    short = df[df.phrase_idx == 1].iloc[0]
    assert short["accuracy"] == 0.0 and short["octave_error_rate"] == 1.0



THREE = MELODY + [[(60, 6.0, 0.5), (62, 6.5, 0.5), (64, 7.0, 0.5), (65, 7.5, 0.5)]]


def moved_middle_phrase(tmp_path, cents):
    """Phrases 0 and 2 sung as written, phrase 1 moved by `cents`: the take's octave stays 0."""
    melody = write_melody(tmp_path / "three.json", THREE)
    segs = sung_as_written(THREE[:1]) + sung_as_written(THREE[1:2], cents) + sung_as_written(THREE[2:])
    df = score.score_take(clean_frames(segs, total_s=9.0), melody, "s1", transpose=0)
    assert (df["transpose"] == 0).all()
    return df.set_index("phrase_idx")


def test_whole_phrase_one_octave_up_counts_as_right_but_not_strict(tmp_path):
    df = moved_middle_phrase(tmp_path, 1200)
    assert df.loc[1, "phrase_octave"] == 1
    assert df.loc[1, "accuracy"] == 1.0
    assert df.loc[1, "accuracy_strict"] == 0.0 and df.loc[1, "octave_error_rate"] == 1.0
    assert df.loc[0, "phrase_octave"] == 0 and df.loc[0, "accuracy"] == 1.0


def test_whole_phrase_one_octave_down_counts_as_right(tmp_path):
    df = moved_middle_phrase(tmp_path, -1200)
    assert df.loc[1, "phrase_octave"] == -1 and df.loc[1, "accuracy"] == 1.0


def test_two_octaves_away_is_not_allowed(tmp_path):
    df = moved_middle_phrase(tmp_path, 2400)
    assert df.loc[1, "phrase_octave"] == 0 and df.loc[1, "accuracy"] == 0.0


def test_phrase_octave_wrong_notes_still_count_wrong(tmp_path):
    """An octave up AND a semitone sharp: the octave is forgiven, the semitone is not."""
    df = moved_middle_phrase(tmp_path, 1300)
    assert df.loc[1, "phrase_octave"] == 1 and df.loc[1, "accuracy"] == 0.0
