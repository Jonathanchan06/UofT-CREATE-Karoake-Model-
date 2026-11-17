"""build_dataset.py: tempo folding, stable anonymous ids, default cover range, cover flags, parquet types."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from build_dataset import CONTRACT, EXTRA, cover_flags, cover_singers, fold_tempo, singer_ids, write_phrases


def test_fold_tempo_into_one_band():
    assert fold_tempo(136.0) == 68.0 and fold_tempo(68.0) == 68.0
    assert fold_tempo(172.3) == pytest.approx(86.2, abs=0.05)
    assert fold_tempo(45.0) == 90.0
    assert np.isnan(fold_tempo(float("nan")))


def test_singer_ids_are_stable_and_anonymous(tmp_path):
    path = tmp_path / "map.csv"
    first = singer_ids(["bob", "alice", "carol"], path)
    assert sorted(first.values()) == ["c0001", "c0002", "c0003"]
    again = singer_ids(["alice", "bob", "carol", "dave"], path)       # a new uploader appears
    assert all(again[name] == first[name] for name in first)
    assert again["dave"] == "c0004"


def test_cover_singers_share_one_default_range():
    singers = cover_singers(["c0002", "c0001", "c0002"], cover_range=(52, 72))
    assert list(singers["singer_id"]) == ["c0001", "c0002"]
    assert (singers["range_low_midi"] == 52).all() and (singers["range_high_midi"] == 72).all()
    assert list(singers.columns) == ["singer_id", "range_low_midi", "range_high_midi", "range_source"]


def test_cover_flags():
    """1 s of singing on a note that is due (exactly on semitones), then 1 s where nothing is due."""
    t = np.round(np.arange(0, 3, 0.01), 6)
    midi = np.where(t < 2, 60.0, np.nan)
    frames = pd.DataFrame({"time_s": t, "midi": midi, "confidence": np.where(np.isnan(midi), 0.1, 0.9)})
    melody = {"phrases": [[{"midi_pitch": 60, "start_s": 0.0, "duration_s": 1.0}]]}
    outside, on_semitone = cover_flags(frames, melody, tuning_cents=0.0)
    assert outside == pytest.approx(0.85 / 2, abs=0.01)                 # 1.15-2.0 s is outside (+-0.15 s pad)
    assert on_semitone == 1.0
    assert cover_flags(frames.assign(midi=midi + 0.3), melody, 0.0)[1] == 0.0   # 30 cents off


def test_parquet_has_the_contract_types(tmp_path):
    row = {"song_id": "s", "singer_id": "c0001", "take_id": 1, "transpose": -12, "phrase_idx": 0,
           "target_notes": [{"midi_pitch": 48, "start_s": 1.0, "duration_s": 0.5}],
           "accuracy": 0.8, "mean_cents_error": 30.0, "timing_accuracy": 0.9,
           "mean_onset_error_ms": -10.0, "missed_note_rate": 0.0,
           **{c: 0 for c in EXTRA}}
    write_phrases(pd.DataFrame([row]), tmp_path / "p.parquet")
    schema = pq.read_schema(tmp_path / "p.parquet")
    for name, typ in CONTRACT:
        assert schema.field(name).type == typ
    assert schema.names == [c for c, _ in CONTRACT] + EXTRA
