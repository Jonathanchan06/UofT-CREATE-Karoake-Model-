import numpy as np
import pandas as pd
import pytest

from evaluate import score, song_folds, split_holdout


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    songs = np.repeat([f"song_{i:02d}" for i in range(20)], 15)
    return pd.DataFrame({"song_id": songs, "accuracy": rng.uniform(0, 1, len(songs))})


def test_holdout_has_no_shared_songs(table):
    train, test = split_holdout(table, test_frac=0.2)
    assert not set(train["song_id"]) & set(test["song_id"])
    assert len(train) + len(test) == len(table)
    assert test["song_id"].nunique() == 4


def test_folds_have_no_shared_songs_and_cover_every_song(table):
    tested = []
    for train, test in song_folds(table, n_splits=5):
        assert not set(train["song_id"]) & set(test["song_id"])
        tested += list(test["song_id"].unique())
    assert sorted(tested) == sorted(table["song_id"].unique())


def test_score():
    s = score([0.2, 0.5, 0.9], [0.3, 0.5, 0.7])
    assert s["mae"] == pytest.approx(0.1)
    assert s["spearman"] == pytest.approx(1.0)


def test_score_constant_prediction_has_zero_spearman():
    assert score([0.2, 0.5, 0.9], [0.5, 0.5, 0.5])["spearman"] == 0.0


def test_load_dataset_counts_phrases_from_melodies(tmp_path):
    import make_fake_data as fake
    from evaluate import load_dataset

    rng = np.random.default_rng(0)
    singers, songs = fake.make_singers(rng), fake.make_songs(rng, count=3)
    song_phrases = fake.make_all_song_phrases(rng, songs)
    phrases = fake.build_phrase_rows(rng, singers, songs, song_phrases, with_timing=False)
    last = phrases.groupby("song_id")["phrase_idx"].transform("max")
    phrases = phrases[phrases["phrase_idx"] < last].reset_index(drop=True)   # drop each song's last phrase

    fake.write_phrases(phrases, tmp_path / "phrases_vT.parquet")
    singers.drop(columns="skill_offset").to_csv(tmp_path / "singers.csv", index=False)
    songs.to_csv(tmp_path / "songs.csv", index=False)
    fake.write_melodies(songs, song_phrases, tmp_path / "melodies")

    table, _, _ = load_dataset(tmp_path, "vT")
    for song_id, rows in table.groupby("song_id"):
        n = len(song_phrases[song_id])
        expected = rows["phrase_idx"] / n
        assert np.allclose(rows["phrase_position"], expected)


def test_load_dataset_requires_a_melody_for_every_song(tmp_path):
    import make_fake_data as fake
    from evaluate import load_dataset

    rng = np.random.default_rng(0)
    singers, songs = fake.make_singers(rng), fake.make_songs(rng, count=2)
    song_phrases = fake.make_all_song_phrases(rng, songs)
    phrases = fake.build_phrase_rows(rng, singers, songs, song_phrases, with_timing=False)
    fake.write_phrases(phrases, tmp_path / "phrases_vT.parquet")
    singers.drop(columns="skill_offset").to_csv(tmp_path / "singers.csv", index=False)
    songs.to_csv(tmp_path / "songs.csv", index=False)

    with pytest.raises(ValueError, match="no melody file"):
        load_dataset(tmp_path, "vT")
