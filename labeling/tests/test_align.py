"""Two-pass alignment on inputs where the right answer is known."""

import numpy as np
import pandas as pd
import pytest

import config
from align import align_backing, find_onsets, recording_octave, song_to_cover_time

HOP = config.CHROMA_HOP_S


def chord_sequence(rng, seconds):
    """Song chroma: random chords (3 active pitch classes) held for 0.5-2 s each."""
    frames = []
    while len(frames) * HOP < seconds:
        chord = np.full(12, 0.05)
        chord[rng.choice(12, 3, replace=False)] = 1.0
        frames += [chord] * int(rng.integers(5, 21))
    chroma = np.array(frames).T
    return chroma / np.linalg.norm(chroma, axis=0)


def test_pass1_recovers_key_cut_and_tempo():
    """Cover = the song from 20 s on, 10% slower, 2 semitones up, with noise."""
    rng = np.random.default_rng(0)
    song = chord_sequence(rng, 180)
    cover_frames = np.arange(int((song.shape[1] - 200) * 1.1))
    true_song_frame = 200 + cover_frames / 1.1
    cover = np.roll(song[:, np.round(true_song_frame).astype(int).clip(max=song.shape[1] - 1)], 2, axis=0)
    cover = np.abs(cover + rng.normal(0, 0.05, cover.shape))
    cover /= np.linalg.norm(cover, axis=0)

    result = align_backing(cover, np.zeros(cover.shape[1]), song)
    assert result["key_shift"] == 2
    assert result["cost"] < 0.1
    error_s = np.abs(result["song_t"] - true_song_frame * HOP)
    assert np.median(error_s) < 0.1 and np.percentile(error_s, 95) < 0.3
    # The cover starts mid-way through a held chord, which chroma cannot place to the frame
    assert result["song_start_s"] == pytest.approx(20, abs=1.0)


def test_song_to_cover_time_is_nan_outside_the_cover():
    alignment = {"cover_t": np.array([0.0, 10.0]), "song_t": np.array([30.0, 40.0]),
                 "song_start_s": 30.0, "song_end_s": 40.0}
    assert song_to_cover_time(alignment, [35.0])[0] == pytest.approx(5.0)
    assert np.isnan(song_to_cover_time(alignment, [10.0, 50.0])).all()


def frames_from(segments, hop=0.01):
    """Singer pitch frames from (midi or None, seconds) segments."""
    midi = np.concatenate([np.full(int(round(s / hop)), np.nan if m is None else m) for m, s in segments])
    return pd.DataFrame({"time_s": np.arange(len(midi)) * hop, "midi": midi})


def notes_at(*notes):
    return pd.DataFrame([{"target_midi": m, "expected_cover_s": t, "duration_s": d} for m, t, d in notes])


def test_pass2_reports_known_delays_exactly():
    # Targets at 1.0 s (60), 2.0 s (64), 3.0 s (62). The singer is 80 ms late, 50 ms early, on time.
    frames = frames_from([(None, 1.08), (60, 0.5), (None, 0.37), (64, 0.5), (None, 0.55), (62, 0.5)])
    notes = find_onsets(notes_at((60, 1.0, 0.5), (64, 2.0, 0.5), (62, 3.0, 0.5)), frames)
    assert notes["onset_found"].all()
    assert notes["onset_error_ms"].round().tolist() == [80, -50, 0]


def test_pass2_ignores_octave_and_small_pitch_errors():
    frames = frames_from([(None, 1.1), (48.6, 0.5)])         # an octave down and 60 cents flat
    notes = find_onsets(notes_at((60, 1.0, 0.5)), frames)
    assert notes["onset_error_ms"].round().tolist() == [100]


def test_pass2_no_onset_when_skipped_or_too_late():
    frames = frames_from([(None, 1.5), (60, 0.5), (None, 1.0)])   # sung 500 ms late: outside +/-300
    notes = find_onsets(notes_at((60, 1.0, 0.5), (67, 2.5, 0.5)), frames)   # second note never sung
    assert not notes["onset_found"].any()


def test_recording_octave():
    frames = frames_from([(None, 1.0), (48.2, 1.0), (52.1, 1.0)])
    notes = notes_at((60, 1.0, 1.0), (64, 2.0, 1.0))
    assert recording_octave(notes, frames) == -12
    assert recording_octave(notes.assign(target_midi=notes["target_midi"] - 12), frames) == 0


def test_pass1_on_real_audio():
    """The pilot original's backing, cut by 15 s, 5% slower and 2 semitones up, must map back."""
    import librosa
    import soundfile as sf
    from align import backing_chroma

    path = config.ORIGINAL_STEMS_DIR / "Adele - Someone Like You" / "no_vocals.wav"
    if not path.exists():
        pytest.skip("pilot stems not on this machine")
    y, sr = librosa.load(path, sr=config.ALIGN_SAMPLE_RATE, mono=True, duration=120)
    fake = librosa.effects.pitch_shift(librosa.effects.time_stretch(y[15 * sr:], rate=1 / 1.05), sr=sr, n_steps=2)

    tmp = config.DERIVED_DIR / "test_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    sf.write(tmp / "song.wav", y, sr)
    sf.write(tmp / "cover.wav", fake, sr)
    song, _ = backing_chroma(tmp / "song.wav")
    cover, cover_loudness = backing_chroma(tmp / "cover.wav")

    result = align_backing(cover, cover_loudness, song)
    assert result["key_shift"] == 2
    true_song_t = 15 + result["cover_t"] / 1.05
    error_s = np.abs(result["song_t"] - true_song_t)
    assert np.median(error_s) < 0.15, f"median error {np.median(error_s):.2f} s"
    assert np.percentile(error_s, 90) < 0.5


def test_pass1_cover_longer_than_song():
    """Cover = 15 s of unrelated intro + the whole song 1 semitone down + 10 s outro."""
    rng = np.random.default_rng(1)
    song = chord_sequence(rng, 120)
    intro, outro = chord_sequence(rng, 15), chord_sequence(rng, 10)
    cover = np.concatenate([intro, np.roll(song, -1, axis=0), outro], axis=1)

    result = align_backing(cover, np.zeros(cover.shape[1]), song)
    assert result["key_shift"] == -1
    inside = (result["cover_t"] >= intro.shape[1] * HOP + 2) & (result["cover_t"] <= (intro.shape[1] + song.shape[1]) * HOP - 2)
    error_s = np.abs(result["song_t"][inside] - (result["cover_t"][inside] - intro.shape[1] * HOP))
    assert np.median(error_s) < 0.1 and np.percentile(error_s, 95) < 0.3
