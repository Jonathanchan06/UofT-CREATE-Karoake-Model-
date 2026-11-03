"""Note maker on synthetic pitch tracks where the right answer is known exactly."""

import numpy as np
import pytest

from targets import frames_to_notes, notes_to_phrases, song_id_for, tuning_cents

HOP_S = 0.01


def track(*segments):
    """Pitch frames from (midi or None for silence, seconds) segments. Returns (midi, time_s)."""
    midi = np.concatenate([np.full(int(round(s / HOP_S)), np.nan if m is None else m, dtype=float)
                           for m, s in segments])
    return midi, np.arange(len(midi)) * HOP_S


def test_song_ids_are_file_safe():
    assert song_id_for("Bon Jovi - Livin’ on a Prayer") == "bon-jovi-livin-on-a-prayer"
    assert song_id_for("Britney Spears - ...Baby One More Time") == "britney-spears-baby-one-more-time"
    assert song_id_for("Beyoncé ft. Jay-Z - Crazy in Love") == "beyonce-ft-jay-z-crazy-in-love"


def test_tuning_cents_is_a_circular_mean():
    assert tuning_cents([60.2, 62.2, 64.2, np.nan]) == pytest.approx(20)
    assert tuning_cents([60.1, 61.9]) == pytest.approx(0, abs=1e-6)      # +10 and -10 cents
    assert abs(tuning_cents([60.49, 61.51])) == pytest.approx(50, abs=1)  # wraps, not 0


def test_clean_notes_and_rests():
    midi, t = track((None, 1.0), (60, 0.5), (64, 0.3), (None, 0.5), (62, 0.4))
    assert frames_to_notes(midi, t) == [(60, 1.0, 0.5), (64, 1.5, 0.3), (62, 2.3, 0.4)]


def test_vibrato_stays_one_note():
    midi, t = track((None, 0.2), (60, 1.0))
    midi[20:] += 0.45 * np.sin(2 * np.pi * 5.5 * t[20:])
    assert frames_to_notes(midi, t) == [(60, 0.2, 1.0)]


def test_glide_and_blip_are_dropped():
    # 60 -> 64 slide over 60 ms (each passing note < MIN_NOTE_S), then a 40 ms blip up at 72
    midi, t = track((60, 0.5), (61, 0.02), (62, 0.02), (63, 0.02), (64, 0.3), (72, 0.04), (64, 0.3))
    notes = frames_to_notes(midi, t)
    assert [n[0] for n in notes] == [60, 64]
    assert notes[1][1] == pytest.approx(0.56)


def test_short_dropout_inside_a_note_is_joined():
    midi, t = track((60, 0.4), (None, 0.03), (60, 0.4))
    assert frames_to_notes(midi, t) == [(60, 0.0, 0.83)]


def test_repeated_note_is_split_at_a_loudness_dip():
    midi, t = track((60, 1.0))
    loudness = np.full(len(t), -20.0)
    loudness[48:53] = -35.0                     # a 15 dB dip at 0.5 s (a new syllable)
    notes = frames_to_notes(midi, t, loudness)
    assert [n[0] for n in notes] == [60, 60]
    assert notes[1][1] == pytest.approx(0.48)
    assert frames_to_notes(midi, t) == [(60, 0.0, 1.0)]   # without loudness it stays one note


def test_phrases_split_at_long_rests():
    notes = [(60, 0.0, 0.5), (62, 0.6, 0.5), (64, 1.6, 0.5), (65, 2.2, 0.3)]
    assert notes_to_phrases(notes) == [notes[:2], notes[2:]]   # 0.5 s rest >= PHRASE_GAP_S
