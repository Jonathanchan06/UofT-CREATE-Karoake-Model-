"""PESTO wrapper: units, every frame kept, and accuracy on tones of known pitch.
These go through PESTO, so they use tolerances, not exact values."""

import numpy as np
import pytest

from pitch import hz_to_midi, midi_to_hz, track_pitch

SR = 44100


def tone(midi, seconds):
    hz = 440 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(seconds * SR)) / SR
    return sum(np.sin(2 * np.pi * hz * h * t) / h for h in range(1, 6)).astype(np.float32) * 0.2


def test_hz_midi_conversions():
    assert hz_to_midi(440) == pytest.approx(69)
    assert hz_to_midi(261.6256) == pytest.approx(60, abs=1e-4)      # middle C
    assert midi_to_hz(81) == pytest.approx(880)


def test_units_and_every_frame_kept():
    audio = np.concatenate([np.zeros(SR, dtype=np.float32), tone(69, 1.0)])   # 1 s silence, 1 s A4
    frames = track_pitch(audio, SR)
    assert list(frames.columns) == ["time_s", "hz", "midi", "confidence", "voiced"]
    assert len(frames) == pytest.approx(201, abs=1)                  # 2 s at 10 ms, nothing removed
    assert np.diff(frames["time_s"]).mean() == pytest.approx(0.01)   # seconds, not ms
    silent, sung = frames[frames["time_s"] < 0.9], frames[frames["time_s"] > 1.1]
    assert not silent["voiced"].any() and silent["midi"].isna().all()
    assert sung["voiced"].all()
    assert sung["hz"].median() == pytest.approx(440, rel=0.005)


# Halfway between PESTO's 33.3-cent bins, where plain PESTO is off by about 16.7 cents
@pytest.mark.parametrize("midi", [52 + 1 / 6, 59 + 5 / 6, 64.5, 71 + 1 / 6])
def test_shifted_copies_beat_the_33_cent_grid(midi):
    def error_cents(copies):
        frames = track_pitch(tone(midi, 1.0), SR, copies=copies)
        return (frames["midi"].iloc[15:-15] - midi).abs().median() * 100

    assert error_cents(copies=1) > 12          # the grid problem is really there
    assert error_cents(copies=3) < 9
