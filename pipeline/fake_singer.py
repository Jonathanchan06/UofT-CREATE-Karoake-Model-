"""Make a fake recording from an answer key, in the same CSV format PESTO writes.

It lets the whole pipeline run with no audio. The fake singer follows the melody but is late, a
little out of tune, drifts through vibrato, and sometimes sings a wrong note. Because we choose
these flaws, we know roughly what the scorer should find.
"""

import json
from pathlib import Path

import numpy as np

HOP_S = 0.01


def simulate_singer(melody_json, out_csv, seed=0, transpose=0, lag_s=0.15, cents_noise=18.0,
                    wrong_note_rate=0.1, wrong_by=2, ms_timestamps=True):
    """Write a pitch CSV (timestamp, frequency, confidence). Returns the planted truth as a dict."""
    rng = np.random.default_rng(seed)
    melody = json.loads(Path(melody_json).read_text())
    notes = [n for phrase in melody["phrases"] for n in phrase]
    n_frames = int((max(n["start_s"] + n["duration_s"]
                   for n in notes) + lag_s + 1.0) / HOP_S)
    t = np.arange(n_frames) * HOP_S

    midi = np.full(n_frames, np.nan)
    wrong = 0
    for n in notes:
        pitch = n["midi_pitch"] + transpose
        if rng.random() < wrong_note_rate:
            pitch += wrong_by * rng.choice([-1, 1])
            wrong += 1
        a = int(round((n["start_s"] + lag_s) / HOP_S))
        b = int(round((n["start_s"] + lag_s + n["duration_s"]) / HOP_S))
        midi[a:b] = pitch

    voiced = np.isfinite(midi)
    # +-15 cents wobble
    vibrato = 0.15 * np.sin(2 * np.pi * 5.5 * t)
    drift = np.convolve(rng.normal(0, cents_noise / 100,
                        n_frames), np.ones(30) / 30, mode="same") * 3
    midi = midi + vibrato + drift

    # silence gets a junk pitch...
    freq = 440.0 * 2 ** ((np.where(voiced, midi, 60.0) - 69) / 12)
    conf = np.where(voiced, rng.uniform(0.85, 0.99, n_frames),
                    rng.uniform(0.05, 0.4, n_frames))
    # ...with low confidence
    stamps = t * 1000.0 if ms_timestamps else t

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, np.column_stack((stamps, freq, conf)), delimiter=",",
               header="timestamp,frequency,confidence", comments="", fmt=["%.4f", "%.3f", "%.3f"])
    return {"n_notes": len(notes), "n_wrong_notes": wrong, "lag_s": lag_s, "transpose": transpose}
