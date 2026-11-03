"""PESTO pitch tracking that keeps every frame.

track_pitch() returns one row per 10 ms frame with explicit units:
    time_s       frame time from the start of the audio, in seconds (PESTO itself gives ms)
    hz           pitch in Hz, NaN when unvoiced
    midi         pitch in fractional MIDI semitones, NaN when unvoiced (midi = 69 + 12*log2(hz/440))
    confidence   PESTO's voicing confidence, 0-1
    voiced       confidence >= CONFIDENCE_THRESHOLD
Low-confidence frames are marked unvoiced, never removed, so frame i is always at i * HOP_MS.
"""

import numpy as np
import pandas as pd
import pesto
import soundfile as sf
import torch

import config

_models = {}


def _model(sample_rate, device):
    key = (sample_rate, device)
    if key not in _models:
        _models[key] = pesto.load_model(config.PESTO_MODEL, step_size=config.HOP_MS,
                                        sampling_rate=sample_rate).to(device).eval()
    return _models[key]


def hz_to_midi(hz):
    return 69 + 12 * np.log2(np.asarray(hz, dtype=float) / 440)


def midi_to_hz(midi):
    return 440 * 2 ** ((np.asarray(midi, dtype=float) - 69) / 12)


def load_mono(path):
    """(mono float32 samples, sample rate)."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def _pesto_midi(samples, sample_rate, shift_cents, device):
    """One PESTO pass on audio pitch-shifted up by about shift_cents. The shift is done by
    telling PESTO the sample rate is higher; it is undone here, in pitch and in time.
    Returns (time_s, midi, confidence)."""
    claimed_rate = int(round(sample_rate * 2 ** (shift_cents / 1200)))
    factor = claimed_rate / sample_rate                   # the exact shift applied
    x = torch.from_numpy(samples).to(device)
    with torch.inference_mode():
        midi, confidence, _, _ = _model(claimed_rate, device)(x, sr=claimed_rate, convert_to_freq=False)
    midi = midi.cpu().numpy() - 12 * np.log2(factor)
    time_s = np.arange(len(midi)) * config.HOP_MS / 1000 * factor
    return time_s, midi, confidence.cpu().numpy()


def track_pitch(samples, sample_rate, copies=None, device=None):
    """Pitch for every frame of a mono signal, as a DataFrame (columns in the module docstring)."""
    copies = config.PITCH_SHIFT_COPIES if copies is None else copies
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bin_cents = 100 / 3                                   # PESTO has 3 bins per semitone
    time_s, midi, confidence = _pesto_midi(samples, sample_rate, 0.0, device)
    passes = [midi]
    for i in range(1, copies):
        shifted_t, shifted_midi, _ = _pesto_midi(samples, sample_rate, i * bin_cents / copies, device)
        passes.append(np.interp(time_s, shifted_t, shifted_midi))
    midi = np.mean(passes, axis=0)

    voiced = confidence >= config.CONFIDENCE_THRESHOLD
    midi = np.where(voiced, midi, np.nan)
    return pd.DataFrame({"time_s": time_s, "hz": midi_to_hz(midi), "midi": midi,
                         "confidence": confidence, "voiced": voiced})


def track_file(path, cache_path=None):
    """track_pitch() on an audio file, cached as parquet when cache_path is given."""
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    frames = track_pitch(*load_mono(path))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frames.to_parquet(cache_path, index=False)
    return frames


def loudness_db(samples, sample_rate, frame_times_s, window_s=None):
    """RMS loudness in dB around each frame time (centred window), aligned with track_pitch() frames."""
    window = int(round((window_s or config.LOUDNESS_WINDOW_S) * sample_rate))
    squares = np.concatenate([[0.0], np.cumsum(samples.astype(np.float64) ** 2)])
    centres = np.round(np.asarray(frame_times_s) * sample_rate).astype(int)
    lo = np.clip(centres - window // 2, 0, len(samples))
    hi = np.clip(centres + window // 2, 0, len(samples))
    energy = (squares[hi] - squares[lo]) / np.maximum(hi - lo, 1)
    return 10 * np.log10(energy + 1e-10)
