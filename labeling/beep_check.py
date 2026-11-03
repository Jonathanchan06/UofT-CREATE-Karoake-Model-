"""Listening check for target notes: the original song with the target notes played as beeps.

If the beeps follow the melody (right notes, right times, nothing missing or extra), the answer
key is good. Also saves a picture of the vocal pitch with the notes drawn on top.

Writes derived/beep_check/<song_id>.wav and <song_id>.png. Run make_targets.py first. Usage:
    python labeling/beep_check.py [--all]
"""

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from pitch import midi_to_hz

FADE_S = 0.01


def render_beeps(phrases, tuning, num_samples, sample_rate):
    """Each target note as a soft tone at its pitch (tuning put back), silent elsewhere."""
    out = np.zeros(num_samples, dtype=np.float32)
    fade = int(FADE_S * sample_rate)
    for phrase in phrases:
        for note in phrase:
            start = int(note["start_s"] * sample_rate)
            length = min(int(note["duration_s"] * sample_rate), num_samples - start)
            if length <= 2 * fade:
                continue
            hz = float(midi_to_hz(note["midi_pitch"] + tuning / 100))
            t = np.arange(length) / sample_rate
            tone = np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(4 * np.pi * hz * t)
            envelope = np.ones(length)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            out[start:start + length] += (tone * envelope / 1.3).astype(np.float32)
    return out


def plot_song(frames, phrases, tuning, path, windows=3, window_s=20.0):
    """A few 20 s windows of vocal pitch (grey) with the target notes (blue bars)."""
    voiced_times = frames.loc[frames["voiced"], "time_s"]
    starts = np.linspace(voiced_times.min(), max(voiced_times.min(), voiced_times.max() - window_s), windows)
    fig, axes = plt.subplots(windows, 1, figsize=(14, 3 * windows))
    for ax, t0 in zip(np.atleast_1d(axes), starts):
        part = frames[(frames["time_s"] >= t0) & (frames["time_s"] < t0 + window_s)]
        ax.plot(part["time_s"], part["midi"] - tuning / 100, ".", ms=1.5, color="0.5", label="vocal pitch")
        for idx, phrase in enumerate(phrases):
            for note in phrase:
                if t0 <= note["start_s"] < t0 + window_s:
                    ax.hlines(note["midi_pitch"], note["start_s"], note["start_s"] + note["duration_s"],
                              color="#3060c0", lw=4)
                    if note is phrase[0]:
                        ax.annotate(f"p{idx}", (note["start_s"], note["midi_pitch"] + 1), fontsize=7)
        ax.set_xlim(t0, t0 + window_s)
        ax.set_ylabel("MIDI")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("time from song start (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    songs = pd.read_csv(config.REAL_DATA_DIR / "songs.csv")
    if not args.all:
        songs = songs[songs["folder"].isin(config.PILOT_SONGS)]
    out_dir = config.DERIVED_DIR / "beep_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    for song in songs.itertuples(index=False):
        melody = json.loads((config.REAL_DATA_DIR / "melodies" / f"{song.song_id}.json").read_text())
        audio, sample_rate = sf.read(config.ORIGINALS_DIR / song.folder / "original.wav",
                                     dtype="float32", always_2d=True)
        beeps = render_beeps(melody["phrases"], song.tuning_cents, len(audio), sample_rate)
        mix = np.clip(config.SONG_LEVEL * audio + config.BEEP_LEVEL * beeps[:, None], -1, 1)
        sf.write(out_dir / f"{song.song_id}.wav", mix, sample_rate, subtype="PCM_16")

        frames = pd.read_parquet(config.DERIVED_DIR / "pitch" / "originals" / f"{song.song_id}.parquet")
        plot_song(frames, melody["phrases"], song.tuning_cents, out_dir / f"{song.song_id}.png")
        print(f"{song.folder}: {out_dir / song.song_id}.wav / .png")


if __name__ == "__main__":
    main()
