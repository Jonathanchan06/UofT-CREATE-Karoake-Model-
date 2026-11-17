"""Labeling accuracy checks with planted errors (CLAUDE.md "Next steps", step 3). Two tests:

A. Synthetic voice, PESTO + scorer. A voice is rendered from a song's target notes (in the song's
   tuning, with vibrato and a small scoop into each note) with a different error planted in each
   phrase (PLANTS, cycling). The exact rendered pitch is known, so:
       truth  = the scorer on the exact pitch     pesto = the scorer on PESTO run on the voice
   Demucs is left out: it treats this synthetic "ah" as an instrument, not a voice.

B. Real voice, full cover pipeline. The same errors are planted into the original singer's own
   vocal stem (pitch-shifted phrases, a wrong note, muted notes, moved notes). It is mixed over the
   original's backing and the first COVER_CUT_S seconds are cut, like a cover with a trimmed intro.
       unplanted = PESTO on the untouched vocal    clean = PESTO on the planted vocal (no mix)
       pipeline  = Demucs -> PESTO -> pass 1 -> pass 1b -> scorer on the mix
   clean vs pipeline is what Demucs and the alignment cost; unplanted vs clean shows each plant.

Writes derived/synthetic/<song_id>/ (audio, stems, labels_a.csv, labels_b.csv). Usage:
    python labeling/synthetic_singer.py [--song "Adele - Someone Like You"]
"""

import argparse
import json
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

import config
from align import align_backing, backing_chroma, onset_envelope, refine_alignment, to_song_timeline
from pitch import load_mono, track_pitch
from targets import song_id_for

sys.path.insert(0, str(config.ROOT))
sys.path.insert(0, str(config.ROOT / "scorer"))
import score  # noqa: E402

SR = 44100
HOP_S = 0.01
COVER_CUT_S = 3.2                       # the "cover" starts this far into the song
VIBRATO_HZ, VIBRATO_CENTS = 5.5, 25.0   # starts 150 ms into notes longer than 300 ms
SCOOP_CENTS, SCOOP_S = -30.0, 0.04      # slides up into each note
FORMANTS = [(730, 90), (1090, 110), (2440, 170)]   # "ah": centre Hz, bandwidth Hz
FADE_S = 0.01
PLANTS = ["clean", "sharp_30", "sharp_60", "timing", "skip_two", "octave_down", "wrong_note", "late_120"]
LABELS = ["accuracy", "accuracy_strict", "timing_accuracy", "mean_onset_error_ms", "missed_note_rate"]


# ---------- Test A: synthetic voice ----------

def planted_notes(phrases):
    """[(phrase_idx, plant, sung_midi, start_s, end_s)] for every note that is sung, after the
    phrase's planted error."""
    out = []
    for p, phrase in enumerate(phrases):
        plant = PLANTS[p % len(PLANTS)]
        sung = []
        for k, note in enumerate(phrase):
            offset, start, end = 0.0, note["start_s"], note["start_s"] + note["duration_s"]
            if plant == "sharp_30":
                offset = 0.30
            elif plant == "sharp_60":
                offset = 0.60
            elif plant == "octave_down":
                offset = -12.0
            elif plant == "wrong_note" and k == 1:
                offset = 2.0
            elif plant == "late_120":
                start, end = start + 0.12, end + 0.12
            elif plant == "timing" and k == 1:
                start += 0.08                           # 80 ms late (ends on time)
            elif plant == "timing" and k == 2:
                start -= 0.05                           # 50 ms early: cuts the previous note short
                if sung:
                    sung[-1][4] = min(sung[-1][4], start)
            if plant == "skip_two" and k in (1, 3):
                continue
            sung.append([p, plant, note["midi_pitch"] + offset, start, end])
        out += sung
    return [tuple(n) for n in out]


def render_voice(notes, total_s, tuning_cents, seed=0):
    """(mono voice at SR, truth frames on a 10 ms grid with time_s, midi, confidence)."""
    rng = np.random.default_rng(seed)
    voice = np.zeros(int(total_s * SR), dtype=np.float64)
    grid_t = np.round(np.arange(0, total_s, HOP_S), 6)
    grid_midi = np.full(len(grid_t), np.nan)
    for _, _, midi, start, end in notes:
        a, b = int(start * SR), min(int(end * SR), len(voice))
        if b - a < int(0.03 * SR):
            continue
        t = np.arange(b - a) / SR
        cents = SCOOP_CENTS * np.clip(1 - t / SCOOP_S, 0, 1)
        if end - start > 0.3:
            ramp = np.clip((t - 0.15) / 0.1, 0, 1)
            cents = cents + ramp * VIBRATO_CENTS * np.sin(2 * np.pi * VIBRATO_HZ * t)
        cents = cents + np.convolve(rng.normal(0, 4, len(t)), np.ones(441) / 441, mode="same")
        f0_midi = midi + tuning_cents / 100 + cents / 100
        f0 = 440 * 2 ** ((f0_midi - 69) / 12)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        tone = np.zeros(len(t))
        for h in range(1, int(5000 / f0.min()) + 1):
            freq = h * f0
            gain = sum(1 / (1 + ((freq - fc) / (bw / 2)) ** 2) for fc, bw in FORMANTS) + 0.02
            tone += gain * h ** -1.2 * np.sin(h * phase)
        envelope = np.minimum(1, np.minimum(t / 0.02, (t[-1] - t) / 0.03))
        voice[a:b] += tone * envelope + rng.normal(0, 0.01, len(t)) * envelope
        # truth frames: the rendered pitch from the start of the note until its release
        frames = (grid_t >= start) & (grid_t < end - 0.03)
        grid_midi[frames] = np.interp(grid_t[frames], start + t, f0_midi)
    truth = pd.DataFrame({"time_s": grid_t, "midi": grid_midi,
                          "confidence": np.where(np.isnan(grid_midi), 0.1, 0.95)})
    return voice / np.abs(voice).max() * 0.5, truth


# ---------- Test B: real voice ----------

def _splice(out, start_s, new):
    """Write `new` into `out` from start_s, crossfading FADE_S at both ends."""
    a = int(round(start_s * SR))
    b = min(a + len(new), len(out))
    new = new[:b - a]
    w = np.ones(b - a)
    fade = min(int(FADE_S * SR), (b - a) // 2)
    if fade:
        w[:fade], w[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    out[a:b] = w * new + (1 - w) * out[a:b]


def plant_into_vocal(vocal, phrases):
    """The original singer's vocal with each phrase's planted error (same PLANTS as test A)."""
    out = vocal.copy()

    def span(t0, t1):
        return vocal[int(round(t0 * SR)):int(round(t1 * SR))]

    for p, phrase in enumerate(phrases):
        plant = PLANTS[p % len(PLANTS)]
        notes = [(n["start_s"], n["start_s"] + n["duration_s"]) for n in phrase]
        a, b = notes[0][0] - 0.1, notes[-1][1] + 0.1
        if plant in ("sharp_30", "sharp_60", "octave_down"):
            steps = {"sharp_30": 0.3, "sharp_60": 0.6, "octave_down": -12.0}[plant]
            _splice(out, a, librosa.effects.pitch_shift(span(a, b), sr=SR, n_steps=steps))
        elif plant == "wrong_note" and len(notes) > 1:
            s, e = notes[1]
            _splice(out, s - 0.02, librosa.effects.pitch_shift(span(s - 0.02, e + 0.02), sr=SR, n_steps=2.0))
        elif plant == "skip_two":
            for k in (1, 3):
                if k < len(notes):
                    s, e = notes[k]
                    _splice(out, s - 0.02, np.zeros(int((e - s + 0.04) * SR)))
        elif plant == "late_120":
            seg = span(a, b)
            _splice(out, a, np.concatenate([np.zeros(int(0.12 * SR)), seg[:-int(0.12 * SR)]]))
        elif plant == "timing" and len(notes) > 2:
            s1, e1 = notes[1]
            _splice(out, s1, np.concatenate([np.zeros(int(0.08 * SR)), span(s1, e1 - 0.08)]))  # 80 ms late
            s2, e2 = notes[2]
            _splice(out, s2 - 0.05, np.concatenate([span(s2, e2), np.zeros(int(0.05 * SR))]))  # 50 ms early
    return out


def run_pipeline(cover_wav, song_folder, stems_dir):
    """Demucs -> PESTO -> pass 1 -> pass 1b on a cover file. Returns (frames on song timeline, alignment)."""
    from demucs.pretrained import get_model
    from separate_covers import MODEL_NAME, separate

    model = get_model(MODEL_NAME)
    model.eval()
    separate(model, cover_wav, stems_dir, "cuda" if torch.cuda.is_available() else "cpu")
    frames = track_pitch(*load_mono(stems_dir / "vocals.wav"))
    song_backing = config.ORIGINAL_STEMS_DIR / song_folder / "no_vocals.wav"
    song_chroma, _ = backing_chroma(song_backing)
    cover_chroma, cover_loudness = backing_chroma(stems_dir / "no_vocals.wav")
    alignment = align_backing(cover_chroma, cover_loudness, song_chroma)
    alignment = refine_alignment(alignment, onset_envelope(stems_dir / "no_vocals.wav"), onset_envelope(song_backing))
    return to_song_timeline(frames, alignment), alignment


# ---------- Comparison ----------

def labels_for(frames, source, melody_path, tuning_cents, transpose=0):
    df = score.score_take(frames, melody_path, source, transpose=transpose, tuning_cents=tuning_cents)
    df["source"] = source
    df["plant"] = df["phrase_idx"].map(lambda p: PLANTS[p % len(PLANTS)])
    return df.drop(columns="target_notes")


def per_plant(labels, sources):
    """Median of each label per plant type and source."""
    table = labels.pivot_table(index="plant", columns="source", values=LABELS[:1] + LABELS[2:], aggfunc="median")
    table = table.reindex(PLANTS)
    table.columns = [f"{label}:{src}" for label, src in table.columns]
    return table[[f"{label}:{src}" for label in LABELS[:1] + LABELS[2:] for src in sources]]


def gap(labels, source, reference):
    """Median per-phrase |source - reference| for each label, over phrases both scored."""
    wide = labels.pivot_table(index="phrase_idx", columns="source", values=LABELS)
    out = {}
    for label in LABELS:
        both = wide[[(label, source), (label, reference)]].dropna()
        out[label] = (both[(label, source)] - both[(label, reference)]).abs().median()
    return pd.Series(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", default="Adele - Someone Like You")
    args = parser.parse_args()
    pd.set_option("display.width", 250)

    song_id = song_id_for(args.song)
    tuning = pd.read_csv(config.REAL_DATA_DIR / "songs.csv").set_index("song_id").loc[song_id, "tuning_cents"]
    melody_path = config.REAL_DATA_DIR / "melodies" / f"{song_id}.json"
    phrases = json.loads(melody_path.read_text())["phrases"]
    work = config.DERIVED_DIR / "synthetic" / song_id
    work.mkdir(parents=True, exist_ok=True)
    stems = config.ORIGINAL_STEMS_DIR / args.song
    backing, sr = sf.read(stems / "no_vocals.wav", dtype="float32", always_2d=True)
    assert sr == SR
    print(f"{args.song}: {len(phrases)} phrases, plants cycle through: {', '.join(PLANTS)}")

    # Test A
    voice, truth = render_voice(planted_notes(phrases), len(backing) / SR, tuning)
    sf.write(work / "voice.wav", voice.astype(np.float32), SR)
    a = pd.concat([labels_for(truth, "truth", melody_path, tuning),
                   labels_for(track_pitch(voice.astype(np.float32), SR), "pesto", melody_path, tuning)])
    a.to_csv(work / "labels_a.csv", index=False)
    print("\n=== Test A: synthetic voice, PESTO vs exact truth ===")
    print(f"phrases scored: {a.groupby('source').size().to_dict()}")
    print(per_plant(a, ["truth", "pesto"]).round(2).to_string())
    print("median per-phrase |pesto - truth|: " + ", ".join(f"{k} {v:.3f}" for k, v in gap(a, "pesto", "truth").items()))

    # Test B
    vocal, _ = load_mono(stems / "vocals.wav")
    planted = plant_into_vocal(vocal.astype(np.float64), phrases).astype(np.float32)
    sf.write(work / "vocal_planted.wav", planted, SR)
    mix = np.clip((backing + planted[:len(backing), None])[int(COVER_CUT_S * SR):], -1, 1)
    sf.write(work / "cover_real.wav", mix, SR, subtype="PCM_16")
    pipeline_frames, alignment = run_pipeline(work / "cover_real.wav", args.song, work / "stems_real")
    map_error_ms = np.abs(alignment["song_t"] - (alignment["cover_t"] + COVER_CUT_S)) * 1000
    b = pd.concat([labels_for(track_pitch(vocal, SR), "unplanted", melody_path, tuning),
                   labels_for(track_pitch(planted, SR), "clean", melody_path, tuning),
                   labels_for(pipeline_frames, "pipeline", melody_path, tuning, alignment["key_shift"])])
    b.to_csv(work / "labels_b.csv", index=False)
    print("\n=== Test B: original singer with planted errors, full cover pipeline ===")
    print(f"pass 1+1b: key {alignment['key_shift']:+d}, map error median {np.median(map_error_ms):.0f} ms, "
          f"p90 {np.percentile(map_error_ms, 90):.0f} ms")
    print(f"phrases scored: {b.groupby('source').size().to_dict()}; "
          f"transpose {b.groupby('source')['transpose'].first().to_dict()}")
    print(per_plant(b, ["unplanted", "clean", "pipeline"]).round(2).to_string())
    print("median per-phrase |pipeline - clean|: " + ", ".join(f"{k} {v:.3f}" for k, v in gap(b, "pipeline", "clean").items()))


if __name__ == "__main__":
    main()
