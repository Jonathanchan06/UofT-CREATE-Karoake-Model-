"""Target notes (the answer key) from an original song's vocal stem.

    pitch frames -> tuning_cents -> notes (midi_pitch, start_s, duration_s) -> phrases

Notes are whole MIDI numbers in the original key, with times in seconds from the start of the
song. tuning_cents is how far the original sits from A440 (e.g. +20 = 20 cents sharp); the
notes are taken after removing it, and the scorer applies it again when comparing a recording.
"""

import re
import unicodedata

import numpy as np
from scipy.ndimage import median_filter

import config


def song_id_for(folder_name):
    """'Bon Jovi - Livin’ on a Prayer' -> 'bon-jovi-livin-on-a-prayer' (safe as a file name)."""
    ascii_name = unicodedata.normalize("NFKD", folder_name).encode("ascii", "ignore").decode()
    ascii_name = ascii_name.replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def tuning_cents(midi):
    """Offset of the voiced frames from the nearest semitone, in cents (-50 to +50).
    A circular mean, so +49 and -49 cents average to +/-50, not 0."""
    midi = np.asarray(midi, dtype=float)
    midi = midi[~np.isnan(midi)]
    if len(midi) == 0:
        return 0.0
    angles = 2 * np.pi * (midi % 1.0)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float(mean_angle / (2 * np.pi) * 100)


def _smooth(midi, hop_s):
    """Median filter over voiced frames only; unvoiced frames stay NaN."""
    size = max(1, int(round(config.SMOOTH_S / hop_s)))
    if size < 2:
        return midi
    filled = np.where(np.isnan(midi), 0.0, midi)
    smoothed = median_filter(filled, size=size, mode="nearest")
    # A median that mixed in unvoiced zeros is not a pitch; keep the raw value there
    near_gap = median_filter(np.isnan(midi).astype(float), size=size, mode="nearest") > 0
    smoothed = np.where(near_gap, midi, smoothed)
    return np.where(np.isnan(midi), np.nan, smoothed)


def _label_frames(midi):
    """Per-frame note number with hysteresis, so vibrato near a note boundary does not flicker.
    -1 = unvoiced."""
    labels = np.full(len(midi), -1)
    current = None
    for i, m in enumerate(midi):
        if np.isnan(m):
            current = None
            continue
        if current is None or abs(m - current) > config.NOTE_HYSTERESIS_SEMITONES:
            current = int(round(m))
        labels[i] = current
    return labels


def _runs(labels):
    """[(start, end_exclusive, label)] for runs of equal labels, unvoiced (-1) runs left out."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != -1:
                runs.append((start, i, int(labels[start])))
            start = i
    return runs


def _split_at_loudness_dips(start, end, loudness, hop_s):
    """Split frames [start, end) at loudness dips at least ONSET_DIP_DB deep (repeated notes)."""
    reach = max(1, int(round(config.ONSET_DIP_WINDOW_S / hop_s)))
    min_len = max(1, int(round(config.MIN_NOTE_S / hop_s)))
    cuts = [start]
    for i in range(start + min_len, end - min_len):
        before = loudness[max(start, i - reach):i]
        after = loudness[i + 1:min(end, i + 1 + reach)]
        if (loudness[i] == loudness[max(start, i - reach):min(end, i + reach + 1)].min()
                and before.max() - loudness[i] >= config.ONSET_DIP_DB
                and after.max() - loudness[i] >= config.ONSET_DIP_DB
                and i - cuts[-1] >= min_len):
            cuts.append(i)
    cuts.append(end)
    return list(zip(cuts[:-1], cuts[1:]))


def frames_to_notes(midi, time_s, loudness=None):
    """Notes [(midi_pitch, start_s, duration_s)] from a pitch track.

    midi: fractional MIDI per frame (NaN = unvoiced), tuning already removed
    time_s: frame times in seconds (evenly spaced)
    loudness: optional loudness in dB per frame, used to split repeated notes on one pitch
    """
    midi = np.asarray(midi, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    hop_s = float(np.median(np.diff(time_s)))
    smoothed = _smooth(midi, hop_s)
    min_frames = int(round(config.MIN_NOTE_S / hop_s))
    merge_frames = int(round(config.MERGE_GAP_S / hop_s))

    # 1. Runs of one note number; short runs (glides, blips) are dropped
    runs = [r for r in _runs(_label_frames(smoothed)) if r[1] - r[0] >= min_frames]

    # 2. Join same-pitch runs split by a short unvoiced gap (tracker dropouts)
    joined = []
    for run in runs:
        if joined and run[2] == joined[-1][2] and run[0] - joined[-1][1] <= merge_frames:
            joined[-1] = (joined[-1][0], run[1], run[2])
        else:
            joined.append(run)

    # 3. Split runs at loudness dips (same pitch sung twice), then measure each note
    notes = []
    for start, end, _ in joined:
        pieces = _split_at_loudness_dips(start, end, loudness, hop_s) if loudness is not None else [(start, end)]
        for a, b in pieces:
            if b - a < min_frames:
                continue
            pitch = int(round(np.nanmedian(smoothed[a:b])))
            notes.append((pitch, round(float(time_s[a]), 3), round(float(b - a) * hop_s, 3)))
    return notes


def notes_to_phrases(notes):
    """Group notes into phrases: a rest of PHRASE_GAP_S or more starts a new phrase."""
    phrases = []
    for note in notes:
        if phrases:
            prev = phrases[-1][-1]
            if note[1] - (prev[1] + prev[2]) < config.PHRASE_GAP_S:
                phrases[-1].append(note)
                continue
        phrases.append([note])
    return phrases


def melody_json(song_id, phrases, tempo_bpm=None):
    """The melodies/<song_id>.json structure from the data contract."""
    return {
        "song_id": song_id,
        "tempo_bpm": tempo_bpm,
        "phrases": [[{"midi_pitch": p, "start_s": s, "duration_s": d} for p, s, d in phrase]
                    for phrase in phrases],
    }
