"""Two-pass alignment of a cover recording to the original song.

Pass 1 (singer-independent): line up the cover's backing track (Demucs no_vocals) with the
original's, using chroma (energy in each of the 12 note names) and DTW. This gives a time map
cover time <-> song time and the cover's key shift. The singer is not involved, so their
mistakes cannot bend the map.

Pass 2 (per note): with the map in place, look within +/-ONSET_SEARCH_S of where each target note
should start for where the singer actually started it. That offset is the timing error.

All times are seconds unless the name says _ms. Pitches are MIDI semitones.
"""

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import config

# DTW steps (cover frames, song frames): normal speed, or up to 2x faster/slower locally
STEP_SIZES = np.array([[1, 1], [1, 2], [2, 1]])
STEP_WEIGHTS = np.array([2.0, 3.0, 3.0])


# ---------- Pass 1 ----------

def backing_chroma(path):
    """(chroma: 12 x frames, unit length per frame; loudness_db per frame) for an audio file."""
    y, sr = librosa.load(path, sr=config.ALIGN_SAMPLE_RATE, mono=True)
    hop = int(round(config.CHROMA_HOP_S * sr))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=config.CHROMA_N_FFT, hop_length=hop, norm=None)
    chroma = np.log1p(config.CHROMA_LOG_GAMMA * chroma) + 1e-3      # +1e-3: silence is not a zero vector
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    rms = librosa.feature.rms(y=y, frame_length=config.CHROMA_N_FFT, hop_length=hop)[0]
    loudness_db = 20 * np.log10(rms + 1e-10)
    return chroma, loudness_db[:chroma.shape[1]]


def key_candidates(cover_chroma, song_chroma, count=None):
    """Key shifts (semitones, -6..+5) ranked by how well the cover's average pitch-class profile,
    shifted back, matches the song's. Positive = the cover is higher than the original."""
    cover_profile, song_profile = cover_chroma.mean(axis=1), song_chroma.mean(axis=1)
    scores = {k: np.corrcoef(np.roll(cover_profile, -k), song_profile)[0, 1] for k in range(-6, 6)}
    return sorted(scores, key=scores.get, reverse=True)[:count or config.KEY_CANDIDATES]


def _dtw(cover_chroma, song_chroma):
    """Subsequence DTW: the shorter recording must match some stretch of the longer one
    (an excerpt inside the song, or the song inside a cover with an extra intro or outro).
    Returns (path as [(cover_frame, song_frame)] ascending, mean cosine distance along it)."""
    cost = 1 - cover_chroma.T @ song_chroma                          # cosine distance
    # librosa needs the query (rows) to be the shorter one; its path comes out wrong otherwise
    swap = cost.shape[0] > cost.shape[1]
    _, path = librosa.sequence.dtw(C=cost.T if swap else cost, subseq=True,
                                   step_sizes_sigma=STEP_SIZES, weights_mul=STEP_WEIGHTS)
    path = path[::-1, ::-1] if swap else path[::-1]
    # Path cost relative to an average (random) frame pair: about 0.3 when the backing really
    # matches, 0.6+ when it is a different arrangement. Scale-free, unlike the raw cost.
    return path, float(cost[path[:, 0], path[:, 1]].mean() / cost.mean())


def _trim(loudness_db):
    """(first, last+1) frames of the backing that are not near-silent."""
    loud = np.flatnonzero(loudness_db > loudness_db.max() + config.TRIM_SILENCE_DB)
    return (int(loud[0]), int(loud[-1]) + 1) if len(loud) else (0, len(loudness_db))


def align_backing(cover_chroma, cover_loudness, song_chroma):
    """Pass 1. Returns a dict:
        key_shift      semitones the cover's backing is above the original (-6..+5)
        cost           path cost / average frame-pair cost (about 0.3 = matched, 0.6+ = did not)
        slope          song seconds per cover second over the map (about 1 = same tempo)
        cover_t, song_t  the time map (seconds), both non-decreasing, one point per cover frame
        song_start_s, song_end_s  the part of the song the cover covers (excerpts are shorter)
    """
    hop = config.CHROMA_HOP_S
    first, last = _trim(cover_loudness)
    cover = cover_chroma[:, first:last]

    best = None
    for key in key_candidates(cover, song_chroma):
        path, cost = _dtw(np.roll(cover, -key, axis=0), song_chroma)
        if best is None or cost < best[2]:
            best = (key, path, cost)
    key, path, cost = best

    # One song time per cover frame, smoothed and kept monotonic. Steps can skip frames, so
    # interpolate along the path; cover frames before/after the path (extra intro or outro)
    # get the song's first/last matched time and are cut from the map.
    path = path[np.r_[True, np.diff(path[:, 0]) > 0]]                 # one point per cover frame
    frames = np.arange(path[0, 0], path[-1, 0] + 1)
    song_frame = np.interp(frames, path[:, 0], path[:, 1])
    size = max(1, int(round(config.MAP_SMOOTH_S / hop)) | 1)
    song_frame = np.maximum.accumulate(median_filter(song_frame, size=size, mode="nearest"))

    cover_t = (frames + first) * hop
    song_t = song_frame * hop
    slope = float((song_t[-1] - song_t[0]) / max(cover_t[-1] - cover_t[0], hop))
    return {"key_shift": int(key), "cost": cost, "slope": slope, "cover_t": cover_t, "song_t": song_t,
            "song_start_s": float(song_t[0]), "song_end_s": float(song_t[-1])}


def song_to_cover_time(alignment, song_s):
    """Map song times to cover times; NaN outside the part of the song the cover covers."""
    song_t = alignment["song_t"] + np.arange(len(alignment["song_t"])) * 1e-9   # strictly increasing
    song_s = np.asarray(song_s, dtype=float)
    out = np.interp(song_s, song_t, alignment["cover_t"])
    inside = (song_s >= alignment["song_start_s"]) & (song_s <= alignment["song_end_s"])
    return np.where(inside, out, np.nan)


# ---------- Pass 2 ----------

def _folded_distance(midi, target):
    """Distance in semitones ignoring octave: 61 vs 73 -> 0, 61 vs 62 -> 1."""
    return np.abs((midi - target + 6) % 12 - 6)


def recording_octave(notes, frames, tuning_cents=0.0):
    """Octave (multiple of 12) the singer sang in relative to the key-shifted notes, from the median
    pitch difference over the notes' expected spans. Used so a man singing a song an octave down
    gets transpose = key - 12 and the range features describe what he actually sang."""
    diffs = []
    times, midi = frames["time_s"].to_numpy(), frames["midi"].to_numpy()
    for note in notes.itertuples(index=False):
        if np.isnan(note.expected_cover_s):
            continue
        inside = (times >= note.expected_cover_s) & (times < note.expected_cover_s + note.duration_s)
        diffs.append(midi[inside] - (note.target_midi + tuning_cents / 100))
    diffs = np.concatenate(diffs) if diffs else np.array([])
    diffs = diffs[~np.isnan(diffs)]
    return int(12 * np.round(np.median(diffs) / 12)) if len(diffs) else 0


def find_onsets(notes, frames, tuning_cents=0.0):
    """Pass 2. For each note, the start of the singer's run near its pitch that is closest to
    where pass 1 expects it, within +/-ONSET_SEARCH_S.

    notes: DataFrame with target_midi (already key-shifted), expected_cover_s, duration_s
    frames: the cover's vocal pitch frames from pitch.track_pitch()
    Adds columns: onset_found (bool), sung_onset_s, onset_error_ms (signed, negative = early).
    Notes with no clear onset (e.g. legato into the same pitch) get onset_found = False.
    """
    times, midi = frames["time_s"].to_numpy(), frames["midi"].to_numpy()
    hop_s = float(np.median(np.diff(times)))
    min_run = max(1, int(round(config.ONSET_MIN_RUN_S / hop_s)))

    found, sung = [], []
    for note in notes.itertuples(index=False):
        onset = np.nan
        if not np.isnan(note.expected_cover_s):
            lo = np.searchsorted(times, note.expected_cover_s - config.ONSET_SEARCH_S)
            hi = np.searchsorted(times, note.expected_cover_s + config.ONSET_SEARCH_S)
            target = note.target_midi + tuning_cents / 100
            near = _folded_distance(midi[max(lo - 1, 0):hi + min_run], target) <= config.ONSET_PITCH_TOLERANCE_SEMITONES
            near &= ~np.isnan(midi[max(lo - 1, 0):hi + min_run])
            offset = max(lo - 1, 0)
            starts = [i for i in range(1, len(near) - min_run + 1)
                      if near[i] and not near[i - 1] and near[i:i + min_run].all()
                      and lo <= offset + i < hi]
            if starts:
                best = min(starts, key=lambda i: abs(times[offset + i] - note.expected_cover_s))
                onset = float(times[offset + best])
        found.append(not np.isnan(onset))
        sung.append(onset)

    notes = notes.copy()
    notes["onset_found"] = found
    notes["sung_onset_s"] = sung
    notes["onset_error_ms"] = (notes["sung_onset_s"] - notes["expected_cover_s"]) * 1000
    return notes


def note_table(melody, alignment):
    """One row per target note with where pass 1 expects it in the cover.
    target_midi is key-shifted (octave not yet applied)."""
    rows = []
    for phrase_idx, phrase in enumerate(melody["phrases"]):
        for note_idx, note in enumerate(phrase):
            rows.append({"phrase_idx": phrase_idx, "note_idx": note_idx,
                         "song_midi": note["midi_pitch"], "start_s": note["start_s"],
                         "duration_s": note["duration_s"]})
    notes = pd.DataFrame(rows)
    notes["target_midi"] = notes["song_midi"] + alignment["key_shift"]
    notes["expected_cover_s"] = song_to_cover_time(alignment, notes["start_s"])
    return notes
