"""Two-pass alignment of a cover recording to the original song.

Pass 1 (singer-independent): line up the cover's backing track (Demucs no_vocals) with the
original's, using chroma (energy in each of the 12 note names) and DTW. This gives a time map
cover time <-> song time and the cover's key shift. The singer is not involved, so their
mistakes cannot bend the map.

Pass 1b (fine-tuning, also singer-independent): chroma is only good to about 100 ms, so every
second the map is nudged by lining up the backing tracks' onsets (drum hits, chord attacks) in
10 ms steps within +/-300 ms.

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


# ---------- Pass 1b: fine-tune the map ----------

def onset_envelope(path):
    """Onset strength of an audio file (peaks at drum hits and chord attacks), one value per
    ONSET_HOP_S, standardised to mean 0 and sd 1."""
    y, sr = librosa.load(path, sr=config.ALIGN_SAMPLE_RATE, mono=True)
    hop = int(round(config.ONSET_HOP_S * sr))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # 220 samples at 22050 Hz is 9.977 ms, not 10: put the frames on an exact ONSET_HOP_S grid,
    # or the error grows along the song (about 30 ms by 45 s)
    frame_s = np.arange(len(env)) * hop / sr
    env = np.interp(np.arange(0, frame_s[-1], config.ONSET_HOP_S), frame_s, env)
    return (env - env.mean()) / (env.std() + 1e-9)


def refine_alignment(alignment, cover_env, song_env):
    """Pass 1b. Chroma (pass 1) finds the right part of the song but is only good to about 100 ms.
    Every REFINE_STEP_S, take REFINE_WINDOW_S of the cover's backing onsets and find the shift
    (within +/-REFINE_MAX_SHIFT_S) that best lines them up with the song's backing onsets around
    where pass 1 maps them. Corrections are median-smoothed and added to the map. Uses only the
    backing tracks, never the singer.

    Returns a new alignment dict with song_t corrected, plus refine_windows (how many),
    refine_used (share with a clear match) and refine_median_abs_ms (typical correction)."""
    hop = config.ONSET_HOP_S
    win = int(round(config.REFINE_WINDOW_S / hop))
    lags_s = np.arange(-round(config.REFINE_MAX_SHIFT_S / hop), round(config.REFINE_MAX_SHIFT_S / hop) + 1) * hop
    cover_t, song_t = alignment["cover_t"], alignment["song_t"]
    song_grid = np.arange(len(song_env)) * hop

    centres, line_at_centre, shifts = [], [], []
    for centre in np.arange(cover_t[0] + config.REFINE_WINDOW_S / 2,
                            cover_t[-1] - config.REFINE_WINDOW_S / 2, config.REFINE_STEP_S):
        c0 = int(round(centre / hop)) - win // 2
        cover_win = cover_env[max(c0, 0):c0 + win]
        window_t = (c0 + np.arange(win)) * hop
        # Pass 1's map wobbles from frame to frame; use a straight line through it in this window
        # (follows tempo differences, drops the wobble)
        slope, intercept = np.polyfit(window_t, np.interp(window_t, cover_t, song_t), 1)
        centres.append(centre)
        line_at_centre.append(slope * centre + intercept)
        if c0 < 0 or len(cover_win) < win:
            shifts.append(np.nan)
            continue
        # The song's onsets along that line, shifted by each candidate lag: one row per lag
        mapped = slope * window_t + intercept
        candidates = np.interp(mapped[None, :] + lags_s[:, None], song_grid, song_env)
        a = candidates - candidates.mean(axis=1, keepdims=True)
        b = cover_win - cover_win.mean()
        r = (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)
        score = np.where(r >= config.REFINE_MIN_CORR, r - config.REFINE_LAG_PENALTY * np.abs(lags_s), -np.inf)
        shifts.append(lags_s[np.argmax(score)] if np.isfinite(score.max()) else np.nan)

    shifts = pd.Series(shifts, dtype=float)
    used = float(shifts.notna().mean()) if len(shifts) else 0.0
    smoothed = shifts.rolling(config.REFINE_SMOOTH_WINDOWS, center=True, min_periods=1).median()
    ok = smoothed.notna().to_numpy()
    refined = dict(alignment)
    if ok.any():
        # New map through one corrected anchor per window; outside the first/last anchor, pass 1's
        # map moved by the nearest correction
        anchor_c = np.array(centres)[ok]
        anchor_s = np.array(line_at_centre)[ok] + smoothed.to_numpy()[ok]
        new_t = np.interp(cover_t, anchor_c, anchor_s)
        before, after = cover_t < anchor_c[0], cover_t > anchor_c[-1]
        new_t[before] = song_t[before] + (anchor_s[0] - np.interp(anchor_c[0], cover_t, song_t))
        new_t[after] = song_t[after] + (anchor_s[-1] - np.interp(anchor_c[-1], cover_t, song_t))
        refined["song_t"] = np.maximum.accumulate(new_t)
        refined["song_start_s"] = float(refined["song_t"][0])
        refined["song_end_s"] = float(refined["song_t"][-1])
    refined.update(refine_centres=np.array(centres), refine_shifts=shifts.to_numpy(),
                   refine_windows=len(centres), refine_used=used,
                   refine_median_abs_ms=float(np.nanmedian(np.abs(smoothed)) * 1000) if ok.any() else 0.0)
    return refined


def to_song_timeline(frames, alignment, hop_s=0.01):
    """A cover's pitch frames (pitch.track_pitch format) moved onto the song's timeline with the
    time map, on a hop_s grid from 0 (nearest cover frame). Frames outside the part of the song the
    cover covers are unvoiced. This is what the scorer takes for covers."""
    cover_t, song_t = alignment["cover_t"], alignment["song_t"]
    inside = frames[(frames["time_s"] >= cover_t[0]) & (frames["time_s"] <= cover_t[-1])]
    mapped = np.interp(inside["time_s"], cover_t, song_t)
    grid = np.round(np.arange(0, mapped.max() + hop_s, hop_s), 6)
    idx = np.clip(np.searchsorted(mapped, grid), 0, len(mapped) - 1)
    close = np.abs(mapped[idx] - grid) <= hop_s
    return pd.DataFrame({"time_s": grid,
                         "midi": np.where(close, inside["midi"].to_numpy()[idx], np.nan),
                         "confidence": np.where(close, inside["confidence"].to_numpy()[idx], 0.0)})


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
