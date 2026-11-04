"""Scorer: turn one recording's pitch track into labeled phrase rows (stage 3 of the pipeline).

Input
    pitch CSV   written by program.py (PESTO): timestamp, frequency, confidence
    melody JSON ml/data/.../melodies/<song_id>.json, the answer key in the ORIGINAL key
Output
    rows in the shared dataset format (phrases_vX.Y.parquet), one row per phrase:
    song_id, singer_id, take_id, transpose, phrase_idx, target_notes (already shifted by
    transpose), accuracy, mean_cents_error, timing_accuracy, mean_onset_error_ms,
    missed_note_rate, plus quality columns (n_scored, bias_cents, lag_s, transpose_conf,
    transpose_source, qc_flag).

How one take is scored
    1. Load pitch, drop readings below the confidence threshold, convert Hz to MIDI.
    2. Remove the whole take's timing offset (late start, headphone latency).
    3. Find the key the singer used, unless the transpose is already known.
    4. For each phrase: line up the singing with the notes using DTW with a tight band, then
       measure the error in cents for every matched reading.
    5. accuracy = share of readings within the tolerance, mean_cents_error = average error.

Usage
    python scorer/score.py --pitch my_file_ORIGINAL.csv --melody melodies/song_00.json \
        --singer-id s1 --take 1 --out phrases_v0.1.parquet
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOP_S = 0.01  # PESTO's default step: one reading every 10 ms


@dataclass
class Config:
    """The label decisions the two teams agreed to settle. Change them here, in one place."""
    conf_threshold: float = 0.70      # readings below this are breaths, silence or consonants
    tolerance_cents: float = 50.0     # a reading is a hit if its error is within this
    # the right note in the wrong octave counts as correct
    octave_tolerant: bool = True
    # ignore the first 50 ms of a note, the pitch is still sliding
    skip_note_start_s: float = 0.05
    # smooth over 100 ms so vibrato doesn't count as wrong notes
    smooth_s: float = 0.10
    # after the global offset, DTW may move time by this much
    band_s: float = 0.10
    step_penalty: float = 0.3         # extra cost for each time-stretching DTW step
    max_lag_s: float = 0.6            # largest whole-take offset we will search for
    window_pad_s: float = 1.0         # extra singing searched on each side of a phrase
    min_scored_frames: int = 10       # drop phrases with fewer scored readings than this
    # a note started "on time" if within this of its start
    onset_tolerance_ms: float = 100
    key_search: int = 6               # try transposes from -6 to +6 semitones


def hz_to_midi(hz):
    """midi = 69 + 12 * log2(hz / 440). Zero or negative Hz (no pitch) becomes NaN."""
    hz = np.asarray(hz, dtype=float)
    out = np.full(hz.shape, np.nan)
    ok = hz > 0
    out[ok] = 69.0 + 12.0 * np.log2(hz[ok] / 440.0)
    return out


def load_pitch(path, cfg):
    """MIDI pitch for every 10 ms frame; NaN where there is no confident pitch."""
    df = pd.read_csv(path)
    t = df["timestamp"].to_numpy(dtype=float)
    if len(t) > 1 and np.median(np.diff(t)) > 1.0:
        t = t / 1000.0  # PESTO writes milliseconds; seconds files have steps of about 0.01
    midi = hz_to_midi(df["frequency"].to_numpy())
    midi[df["confidence"].to_numpy() < cfg.conf_threshold] = np.nan

    grid = np.full(int(round(t.max() / HOP_S)) + 1, np.nan)
    grid[np.round(t / HOP_S).astype(int)] = midi
    return grid


def smooth_voiced(midi, width):
    """Moving average inside each unbroken run of voiced frames (removes vibrato wobble)."""
    out = midi.copy()
    voiced = np.flatnonzero(np.isfinite(midi))
    if len(voiced) == 0 or width <= 1:
        return out
    for run in np.split(voiced, np.flatnonzero(np.diff(voiced) > 1) + 1):
        k = min(width, len(run))
        padded = np.pad(midi[run], (k // 2, k - 1 - k // 2), mode="edge")
        out[run] = np.convolve(padded, np.ones(k) / k, mode="valid")
    return out

def load_melody(path):
    """Returns (song_id, tempo_bpm, phrases), phrases = list of lists of (midi, start_s, dur_s)."""
    melody = json.loads(Path(path).read_text())
    phrases = [[(float(n["midi_pitch"]), float(n["start_s"]), float(n["duration_s"])) for n in notes]
               for notes in melody["phrases"]]
    return melody["song_id"], melody.get("tempo_bpm"), phrases


def notes_to_curve(notes, n_frames):
    """Target MIDI for each frame (NaN = rest) and seconds since the current note began."""
    target = np.full(n_frames, np.nan)
    since = np.full(n_frames, np.nan)
    for pitch, start, dur in notes:
        a = max(int(round(start / HOP_S)), 0)
        b = min(int(round((start + dur) / HOP_S)), n_frames)
        if b > a:
            target[a:b] = pitch
            since[a:b] = np.arange(b - a) * HOP_S
    return target, since


def estimate_global_lag(sung, target, max_lag_s):
    """Offset in frames for the whole take (positive = the singer is late).

    Uses only WHEN there is singing versus when a note is due, never the pitch, so it cannot
    hide a wrong note."""
    s, t = np.isfinite(sung).astype(float), np.isfinite(target).astype(float)
    n = min(len(s), len(t))
    s, t = s[:n], t[:n]
    if s.sum() == 0 or t.sum() == 0:
        return 0
    max_lag = int(round(max_lag_s / HOP_S))
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            score = float(np.dot(t[: n - lag], s[lag:]))
        else:
            score = float(np.dot(t[-lag:], s[: n + lag]))
        score -= 1e-3 * abs(lag)  # on ties prefer the smaller offset
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def shift_left(x, lag):
    """out[i] = x[i + lag], NaN where there is no data."""
    out = np.full(len(x), np.nan)
    if lag >= 0:
        out[: len(x) - lag] = x[lag:]
    else:
        out[-lag:] = x[: len(x) + lag]
    return out


def fold_cents(err):
    """Wrap an error into [-600, 600) cents so a whole octave off counts as zero."""
    return (err + 600.0) % 1200.0 - 600.0


def estimate_transpose(sung, target, cfg):
    """Which shift of the original key matches the singing best, with a confidence.

    Tries every shift from -key_search to +key_search and counts the share of readings that
    land within tolerance. Confidence = best share minus the runner up."""
    both = np.isfinite(sung[: len(target)]) & np.isfinite(target[: len(sung)])
    n = min(len(sung), len(target))
    diff = (sung[:n] - target[:n])[both[:n]]
    if len(diff) == 0:
        return 0, 0.0
    shifts = list(range(-cfg.key_search, cfg.key_search + 1))
    shares = np.array([np.mean(np.abs(fold_cents(100.0 * (diff - k))) <= cfg.tolerance_cents)
                       for k in shifts])
    # ties go to the shift closest to the original key
    order = sorted(range(len(shifts)),
                   key=lambda i: (-round(shares[i], 6), abs(shifts[i])))
    runner_up = shares[order[1]] if len(order) > 1 else 0.0
    return shifts[order[0]], float(shares[order[0]] - runner_up)


# --------------------------------------------------------------------------------------
# Step 4: align one phrase with a tight DTW band
# --------------------------------------------------------------------------------------
def cost_matrix(target_frames, window, octave_tolerant, cap=3.0, mismatch=1.5):
    """Cost of pairing each target frame (rows) with each singing frame (columns), in semitones."""
    t_ok = np.isfinite(target_frames)[:, None]
    s_ok = np.isfinite(window)[None, :]
    diff = np.nan_to_num(target_frames)[
        :, None] - np.nan_to_num(window)[None, :]
    if octave_tolerant:
        diff = (diff + 6.0) % 12.0 - 6.0
    cost = np.minimum(np.abs(diff), cap)
    return np.where(t_ok & s_ok, cost, np.where(t_ok ^ s_ok, mismatch, 0.0))


def dtw_align(cost, center, radius, penalty):
    """DTW restricted to a band around the diagonal. Diagonal steps are free, steps that stretch
    time cost `penalty`. Returns matched (target_frame, singing_frame) index arrays, or None."""
    n, m = cost.shape
    D = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(max(0, i + center - radius), min(m - 1, i + center + radius) + 1):
            if i == 0:
                D[i, j] = cost[i, j]
                continue
            best = D[i - 1, j] + penalty
            if j > 0:
                best = min(best, D[i - 1, j - 1], D[i, j - 1] + penalty)
            D[i, j] = cost[i, j] + best

    if not np.isfinite(D[-1]).any():
        return None
    i, j = n - 1, int(np.argmin(D[-1]))
    path = [(i, j)]
    while i > 0:
        options = [(D[i - 1, j] + penalty, i - 1, j)]
        if j > 0:
            options += [(D[i - 1, j - 1], i - 1, j - 1),
                        (D[i, j - 1] + penalty, i, j - 1)]
        _, i, j = min(options)
        path.append((i, j))
    path.reverse()
    return np.array([p[0] for p in path]), np.array([p[1] for p in path])

def summarize_errors(sung_midi, target_midi, tolerance_cents=50.0, octave_tolerant=True):
    """The phrase label from matched readings, or None if there is nothing to score."""
    err = 100.0 * (np.asarray(sung_midi, dtype=float) -
                   np.asarray(target_midi, dtype=float))
    if len(err) == 0:
        return None
    used = fold_cents(err) if octave_tolerant else err
    return {
        "accuracy": float(np.mean(np.abs(used) <= tolerance_cents)),
        "mean_cents_error": float(np.mean(np.abs(used))),
        "n_scored": int(len(err)),
        # positive = tends sharp, negative = tends flat
        "bias_cents": float(np.mean(used)),
    }


def score_phrase(sung_padded, target_padded, since_padded, a, b, pad, cfg):
    """Align the phrase made of frames a..b with the singing around it, then score it."""
    target_frames = target_padded[a + pad: b + pad]
    since_frames = since_padded[a + pad: b + pad]
    window = sung_padded[a: b + 2 * pad]

    radius = int(round(cfg.band_s / HOP_S))
    cost = cost_matrix(target_frames, window, cfg.octave_tolerant)
    aligned = dtw_align(cost, pad, radius, cfg.step_penalty)
    if aligned is None:
        return None
    rows, cols = aligned

    sung_vals, target_vals = [], []
    # one singing frame can match several target frames: take the middle one
    for j in np.unique(cols):
        i = int(np.median(rows[cols == j]))
        sung, target, since = window[j], target_frames[i], since_frames[i]
        if np.isfinite(sung) and np.isfinite(target) and since >= cfg.skip_note_start_s:
            sung_vals.append(sung)
            target_vals.append(target)
    return summarize_errors(sung_vals, target_vals, cfg.tolerance_cents, cfg.octave_tolerant)


def onset_stats(sung, notes, cfg):
    """Timing labels for a phrase: how late or early each note started, and which were missed.

    A note "starts" at the first run of 3 readings (30 ms) within the pitch tolerance of the note.
    The tolerance must stay below a semitone, or a neighbouring note would look like this one."""
    errors, missed = [], 0
    for pitch, start, dur in notes:
        a = max(int((start - 0.3) / HOP_S), 0)
        b = min(int((start + dur) / HOP_S), len(sung))
        segment = sung[a:b]
        close = np.isfinite(segment) & (
            np.abs(fold_cents(100.0 * (segment - pitch))) <= cfg.tolerance_cents)
        runs = np.flatnonzero(np.convolve(close, np.ones(
            3), mode="valid") == 3) if len(close) >= 3 else []
        if len(runs) == 0:
            missed += 1
        else:
            errors.append((a + runs[0]) * HOP_S - start)
    n = len(notes)
    on_time = sum(abs(e) * 1000 <= cfg.onset_tolerance_ms for e in errors)
    return {
        "timing_accuracy": on_time / n,
        "mean_onset_error_ms": float(np.mean(errors) * 1000) if errors else float("nan"),
        "missed_note_rate": missed / n,
    }


def shift_notes(notes, transpose):
    """Notes in the dataset's format, already moved by the transpose."""
    return [{"midi_pitch": int(round(p + transpose)), "start_s": s, "duration_s": d} for p, s, d in notes]


COLUMNS = ["song_id", "singer_id", "take_id", "transpose", "phrase_idx", "target_notes", "accuracy",
           "mean_cents_error", "timing_accuracy", "mean_onset_error_ms", "missed_note_rate",
           "n_scored", "bias_cents", "lag_s", "transpose_conf", "transpose_source", "qc_flag"]


def score_take(pitch_csv, melody_json, singer_id, take_id=1, transpose=None, cfg=None):
    """Score one recording. Pass `transpose` when it is known; otherwise it is estimated.
    Returns a DataFrame in the dataset format. df.attrs["dropped_phrases"] counts skipped phrases."""
    cfg = cfg or Config()
    song_id, _, phrases = load_melody(melody_json)
    all_notes = [n for phrase in phrases for n in phrase]
    sung = smooth_voiced(load_pitch(pitch_csv, cfg),
                         int(round(cfg.smooth_s / HOP_S)))

    n_frames = max(len(sung), int(
        np.ceil(max(s + d for _, s, d in all_notes) / HOP_S)) + 1)
    target, _ = notes_to_curve(all_notes, n_frames)
    sung_full = np.full(n_frames, np.nan)
    sung_full[: len(sung)] = sung

    lag = estimate_global_lag(sung_full, target, cfg.max_lag_s)
    sung_full = shift_left(sung_full, lag)

    if transpose is None:
        transpose, confidence = estimate_transpose(sung_full, target, cfg)
        source = "estimated"
    else:
        confidence, source = float("nan"), "given"

    flags = []
    due = np.isfinite(target)
    if due.any() and np.isfinite(sung_full[due]).mean() < 0.15:
        flags.append("mostly_silent")
    if source == "estimated" and confidence < 0.05:
        flags.append("unsure_key")
    if abs(lag) >= int(round(cfg.max_lag_s / HOP_S)) - 2:
        flags.append("lag_at_limit")

    # shift the sung pitch back by the transpose, so every phrase is scored against the answer
    # key in the original key. Equivalent to shifting the notes, and keeps the curves simple.
    pad = int(round(cfg.window_pad_s / HOP_S))
    sung_padded = np.full(n_frames + 2 * pad, np.nan)
    sung_padded[pad: pad + n_frames] = sung_full - transpose

    rows, dropped = [], 0
    for idx, notes in enumerate(phrases):
        a = int(round(notes[0][1] / HOP_S))
        b = int(round((notes[-1][1] + notes[-1][2]) / HOP_S))
        phrase_target, phrase_since = notes_to_curve(notes, n_frames)
        target_padded = np.concatenate(
            [np.full(pad, np.nan), phrase_target, np.full(pad, np.nan)])
        since_padded = np.concatenate(
            [np.full(pad, np.nan), phrase_since, np.full(pad, np.nan)])

        result = score_phrase(sung_padded, target_padded,
                              since_padded, a, b, pad, cfg)
        if result is None or result["n_scored"] < cfg.min_scored_frames:
            dropped += 1
            continue
        timing = onset_stats(sung_full - transpose, notes, cfg)
        rows.append({
            "song_id": song_id, "singer_id": singer_id, "take_id": int(take_id),
            "transpose": int(transpose), "phrase_idx": idx,
            "target_notes": shift_notes(notes, transpose),
            **{k: result[k] for k in ("accuracy", "mean_cents_error")}, **timing,
            "n_scored": result["n_scored"], "bias_cents": result["bias_cents"],
            "lag_s": lag * HOP_S, "transpose_conf": confidence, "transpose_source": source,
            "qc_flag": ";".join(flags),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.attrs["dropped_phrases"] = dropped
    return df


def append_to_dataset(df, out_path):
    """Add rows to a parquet file, replacing any rows for the same take and phrase."""
    out_path = Path(out_path)
    if out_path.exists():
        df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
        df = df.drop_duplicates(
            ["song_id", "singer_id", "take_id", "transpose", "phrase_idx"], keep="last")
    df.to_parquet(out_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pitch", required=True, type=Path,
                        help="pitch CSV from program.py")
    parser.add_argument("--melody", required=True, type=Path,
                        help="melodies/<song_id>.json")
    parser.add_argument("--singer-id", required=True)
    parser.add_argument("--take", type=int, default=1)
    parser.add_argument("--transpose", type=int, default=None,
                        help="leave out to estimate the key")
    parser.add_argument("--out", type=Path,
                        help="parquet file to create or add to")
    args = parser.parse_args()

    df = score_take(args.pitch, args.melody, args.singer_id,
                    args.take, args.transpose)
    print(df.drop(columns="target_notes").round(2).to_string(index=False))
    print(
        f"dropped phrases (too little singing): {df.attrs['dropped_phrases']}")
    if args.out:
        append_to_dataset(df, args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
