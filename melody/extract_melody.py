"""Answer keys: turn a song's vocals (or a karaoke MIDI file) into melodies/<song_id>.json.

The scorer compares every singer against this file, so a wrong note here becomes a wrong label for
everyone who sings the song. The steps are:

    vocals.wav --basic-pitch--> raw notes --cleanup--> one melody line --phrases--> melody JSON
    song.mid  ------------------------------^

Cleanup rules (all thresholds live in Config):
    1. drop notes outside a singable range or too quiet
    2. when two notes overlap, keep only the louder one (this removes ghost notes, which can be an
       octave or any other harmonic of the real note), and trim small overlaps between neighbours
    3. merge fragments of the same pitch separated by a tiny gap (one note split in two)
    4. drop notes that are too short to be sung, then merge again

Output format matches ml/data/.../melodies/<song_id>.json:
    {"song_id": ..., "tempo_bpm": ..., "phrases": [[{"midi_pitch", "start_s", "duration_s"}, ...], ...]}

Usage
    python melody/extract_melody.py --audio separated/htdemucs/<song>/vocals.wav --song-id song_00
    python melody/extract_melody.py --midi karaoke.mid --song-id song_00 --track Vocals
Add --tempo-audio original.mp3 to estimate the tempo from the full song (vocals alone are poor at it).
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class Config:
    """Thresholds for the cleanup and the phrase rules. Change them here, in one place."""
    min_midi: int = 36             # lowest note a singer could produce (C2)
    max_midi: int = 88             # highest (E6), anything above is almost certainly a ghost
    min_amplitude: float = 0.2     # basic-pitch confidence, 0 to 1
    overlap_frac: float = 0.5      # overlapping by more than this share of the shorter note = conflict
    merge_gap_s: float = 0.08      # same pitch closer together than this is one note
    min_note_s: float = 0.08       # shorter notes are blips, not singing
    phrase_gap_s: float = 0.6      # a silence this long starts a new phrase
    max_phrase_s: float = 10.0     # longer phrases are split at their longest gap
    max_range_semitones: int = 36  # a melody wider than this is suspicious (QC flag)
    min_notes: int = 20            # fewer notes than this is suspicious (QC flag)


class Note:
    """One note: pitch as a MIDI number, start and end in seconds, amplitude from 0 to 1."""
    __slots__ = ("pitch", "start", "end", "amp")

    def __init__(self, pitch, start, end, amp=1.0):
        self.pitch, self.start, self.end, self.amp = int(pitch), float(start), float(end), float(amp)

    @property
    def dur(self):
        return self.end - self.start

    def __repr__(self):
        return f"Note({self.pitch}, {self.start:.2f}-{self.end:.2f}, amp={self.amp:.2f})"



def transcribe(audio_path, cfg):
    """Raw notes from basic-pitch, a neural network that turns audio into note events."""
    from basic_pitch.inference import predict  # imported here so the rest works without it

    _, _, events = predict(
        str(audio_path),
        minimum_note_length=cfg.min_note_s * 1000,  # basic-pitch wants milliseconds
        minimum_frequency=440.0 * 2 ** ((cfg.min_midi - 69) / 12),
        maximum_frequency=440.0 * 2 ** ((cfg.max_midi - 69) / 12),
    )
    return [Note(pitch, start, end, amp) for start, end, pitch, amp, *_ in events]


def read_midi(path, track=None):
    """Raw notes from a karaoke MIDI file. Picks the vocal track unless `track` names one.

    Without a name it prefers a track called vocal, melody, lead or voice, otherwise the track whose
    notes overlap least (a melody is one note at a time) and that has the most notes."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    tracks = [t for t in midi.instruments if not t.is_drum and t.notes]
    if not tracks:
        raise ValueError(f"{path} has no pitched tracks")

    def one_at_a_time(t):
        notes = sorted(t.notes, key=lambda n: n.start)
        clashes = sum(a.end > b.start + 0.01 for a, b in zip(notes, notes[1:]))
        return 1 - clashes / max(len(notes), 1)

    if track is not None:
        chosen = [t for t in tracks if t.name.lower() == track.lower()]
        if not chosen:
            raise ValueError(f"no track named {track!r}; tracks are {[t.name for t in tracks]}")
    else:
        named = [t for t in tracks if any(w in t.name.lower() for w in ("vocal", "melody", "lead", "voice"))]
        chosen = named or sorted(tracks, key=lambda t: (one_at_a_time(t), len(t.notes)), reverse=True)
    picked = chosen[0]
    print(f"MIDI track used: {picked.name!r} ({len(picked.notes)} notes). Tracks: {[t.name for t in tracks]}")
    return [Note(n.pitch, n.start, n.end, n.velocity / 127.0) for n in picked.notes]



def overlap_s(a, b):
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def keep_one_line(notes, cfg):
    """Make the notes a single melody line: the louder of two overlapping notes wins."""
    kept = []
    for note in sorted(notes, key=lambda n: (n.start, -n.amp)):
        clashes = [k for k in kept if overlap_s(k, note) > cfg.overlap_frac * min(k.dur, note.dur)]
        if any(k.amp >= note.amp for k in clashes):
            continue  # a louder (or equal) note already holds this spot
        kept = [k for k in kept if k not in clashes] + [note]
    kept.sort(key=lambda n: n.start)
    for a, b in zip(kept, kept[1:]):  # small overlaps between neighbours: the earlier note ends sooner
        if a.end > b.start:
            a.end = b.start
    return kept


def merge_fragments(notes, cfg):
    """Join consecutive notes of the same pitch with a tiny gap between them."""
    merged = []
    for note in notes:
        last = merged[-1] if merged else None
        if last and last.pitch == note.pitch and note.start - last.end < cfg.merge_gap_s:
            last.end, last.amp = max(last.end, note.end), max(last.amp, note.amp)
        else:
            merged.append(Note(note.pitch, note.start, note.end, note.amp))
    return merged


def clean_notes(raw, cfg):
    """Raw notes in, one clean melody line out. Returns (notes, counts of what was dropped)."""
    stats = Counter(raw=len(raw))
    notes = [n for n in raw if cfg.min_midi <= n.pitch <= cfg.max_midi and n.amp >= cfg.min_amplitude]
    stats["dropped_range_or_quiet"] = len(raw) - len(notes)

    before = len(notes)
    notes = keep_one_line(notes, cfg)
    stats["dropped_overlap"] = before - len(notes)

    before = len(notes)
    notes = merge_fragments(notes, cfg)
    stats["merged_fragments"] = before - len(notes)

    before = len(notes)
    notes = [n for n in notes if n.dur >= cfg.min_note_s]
    stats["dropped_short"] = before - len(notes)

    notes = merge_fragments(notes, cfg)  # blips between two fragments are gone now, so merge again
    stats["final"] = len(notes)
    return notes, stats



def split_phrases(notes, cfg):
    """Group notes into phrases at silences, then cut any phrase that is too long."""
    phrases = []
    for note in notes:
        if phrases and note.start - phrases[-1][-1].end < cfg.phrase_gap_s:
            phrases[-1].append(note)
        else:
            phrases.append([note])

    result = []
    while phrases:
        phrase = phrases.pop(0)
        if len(phrase) > 1 and phrase[-1].end - phrase[0].start > cfg.max_phrase_s:
            gaps = [b.start - a.end for a, b in zip(phrase, phrase[1:])]
            cut = int(np.argmax(gaps)) + 1  # cut where the silence is longest
            phrases[:0] = [phrase[:cut], phrase[cut:]]
        else:
            result.append(phrase)
    return result



def estimate_tempo(audio_path):
    """Beats per minute. Reliable on a full song, poor on vocals alone."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return int(round(float(np.atleast_1d(tempo)[0])))


def guess_key(notes):
    """Best guess at the key's root note, from how long each pitch class is sung.

    Compares against the Krumhansl major and minor profiles. It's a guess: fix it by hand."""
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.dur
    if hist.sum() == 0:
        return None, None
    best = max(((np.corrcoef(hist, np.roll(profile, k))[0, 1], k, mode)
                for profile, mode in ((major, "major"), (minor, "minor")) for k in range(12)))
    return NOTE_NAMES[best[1]], best[2]


def write_melody_json(path, song_id, tempo, phrases):
    data = {"song_id": song_id, "tempo_bpm": tempo,
            "phrases": [[{"midi_pitch": n.pitch, "start_s": round(n.start, 4), "duration_s": round(n.dur, 4)}
                         for n in phrase] for phrase in phrases]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1))


def write_preview(path, phrases, sr=22050):
    """A WAV of the extracted melody (plain tones) to play next to the original and check by ear."""
    import soundfile as sf

    notes = [n for phrase in phrases for n in phrase]
    y = np.zeros(int((max(n.end for n in notes) + 0.5) * sr))
    for n in notes:
        t = np.arange(int(n.dur * sr)) / sr
        freq = 440.0 * 2 ** ((n.pitch - 69) / 12)
        tone = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3))
        fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fades so notes don't click
        start = int(n.start * sr)
        y[start: start + len(tone)] += 0.2 * tone * fade
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (y / max(1.0, np.abs(y).max())).astype(np.float32), sr)


def qc_report(notes, phrases, stats, cfg, tempo_source):
    """Numbers to look at before trusting an answer key, plus warning flags."""
    pitches = [n.pitch for n in notes]
    flags = []
    if len(notes) < cfg.min_notes:
        flags.append("very_few_notes")
    if pitches and max(pitches) - min(pitches) > cfg.max_range_semitones:
        flags.append("pitch_range_suspicious")
    if tempo_source in ("vocals", "default"):  # a guess from a weak source, worth a second look
        flags.append("tempo_from_" + tempo_source)
    report = {
        **{k: int(v) for k, v in stats.items()},
        "n_phrases": len(phrases),
        "lowest_midi": min(pitches) if pitches else None,
        "highest_midi": max(pitches) if pitches else None,
        "sung_seconds": round(sum(n.dur for n in notes), 1),
        "longest_phrase_s": round(max((p[-1].end - p[0].start for p in phrases), default=0), 1),
        "flags": flags,
    }
    return report



def extract(song_id, out_dir, audio=None, midi=None, track=None, tempo_audio=None, tempo=None,
            check_dir=None, cfg=None):
    """Make melodies/<song_id>.json (plus a listen-to file and QC report) for one song."""
    cfg = cfg or Config()
    if (audio is None) == (midi is None):
        raise ValueError("give exactly one of audio or midi")
    raw = transcribe(audio, cfg) if audio is not None else read_midi(midi, track)

    notes, stats = clean_notes(raw, cfg)
    if not notes:
        raise ValueError("no notes left after cleanup; is this the vocals file?")
    phrases = split_phrases(notes, cfg)

    if tempo is not None:
        tempo_source = "given"
    elif tempo_audio is not None:
        tempo, tempo_source = estimate_tempo(tempo_audio), "audio"
    elif audio is not None:
        tempo, tempo_source = estimate_tempo(audio), "vocals"
    else:
        tempo, tempo_source = 120, "default"

    out_path = Path(out_dir) / f"{song_id}.json"
    write_melody_json(out_path, song_id, tempo, phrases)

    report = qc_report(notes, phrases, stats, cfg, tempo_source)
    key, mode = guess_key(notes)
    report.update(tempo_bpm=tempo, key_guess=f"{key} {mode}" if key else None)
    if check_dir is not None:
        check = Path(check_dir) / song_id
        check.mkdir(parents=True, exist_ok=True)
        write_preview(check / "preview.wav", phrases)
        (check / "qc.json").write_text(json.dumps(report, indent=1))
    return out_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="separated vocals (Demucs vocals.wav)")
    source.add_argument("--midi", type=Path, help="karaoke MIDI file")
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--track", help="MIDI track name, if the automatic pick is wrong")
    parser.add_argument("--tempo-audio", type=Path, help="the full original song, for the tempo")
    parser.add_argument("--tempo", type=int, help="tempo in BPM if you know it")
    parser.add_argument("--out-dir", type=Path, default=Path("melodies"))
    parser.add_argument("--check-dir", type=Path, default=Path("melody_checks"))
    args = parser.parse_args()

    path, report = extract(args.song_id, args.out_dir, args.audio, args.midi, args.track,
                           args.tempo_audio, args.tempo, args.check_dir)
    print(f"saved {path}")
    print(json.dumps(report, indent=1))
    print(f"listen to {args.check_dir / args.song_id / 'preview.wav'} next to the original song")
    print("songs.csv row (fix the blanks and the key): "
          f"{args.song_id},{report['tempo_bpm']},{(report['key_guess'] or '?').split()[0]},<language>,<genre>")
    if report["flags"]:
        print("WARNING, check this answer key:", ", ".join(report["flags"]))


if __name__ == "__main__":
    main()