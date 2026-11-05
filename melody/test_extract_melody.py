"""Tests for extract_melody.py. The basic-pitch round trip is skipped if basic-pitch isn't installed."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scorer"))
import extract_melody as em  # noqa: E402

CFG = em.Config()


def N(pitch, start, end, amp=0.8):
    return em.Note(pitch, start, end, amp)


def tiny_example():
    return [N(64, 0.00, 0.31), N(64, 0.33, 0.52, 0.7),   # one note split in two
            N(66, 0.55, 0.60, 0.5),                       # 50 ms blip
            N(67, 0.60, 1.05, 0.9), N(55, 0.62, 1.00, 0.4),  # a quieter ghost under a real note
            N(69, 2.30, 2.90)]                            # after a long silence


def test_tiny_example_cleanup():
    notes, stats = em.clean_notes(tiny_example(), CFG)
    assert [(n.pitch, round(n.start, 2), round(n.end, 2)) for n in notes] == [
        (64, 0.00, 0.52), (67, 0.60, 1.05), (69, 2.30, 2.90)]
    assert stats["merged_fragments"] == 1
    assert stats["dropped_overlap"] == 1   # the ghost
    assert stats["dropped_short"] == 1     # the blip


def test_tiny_example_phrases():
    notes, _ = em.clean_notes(tiny_example(), CFG)
    phrases = em.split_phrases(notes, CFG)
    assert [[n.pitch for n in p] for p in phrases] == [[64, 67], [69]]


def test_ghost_at_any_harmonic_is_removed():
    """The probe on synthetic singing found a faint note 28 semitones above a real one."""
    notes, _ = em.clean_notes([N(69, 4.0, 4.52, 0.88), N(97, 4.01, 4.4, 0.34)], CFG)
    assert [n.pitch for n in notes] == [69]


def test_out_of_range_and_quiet_notes_dropped():
    notes, stats = em.clean_notes([N(20, 0, 1), N(100, 1, 2), N(60, 2, 3, 0.05), N(60, 3, 4)], CFG)
    assert [n.pitch for n in notes] == [60] and stats["dropped_range_or_quiet"] == 3


def test_small_overlap_between_neighbours_is_trimmed():
    notes, _ = em.clean_notes([N(60, 0.0, 1.05), N(62, 1.0, 1.5)], CFG)
    assert notes[0].end == pytest.approx(1.0) and notes[1].start == pytest.approx(1.0)


def test_louder_note_wins_even_if_it_starts_later():
    notes, _ = em.clean_notes([N(55, 0.0, 1.0, 0.3), N(67, 0.1, 1.0, 0.9)], CFG)
    assert [n.pitch for n in notes] == [67]


def test_same_pitch_far_apart_is_not_merged():
    notes, _ = em.clean_notes([N(60, 0.0, 0.5), N(60, 0.8, 1.3)], CFG)
    assert len(notes) == 2


def test_long_phrase_is_cut_at_its_longest_gap():
    notes = [N(60, i * 1.0, i * 1.0 + 0.9) for i in range(8)]       # eight notes, tiny 0.1 s gaps
    notes[4] = N(60, 4.5, 5.4)                                      # a bigger 0.5 s gap before note 4
    cfg = em.Config(max_phrase_s=6.0)
    phrases = em.split_phrases(notes, cfg)
    assert len(phrases) == 2 and len(phrases[0]) == 4
    assert all(p[-1].end - p[0].start <= 6.0 for p in phrases)


def test_no_phrase_when_no_notes_means_error(tmp_path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(pretty_midi.Instrument(0, name="Vocals"))
    midi.write(str(tmp_path / "empty.mid"))
    with pytest.raises(ValueError):
        em.extract("x", tmp_path, midi=tmp_path / "empty.mid")


def make_midi(path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    vocals, bass = pretty_midi.Instrument(0, name="Vocals"), pretty_midi.Instrument(33, name="Bass")
    for i, p in enumerate([60, 62, 64, 65, 67, 69, 67, 64] * 3):
        vocals.notes.append(pretty_midi.Note(80, p, i * 0.6, i * 0.6 + 0.5))
    for i in range(24):  # a busy bass line with more notes, which must not be chosen
        bass.notes.append(pretty_midi.Note(80, 40, i * 0.3, i * 0.3 + 0.28))
    midi.instruments += [bass, vocals]
    midi.write(str(path))
    return path


def test_midi_input_picks_the_vocal_track(tmp_path):
    path, report = em.extract("song", tmp_path / "melodies", midi=make_midi(tmp_path / "k.mid"),
                              tempo=100, check_dir=tmp_path / "checks")
    data = json.loads(path.read_text())
    notes = [n for phrase in data["phrases"] for n in phrase]
    assert [n["midi_pitch"] for n in notes][:3] == [60, 62, 64]
    assert report["n_phrases"] == len(data["phrases"]) and report["flags"] == []


def test_json_matches_the_repo_format_and_the_scorer_can_read_it(tmp_path):
    import score  # from scorer/
    path, _ = em.extract("song", tmp_path, midi=make_midi(tmp_path / "k.mid"), tempo=100)
    data = json.loads(path.read_text())
    assert set(data) == {"song_id", "tempo_bpm", "phrases"}
    assert set(data["phrases"][0][0]) == {"midi_pitch", "start_s", "duration_s"}
    song_id, tempo, phrases = score.load_melody(path)
    assert song_id == "song" and tempo == 100 and len(phrases) == len(data["phrases"])


def test_preview_and_qc_files_are_written(tmp_path):
    em.extract("song", tmp_path / "m", midi=make_midi(tmp_path / "k.mid"), tempo=90, check_dir=tmp_path / "c")
    assert (tmp_path / "c" / "song" / "preview.wav").stat().st_size > 10_000
    assert "flags" in json.loads((tmp_path / "c" / "song" / "qc.json").read_text())


def test_qc_flags_a_suspicious_key(tmp_path):
    notes = [N(40 + 3 * i, i * 1.0, i * 1.0 + 0.5) for i in range(25)]  # steps up more than 3 octaves
    phrases = em.split_phrases(notes, CFG)
    report = em.qc_report(notes, phrases, {}, CFG, "vocals")
    assert "pitch_range_suspicious" in report["flags"] and "tempo_from_vocals" in report["flags"]


def test_key_guess_on_a_c_major_tune():
    scale = [N(p, i, i + 1) for i, p in enumerate([60, 64, 67, 60, 62, 64, 65, 67, 67, 60])]
    assert em.guess_key(scale)[0] == "C"


def test_synthetic_singing_round_trip(tmp_path):
    pytest.importorskip("basic_pitch")
    import soundfile as sf

    sr = 22050
    truth = [(60, 0.5, 0.5), (62, 1.0, 0.5), (64, 1.5, 0.5), (65, 2.0, 0.5),
             (67, 3.5, 0.5), (69, 4.0, 0.5), (67, 4.5, 0.5), (64, 5.0, 0.5)]
    t = np.arange(int(7 * sr)) / sr
    freq, amp = np.zeros_like(t), np.zeros_like(t)
    for pitch, start, dur in truth:
        a, b = int(start * sr), int((start + dur) * sr)
        freq[a:b] = 440 * 2 ** ((pitch - 69) / 12)
        amp[a:b] = np.minimum(1, np.minimum(np.arange(b - a), np.arange(b - a)[::-1]) / (0.03 * sr))
    freq *= 2 ** ((15 / 1200) * np.sin(2 * np.pi * 5.5 * t))  # vibrato
    phase = 2 * np.pi * np.cumsum(freq) / sr
    voice = sum(np.sin(h * phase) / h ** 1.2 for h in range(1, 8)) * amp
    sf.write(str(tmp_path / "vocals.wav"), (0.3 * voice / np.abs(voice).max()).astype(np.float32), sr)

    path, report = em.extract("synth", tmp_path / "m", audio=tmp_path / "vocals.wav", tempo=100)
    found = [n for phrase in json.loads(path.read_text())["phrases"] for n in phrase]
    assert len(found) == len(truth), found  # all eight notes, and no ghost
    for n, (pitch, start, dur) in zip(found, truth):
        assert n["midi_pitch"] == pitch
        assert n["start_s"] == pytest.approx(start, abs=0.1)
        assert n["duration_s"] == pytest.approx(dur, abs=0.15)
    assert report["n_phrases"] == 2