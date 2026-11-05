# Answer keys (melody extractor)

Turns a song's vocals into `melodies/<song_id>.json`, the file the scorer compares every singer
against. A wrong note here becomes a wrong label for everyone who sings that song, so the
extractor cleans up basic-pitch's output and makes it easy to check by ear.

## Run it

```
demucs "D:\songs\perfect.mp3"
python melody/extract_melody.py --audio separated/htdemucs/perfect/vocals.wav --song-id perfect_ed_sheeran ^
    --tempo-audio "D:\songs\perfect.mp3" --out-dir ml/data/melodies
python -m pytest melody
```

Or, if you have a karaoke MIDI file (usually more accurate than basic-pitch):

```
python melody/extract_melody.py --midi perfect.mid --song-id perfect_ed_sheeran --track Vocals
```

It writes three things:

| File | What it is |
|---|---|
| `melodies/<song_id>.json` | the answer key, same format as `ml/data/fake/melodies/` |
| `melody_checks/<song_id>/preview.wav` | the extracted melody as plain tones |
| `melody_checks/<song_id>/qc.json` | counts, pitch range, and warning flags |

It also prints a suggested row for `songs.csv`. Fix the key, language and genre by hand.

## Check every answer key by ear

Play `preview.wav` next to the original song. If the tune is recognisable and the notes line up,
the key is good. Wrong notes, missing words, or a melody that jumps around mean the answer key is
bad, and nobody should be scored against it. The QC flags are only a first filter.

## The design

1. **basic-pitch** is a small neural network (from Spotify) that turns audio into note events. It
   hears several notes at once, so on a single voice it also invents ghost notes.
2. **Keep one line.** When two notes overlap, the louder one wins. This removes ghosts at any
   pitch, not just octaves. We found one 28 semitones above the real note when we tried it.
3. **Merge and drop.** Same-pitch pieces under 80 ms apart become one note, then notes under
   80 ms are dropped, then merging runs again.
4. **Phrases** start after a silence of 0.6 s, and any phrase over 10 s is cut at its longest gap.
5. **Every threshold lives in `Config`.** Change them in one place.

## If `pip install basic-pitch` fails

On some Python versions pip tries to compile an old numpy and stops. This worked instead:

```
pip install --no-deps basic-pitch
pip install onnxruntime resampy mir_eval scikit-learn librosa pretty_midi soundfile
```

## Known limits

- Tested on synthetic singing and the cleanup rules only. The first real Demucs vocal is the real
  test, so listen to its `preview.wav`.
- Tempo from the vocals alone is a guess, and gets the flag `tempo_from_vocals`. Pass
  `--tempo-audio` with the full song to estimate it properly.
- The key guess comes from which notes are sung most. Treat it as a hint.
- Backing vocals that survive Demucs can still leak in. They are usually quieter, so the
  louder-note-wins rule removes most of them, but not all.
- The CI run skips the basic-pitch test, because installing it there is heavy. Run
  `python -m pytest melody` locally to include it.