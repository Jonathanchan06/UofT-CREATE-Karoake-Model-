"""Every tunable rule of the labeling pipeline, in one place. Units are in the names:
_s = seconds, _ms = milliseconds, _cents = cents (100 cents = 1 semitone), _db = decibels."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS_DIR = ROOT / "originals_data"
ORIGINAL_STEMS_DIR = ROOT / "separated" / "originals" / "htdemucs"   # <song folder>/vocals.wav, no_vocals.wav
COVER_STEMS_DIR = ROOT / "separated" / "htdemucs"                   # <song folder>/<video id>/vocals.wav
DERIVED_DIR = ROOT / "derived"                                       # caches and listening checks (not in Git)
REAL_DATA_DIR = ROOT / "ml" / "data" / "real"                        # contract-format output (not in Git)

PILOT_SONGS = [
    "Adele - Someone Like You",
    "Bruno Mars - When I Was Your Man",
    "Coldplay - Fix You",
    "ABBA - Dancing Queen",
    "a-ha - Take On Me",
]
LEFT_OUT_SONGS = {"Whitney Houston - I Will Always Love You": "original.wav missing"}

# --- Pitch tracking (PESTO) ---
PESTO_MODEL = "mir-1k_g7"
HOP_MS = 10.0                       # PESTO step size; its timestamps are in ms
CONFIDENCE_THRESHOLD = 0.5          # frames below this are marked unvoiced (kept, never deleted)
# PESTO snaps to a 33.3-cent grid. Averaging copies shifted by a known amount fills it in:
# 3 copies halve the error on known tones (8.7 -> 4.5 cents mean). 1 = plain PESTO.
PITCH_SHIFT_COPIES = 3

# --- Target notes from the original vocal stem ---
SMOOTH_S = 0.05                     # median filter on pitch before splitting into notes
NOTE_HYSTERESIS_SEMITONES = 0.75    # move to a new note only when pitch is this far from the current one
MIN_NOTE_S = 0.08                   # shorter notes are dropped (glides, tracker blips)
MERGE_GAP_S = 0.06                  # same-pitch notes split by a shorter unvoiced gap are joined
ONSET_DIP_DB = 6.0                  # a loudness dip this deep inside a note splits it (repeated notes)
ONSET_DIP_WINDOW_S = 0.06           # how far either side of the dip to look for the louder parts
LOUDNESS_WINDOW_S = 0.046           # RMS window for the loudness curve
PHRASE_GAP_S = 0.4                  # a rest at least this long starts a new phrase
SELF_SCORE_MIN_ACCURACY = 0.85      # per-song flag: original's own stem vs its targets (circular check)

# --- Pass 1: align the cover's backing track to the original's (chroma + DTW) ---
ALIGN_SAMPLE_RATE = 22050
CHROMA_HOP_S = 0.1                  # one chroma frame per 100 ms; pass 2 refines timing
CHROMA_N_FFT = 4096
CHROMA_LOG_GAMMA = 100.0            # log compression, so quiet parts still count
TRIM_SILENCE_DB = -40.0             # leading/trailing backing quieter than this (vs its peak) is cut
# Keys (ranked by pitch-class profile) that get a full DTW; the lowest-cost one wins. All 12:
# on the pilot the profile ranking missed the true key for 6 of 10 Bruno Mars covers (a fourth
# up looks similar in a piano ballad). About 0.5 s per key per cover.
KEY_CANDIDATES = 12
MAP_SMOOTH_S = 0.5                  # median filter on the time map
MAX_ALIGN_COST = 0.5                # path cost / random-pair cost; matched ~0.3, different backing 0.6+
MAX_SLOPE_ERROR = 0.1               # reject if the map's overall tempo ratio is outside 1 +/- this

# --- Pass 2: find where the singer actually started each note ---
ONSET_SEARCH_S = 0.3                # look this far either side of where pass 1 puts the note
ONSET_PITCH_TOLERANCE_SEMITONES = 1.0   # "singing this note" for onset finding (octave-folded)
ONSET_MIN_RUN_S = 0.03              # the singer must stay near the note this long to count as an onset

# --- Beep listening check ---
BEEP_LEVEL = 0.25
SONG_LEVEL = 0.6
