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
# A phrase that nearly every cover fails points at a wrong answer key (e.g. the note maker followed
# a harmony line), which the self-score cannot see. Such phrases are dropped from every cover.
CONSENSUS_MIN_COVERS = 3            # only judge phrases with at least this many scored covers
CONSENSUS_MAX_MEDIAN_ACCURACY = 0.15  # drop when the covers' median accuracy is below this

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

# --- Pass 1b: fine-tune the time map with backing-track onsets (drum hits, chord attacks) ---
ONSET_HOP_S = 0.01                  # onset-strength frames
REFINE_WINDOW_S = 4.0               # length of each stretch of backing that is lined up again
REFINE_STEP_S = 1.0                 # one correction per this much cover time
REFINE_MAX_SHIFT_S = 0.3            # largest correction searched either way
REFINE_MIN_CORR = 0.3               # weaker matches (quiet or beatless stretches) give no correction
REFINE_LAG_PENALTY = 0.2            # correlation units per second of shift: between two near-equal
                                    # peaks (e.g. one beat apart), prefer the smaller shift
REFINE_SMOOTH_WINDOWS = 5           # median of this many neighbouring corrections

# --- Pass 2: find where the singer actually started each note ---
ONSET_SEARCH_S = 0.3                # look this far either side of where pass 1 puts the note
ONSET_PITCH_TOLERANCE_SEMITONES = 1.0   # "singing this note" for onset finding (octave-folded)
ONSET_MIN_RUN_S = 0.03              # the singer must stay near the note this long to count as an onset

# --- Dataset (build_dataset.py) ---
DATASET_VERSION = "v1.0"
PRIVATE_DIR = DERIVED_DIR / "private"     # uploader -> singer_id and recording -> video maps (never share)
# Cover singers have no range sweep, so every one gets this same comfortable range (decided
# 2026-11-17). Not from what they sang accurately (leaks the label), and not voice types from their
# median sung pitch: covers keep the song's key and octave, so the median measures the song, not the
# singer. With one fixed range the range features mean "how high/low for a typical amateur".
# Team members get their own range from a sweep (range_source "sweep").
COVER_RANGE_MIDI = (52, 72)         # E3-C5; pilot target notes are 52-74 (10th-90th percentile)
# Beat trackers cannot tell 68 BPM from 136. Every tempo is folded into [min, 2*min) so the same
# song always gets the same number (the feature only needs to be consistent).
TEMPO_BAND_MIN_BPM = 60.0
DEFAULT_LANGUAGE = "en"
# Simple quality flags per cover (marked, not dropped):
DUET_FLAG_SHARE = 0.35              # share of the cover's singing where the original lead has no note
PITCH_CORRECTION_CENTS = 15.0       # "on a semitone" = within this of one (after tuning)...
PITCH_CORRECTION_FLAG_SHARE = 0.55  # ...flag when more of the singing than this is (natural ~0.3)

# --- Beep listening check ---
BEEP_LEVEL = 0.25
SONG_LEVEL = 0.6
