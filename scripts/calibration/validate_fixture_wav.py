"""Validate WAV files against the DEMON fixture spec.

Runs on any platform with Python + soundfile + numpy. No CUDA, no
torch, no DEMON env required. Intended for catching format mistakes
before uploading to the daydreamlive/demon-fixtures HF dataset.

Checks per file:
    1. File loads without error.
    2. Sample rate is exactly 48000 Hz.
    3. Channel count is exactly 2 (stereo).
    4. Duration is ~60s (default; configurable via --duration).
    5. Filename trailing token parses to a valid key (e.g. _gsm, _enm, _aM).
    6. Loop discontinuity at the splice point (soft warning, not a failure).

Usage:
    python3 scripts/calibration/validate_fixture_wav.py file1.wav [file2.wav ...]
    python3 scripts/calibration/validate_fixture_wav.py --duration 60 *.wav

Install deps on macOS (one-time):
    pip3 install soundfile numpy

Exit codes:
    0 - all files pass hard checks
    1 - one or more files failed
    2 - missing dependency or bad CLI args
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:
    print(
        "ERROR: missing dependency. Install with:\n"
        "    pip3 install soundfile numpy\n"
        f"  ({e})",
        file=sys.stderr,
    )
    sys.exit(2)


EXPECTED_SR = 48000
EXPECTED_CHANNELS = 2
DEFAULT_DURATION = 60.0
DURATION_TOLERANCE = 2.0  # +/- seconds; precompute trims to mod-9600 anyway

LOOP_WARN_THRESHOLD = 0.05
LOOP_FAIL_THRESHOLD = 0.20

NOTE_LETTERS = set("abcdefg")
MODIFIERS = {"s": "#", "n": "", "f": "b"}
MODES = {"m": "minor", "M": "major"}


def parse_key_suffix(suffix: str):
    """Returns (pitch_str, mode_str) or None for unrecognized suffixes.

    Mirrors acestep/fixtures.py::_parse_key_suffix so this script stays
    standalone (no demon imports).
    """
    if not suffix:
        return None
    mode_char = suffix[-1]
    if mode_char not in MODES:
        return None
    mode = MODES[mode_char]
    rest = suffix[:-1]
    if not rest or rest[0] not in NOTE_LETTERS:
        return None
    note = rest[0].upper()
    rest = rest[1:]
    accidental = ""
    if rest:
        if rest[0] not in MODIFIERS:
            return None
        accidental = MODIFIERS[rest[0]]
        rest = rest[1:]
    if rest:
        return None
    return (f"{note}{accidental}", mode)


def parse_filename_key(path: Path):
    """Extracts a human-readable key from the trailing filename token.

    Returns "G# minor" / "C major" / etc, or None if unparseable.
    """
    stem = path.stem
    if "_" not in stem:
        return None
    suffix = stem.rsplit("_", 1)[-1]
    parsed = parse_key_suffix(suffix)
    if parsed is None:
        return None
    pitch, mode = parsed
    return f"{pitch} {mode}"


def loop_discontinuity(audio: np.ndarray, window: int = 480) -> float:
    """Peak abs-difference between the last `window` samples and the first
    `window` samples, normalized by the file's peak amplitude.

    Default window is 480 samples (10ms at 48kHz). The intuition: a
    perfect loop has the waveform at the end matching what the start
    is about to play, so wrapping around is invisible. A click happens
    when the two don't match.

    audio: shape (channels, samples). Returns 0.0 for silent or too-short files.
    """
    if audio.shape[1] < window * 2:
        return 0.0
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return 0.0
    end = audio[:, -window:]
    start = audio[:, :window]
    return float(np.max(np.abs(end - start))) / peak


def validate(path: Path, expected_duration: float) -> bool:
    """Returns True if file passes all hard checks. Prints results inline."""
    print(f"\n[{path.name}]")

    if not path.is_file():
        print("  FAIL: file not found")
        return False

    try:
        info = sf.info(str(path))
    except Exception as e:
        print(f"  FAIL: can't read file: {e}")
        return False

    ok = True

    # Sample rate
    if info.samplerate != EXPECTED_SR:
        print(f"  FAIL: sample rate is {info.samplerate} Hz; need {EXPECTED_SR}")
        ok = False
    else:
        print(f"  ok   sample rate: {info.samplerate} Hz")

    # Channels
    if info.channels != EXPECTED_CHANNELS:
        print(f"  FAIL: channels = {info.channels}; need {EXPECTED_CHANNELS} (stereo)")
        ok = False
    else:
        print("  ok   channels: stereo")

    # Duration
    duration = info.frames / info.samplerate if info.samplerate > 0 else 0.0
    if abs(duration - expected_duration) > DURATION_TOLERANCE:
        print(
            f"  FAIL: duration is {duration:.2f}s; want ~{expected_duration:.0f}s "
            f"(+/- {DURATION_TOLERANCE:.0f}s)"
        )
        ok = False
    else:
        print(f"  ok   duration: {duration:.2f}s")

    # Format diagnostic
    print(f"  info format: {info.format} {info.subtype}")

    # Filename key parse
    parsed_key = parse_filename_key(path)
    if parsed_key is None:
        print("  FAIL: filename doesn't end with a valid key suffix")
        print("       Expected pattern: <name>_loop_60s_<key>.wav")
        print("       Key examples:")
        print("         gsm  -> G# minor          dnm  -> D minor")
        print("         enm  -> E minor           aM   -> A major")
        print("         dfM  -> Db major          cnm  -> C minor")
        ok = False
    else:
        print(f"  ok   filename key: {parsed_key}")

    if not ok:
        return False

    # Loop quality (soft check, never fails the file)
    try:
        audio, _ = sf.read(str(path), always_2d=True)
        disc = loop_discontinuity(audio.T)
        pct = disc * 100
        if disc > LOOP_FAIL_THRESHOLD:
            print(
                f"  WARN loop splice has large discontinuity ({pct:.1f}% of peak). "
                f"Will likely click when looping. Crossfade the boundary in "
                f"Audacity or pick a different cut point."
            )
        elif disc > LOOP_WARN_THRESHOLD:
            print(
                f"  note loop splice has minor discontinuity ({pct:.1f}% of peak). "
                f"May click on some material. Preview with Shift+Space in Audacity."
            )
        else:
            print(f"  ok   loop splice clean ({pct:.2f}% discontinuity)")
    except Exception as e:
        print(f"  note couldn't compute loop quality: {e}")

    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="WAV file(s) to validate",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Target duration in seconds (default: {DEFAULT_DURATION})",
    )
    args = p.parse_args()

    results = [validate(f, args.duration) for f in args.files]
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 50)
    print(f"{passed}/{total} passed hard checks")
    if passed < total:
        print("(WARN/note items above are advisory only; they don't fail validation.)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
