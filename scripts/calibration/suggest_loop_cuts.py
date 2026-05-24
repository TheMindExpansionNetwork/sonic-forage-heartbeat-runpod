"""Suggest 60s loop cuts for a long audio file.

Analyzes a music file (MP3 or WAV, any sample rate, mono or stereo)
and proposes 60-second windows that:
  - Start and end on a bar boundary (computed from detected BPM).
  - Have low waveform discontinuity at the splice (won't click when looping).
  - Have matching RMS levels at start and end (no loud-to-quiet jolt).
  - Have similar spectral content at start and end (no vocal phrase
    cut mid-syllable).
  - Cut during a relatively quiet onset moment (between hits, not on one).

Outputs the top N candidates ranked by an aggregate score, then either
prints next-step commands or directly writes the cut WAV to disk in
DEMON fixture spec (48kHz stereo).

Two modes:

    # Mode A: analyze and rank
    python3 scripts/calibration/suggest_loop_cuts.py raw_track.mp3

    # Mode B: cut a chosen window to a fixture-spec WAV
    python3 scripts/calibration/suggest_loop_cuts.py raw_track.mp3 \\
        --cut 32.45 --out funk_loop_60s_anm.wav

Install deps on macOS (one-time):
    pip3 install librosa soundfile numpy

Limitations the operator should understand:
  - I can't listen. Scoring is mathematical, not musical. The top-scored
    candidate is mathematically clean; you still need to listen to
    confirm it's musically appropriate.
  - BPM detection can be off on rubato/free-tempo material (Amazing
    Grace, classical, etc). Pass --bpm <N> to override.
  - Default time signature is 4/4. Pass --time-signature 3 for waltzes,
    --time-signature 6 for 6/8, etc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import librosa
    import numpy as np
    import soundfile as sf
except ImportError as e:
    print(
        "ERROR: missing dependency. Install with:\n"
        "    pip3 install librosa soundfile numpy\n"
        f"  ({e})",
        file=sys.stderr,
    )
    sys.exit(2)


TARGET_DURATION = 60.0
DURATION_TOLERANCE = 2.0  # candidates within +/- of target are eligible
OUTPUT_SR = 48000
OUTPUT_CHANNELS = 2

SPLICE_WINDOW = 480  # samples at OUTPUT_SR (10ms) for waveform-disc check
RMS_WINDOW_MS = 100  # window for RMS comparison
SPEC_WINDOW = 4096   # samples for FFT-based spectral comparison
ONSET_WINDOW_MS = 50  # window around splice for onset-energy check


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio at `target_sr`, return shape (channels, samples) stereo."""
    # librosa returns mono by default with sr resampling; we want stereo
    audio, _ = librosa.load(str(path), sr=target_sr, mono=False)
    if audio.ndim == 1:
        # mono -> duplicate to stereo
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        # >2 channels -> downmix to first 2
        audio = audio[:2]
    return audio.astype(np.float32)


def detect_bpm(audio_mono: np.ndarray, sr: int, override: float | None) -> tuple[float, str]:
    """Returns (bpm, source). source is 'override' or 'librosa'."""
    if override is not None and override > 0:
        return float(override), "override"
    bpm_raw, _ = librosa.beat.beat_track(y=audio_mono, sr=sr)
    bpm = float(np.asarray(bpm_raw).flat[0])
    return bpm, "librosa"


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------

def waveform_discontinuity(audio: np.ndarray, start: int, end: int) -> float:
    """Peak |end_chunk - start_chunk| / peak amplitude. Lower = cleaner splice."""
    if end - SPLICE_WINDOW < start + SPLICE_WINDOW:
        return 1.0
    peak = float(np.max(np.abs(audio[:, start:end])))
    if peak < 1e-6:
        return 0.0
    end_chunk = audio[:, end - SPLICE_WINDOW:end]
    start_chunk = audio[:, start:start + SPLICE_WINDOW]
    return float(np.max(np.abs(end_chunk - start_chunk))) / peak


def rms_mismatch_db(audio: np.ndarray, start: int, end: int, sr: int) -> float:
    """Absolute dB difference between last and first windows. Lower = smoother."""
    n = int(sr * RMS_WINDOW_MS / 1000)
    if end - n < start + n:
        return 0.0
    end_chunk = audio[:, end - n:end]
    start_chunk = audio[:, start:start + n]
    end_rms = float(np.sqrt(np.mean(end_chunk ** 2))) + 1e-9
    start_rms = float(np.sqrt(np.mean(start_chunk ** 2))) + 1e-9
    return abs(20 * np.log10(end_rms / start_rms))


def spectral_dissimilarity(audio: np.ndarray, start: int, end: int) -> float:
    """1 - cosine similarity of FFT magnitudes at the boundary windows.
    Lower = more similar timbral content (good loop)."""
    if end - SPEC_WINDOW < start + SPEC_WINDOW:
        return 1.0
    end_mono = audio[:, end - SPEC_WINDOW:end].mean(axis=0)
    start_mono = audio[:, start:start + SPEC_WINDOW].mean(axis=0)
    end_mag = np.abs(np.fft.rfft(end_mono * np.hanning(SPEC_WINDOW)))
    start_mag = np.abs(np.fft.rfft(start_mono * np.hanning(SPEC_WINDOW)))
    e_norm = np.linalg.norm(end_mag) + 1e-9
    s_norm = np.linalg.norm(start_mag) + 1e-9
    cos_sim = float(np.dot(end_mag, start_mag) / (e_norm * s_norm))
    return max(0.0, 1.0 - cos_sim)


def onset_activity(onset_env: np.ndarray, hop: int, sr: int, start: int, end: int) -> float:
    """Mean onset envelope around the splice. Lower = cutting between hits."""
    win = int(sr * ONSET_WINDOW_MS / 1000)
    # convert sample positions to onset frames
    s_frame = max(0, (start - win) // hop)
    e_frame = min(len(onset_env), (end + win) // hop + 1)
    if e_frame <= s_frame:
        return 0.0
    # take onset energy at both the end region and start region
    end_frame_lo = max(0, (end - win) // hop)
    end_frame_hi = min(len(onset_env), (end + win) // hop + 1)
    start_frame_lo = max(0, (start - win) // hop)
    start_frame_hi = min(len(onset_env), (start + win) // hop + 1)
    end_act = float(np.mean(onset_env[end_frame_lo:end_frame_hi]) if end_frame_hi > end_frame_lo else 0)
    start_act = float(np.mean(onset_env[start_frame_lo:start_frame_hi]) if start_frame_hi > start_frame_lo else 0)
    peak_env = float(np.max(onset_env)) + 1e-9
    return (end_act + start_act) / (2 * peak_env)


def aggregate_score(disc: float, rms_db: float, spec: float, onset: float) -> float:
    """Lower = better. Weighted sum; tweak weights to taste."""
    # Normalize each to roughly 0-1
    disc_n = min(1.0, disc / 0.25)              # 25% disc = 1.0
    rms_n  = min(1.0, rms_db / 6.0)             # 6 dB diff = 1.0
    spec_n = min(1.0, spec / 0.5)               # 0.5 (1-cos) = 1.0
    onset_n = min(1.0, onset / 0.5)             # half of peak onset = 1.0
    return 0.40 * disc_n + 0.20 * rms_n + 0.20 * spec_n + 0.20 * onset_n


# ---------------------------------------------------------------------------
# Candidate generation and ranking
# ---------------------------------------------------------------------------

def generate_candidates(
    total_samples: int, sr: int, bpm: float, time_sig: int, target_duration: float
):
    """Yields (start_sample, end_sample, n_bars) tuples for each candidate
    start-bar position. Both start and end land on bar boundaries."""
    bar_seconds = (60.0 / bpm) * time_sig
    bar_samples = int(round(bar_seconds * sr))

    # How many bars makes a loop closest to target_duration?
    n_bars_target = max(1, int(round(target_duration / bar_seconds)))

    # Also try one bar less and one bar more, keep whichever lands closest
    # to target_duration AND inside tolerance.
    candidates_n = []
    for delta in (-1, 0, 1):
        n = n_bars_target + delta
        if n < 1:
            continue
        dur = n * bar_seconds
        if abs(dur - target_duration) <= DURATION_TOLERANCE:
            candidates_n.append(n)
    if not candidates_n:
        # No bar-count fits inside +/- 2s tolerance. Use the closest anyway.
        candidates_n = [n_bars_target]

    seen = set()
    for n_bars in candidates_n:
        loop_samples = n_bars * bar_samples
        if loop_samples >= total_samples:
            continue
        max_start_bar = (total_samples - loop_samples) // bar_samples
        for start_bar in range(int(max_start_bar) + 1):
            start = start_bar * bar_samples
            end = start + loop_samples
            if end > total_samples:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            yield (start, end, n_bars)


def analyze(path: Path, *, bpm_override, time_sig, target_duration, top_n):
    print(f"Loading {path.name} at {OUTPUT_SR} Hz...")
    audio = load_audio(path, OUTPUT_SR)
    total = audio.shape[1]
    duration = total / OUTPUT_SR
    print(f"  {duration:.1f}s, {audio.shape[0]} channel(s), {total} samples")

    if duration < target_duration + DURATION_TOLERANCE:
        print(
            f"ERROR: file is {duration:.1f}s; need at least "
            f"{target_duration + DURATION_TOLERANCE:.0f}s to extract a {target_duration:.0f}s loop.",
            file=sys.stderr,
        )
        return None

    # BPM detection on mono mix
    mono = audio.mean(axis=0)
    bpm, bpm_src = detect_bpm(mono, OUTPUT_SR, bpm_override)
    bar_seconds = (60.0 / bpm) * time_sig
    print(f"  BPM: {bpm:.1f} ({bpm_src})  time sig: {time_sig}/4  bar: {bar_seconds:.3f}s")

    # Onset envelope (precomputed once)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=mono, sr=OUTPUT_SR, hop_length=hop)

    # Generate and score all candidates
    print(f"\nGenerating candidates (target {target_duration:.0f}s +/- {DURATION_TOLERANCE:.0f}s)...")
    cands = list(generate_candidates(total, OUTPUT_SR, bpm, time_sig, target_duration))
    if not cands:
        print("ERROR: no candidates fit. Try --bpm or --time-signature.", file=sys.stderr)
        return None
    print(f"  {len(cands)} candidate windows")

    scored = []
    for start, end, n_bars in cands:
        disc = waveform_discontinuity(audio, start, end)
        rms_db = rms_mismatch_db(audio, start, end, OUTPUT_SR)
        spec = spectral_dissimilarity(audio, start, end)
        onset = onset_activity(onset_env, hop, OUTPUT_SR, start, end)
        score = aggregate_score(disc, rms_db, spec, onset)
        scored.append((score, start, end, n_bars, disc, rms_db, spec, onset))

    scored.sort(key=lambda x: x[0])

    print(f"\nTop {top_n} candidates (lower aggregate score = better loop math):\n")
    for i, (score, start, end, n_bars, disc, rms_db, spec, onset) in enumerate(scored[:top_n]):
        start_s = start / OUTPUT_SR
        end_s = end / OUTPUT_SR
        dur_s = end_s - start_s
        print(
            f"  #{i+1}  start={start_s:6.2f}s  end={end_s:6.2f}s  ({dur_s:.2f}s, {n_bars} bars)"
        )
        print(
            f"        score={score:.3f}  | "
            f"splice-disc={disc*100:.1f}%  "
            f"RMS-Δ={rms_db:.2f}dB  "
            f"spec-dissim={spec:.3f}  "
            f"onset={onset:.3f}"
        )

    best = scored[0]
    best_start = best[1] / OUTPUT_SR
    print(f"\nTo cut #1, run (fill in the key suffix):")
    print(
        f"  python3 scripts/calibration/suggest_loop_cuts.py {path} \\\n"
        f"      --cut {best_start:.2f} --out <name>_loop_60s_<key>.wav"
    )
    print()
    return scored


def cut_and_write(path: Path, cut_start_s: float, out_path: Path, target_duration: float):
    """Cut a window starting at `cut_start_s` of length `target_duration` and
    write to `out_path` in DEMON fixture spec (48kHz stereo)."""
    print(f"Loading {path.name}...")
    audio = load_audio(path, OUTPUT_SR)
    total = audio.shape[1]

    start = int(round(cut_start_s * OUTPUT_SR))
    end = start + int(round(target_duration * OUTPUT_SR))

    if start < 0 or end > total:
        print(
            f"ERROR: cut range [{start}, {end}] outside file [0, {total}] "
            f"(file is {total / OUTPUT_SR:.2f}s).",
            file=sys.stderr,
        )
        sys.exit(1)

    clip = audio[:, start:end]
    # Soundfile expects shape (samples, channels)
    sf.write(str(out_path), clip.T, OUTPUT_SR, subtype="PCM_16")
    print(f"Wrote {out_path} ({clip.shape[1] / OUTPUT_SR:.2f}s, {clip.shape[0]}ch, 48kHz)")
    print()
    print(f"Validate with:")
    print(f"  python3 scripts/calibration/validate_fixture_wav.py {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input audio file (MP3 / WAV / etc)")
    p.add_argument("--top", type=int, default=5, help="How many candidates to print (default 5)")
    p.add_argument("--target-duration", type=float, default=TARGET_DURATION,
                   help=f"Target loop length in seconds (default {TARGET_DURATION})")
    p.add_argument("--bpm", type=float, default=None,
                   help="Override BPM detection (use when librosa misdetects)")
    p.add_argument("--time-signature", type=int, default=4,
                   help="Time signature numerator (default 4 for 4/4; 3 for 3/4 waltz, etc)")
    p.add_argument("--cut", type=float, default=None, metavar="START_S",
                   help="Skip analysis, just cut this window and write it")
    p.add_argument("--out", type=Path, default=None,
                   help="Output WAV path (required with --cut)")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.cut is not None:
        if args.out is None:
            print("ERROR: --cut requires --out PATH", file=sys.stderr)
            sys.exit(2)
        cut_and_write(args.input, args.cut, args.out, args.target_duration)
        return 0

    result = analyze(
        args.input,
        bpm_override=args.bpm,
        time_sig=args.time_signature,
        target_duration=args.target_duration,
        top_n=args.top,
    )
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
