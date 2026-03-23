#!/usr/bin/env python3
"""
pump.py — Batch vinyl restoration with tempo-aware compression
========================================================================
Processes AIFF files through a restoration chain designed for digitised
vinyl (dance music): rumble removal → tempo-aware compression →
two-pass EBU R128 loudness normalisation → true-peak limiting.

Compression attack, release, and ratio are all derived from detected BPM
so the compressor behaves musically at any tempo from ~60–180 BPM.

Dependencies:
    pip install librosa

Usage:
    python3 pump.py <input_dir> [output_dir] [options]

Options:
    --lufs          Target integrated loudness in LUFS (default: -11)
    --peak          True peak ceiling in dBTP (default: -0.5)
    --lra           Loudness range target (default: 7)
    --jobs          Parallel workers (default: 4)
    --format        Output format: aiff or wav (default: aiff)
    --bit-depth     Output bit depth: 16 or 24 (default: 16)
    --dry-run       Print what would be processed without writing files
    --no-compress   Skip compression (only normalise + limit)
    --bpm FLOAT     Override BPM for all files (skips auto-detection)

Examples:
    python3 pump.py ~/Records/scans
    python3 pump.py ~/Records/scans ~/Records/processed --lufs -11 --jobs 8
    python3 pump.py ~/Records/scans --bpm 174   # force DnB settings
    python3 pump.py ~/Records/scans --dry-run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ── Defaults ──────────────────────────────────────────────────────────────────

TARGET_LUFS  = -11
TRUE_PEAK    = -0.5
LRA          = 7
HIGHPASS_HZ  = 25
JOBS         = 4
BIT_DEPTH    = 16
FALLBACK_BPM = 124.0
EXTENSIONS   = {'.aif', '.aiff', '.AIF', '.AIFF'}


# ── Tempo-aware compression ────────────────────────────────────────────────────

@dataclass
class CompressorParams:
    bpm:     float
    attack:  float   # ms
    release: float   # ms
    ratio:   float
    makeup:  float   # dB
    knee:    float   # dB

    def describe(self) -> str:
        return (
            f"BPM={self.bpm:.1f}  "
            f"attack={self.attack:.0f}ms  "
            f"release={self.release:.0f}ms  "
            f"ratio={self.ratio:.1f}:1"
        )


def bpm_to_compressor_params(bpm: float) -> CompressorParams:
    """
    Derive musically appropriate compressor settings from tempo.

    Core principle: the compressor should recover between beats so it
    doesn't fight the groove. We target recovery at ~60% of the beat
    duration — tight enough to act on each hit, relaxed enough not to pump.

    Reference points:
        60  BPM (downtempo/ambient): beat=1000ms → release=500ms, attack=16ms, ratio=2.8
        80  BPM (krautrock/slow):    beat=750ms  → release=450ms, attack=13ms, ratio=2.6
        100 BPM (slow house/disco):  beat=600ms  → release=360ms, attack=11ms, ratio=2.5
        124 BPM (house):             beat=484ms  → release=290ms, attack=10ms, ratio=2.3
        138 BPM (techno/trance):     beat=435ms  → release=261ms, attack=8ms,  ratio=2.1
        150 BPM (jungle/breaks):     beat=400ms  → release=240ms, attack=7ms,  ratio=2.0
        174 BPM (DnB):               beat=345ms  → release=207ms, attack=5ms,  ratio=1.9
        180 BPM (fast DnB):          beat=333ms  → release=200ms, attack=5ms,  ratio=1.8
    """
    beat_ms = 60_000 / bpm

    # Release: 60% of beat duration, clamped
    release = round(beat_ms * 0.60)
    release = max(150, min(600, release))

    # Attack: inversely proportional to BPM (16ms at 60BPM → 5ms at 180BPM)
    bpm_clamped = max(60.0, min(200.0, bpm))
    attack = 16 - ((bpm_clamped - 60) / 120) * 11
    attack = round(max(4.0, min(16.0, attack)), 1)

    # Ratio: slightly gentler at higher tempos (2.8:1 → 1.8:1 over 60–180 BPM)
    ratio = 2.8 - ((bpm_clamped - 60) / 120) * 1.0
    ratio = round(max(1.8, min(2.8, ratio)), 1)

    return CompressorParams(
        bpm=bpm,
        attack=attack,
        release=release,
        ratio=ratio,
        makeup=3.0,
        knee=6.0,
    )


def detect_bpm(path: Path, sample_duration_s: int = 90) -> float | None:
    """
    Detect BPM using librosa's beat tracker.

    Loads audio at 22050 Hz (mono) for speed — sufficient for tempo
    detection. Skips the first 10% of the track to avoid silent intros
    and run-in grooves, then analyses up to sample_duration_s seconds.

    librosa.beat.beat_track uses a dynamic programming beat tracker
    that is more robust than simple autocorrelation at slow tempos,
    which makes it well suited to the 60–180 BPM range here.
    """
    if not LIBROSA_AVAILABLE:
        return None

    sr = 22050  # Half native rate — fine for tempo, loads faster

    try:
        duration = librosa.get_duration(path=str(path))

        # Skip the first 10% (intro / run-in groove), cap total analysis
        offset = duration * 0.10
        analyse_for = min(sample_duration_s, duration - offset)

        if analyse_for < 10:
            return None  # Track too short to get a reliable reading

        y, _ = librosa.load(
            str(path),
            sr=sr,
            mono=True,
            offset=offset,
            duration=analyse_for,
        )

        # beat_track returns (tempo_scalar, beat_frame_indices)
        # start_bpm seeds the search — 90 BPM is a neutral midpoint for
        # our 60–180 range and stops the tracker anchoring on 120 by default
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=90, units="time")

        # librosa ≥ 0.10 may return a 1-element ndarray
        bpm = float(np.atleast_1d(tempo)[0])

        if bpm < 40 or bpm > 220:
            return None

        # Correct half-time / double-time.
        # Threshold at 80 — anything below is assumed to be a half-time
        # misread and doubled. Genuine sub-80 BPM material is unlikely
        # in this collection.
        if bpm < 80:
            bpm *= 2
        elif bpm > 185:
            bpm /= 2

        return round(bpm, 1)

    except Exception:
        return None


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def check_ffmpeg():
    if not shutil.which('ffmpeg'):
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        sys.exit(1)


def build_filter_chain(
    target_lufs: float,
    true_peak: float,
    lra: float,
    highpass_hz: int,
    compress_params,
    measured: dict | None = None,
) -> str:
    filters = []

    # 1. DC offset removal (5 Hz highpass — inaudible, removes ADC bias)
    filters.append("highpass=f=5:poles=2")

    # 2. Subsonic high-pass (removes turntable rumble, frees headroom)
    filters.append(f"highpass=f={highpass_hz}:poles=2")

    # 3. Tempo-aware broadband compression
    if compress_params:
        p = compress_params
        filters.append(
            f"acompressor="
            f"threshold=-24dB:"
            f"ratio={p.ratio}:"
            f"attack={p.attack}:"
            f"release={p.release}:"
            f"makeup={p.makeup}:"
            f"knee={p.knee}"
        )

    # 4. EBU R128 two-pass loudnorm
    if measured:
        filters.append(
            f"loudnorm="
            f"I={target_lufs}:"
            f"TP={true_peak}:"
            f"LRA={lra}:"
            f"measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            f"linear=true:"
            f"print_format=none"
        )
    else:
        filters.append(
            f"loudnorm="
            f"I={target_lufs}:"
            f"TP={true_peak}:"
            f"LRA={lra}:"
            f"print_format=json"
        )

    # 5. True-peak limiter (render pass only)
    if measured:
        limit_linear = round(10 ** (true_peak / 20), 4)
        filters.append(
            f"alimiter="
            f"limit={limit_linear}:"
            f"level=true:"
            f"attack=5:"
            f"release=50:"
            f"asc=true"
        )

    return ",".join(filters)


def analyse_file(path, compress_params, args) -> dict | None:
    chain = build_filter_chain(
        args.lufs, args.peak, args.lra, HIGHPASS_HZ, compress_params, measured=None
    )
    cmd = ["ffmpeg", "-hide_banner", "-i", str(path), "-af", chain, "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        m = re.search(r'\{[^{}]+\}', result.stderr, re.DOTALL)
        if not m:
            print(f"  [WARN] Could not parse loudnorm output for {path.name}")
            return None
        return json.loads(m.group())
    except Exception as e:
        print(f"  [ERROR] Analysis failed for {path.name}: {e}")
        return None


def render_file(input_path, output_path, measured, compress_params, args) -> bool:
    chain = build_filter_chain(
        args.lufs, args.peak, args.lra, HIGHPASS_HZ, compress_params, measured=measured
    )
    codec = f"pcm_s{args.bit_depth}be" if args.format == "aiff" else f"pcm_s{args.bit_depth}le"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-af", chain,
        "-c:a", codec,
        "-ar", "44100",
        "-y", str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  [ERROR] Render failed for {input_path.name}: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] {input_path.name}: {e}")
        return False


# ── Core processing ────────────────────────────────────────────────────────────

def process_file(input_path, output_dir, input_dir, args):
    """Returns (filename, success, bpm_info)"""
    name = input_path.name
    suffix = ".aiff" if args.format == "aiff" else ".wav"

    # Mirror the subdirectory structure of input_dir inside output_dir
    relative   = input_path.relative_to(input_dir)
    output_path = output_dir / relative.parent / (input_path.stem + "_restored" + suffix)

    if args.dry_run:
        src = f"forced {args.bpm} BPM" if args.bpm else ("librosa" if LIBROSA_AVAILABLE else f"fallback {FALLBACK_BPM}")
        print(f"  [DRY RUN] {relative}  (BPM source: {src})")
        return name, True, ""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  [SKIP]    {name}")
        return name, True, "skipped"

    # ── Determine compressor settings ─────────────────────────────────────────
    if not args.compress:
        compress_params = None
        bpm_info = "compression disabled"
    elif args.bpm:
        compress_params = bpm_to_compressor_params(args.bpm)
        bpm_info = f"forced → {compress_params.describe()}"
    else:
        print(f"  [BPM]     Detecting {name}…")
        detected = detect_bpm(input_path)
        if detected:
            compress_params = bpm_to_compressor_params(detected)
            bpm_info = f"detected → {compress_params.describe()}"
        else:
            compress_params = bpm_to_compressor_params(FALLBACK_BPM)
            bpm_info = f"undetected → fallback {FALLBACK_BPM:.0f} BPM → {compress_params.describe()}"

    print(f"  [1/3]     {name}: {bpm_info}")

    # ── Pass 1: loudnorm analysis ──────────────────────────────────────────────
    print(f"  [2/3]     Analysing loudness: {name}")
    measured = analyse_file(input_path, compress_params, args)

    if measured is None:
        measured = {
            "input_i": "-99", "input_tp": "-99",
            "input_lra": "0",  "input_thresh": "-99",
            "target_offset": "0",
        }

    # ── Pass 2: render ─────────────────────────────────────────────────────────
    print(
        f"  [3/3]     Rendering: {name}  "
        f"({measured.get('input_i', '?')} LUFS → {args.lufs} LUFS)"
    )
    success = render_file(input_path, output_path, measured, compress_params, args)
    print(f"  {'[OK]' if success else '[FAIL]'}      {name}")
    return name, success, bpm_info


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch vinyl restoration with tempo-aware compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_dir",  type=Path)
    p.add_argument("output_dir", type=Path, nargs="?", default=None)
    p.add_argument("--lufs",        type=float, default=TARGET_LUFS)
    p.add_argument("--peak",        type=float, default=TRUE_PEAK)
    p.add_argument("--lra",         type=float, default=LRA)
    p.add_argument("--jobs",        type=int,   default=JOBS)
    p.add_argument("--format",      choices=["aiff", "wav"], default="aiff")
    p.add_argument("--bit-depth",   type=int, choices=[16, 24], default=BIT_DEPTH, dest="bit_depth")
    p.add_argument("--dry-run",     action="store_true", dest="dry_run")
    p.add_argument("--no-compress", action="store_false", dest="compress", default=True)
    p.add_argument("--bpm",         type=float, default=None,
                   help="Override BPM for all files (e.g. --bpm 174 for a DnB folder)")
    return p.parse_args()


def main():
    args = parse_args()
    check_ffmpeg()

    if args.compress and not args.bpm and not LIBROSA_AVAILABLE:
        print(
            "\nWARNING: librosa not installed — BPM detection unavailable.\n"
            f"         Using fallback: {FALLBACK_BPM} BPM for all files.\n"
            "         To fix: pip install librosa\n"
            "         Or override: --bpm 174\n"
        )

    input_dir  = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "processed").resolve()

    if not input_dir.is_dir():
        print(f"ERROR: not a directory: {input_dir}")
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.rglob("*")
                   if f.is_file() and f.suffix in EXTENSIONS)

    if not files:
        print(f"No AIFF files found in {input_dir}")
        sys.exit(0)

    total    = len(files)
    bpm_src  = (
        f"forced {args.bpm} BPM" if args.bpm
        else ("auto-detect via librosa" if LIBROSA_AVAILABLE else f"fallback {FALLBACK_BPM} BPM")
    )

    print(f"\n{'='*65}")
    print(f"  Vinyl Restoration — Tempo-Aware Batch Processor")
    print(f"{'='*65}")
    print(f"  Input:        {input_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  Files:        {total}")
    print(f"  Target LUFS:  {args.lufs}")
    print(f"  True peak:    {args.peak} dBTP")
    print(f"  Compression:  {'yes — ' + bpm_src if args.compress else 'disabled'}")
    print(f"  Workers:      {args.jobs}")
    print(f"  Format:       {args.bit_depth}-bit {args.format.upper()}")
    if args.dry_run:
        print(f"  *** DRY RUN ***")
    print(f"{'='*65}\n")

    succeeded, failed = 0, []

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_file, f, output_dir, input_dir, args): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            name, ok, _ = future.result()
            if ok:
                succeeded += 1
            else:
                failed.append(name)
            print(f"  ── Progress: {i}/{total}  ({succeeded} ok, {len(failed)} failed)\n")

    print(f"\n{'='*65}")
    print(f"  Complete: {succeeded}/{total} files processed")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
