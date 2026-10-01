#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ReviewerDemo Stage 04 — prepare a dense temporal emotion trajectory
===================================================================

This script creates overlapping 5-second EEG/Audio/Video replay windows from one
Stage-01 validated EAV Speaking trial.

It lives entirely inside ReviewerDemo and does NOT modify any final-deployment
module, checkpoint, quality formula, routing rule, threshold, or fusion weight.

Default temporal policy
-----------------------
    frozen model window length : 5.0 s   (UNCHANGED)
    trajectory stride          : 1.0 s
    EAV trial duration         : 20.0 s
    generated windows          : 16

Thus:
    step 00 :  0.0 --  5.0 s   center  2.5 s
    step 01 :  1.0 --  6.0 s   center  3.5 s
    ...
    step 15 : 15.0 -- 20.0 s   center 17.5 s

The output is a source adapter only. Each generated descriptor is consumed later
by the unchanged final-deployment main.py:

    EEG   -> .npy [30, 2500], kind=eeg_window
    Audio -> 5-s PCM16 WAV,    kind=audio_window
    Video -> 150-frame FFV1,   kind=video_window

QualityController receives an explicit `quality_window` with the exact
trial-relative time span, so no fake wall-clock timestamp is invented.

Reference emotion may be copied into the audit report for reviewer display, but
it is NEVER written to the runtime descriptors and therefore is not model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

VERSION = "EAV-REVIEWER-TEMPORAL-PREPARE.1.0.1"
SOURCE_SCHEMA = "eav.system.source_window.v1"
OUTPUT_SCHEMA = "eav.reviewer_demo.temporal_manifest.v1"

MODALITIES = ("eeg", "audio", "video")
EMOTIONS = ("Neutral", "Sadness", "Anger", "Happiness", "Calmness")
TEST_SUBJECTS = {
    "subject03", "subject05", "subject20",
    "subject31", "subject35", "subject39",
}
SUBJECT_RE = re.compile(r"^subject0*(\d+)$", re.IGNORECASE)

CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "CZ", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "PZ", "P4", "P8",
    "PO9", "O1", "OZ", "O2", "PO10",
]

EEG_FS = 500
AUDIO_FS = 16000
VIDEO_FPS = 30
TRIAL_SECONDS = 20.0
WINDOW_SECONDS = 5.0


class TemporalPreparationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TemporalPreparationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise TemporalPreparationError(
                    f"Invalid JSONL {path}:{line_no}: {exc}"
                ) from exc
            require(isinstance(obj, dict), f"JSONL row {line_no} is not an object")
            rows.append(obj)
    require(rows, f"No records found: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalize_subject(value: str) -> str:
    m = SUBJECT_RE.fullmatch(str(value).strip())
    require(m is not None, f"Invalid subject: {value!r}")
    n = int(m.group(1))
    require(1 <= n <= 42, f"Subject outside EAV range: {n}")
    return f"subject{n:02d}"


def resolve_stage01_manifest(
    script_dir: Path,
    prepared: str | None,
    manifest: str | None,
) -> Path:
    require(not (prepared and manifest), "Choose --prepared OR --manifest")

    if manifest:
        p = Path(manifest).expanduser().resolve()
        require(p.is_file(), f"Manifest not found: {p}")
        return p

    if prepared:
        p = Path(prepared).expanduser().resolve()
        if p.is_file():
            return p
        require(p.is_dir(), f"Prepared directory not found: {p}")
        q = p / "reviewer_windows.jsonl"
        require(q.is_file(), f"reviewer_windows.jsonl not found: {q}")
        return q.resolve()

    root = script_dir / "prepared"
    require(root.is_dir(), f"Stage-01 prepared root not found: {root}")
    candidates = [
        p.resolve()
        for p in root.rglob("reviewer_windows.jsonl")
        if p.is_file()
    ]
    require(candidates, f"No Stage-01 reviewer_windows.jsonl under {root}")
    candidates.sort(
        key=lambda p: (p.stat().st_mtime_ns, str(p).lower()),
        reverse=True,
    )
    return candidates[0]


def extract_reference_emotion(stage01_manifest: Path) -> str | None:
    audit_path = stage01_manifest.parent / "input_audit.json"
    if not audit_path.is_file():
        return None
    try:
        audit = read_json(audit_path)
        selection = audit.get("selection")
        if isinstance(selection, Mapping):
            pair = selection.get("trial_pair")
            if isinstance(pair, Mapping) and pair.get("emotion") in EMOTIONS:
                return str(pair["emotion"])
    except Exception:
        return None
    return None


def validate_stage01(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 4, f"Stage 01 must contain 4 baseline windows, got {len(rows)}")

    subjects: set[str] = set()
    pair_keys: set[str] = set()
    missing_pair_key_rows = 0
    eeg_paths: set[str] = set()
    eeg_vars: set[str] = set()
    eeg_trials: set[int] = set()
    audio_paths: set[str] = set()
    video_paths: set[str] = set()
    instances: set[int] = set()

    indices: list[int] = []

    for row in rows:
        require(row.get("schema") == SOURCE_SCHEMA, "Wrong Stage-01 descriptor schema")
        require(row.get("window_seconds") == 5.0, "Stage-01 window must be 5 s")
        require(str(row.get("task_condition", "")).lower() == "speaking", "Speaking only")
        require(row.get("split") == "replay", "Stage-01 split must be replay")

        identity = row.get("identity")
        require(isinstance(identity, Mapping), "Stage-01 identity missing")
        subject = normalize_subject(str(identity.get("subject")))
        subjects.add(subject)

        pair = identity.get("pair_key")
        if pair is None:
            # Stage-01 V2.1 descriptors did not yet contain pair_key.
            # This is source identity metadata only; derive it deterministically
            # from the already validated subject + media instance below.
            missing_pair_key_rows += 1
        else:
            require(
                isinstance(pair, str) and pair.strip() == pair and pair,
                f"Invalid Stage-01 pair_key: {pair!r}",
            )
            pair_keys.add(pair)

        instance = identity.get("media_instance")
        require(type(instance) is int and 1 <= instance <= 200, "Invalid media_instance")
        instances.add(instance)

        wi = identity.get("window_idx_0based")
        require(type(wi) is int and 0 <= wi < 4, "Invalid Stage-01 window_idx_0based")
        indices.append(wi)

        mods = row.get("modalities")
        require(isinstance(mods, Mapping) and set(mods) == set(MODALITIES),
                "Stage-01 requires exactly three modalities")

        eeg = mods["eeg"]
        audio = mods["audio"]
        video = mods["video"]
        for name, item in (("eeg", eeg), ("audio", audio), ("video", video)):
            require(isinstance(item, Mapping) and item.get("present") is True,
                    f"Stage-01 {name} must be present")
            p = Path(str(item.get("path", ""))).expanduser().resolve()
            require(p.is_file(), f"Stage-01 {name} source missing: {p}")

        eeg_paths.add(str(Path(str(eeg["path"])).resolve()))
        eeg_vars.add(str(eeg.get("variable", "seg")))
        eeg_trials.add(int(eeg["trial_index"]))
        audio_paths.add(str(Path(str(audio["path"])).resolve()))
        video_paths.add(str(Path(str(video["path"])).resolve()))

    require(sorted(indices) == [0, 1, 2, 3], f"Stage-01 indices invalid: {indices}")
    require(len(subjects) == 1 and len(instances) == 1, "Mixed trial identity")
    require(len(eeg_paths) == len(eeg_vars) == len(eeg_trials) == 1, "Mixed EEG source")
    require(len(audio_paths) == len(video_paths) == 1, "Mixed media source")

    subject = next(iter(subjects))
    instance = next(iter(instances))
    canonical_pair_key = f"{subject}_instance{instance:03d}"

    require(
        not pair_keys or pair_keys == {canonical_pair_key},
        (
            "Stage-01 pair_key disagrees with validated subject/media instance. "
            f"Expected {canonical_pair_key!r}, got {sorted(pair_keys)!r}"
        ),
    )
    require(
        missing_pair_key_rows in (0, len(rows)),
        (
            "Mixed Stage-01 pair_key state rejected: either all four rows must "
            "contain the same pair_key or all four must be legacy rows without it."
        ),
    )

    require(subject not in TEST_SUBJECTS,
            f"{subject} is a frozen independent TEST subject; temporal ReviewerDemo refuses it")

    return {
        "subject": subject,
        "pair_key": canonical_pair_key,
        "pair_key_was_missing": missing_pair_key_rows == len(rows),
        "instance": instance,
        "eeg_path": Path(next(iter(eeg_paths))),
        "eeg_variable": next(iter(eeg_vars)),
        "eeg_trial_index": next(iter(eeg_trials)),
        "audio_path": Path(next(iter(audio_paths))),
        "video_path": Path(next(iter(video_paths))),
    }


def locate_executable(value: str, name: str) -> str:
    found = shutil.which(value)
    if found:
        return str(Path(found).resolve())
    p = Path(value).expanduser()
    require(p.is_file(), f"{name} executable not found: {value}")
    return str(p.resolve())


def run_checked(cmd: Sequence[str], timeout: float = 180.0) -> str:
    proc = subprocess.run(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode:
        raise TemporalPreparationError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stderr[-6000:]}"
        )
    return proc.stdout


def parse_rate(value: str) -> float:
    if "/" in value:
        a, b = value.split("/", 1)
        return float(a) / float(b)
    return float(value)


def probe_video(path: Path, ffprobe: str, *, count_frames: bool = False) -> dict[str, Any]:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd += [
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of", "json", str(path),
    ]
    data = json.loads(run_checked(cmd))
    streams = data.get("streams")
    require(isinstance(streams, list) and len(streams) == 1,
            f"Expected one video stream: {path}")
    s = streams[0]
    fps = parse_rate(str(s["avg_frame_rate"]))
    nominal = parse_rate(str(s["r_frame_rate"]))
    frame_value = s.get("nb_read_frames") if count_frames else s.get("nb_frames")
    frames = int(frame_value) if frame_value not in (None, "N/A") else None
    return {
        "fps": fps,
        "nominal_fps": nominal,
        "frames": frames,
        "width": int(s["width"]),
        "height": int(s["height"]),
        "duration": None if s.get("duration") in (None, "N/A") else float(s["duration"]),
    }


def canonical_eeg_tensor(path: Path, variable: str) -> tuple[np.ndarray, str]:
    from scipy.io import loadmat

    obj = loadmat(str(path), variable_names=[variable])
    actual = variable
    if variable not in obj and variable in ("seg", "seg1"):
        actual = "seg1" if variable == "seg" else "seg"
        obj = loadmat(str(path), variable_names=[actual])

    require(actual in obj, f"EEG variable missing: requested={variable}, path={path}")
    a = np.asarray(obj[actual])
    require(a.ndim == 3 and set(a.shape) == {10000, 30, 200},
            f"Unexpected EAV EEG tensor shape: {a.shape}")
    a = a.transpose(a.shape.index(10000), a.shape.index(30), a.shape.index(200))
    return a, actual


def load_audio_trial(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    require(x.ndim in (1, 2) and np.isfinite(x).all(), "Invalid Audio source")
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    source_sr = int(sr)
    if source_sr != AUDIO_FS:
        g = math.gcd(source_sr, AUDIO_FS)
        x = resample_poly(x, AUDIO_FS // g, source_sr // g).astype(np.float32)

    required = int(TRIAL_SECONDS * AUDIO_FS)
    require(x.shape[0] >= required,
            f"Audio shorter than {TRIAL_SECONDS:.0f}s after resampling: {x.shape[0]}")
    return x[:required].copy(), source_sr


def stride_steps(stride_seconds: float) -> list[tuple[int, float, float, float]]:
    require(math.isfinite(stride_seconds) and stride_seconds > 0,
            "--stride-seconds must be positive")

    for rate, label in ((EEG_FS, "EEG"), (AUDIO_FS, "Audio"), (VIDEO_FPS, "Video")):
        value = stride_seconds * rate
        require(abs(value - round(value)) < 1e-9,
                f"Stride {stride_seconds}s does not align to integer {label} samples/frames")

    count_float = (TRIAL_SECONDS - WINDOW_SECONDS) / stride_seconds
    require(abs(count_float - round(count_float)) < 1e-9,
            "Stride must tile the 0..15s start range exactly")
    count = int(round(count_float)) + 1

    out = []
    for i in range(count):
        start = i * stride_seconds
        end = start + WINDOW_SECONDS
        out.append((i, start, end, (start + end) / 2.0))
    require(out[-1][2] <= TRIAL_SECONDS + 1e-9, "Last temporal window exceeds trial")
    return out


def prepare_temporal(
    *,
    info: Mapping[str, Any],
    stage01_manifest: Path,
    output: Path,
    stride_seconds: float,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)

    eeg_dir = output / "eeg"
    audio_dir = output / "audio"
    video_dir = output / "video"
    eeg_dir.mkdir()
    audio_dir.mkdir()
    video_dir.mkdir()

    steps = stride_steps(stride_seconds)
    subject = str(info["subject"])
    pair_key = str(info["pair_key"])
    instance = int(info["instance"])

    # ---------- EEG ----------
    eeg_tensor, actual_var = canonical_eeg_tensor(
        Path(info["eeg_path"]), str(info["eeg_variable"])
    )
    trial = int(info["eeg_trial_index"])
    require(0 <= trial < 200, "Invalid EEG trial index")
    eeg_trial = np.asarray(eeg_tensor[:, :, trial].T)
    require(eeg_trial.shape == (30, 10000), f"EEG trial shape mismatch: {eeg_trial.shape}")

    # ---------- Audio ----------
    audio_full, source_audio_sr = load_audio_trial(Path(info["audio_path"]))

    # ---------- Video ----------
    video_source = Path(info["video_path"])
    video_probe = probe_video(video_source, ffprobe, count_frames=True)
    require(abs(video_probe["fps"] - VIDEO_FPS) < 1e-6,
            f"Temporal replay expects recorded 30fps video; got {video_probe['fps']}")
    require(abs(video_probe["nominal_fps"] - VIDEO_FPS) < 1e-6,
            f"Temporal replay expects nominal 30fps video; got {video_probe['nominal_fps']}")
    require(video_probe["frames"] is not None and int(video_probe["frames"]) >= 600,
            f"Video must contain at least 600 frames; got {video_probe['frames']}")

    import soundfile as sf

    session_id = f"reviewer_trajectory_{pair_key}"
    clock_id = f"eav_trial_relative:{pair_key}:stride_{stride_seconds:g}s"

    descriptors: list[dict[str, Any]] = []
    window_audit: list[dict[str, Any]] = []

    for step, start_s, end_s, center_s in steps:
        eeg_start = int(round(start_s * EEG_FS))
        eeg_end = eeg_start + int(WINDOW_SECONDS * EEG_FS)
        audio_start = int(round(start_s * AUDIO_FS))
        audio_end = audio_start + int(WINDOW_SECONDS * AUDIO_FS)
        video_start_frame = int(round(start_s * VIDEO_FPS))
        video_end_frame = video_start_frame + int(WINDOW_SECONDS * VIDEO_FPS)

        eeg_window = np.ascontiguousarray(eeg_trial[:, eeg_start:eeg_end])
        require(eeg_window.shape == (30, 2500), f"EEG slice {step} shape mismatch")
        eeg_path = eeg_dir / f"step_{step:03d}_{start_s:05.1f}_{end_s:05.1f}.npy"
        np.save(eeg_path, eeg_window, allow_pickle=False)

        audio_window = np.asarray(audio_full[audio_start:audio_end], dtype=np.float32)
        require(audio_window.shape == (80000,), f"Audio slice {step} shape mismatch")
        audio_path = audio_dir / f"step_{step:03d}_{start_s:05.1f}_{end_s:05.1f}.wav"
        # Same quantization family as the formal raw EAV A0 replay policy.
        sf.write(str(audio_path), audio_window, AUDIO_FS, subtype="PCM_16")

        video_path = video_dir / f"step_{step:03d}_{start_s:05.1f}_{end_s:05.1f}.mkv"
        vf = (
            f"trim=start_frame={video_start_frame}:end_frame={video_end_frame},"
            "setpts=PTS-STARTPTS"
        )
        run_checked([
            ffmpeg, "-nostdin", "-v", "error",
            "-i", str(video_source),
            "-map", "0:v:0",
            "-vf", vf,
            "-an",
            "-c:v", "ffv1",
            "-level", "3",
            "-g", "1",
            "-threads", "1",
            "-fps_mode", "passthrough",
            str(video_path),
        ])
        vp = probe_video(video_path, ffprobe, count_frames=True)
        require(vp["frames"] == 150, f"Temporal video step {step} not exactly 150 frames")
        require(abs(vp["fps"] - VIDEO_FPS) < 1e-6, f"Temporal video step {step} not 30fps")

        window_id = f"{pair_key.upper()}_TRAJECTORY_T{step:03d}"
        quality_window = {
            "session_id": session_id,
            "window_id": window_id,
            "clock_id": clock_id,
            "start_seconds": float(start_s),
            "end_seconds": float(end_s),
        }

        descriptor = {
            "schema": SOURCE_SCHEMA,
            "window_id": window_id,
            "session_id": session_id,
            "window_seconds": WINDOW_SECONDS,
            "task_condition": "Speaking",
            "split": "replay",
            "identity": {
                "subject": subject,
                "pair_key": pair_key,
                "source_type": "EAV_REVIEWER_TEMPORAL_TRAJECTORY",
                "media_instance": instance,
                "trajectory_step_0based": step,
                "start_seconds": float(start_s),
                "end_seconds": float(end_s),
                "center_seconds": float(center_s),
                "stride_seconds": float(stride_seconds),
            },
            "quality_window": quality_window,
            # No reference_label: reviewer ground truth is not a model input.
            "modalities": {
                "eeg": {
                    "present": True,
                    "path": str(eeg_path.resolve()),
                    "kind": "eeg_window",
                    "sample_rate": EEG_FS,
                    "input_unit": "training_native",
                    "channel_names": CHANNELS,
                },
                "audio": {
                    "present": True,
                    "path": str(audio_path.resolve()),
                    "kind": "audio_window",
                },
                "video": {
                    "present": True,
                    "path": str(video_path.resolve()),
                    "kind": "video_window",
                },
            },
        }
        descriptors.append(descriptor)
        window_audit.append({
            "step": step,
            "start_seconds": start_s,
            "end_seconds": end_s,
            "center_seconds": center_s,
            "eeg_sample_range": [eeg_start, eeg_end],
            "audio_sample_range_16k": [audio_start, audio_end],
            "video_frame_range_30fps": [video_start_frame, video_end_frame],
            "eeg_path": str(eeg_path.resolve()),
            "audio_path": str(audio_path.resolve()),
            "video_path": str(video_path.resolve()),
        })

    manifest = output / "temporal_windows.jsonl"
    write_jsonl(manifest, descriptors)

    reference_emotion = extract_reference_emotion(stage01_manifest)
    audit = {
        "schema": OUTPUT_SCHEMA,
        "version": VERSION,
        "status": "PASS",
        "created_utc": utc_now(),
        "purpose": "Create dense overlapping 5-s replay windows for an emotion-over-time trajectory.",
        "stage01_manifest": str(stage01_manifest),
        "stage01_manifest_sha256": sha256_file(stage01_manifest),
        "subject": subject,
        "pair_key": pair_key,
        "pair_key_was_missing_in_stage01": bool(info.get("pair_key_was_missing")),
        "pair_key_resolution": (
            "DERIVED_FROM_VALIDATED_SUBJECT_AND_MEDIA_INSTANCE"
            if info.get("pair_key_was_missing")
            else "PRESERVED_FROM_STAGE01"
        ),
        "media_instance": instance,
        "reference_emotion": reference_emotion,
        "reference_emotion_usage": "AUDIT_ONLY_NOT_WRITTEN_TO_RUNTIME_DESCRIPTORS",
        "temporal_policy": {
            "trial_seconds": TRIAL_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": stride_seconds,
            "window_count": len(descriptors),
            "first_center_seconds": descriptors[0]["identity"]["center_seconds"],
            "last_center_seconds": descriptors[-1]["identity"]["center_seconds"],
            "smoothing_applied": False,
            "majority_vote_applied": False,
            "new_model_logic_added": False,
        },
        "source": {
            "eeg_path": str(Path(info["eeg_path"]).resolve()),
            "eeg_variable_requested": str(info["eeg_variable"]),
            "eeg_variable_actual": actual_var,
            "eeg_trial_index_0based": trial,
            "audio_path": str(Path(info["audio_path"]).resolve()),
            "audio_source_sample_rate": source_audio_sr,
            "audio_runtime_sample_rate": AUDIO_FS,
            "audio_output_encoding": "PCM_16",
            "video_path": str(video_source.resolve()),
            "video_probe": video_probe,
        },
        "window_audit": window_audit,
        "output_manifest": str(manifest),
        "output_manifest_sha256": sha256_file(manifest),
        "stage_boundary": {
            "model_inference": False,
            "quality_inference": False,
            "fusion": False,
            "robot_action": False,
            "final_deployment_files_modified": False,
        },
    }
    write_json(output / "temporal_audit.json", audit)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Prepare overlapping 5-s EAV windows for ReviewerDemo emotion trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prepared", default=None, help="Stage-01 prepared directory or reviewer_windows.jsonl")
    p.add_argument("--manifest", default=None, help="Explicit Stage-01 reviewer_windows.jsonl")
    p.add_argument("--stride-seconds", type=float, default=1.0)
    p.add_argument("--output", default=None)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--version", action="version", version=VERSION)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent

    manifest = resolve_stage01_manifest(script_dir, args.prepared, args.manifest)
    rows = read_jsonl(manifest)
    info = validate_stage01(rows)

    ffmpeg = locate_executable(args.ffmpeg, "ffmpeg")
    ffprobe = locate_executable(args.ffprobe, "ffprobe")

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        suffix = f"5s_stride{args.stride_seconds:g}s"
        output = (
            script_dir
            / "temporal_prepared"
            / f"{info['pair_key']}_{suffix}"
        ).resolve()

    print("=" * 110)
    print("EAV REVIEWER DEMO — STAGE 04: TEMPORAL TRAJECTORY PREPARATION")
    print("=" * 110)
    print(f"Version       : {VERSION}")
    print(f"Stage-01 input: {manifest}")
    print(f"Subject       : {info['subject']}")
    print(f"Pair key      : {info['pair_key']}")
    if info.get("pair_key_was_missing"):
        print("Compatibility : Stage-01 legacy manifest had no pair_key; "
              "derived deterministically from subject + media instance")
    print(f"Window        : {WINDOW_SECONDS:.1f} s (frozen)")
    print(f"Stride        : {args.stride_seconds:.1f} s")
    print(f"Output        : {output}")
    print()

    audit = prepare_temporal(
        info=info,
        stage01_manifest=manifest,
        output=output,
        stride_seconds=args.stride_seconds,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )

    print()
    print("=" * 110)
    print("STAGE 04 COMPLETE")
    print("=" * 110)
    print("STATUS          : PASS")
    print(f"WINDOWS         : {audit['temporal_policy']['window_count']}")
    print(f"TIME CENTERS    : {audit['temporal_policy']['first_center_seconds']:.1f}s -> "
          f"{audit['temporal_policy']['last_center_seconds']:.1f}s")
    print(f"MANIFEST        : {audit['output_manifest']}")
    print(f"AUDIT           : {output / 'temporal_audit.json'}")
    print("MODEL INFERENCE : NO")
    print("QUALITY/FUSION  : NO")
    print("CORE MODIFIED   : NO")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TemporalPreparationError as exc:
        print()
        print("=" * 110, file=sys.stderr)
        print("STAGE 04 FAILED", file=sys.stderr)
        print("=" * 110, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 110, file=sys.stderr)
        raise SystemExit(2)
