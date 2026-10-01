#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ReviewerDemo Stage 05 — run and save an emotion-over-time trajectory
===================================================================

Consumes Stage-04 `temporal_windows.jsonl`, calls the SAME frozen final-deployment
main.py replay runtime, and exports the 5-s sliding-window predictions as a
time-indexed trajectory.

No model, checkpoint, quality formula, threshold, route, or fusion logic is
implemented here.

Outputs:
    emotion_trajectory.jsonl
    emotion_trajectory.csv
    emotion_trajectory_summary.json
    main_console.log
    runtime_run/...

The trajectory is the raw sequence of frozen-model outputs. This script does NOT
apply smoothing, majority vote, HMMs, persistence rules, interpolation, or a new
20-s classifier.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = "EAV-REVIEWER-TEMPORAL-INFERENCE.1.0"
SOURCE_SCHEMA = "eav.system.source_window.v1"
TRAJECTORY_SCHEMA = "eav.reviewer_demo.emotion_trajectory.v1"
EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
MODALITIES = ("eeg", "audio", "video")
TEST_SUBJECTS = {
    "subject03", "subject05", "subject20",
    "subject31", "subject35", "subject39",
}


class TemporalInferenceError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TemporalInferenceError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise TemporalInferenceError(
                    f"Invalid JSONL {path}:{line_no}: {exc}"
                ) from exc
            require(isinstance(value, dict), f"JSONL row {line_no} is not an object")
            rows.append(value)
    if not allow_empty:
        require(rows, f"No JSONL records found: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_stage02_helpers(script_dir: Path):
    path = script_dir / "02_run_eav_reviewer_inference.py"
    require(path.is_file(), f"Stage 02 helper missing: {path}")
    spec = importlib.util.spec_from_file_location("_reviewer_stage02_helpers", path)
    require(spec is not None and spec.loader is not None, f"Cannot import Stage 02: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise TemporalInferenceError(
            f"Could not load Stage 02 helper module: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def resolve_temporal_manifest(
    script_dir: Path,
    prepared: str | None,
    manifest: str | None,
) -> Path:
    require(not (prepared and manifest), "Choose --prepared OR --manifest")

    if manifest:
        p = Path(manifest).expanduser().resolve()
        require(p.is_file(), f"Temporal manifest not found: {p}")
        return p

    if prepared:
        p = Path(prepared).expanduser().resolve()
        if p.is_file():
            return p
        require(p.is_dir(), f"Temporal prepared directory not found: {p}")
        q = p / "temporal_windows.jsonl"
        require(q.is_file(), f"temporal_windows.jsonl not found: {q}")
        return q.resolve()

    root = script_dir / "temporal_prepared"
    require(root.is_dir(), f"Temporal prepared root not found: {root}")
    candidates = [
        p.resolve()
        for p in root.rglob("temporal_windows.jsonl")
        if p.is_file()
    ]
    require(candidates, f"No temporal_windows.jsonl found under {root}")
    candidates.sort(
        key=lambda p: (p.stat().st_mtime_ns, str(p).lower()),
        reverse=True,
    )
    return candidates[0]


def validate_temporal_manifest(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    require(len(rows) >= 2, "Temporal manifest needs at least two windows")

    subjects: set[str] = set()
    pair_keys: set[str] = set()
    session_ids: set[str] = set()
    starts: list[float] = []
    ends: list[float] = []
    centers: list[float] = []
    steps: list[int] = []

    for i, row in enumerate(rows):
        require(row.get("schema") == SOURCE_SCHEMA, f"Row {i}: wrong schema")
        require(row.get("window_seconds") == 5.0, f"Row {i}: window must remain 5 s")
        require(str(row.get("task_condition", "")).lower() == "speaking", f"Row {i}: Speaking only")
        require(row.get("split") == "replay", f"Row {i}: split must be replay")
        require("reference_label" not in row, f"Row {i}: reference_label must not enter trajectory runtime")

        session = row.get("session_id")
        require(isinstance(session, str) and session, f"Row {i}: session_id missing")
        session_ids.add(session)

        identity = row.get("identity")
        require(isinstance(identity, Mapping), f"Row {i}: identity missing")
        subject = str(identity.get("subject"))
        subjects.add(subject)
        pair = identity.get("pair_key")
        require(isinstance(pair, str) and pair, f"Row {i}: pair_key missing")
        pair_keys.add(pair)

        step = identity.get("trajectory_step_0based")
        require(type(step) is int and step >= 0, f"Row {i}: invalid trajectory step")
        steps.append(step)

        start = float(identity["start_seconds"])
        end = float(identity["end_seconds"])
        center = float(identity["center_seconds"])
        require(math.isfinite(start) and math.isfinite(end) and math.isfinite(center), "Nonfinite time")
        require(abs((end - start) - 5.0) < 1e-9, f"Row {i}: temporal span is not 5 s")
        require(abs(center - (start + end) / 2.0) < 1e-9, f"Row {i}: wrong center time")
        starts.append(start); ends.append(end); centers.append(center)

        qw = row.get("quality_window")
        require(isinstance(qw, Mapping), f"Row {i}: quality_window missing")
        require(qw.get("session_id") == session and qw.get("window_id") == row.get("window_id"),
                f"Row {i}: quality_window identity mismatch")
        require(float(qw["start_seconds"]) == start and float(qw["end_seconds"]) == end,
                f"Row {i}: quality_window span mismatch")

        mods = row.get("modalities")
        require(isinstance(mods, Mapping) and set(mods) == set(MODALITIES),
                f"Row {i}: exactly three modalities required")
        for m in MODALITIES:
            item = mods[m]
            require(isinstance(item, Mapping) and item.get("present") is True,
                    f"Row {i}: {m} must be present")
            p = Path(str(item.get("path", ""))).expanduser().resolve()
            require(p.is_file(), f"Row {i}: {m} window missing: {p}")

    require(len(subjects) == len(pair_keys) == len(session_ids) == 1, "Mixed trajectory identity")
    require(sorted(steps) == list(range(len(rows))), "Trajectory steps are not contiguous 0..N-1")
    require(all(b > a for a, b in zip(starts, starts[1:])), "Start times not increasing")

    stride_values = [b - a for a, b in zip(starts, starts[1:])]
    stride = stride_values[0]
    require(all(abs(x - stride) < 1e-9 for x in stride_values), "Trajectory stride is inconsistent")

    subject = next(iter(subjects))
    require(subject not in TEST_SUBJECTS,
            f"{subject} is frozen TEST data; ReviewerDemo trajectory refuses it")

    return {
        "rows": rows,
        "subject": subject,
        "pair_key": next(iter(pair_keys)),
        "session_id": next(iter(session_ids)),
        "window_count": len(rows),
        "stride_seconds": stride,
        "first_start_seconds": starts[0],
        "last_end_seconds": ends[-1],
        "first_center_seconds": centers[0],
        "last_center_seconds": centers[-1],
    }


def choose_output(script_dir: Path, explicit: str | None, pair_key: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = (
            script_dir
            / "trajectory_runs"
            / f"{pair_key}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        ).resolve()
    require(not p.exists(), f"Output already exists: {p}")
    p.mkdir(parents=True)
    return p


def run_and_tee(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    print()
    print("=" * 116)
    print("LAUNCHING FROZEN MAIN.PY FOR TEMPORAL TRAJECTORY")
    print("=" * 116)
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    print("=" * 116)

    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(proc.wait())


def modality_probabilities(report: Mapping[str, Any], modality: str) -> list[float] | None:
    emotion = report.get("emotion")
    if isinstance(emotion, Mapping):
        for key in (f"{modality}_probs", "probabilities"):
            value = emotion.get(key)
            if isinstance(value, list) and len(value) == 5:
                return [float(x) for x in value]

    block = report.get(f"{modality}_emotion")
    if isinstance(block, Mapping):
        for key in (f"{modality}_probs", "probabilities"):
            value = block.get(key)
            if isinstance(value, list) and len(value) == 5:
                return [float(x) for x in value]

    packet = report.get("packet")
    if isinstance(packet, Mapping):
        value = packet.get("probabilities")
        if isinstance(value, list) and len(value) == 5:
            return [float(x) for x in value]
    return None


def prediction_from_probs(probs: list[float] | None) -> tuple[str | None, float | None]:
    if not probs:
        return None, None
    idx = max(range(5), key=probs.__getitem__)
    return EMOTIONS[idx], float(probs[idx])


def build_trajectory(runtime_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trajectory: list[dict[str, Any]] = []

    for row in runtime_rows:
        source = row.get("source_window")
        require(isinstance(source, Mapping), "Runtime row missing source_window")
        ident = source.get("identity")
        require(isinstance(ident, Mapping), "Runtime source identity missing")

        fusion = row.get("fusion")
        require(isinstance(fusion, Mapping), "Runtime row missing fusion")
        final = fusion.get("final")
        require(isinstance(final, Mapping), "Runtime fusion final missing")

        final_probs = final.get("probabilities")
        if final_probs is not None:
            require(isinstance(final_probs, list) and len(final_probs) == 5,
                    "Invalid final probability vector")
            final_probs = [float(x) for x in final_probs]

        quality = fusion.get("quality_for_fusion")
        quality = quality if isinstance(quality, Mapping) else {}
        availability = fusion.get("availability")
        availability = availability if isinstance(availability, Mapping) else {}

        reports = row.get("reports")
        reports = reports if isinstance(reports, Mapping) else {}

        modality_rows: dict[str, Any] = {}
        for modality in MODALITIES:
            report = reports.get(modality)
            report = report if isinstance(report, Mapping) else {}
            probs = modality_probabilities(report, modality)
            emotion, confidence = prediction_from_probs(probs)
            available = bool(availability.get(modality, report.get("status") == "OK"))
            modality_rows[modality] = {
                "available": available,
                "quality": None if quality.get(modality) is None else float(quality[modality]),
                "emotion": emotion if available else "UNAVAILABLE",
                "confidence": confidence if available else None,
                "probabilities": probs,
            }

        item = {
            "schema": TRAJECTORY_SCHEMA,
            "step": int(ident["trajectory_step_0based"]),
            "window_id": row.get("window_id"),
            "start_seconds": float(ident["start_seconds"]),
            "end_seconds": float(ident["end_seconds"]),
            "center_seconds": float(ident["center_seconds"]),
            "route": fusion.get("active_branch"),
            "fusion_status": fusion.get("status"),
            "final_emotion": final.get("emotion"),
            "final_confidence": final.get("confidence"),
            "final_probabilities": final_probs,
            "class_order": EMOTIONS,
            "modalities": modality_rows,
            "no_decision": fusion.get("status") == "NO_DECISION",
            "robot_action_authorized": False,
            "smoothing_applied": False,
        }
        trajectory.append(item)

    trajectory.sort(key=lambda x: x["step"])
    require(
        [x["step"] for x in trajectory] == list(range(len(trajectory))),
        "Runtime trajectory steps are incomplete/out of order",
    )
    return trajectory


def write_trajectory_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")

    fields = [
        "step", "start_seconds", "end_seconds", "center_seconds",
        "route", "fusion_status", "final_emotion", "final_confidence",
        "p_neutral", "p_sadness", "p_anger", "p_happiness", "p_calmness",
        "eeg_emotion", "eeg_confidence", "q_eeg",
        "audio_emotion", "audio_confidence", "q_audio",
        "video_emotion", "video_confidence", "q_video",
    ]
    for m in MODALITIES:
        fields += [f"{m}_p_{e.lower()}" for e in EMOTIONS]

    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in rows:
            final_probs = item.get("final_probabilities") or [None] * 5
            out: dict[str, Any] = {
                "step": item["step"],
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "center_seconds": item["center_seconds"],
                "route": item["route"],
                "fusion_status": item["fusion_status"],
                "final_emotion": item["final_emotion"],
                "final_confidence": item["final_confidence"],
                "p_neutral": final_probs[0],
                "p_sadness": final_probs[1],
                "p_anger": final_probs[2],
                "p_happiness": final_probs[3],
                "p_calmness": final_probs[4],
            }
            for m in MODALITIES:
                x = item["modalities"][m]
                out[f"{m}_emotion"] = x["emotion"]
                out[f"{m}_confidence"] = x["confidence"]
                out[f"q_{m}"] = x["quality"]
                probs = x.get("probabilities") or [None] * 5
                for i, emotion in enumerate(EMOTIONS):
                    out[f"{m}_p_{emotion.lower()}"] = probs[i]
            writer.writerow(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent

    p = argparse.ArgumentParser(
        description="Run frozen main.py on Stage-04 temporal windows and save emotion trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prepared", default=None, help="Stage-04 directory or temporal_windows.jsonl")
    p.add_argument("--manifest", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--config", default=str(deployment_root / "system_config.json"))
    p.add_argument("--main", default=str(deployment_root / "main.py"))
    p.add_argument("--release", default=None)
    p.add_argument("--fusion-script", default=str(deployment_root / "Fusion" / "融合层部署模块.py"))
    p.add_argument("--assets-dir", default=str(deployment_root / "Fusion"))
    p.add_argument("--fusion-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--eeg-unit", choices=("V", "mV", "uV"), default="uV")
    p.add_argument("--error-policy", choices=("raise", "exclude"), default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--version", action="version", version=VERSION)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent

    manifest = resolve_temporal_manifest(script_dir, args.prepared, args.manifest)
    temporal = validate_temporal_manifest(manifest)

    python_exe = Path(args.python).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    main_py = Path(args.main).expanduser().resolve()
    fusion_script = Path(args.fusion_script).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()

    for p, label, want_dir in (
        (python_exe, "Python", False),
        (config, "system_config.json", False),
        (main_py, "main.py", False),
        (fusion_script, "fusion script", False),
        (assets_dir, "fusion assets", True),
    ):
        require(p.is_dir() if want_dir else p.is_file(), f"{label} missing: {p}")

    stage02 = load_stage02_helpers(script_dir)
    config_info = stage02.inspect_config(config, eeg_unit_override=args.eeg_unit)
    release, release_source, release_info = stage02.resolve_release(
        deployment_root=deployment_root,
        config=config,
        config_info=config_info,
        explicit_release=args.release,
    )

    output = choose_output(script_dir, args.output, temporal["pair_key"])
    runtime_dir = output / "runtime_run"
    console_log = output / "main_console.log"

    cmd = [
        str(python_exe),
        str(main_py),
        "--mode", "replay",
        "--config", str(config),
        "--release", str(release),
        "--manifest", str(manifest),
        "--fusion-script", str(fusion_script),
        "--assets-dir", str(assets_dir),
        "--fusion-device", args.fusion_device,
        "--output", str(runtime_dir),
        "--eeg-unit", args.eeg_unit,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.error_policy:
        cmd += ["--error-policy", args.error_policy]

    print("=" * 116)
    print("EAV REVIEWER DEMO — STAGE 05: TEMPORAL EMOTION TRAJECTORY")
    print("=" * 116)
    print(f"Version        : {VERSION}")
    print(f"Pair key       : {temporal['pair_key']}")
    print(f"Temporal input : {manifest}")
    print(f"Windows        : {temporal['window_count']}")
    print(f"Stride         : {temporal['stride_seconds']:.3f} s")
    print(f"Centers        : {temporal['first_center_seconds']:.1f} -> {temporal['last_center_seconds']:.1f} s")
    print(f"Frozen release : {release}")
    print(f"Output         : {output}")

    rc = run_and_tee(cmd, deployment_root, console_log)
    if rc != 0:
        diagnostics = {}
        if (runtime_dir / "run_summary.json").is_file():
            diagnostics["run_summary"] = read_json(runtime_dir / "run_summary.json")
        if (runtime_dir / "fatal_error.json").is_file():
            diagnostics["fatal_error"] = read_json(runtime_dir / "fatal_error.json")
        write_json(output / "trajectory_failure.json", {
            "schema": "eav.reviewer_demo.emotion_trajectory.failure.v1",
            "version": VERSION,
            "status": "FAILED_RUNTIME",
            "return_code": rc,
            "diagnostics": diagnostics,
        })
        raise TemporalInferenceError(
            f"Frozen main.py exited with code {rc}. See {console_log}"
        )

    summary = read_json(runtime_dir / "run_summary.json")
    require(summary.get("status") == "PASS",
            f"Frozen runtime status is not PASS: {summary.get('status')!r}")
    require(summary.get("n_processed") == temporal["window_count"],
            f"Processed {summary.get('n_processed')} windows; expected {temporal['window_count']}")

    runtime_rows = read_jsonl(runtime_dir / "window_results.jsonl")
    require(len(runtime_rows) == temporal["window_count"],
            f"window_results count mismatch: {len(runtime_rows)}")

    trajectory = build_trajectory(runtime_rows)

    jsonl_path = output / "emotion_trajectory.jsonl"
    csv_path = output / "emotion_trajectory.csv"
    summary_path = output / "emotion_trajectory_summary.json"

    write_jsonl(jsonl_path, trajectory)
    write_trajectory_csv(csv_path, trajectory)

    routes = Counter(str(x.get("route")) for x in trajectory)
    emotions = Counter(str(x.get("final_emotion")) for x in trajectory)
    no_decision = sum(int(bool(x.get("no_decision"))) for x in trajectory)

    temporal_audit_path = manifest.parent / "temporal_audit.json"
    temporal_audit = read_json(temporal_audit_path) if temporal_audit_path.is_file() else None
    reference = temporal_audit.get("reference_emotion") if isinstance(temporal_audit, Mapping) else None

    trajectory_summary = {
        "schema": "eav.reviewer_demo.emotion_trajectory.summary.v1",
        "version": VERSION,
        "status": "PASS",
        "created_utc": utc_now(),
        "subject": temporal["subject"],
        "pair_key": temporal["pair_key"],
        "reference_emotion": reference,
        "reference_emotion_usage": "AUDIT_ONLY_NOT_MODEL_INPUT",
        "temporal_policy": {
            "window_seconds": 5.0,
            "stride_seconds": temporal["stride_seconds"],
            "window_count": temporal["window_count"],
            "first_center_seconds": temporal["first_center_seconds"],
            "last_center_seconds": temporal["last_center_seconds"],
            "smoothing_applied": False,
            "majority_vote_applied": False,
            "new_20s_classifier_created": False,
        },
        "route_counts": dict(routes),
        "final_emotion_counts": dict(emotions),
        "no_decision_windows": no_decision,
        "runtime": {
            "main_version": summary.get("version"),
            "status": summary.get("status"),
            "release_sha256": summary.get("release_sha256"),
            "policy_sha256": summary.get("policy_sha256"),
            "raw_modality_invocation_counts": summary.get("raw_modality_invocation_counts"),
            "assets_verified_after_run": summary.get("assets_verified_after_run"),
            "training_performed": summary.get("training_performed"),
            "robot_actions_performed": summary.get("robot_actions_performed"),
        },
        "release": {
            "path": str(release),
            "resolution": release_source,
            "embedded_release_sha256": release_info["embedded_release_sha256"],
        },
        "outputs": {
            "emotion_trajectory_jsonl": str(jsonl_path),
            "emotion_trajectory_csv": str(csv_path),
            "runtime_run": str(runtime_dir),
            "main_console_log": str(console_log),
        },
    }
    write_json(summary_path, trajectory_summary)

    print()
    print("=" * 116)
    print("STAGE 05 COMPLETE")
    print("=" * 116)
    print("STATUS            : PASS")
    print(f"TRAJECTORY POINTS : {len(trajectory)}")
    print(f"ROUTES            : {dict(routes)}")
    print(f"EMOTIONS          : {dict(emotions)}")
    print(f"NO DECISION       : {no_decision}")
    print(f"JSONL             : {jsonl_path}")
    print(f"CSV               : {csv_path}")
    print(f"SUMMARY           : {summary_path}")
    print("SMOOTHING         : NO")
    print("CORE MODIFIED     : NO")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TemporalInferenceError as exc:
        print()
        print("=" * 116, file=sys.stderr)
        print("STAGE 05 FAILED", file=sys.stderr)
        print("=" * 116, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 116, file=sys.stderr)
        raise SystemExit(2)
