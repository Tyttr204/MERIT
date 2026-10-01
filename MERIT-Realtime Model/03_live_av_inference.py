#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
03_live_av_inference.py
=======================

EAV LiveInteraction — Stage 03
Real-time laptop Audio + Video capture -> frozen main.py -> emotion inference.

Architecture
------------
Laptop camera + microphone
        |
        | continuous acquisition on one time.monotonic() clock
        v
strict synchronized 5-second windows
        |
        | EEG is explicitly unavailable
        v
atomic live inbox descriptor
        |
        v
FROZEN ../main.py --mode live
        |
        +--> existing Audio emotion / quality heads
        +--> existing Video emotion / quality heads
        +--> existing KEEP_V1 QualityController
        +--> EEG unavailable -> robust route
        +--> existing AF4-B fusion
        v
live five-class emotion result

This script deliberately DOES NOT copy or reimplement any emotion model,
quality formula, threshold, router, or fusion model.  The frozen main.py remains
the only inference runtime.

Why a session-local runtime config is created
---------------------------------------------
The frozen main.py source and all learned/model assets remain untouched.  Its
default live scheduling freshness/deadline values were engineering placeholders
for a very short latency budget.  Real Audio+Video neural inference can take
longer than that.  Therefore this script creates a NEW session-local config
whose ONLY intentional semantic changes are:

    runtime.stale_after_sec
    runtime.deadline_sec

These are live scheduling/freshness budgets, not learned quality thresholds,
not q mapping parameters, and not fusion weights.  The original
system_config.json is never overwritten.  The exact before/after values and
SHA256 hashes are recorded in runtime_config_delta.json.

Default behavior
----------------
For a safe first integration test, capture 3 consecutive five-second windows.
Use --windows 0 for continuous operation until Q/ESC/Ctrl+C.

Recommended run
---------------
    python .\LiveInteraction\03_live_av_inference.py

Continuous:
    python .\LiveInteraction\03_live_av_inference.py --windows 0

Requirements
------------
Sibling file:
    MERIT-Realtime Model/02_capture_synchronized_av_window.py

Deployment root:
    ../main.py
    ../system_config.json
    ../Quality/
    ../Fusion/
    ../audio/
    ../video/
    ../eeg/

Stage-01 device config:
    MERIT-Realtime Model/config/detected_av_devices.json

No robot action is authorized by this script.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "EAV-LIVE-AV-INFERENCE.1.0"
SUMMARY_SCHEMA = "eav.live_interaction.live_av_inference.v1"

EXPECTED_MAIN_VERSION = "EAV-MAIN-INTEGRATION.2.1.0-FINAL-TEST"
EXPECTED_STAGE02_VERSION_PREFIX = "EAV-LIVE-SYNC-CAPTURE."

DEFAULT_WINDOWS = 3
DEFAULT_LEAD_SEC = 0.50
DEFAULT_SETTLE_SEC = 0.12
DEFAULT_RETENTION_SEC = 15.0

# Operational LIVE scheduling budget only.
# main.py itself enforces <= 30 seconds.
DEFAULT_LIVE_STALE_SEC = 12.0
DEFAULT_LIVE_DEADLINE_SEC = 10.0

DEFAULT_MAIN_READY_TIMEOUT_SEC = 600.0

EEG_UNIT = "uV"
EEG_UNIT_EVIDENCE = (
    "Historical eq1_feature_contract.json: unit=uV; volts_per_source_unit=1e-6; "
    "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED"
)

EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]


class LiveInferenceError(RuntimeError):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise LiveInferenceError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def atomic_write_json(path: Path, obj: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        last_error = None
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                if attempt == 19:
                    break
                time.sleep(min(0.02 * (attempt + 1), 0.20))
        if last_error is not None:
            raise LiveInferenceError(
                f"Could not atomically publish JSON: {path}"
            ) from last_error
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def import_module_from_path(path: Path, name: str):
    require(path.is_file(), f"Missing Python module: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def discover_release(
    deployment_root: Path,
    configured: str | None,
    explicit: str | None,
) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p.is_file(), f"--release does not exist: {p}")
        return p

    if configured:
        p = Path(configured).expanduser().resolve()
        if p.is_file():
            return p

    preferred = [
        deployment_root / "Quality" / "quality_layer_release_candidate.json",
        deployment_root / "quality_layer_release_candidate.json",
    ]
    for p in preferred:
        if p.is_file():
            return p.resolve()

    candidates = list(deployment_root.glob("system_checks/**/quality_layer_release_candidate.json"))
    valid: list[tuple[Path, str | None]] = []
    for p in candidates[:500]:
        try:
            obj = read_json(p)
        except Exception:
            continue
        if (
            obj.get("schema") == "eav.quality_layer.release_candidate.v1"
            and obj.get("variant") == "KEEP_V1"
        ):
            valid.append((p.resolve(), obj.get("release_sha256")))

    require(
        valid,
        "Could not locate frozen quality_layer_release_candidate.json. "
        "Pass it explicitly with --release.",
    )

    hashes = {h for _, h in valid if h}
    require(
        len(hashes) <= 1,
        "Multiple non-identical KEEP_V1 release candidates were found. "
        "Pass the intended file explicitly with --release.",
    )
    valid.sort(key=lambda x: (len(str(x[0])), str(x[0])))
    return valid[0][0]


def prepare_live_runtime_config(
    *,
    main_mod,
    original_config: Path,
    release_path: Path,
    out_path: Path,
    delta_path: Path,
    stale_sec: float,
    deadline_sec: float,
) -> dict[str, Any]:
    require(0 < stale_sec <= 30.0, "--live-stale-sec must be in (0, 30]")
    require(0 <= deadline_sec <= stale_sec, "--live-deadline-sec must be <= stale budget")

    # Use the frozen main.py's own config resolver so path semantics cannot drift.
    resolved = main_mod.load_config(original_config)
    before_runtime = copy.deepcopy(resolved["runtime"])

    resolved["quality_layer"]["release"] = str(release_path.resolve())
    resolved["runtime"]["stale_after_sec"] = float(stale_sec)
    resolved["runtime"]["deadline_sec"] = float(deadline_sec)

    # Internal resolver fields are not part of the JSON schema.
    resolved.pop("_base", None)
    resolved.pop("_config_path", None)

    # Record exactly what was intentionally changed.
    delta = {
        "schema": "eav.live_interaction.runtime_config_delta.v1",
        "created_utc": utc_now(),
        "original_config": str(original_config.resolve()),
        "original_config_sha256": sha256_file(original_config),
        "session_runtime_config": str(out_path.resolve()),
        "intentional_semantic_changes": {
            "runtime.stale_after_sec": {
                "before": before_runtime.get("stale_after_sec"),
                "after": float(stale_sec),
                "reason": "LIVE_INFERENCE_LATENCY_BUDGET_ONLY",
            },
            "runtime.deadline_sec": {
                "before": before_runtime.get("deadline_sec"),
                "after": float(deadline_sec),
                "reason": "LIVE_INFERENCE_LATENCY_BUDGET_ONLY",
            },
        },
        "quality_formula_changed": False,
        "quality_thresholds_changed": False,
        "fusion_weights_changed": False,
        "model_paths_changed_by_design": False,
        "main_source_modified": False,
        "original_config_overwritten": False,
        "note": (
            "Path strings are normalized to absolute paths using frozen main.py "
            "load_config(); the two runtime timing values above are the only "
            "intentional semantic live-operation changes."
        ),
    }

    atomic_write_json(out_path, resolved)
    delta["session_runtime_config_sha256"] = sha256_file(out_path)
    atomic_write_json(delta_path, delta)
    return resolved


class MainConsolePump:
    def __init__(
        self,
        process: subprocess.Popen,
        log_path: Path,
        *,
        verbose: bool,
    ):
        self.process = process
        self.log_path = log_path
        self.verbose = verbose
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.tail = collections.deque(maxlen=120)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.log_path.open("w", encoding="utf-8", newline="\n") as log:
                stream = self.process.stdout
                if stream is None:
                    return
                for raw in stream:
                    line = raw.rstrip("\r\n")
                    self.tail.append(line)
                    log.write(line + "\n")
                    log.flush()

                    if "[SHADOW INPUT]" in line:
                        self.ready.set()

                    if (
                        self.verbose
                        or line.startswith("[LOAD]")
                        or "[SHADOW INPUT]" in line
                        or "ERROR" in line.upper()
                        or "FAILED" in line.upper()
                    ):
                        print(f"[MAIN] {line}", flush=True)
        finally:
            self.finished.set()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self.ready.is_set():
            rc = self.process.poll()
            if rc is not None:
                tail = "\n".join(self.tail)
                raise LiveInferenceError(
                    f"Frozen main.py exited before live runtime became ready "
                    f"(return code {rc}).\nLast console lines:\n{tail}"
                )
            if time.monotonic() >= deadline:
                raise LiveInferenceError(
                    "Timed out waiting for frozen main.py to finish model loading "
                    "and enter live mode."
                )
            time.sleep(0.05)


class ResultMonitor:
    """Tail frozen main.py window_results.jsonl and expose latest decision."""

    def __init__(self, main_output: Path):
        self.main_output = main_output
        self.path = main_output / "window_results.jsonl"
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.results: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, Any] | None = None
        self.position = 0

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        partial = ""
        while not self.stop_event.is_set():
            try:
                if self.path.is_file():
                    with self.path.open("r", encoding="utf-8") as f:
                        f.seek(self.position)
                        chunk = f.read()
                        self.position = f.tell()
                    if chunk:
                        text = partial + chunk
                        lines = text.split("\n")
                        partial = lines.pop()
                        for line in lines:
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            self._accept(row)
            except Exception:
                pass
            time.sleep(0.05)

        # One last pass after stop.
        try:
            if self.path.is_file():
                with self.path.open("r", encoding="utf-8") as f:
                    f.seek(self.position)
                    chunk = f.read()
                text = partial + chunk
                for line in text.splitlines():
                    if line.strip():
                        try:
                            self._accept(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass

    def _accept(self, row: dict[str, Any]) -> None:
        wid = str(row.get("window_id", ""))
        if not wid:
            return

        fusion = row.get("fusion") or {}
        final = fusion.get("final") or {}
        record = {
            "window_id": wid,
            "status": row.get("status"),
            "active_branch": fusion.get("active_branch"),
            "emotion": final.get("emotion"),
            "confidence": final.get("confidence"),
            "probabilities": final.get("probabilities"),
            "availability": fusion.get("availability"),
            "quality_for_fusion": fusion.get("quality_for_fusion"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "elapsed_since_capture_end_seconds": row.get(
                "elapsed_since_capture_end_seconds"
            ),
            "raw": row,
        }
        with self.lock:
            if wid in self.results:
                return
            self.results[wid] = record
            self.latest = record

        conf = record["confidence"]
        conf_text = f"{100.0 * float(conf):.1f}%" if conf is not None else "N/A"
        print(
            f"[RESULT] {wid} | {record['status']} | "
            f"{record['active_branch']} | {record['emotion']} | "
            f"confidence={conf_text} | "
            f"latency_from_T1={record['elapsed_since_capture_end_seconds']}",
            flush=True,
        )

    def snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            return copy.deepcopy(self.latest)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def draw_live_overlay(
    cv2,
    frame,
    *,
    session_id: str,
    window_text: str,
    phase: str,
    remaining: float | None,
    latest_result: dict[str, Any] | None,
):
    canvas = frame.copy()
    lines = [
        "EAV LiveInteraction - Stage 03",
        f"Session: {session_id}",
        f"Capture: {window_text} | {phase}",
    ]
    if remaining is not None:
        lines.append(f"Remaining: {max(0.0, remaining):.1f} s")

    if latest_result is None:
        lines.extend(
            [
                "Emotion: waiting for first inference...",
                "EEG: UNAVAILABLE | route expected: AF4-B",
            ]
        )
    else:
        emotion = latest_result.get("emotion") or "NO DECISION"
        conf = latest_result.get("confidence")
        conf_text = f"{100.0 * float(conf):.1f}%" if conf is not None else "N/A"
        lines.extend(
            [
                f"Latest emotion: {emotion} | confidence: {conf_text}",
                f"Route: {latest_result.get('active_branch')} | status: {latest_result.get('status')}",
                "EEG: UNAVAILABLE",
            ]
        )

    lines.append("Q / ESC: stop live capture")

    y = 28
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return canvas


def preview_wait(
    cv2,
    video,
    result_monitor: ResultMonitor,
    *,
    session_id: str,
    window_text: str,
    phase: str,
    target_time: float,
    preview: bool,
) -> bool:
    """Wait to target_time while keeping live preview responsive."""
    window_name = "EAV LiveInteraction - Stage 03 live inference"

    while time.monotonic() < target_time:
        if video.thread_error:
            raise LiveInferenceError(
                f"Camera acquisition thread failed: {video.thread_error}"
            )

        if preview:
            frame = video.get_latest()
            if frame is not None:
                remaining = target_time - time.monotonic()
                canvas = draw_live_overlay(
                    cv2,
                    frame,
                    session_id=session_id,
                    window_text=window_text,
                    phase=phase,
                    remaining=remaining,
                    latest_result=result_monitor.snapshot(),
                )
                try:
                    cv2.imshow(window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        return True
                except Exception:
                    preview = False
        time.sleep(0.003)

    return False


def find_main_version(python_exe: str, main_path: Path, deployment_root: Path) -> str:
    proc = subprocess.run(
        [python_exe, str(main_path), "--version"],
        cwd=str(deployment_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    require(proc.returncode == 0, f"Could not query main.py version:\n{proc.stdout}")
    return proc.stdout.strip()


def wait_for_main_exit(
    process: subprocess.Popen,
    *,
    timeout: float,
) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rc = process.poll()
        if rc is not None:
            return rc
        time.sleep(0.05)
    return None


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Real-time laptop Audio+Video capture and emotion inference through "
            "the existing frozen main.py live runtime."
        )
    )
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_WINDOWS,
        help=(
            f"Number of consecutive 5-second windows (default: {DEFAULT_WINDOWS}). "
            "Use 0 for continuous capture until Q/ESC/Ctrl+C."
        ),
    )
    p.add_argument(
        "--device-config",
        default=None,
        help="Stage-01 detected_av_devices.json.",
    )
    p.add_argument(
        "--system-config",
        default=None,
        help="Frozen deployment system_config.json.",
    )
    p.add_argument(
        "--release",
        default=None,
        help="Frozen quality_layer_release_candidate.json. Auto-discovered if omitted.",
    )
    p.add_argument(
        "--main",
        default=None,
        help="Frozen deployment main.py. Default: ../main.py.",
    )
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Neural inference device passed to frozen main.py (default: cuda).",
    )
    p.add_argument(
        "--fusion-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Fusion device passed to frozen main.py (default: cpu).",
    )
    p.add_argument(
        "--live-stale-sec",
        type=float,
        default=DEFAULT_LIVE_STALE_SEC,
        help=f"Session-local live freshness budget (default: {DEFAULT_LIVE_STALE_SEC}).",
    )
    p.add_argument(
        "--live-deadline-sec",
        type=float,
        default=DEFAULT_LIVE_DEADLINE_SEC,
        help=f"Session-local live completion deadline (default: {DEFAULT_LIVE_DEADLINE_SEC}).",
    )
    p.add_argument(
        "--lead-sec",
        type=float,
        default=DEFAULT_LEAD_SEC,
        help=f"Lead before first synchronized T0 (default: {DEFAULT_LEAD_SEC}).",
    )
    p.add_argument(
        "--settle-sec",
        type=float,
        default=DEFAULT_SETTLE_SEC,
        help=f"Post-T1 callback settle time (default: {DEFAULT_SETTLE_SEC}).",
    )
    p.add_argument(
        "--main-ready-timeout",
        type=float,
        default=DEFAULT_MAIN_READY_TIMEOUT_SEC,
        help="Maximum wait for frozen main.py model loading and live readiness.",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Explicit new Stage-03 session directory.",
    )
    p.add_argument("--session-id", default=None)
    p.add_argument("--clock-id", default=None)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument(
        "--verbose-main",
        action="store_true",
        help="Echo every frozen main.py console line instead of only important lines.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require(0 <= args.windows <= 10000, "--windows must be 0..10000")
    require(0.10 <= args.lead_sec <= 5.0, "--lead-sec must be in [0.10, 5.0]")
    require(0.02 <= args.settle_sec <= 1.0, "--settle-sec must be in [0.02, 1.0]")
    require(10 <= args.main_ready_timeout <= 3600, "--main-ready-timeout must be 10..3600 seconds")
    require(0 < args.live_stale_sec <= 30, "--live-stale-sec must be in (0, 30]")
    require(
        0 <= args.live_deadline_sec <= args.live_stale_sec,
        "--live-deadline-sec must be <= --live-stale-sec",
    )

    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent

    stage02_path = script_dir / "02_capture_synchronized_av_window.py"
    stage02 = import_module_from_path(stage02_path, "eav_live_stage02")
    require(
        str(getattr(stage02, "VERSION", "")).startswith(EXPECTED_STAGE02_VERSION_PREFIX),
        "Unexpected/missing Stage-02 capture module version",
    )
    cv2, sd = stage02.load_dependencies()

    main_path = (
        Path(args.main).expanduser().resolve()
        if args.main
        else deployment_root / "main.py"
    )
    system_config = (
        Path(args.system_config).expanduser().resolve()
        if args.system_config
        else deployment_root / "system_config.json"
    )
    device_config = (
        Path(args.device_config).expanduser().resolve()
        if args.device_config
        else script_dir / "config" / "detected_av_devices.json"
    )

    require(main_path.is_file(), f"Frozen main.py not found: {main_path}")
    require(system_config.is_file(), f"system_config.json not found: {system_config}")
    require(device_config.is_file(), f"Stage-01 device config not found: {device_config}")

    # Import the frozen main only for its own config resolver and version constant.
    main_mod = import_module_from_path(main_path, "eav_frozen_main_for_live_config")
    main_version = str(getattr(main_mod, "VERSION", ""))
    require(
        main_version == EXPECTED_MAIN_VERSION,
        f"Wrong frozen main.py version: {main_version!r}; expected {EXPECTED_MAIN_VERSION!r}",
    )
    require(hasattr(main_mod, "FileInboxSource") and hasattr(main_mod, "LiveScheduler"),
            "Frozen main.py does not expose the reviewed live inbox/scheduler path")

    queried_version = find_main_version(sys.executable, main_path, deployment_root)
    require(
        queried_version == EXPECTED_MAIN_VERSION,
        f"Subprocess main.py version mismatch: {queried_version!r}",
    )

    device_cfg = read_json(device_config)
    require(
        device_cfg.get("schema") == "eav.live_interaction.detected_av_devices.v1",
        "Unexpected Stage-01 device config schema",
    )
    require(
        device_cfg.get("probe_status") in ("PASS", "PASS_WITH_WARNINGS"),
        "Stage-01 AV device probe is not in a passing state",
    )

    # Resolve the original deployment config using frozen main.py itself, then
    # discover the accepted release if it is not already configured.
    resolved_original = main_mod.load_config(system_config)
    release_path = discover_release(
        deployment_root,
        resolved_original.get("quality_layer", {}).get("release"),
        args.release,
    )

    session_id = args.session_id or f"live_av_{tag_now()}"
    require(session_id.strip() == session_id and session_id, "Invalid session-id")
    clock_id = args.clock_id or f"python_monotonic_{session_id}"
    require(clock_id.strip() == clock_id and clock_id, "Invalid clock-id")

    session_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else script_dir / "runs" / f"live_inference_{tag_now()}"
    )
    require(not session_root.exists(), f"Output already exists: {session_root}")
    session_root.mkdir(parents=True, exist_ok=False)

    capture_root = session_root / "captures"
    inbox_dir = session_root / "inbox"
    main_output = session_root / "main_runtime"
    runtime_config = session_root / "runtime_config_live.json"
    runtime_delta = session_root / "runtime_config_delta.json"
    main_console_log = session_root / "main_console.log"
    capture_root.mkdir()
    inbox_dir.mkdir()

    effective_config = prepare_live_runtime_config(
        main_mod=main_mod,
        original_config=system_config,
        release_path=release_path,
        out_path=runtime_config,
        delta_path=runtime_delta,
        stale_sec=args.live_stale_sec,
        deadline_sec=args.live_deadline_sec,
    )

    print("=" * 104)
    print("EAV LiveInteraction — Stage 03 real-time AV emotion inference")
    print("=" * 104)
    print(f"Version                  : {VERSION}")
    print(f"Frozen main              : {main_version}")
    print(f"Frozen main SHA256       : {sha256_file(main_path)}")
    print(f"Python                   : {sys.version.split()[0]}")
    print(f"Platform                 : {platform.platform()}")
    print(f"Session ID               : {session_id}")
    print(f"Clock ID                 : {clock_id}")
    print(
        f"Windows                  : "
        f"{'continuous until stop' if args.windows == 0 else args.windows}"
    )
    print(f"Window seconds           : {stage02.WINDOW_SECONDS:.1f}")
    print(f"EEG                      : UNAVAILABLE")
    print(f"Camera + microphone      : REAL LOCAL HARDWARE")
    print(f"Emotion inference        : FROZEN main.py")
    print(f"Quality/Fusion           : KEEP_V1 / F4-AF4B frozen core")
    print(f"Expected live route      : AF4-B because EEG is unavailable")
    print(f"Live stale budget        : {args.live_stale_sec:.2f} s")
    print(f"Live deadline            : {args.live_deadline_sec:.2f} s")
    print(f"Robot action             : DISABLED")
    print(f"Session output           : {session_root}")
    print("=" * 104)

    # Start frozen main.py before hardware capture so model cold-loading does not
    # consume/age the first real source window.
    cmd = [
        sys.executable,
        str(main_path),
        "--mode", "live",
        "--config", str(runtime_config),
        "--release", str(release_path),
        "--inbox", str(inbox_dir),
        "--session-id", session_id,
        "--clock-id", clock_id,
        "--device", args.device,
        "--fusion-device", args.fusion_device,
        "--eeg-unit", EEG_UNIT,
        "--unit-evidence", EEG_UNIT_EVIDENCE,
        "--output", str(main_output),
    ]

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        cwd=str(deployment_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    console = MainConsolePump(
        process,
        main_console_log,
        verbose=args.verbose_main,
    )
    console.start()

    result_monitor = ResultMonitor(main_output)
    result_monitor.start()

    cap = None
    video = None
    audio = None
    capture_records: list[dict[str, Any]] = []
    fatal_error: str | None = None
    user_stopped = False
    descriptors_published = 0
    audio_startup_seconds = None
    camera_info = None
    started_utc = utc_now()

    try:
        print("\nLoading frozen model runtime before starting live capture...", flush=True)
        console.wait_ready(args.main_ready_timeout)
        print("Frozen main.py is ready. Starting real camera + microphone capture.\n", flush=True)

        camera_cfg = device_cfg["camera"]
        audio_cfg = device_cfg["audio"]
        capture_sr = int(round(float(audio_cfg["capture_samplerate"])))
        audio_device_index = int(audio_cfg["device_index"])

        cap, camera_info = stage02.open_camera(cv2, camera_cfg)
        retention = max(
            DEFAULT_RETENTION_SEC,
            stage02.WINDOW_SECONDS + args.settle_sec + 3.0,
        )
        video = stage02.VideoCollector(
            cap,
            retention_seconds=retention,
        )
        audio = stage02.AudioCollector(
            sd,
            device_index=audio_device_index,
            samplerate=capture_sr,
            retention_seconds=retention,
        )

        video.start()
        audio_startup_seconds = audio.start()

        require(
            video.first_frame_event.wait(timeout=5.0),
            "No camera frame arrived after live collector start",
        )
        require(
            audio.first_block_event.wait(timeout=5.0),
            "No microphone block arrived after live collector start",
        )

        print(
            f"Camera                  : index {camera_info['index']} / "
            f"{camera_info['backend_actual']} / "
            f"{camera_info['reported_width']}x{camera_info['reported_height']}"
        )
        print(
            f"Microphone              : device {audio_device_index} / "
            f"{audio_cfg.get('device_name')}"
        )
        print(f"Audio capture           : {capture_sr} Hz mono")
        print(f"Audio stream startup    : {audio_startup_seconds:.3f} s")
        print()
        print("LIVE CAPTURE + INFERENCE STARTED")
        if not args.no_preview:
            print("Press Q or ESC in the preview window to stop.")
        print()

        first_t0 = time.monotonic() + args.lead_sec
        idx = 0

        while args.windows == 0 or idx < args.windows:
            idx += 1
            window_id = f"window_{idx:06d}"
            t0 = first_t0 + (idx - 1) * stage02.WINDOW_SECONDS
            t1 = t0 + stage02.WINDOW_SECONDS

            # If local file encoding ever takes longer than an entire subsequent
            # 5-second interval, we refuse to relabel time or skip silently.
            if time.monotonic() > t1:
                raise LiveInferenceError(
                    f"Capture publisher fell behind real time before {window_id}; "
                    "no timestamp relabelling is allowed."
                )

            user_stopped = preview_wait(
                cv2,
                video,
                result_monitor,
                session_id=session_id,
                window_text=window_id,
                phase="WAITING FOR T0",
                target_time=t0,
                preview=not args.no_preview,
            )
            if user_stopped:
                break

            start_overflows = audio.input_overflows
            start_callback_errors = audio.callback_errors

            print(
                f"[CAPTURE] {window_id} | "
                f"T0={t0:.6f} T1={t1:.6f}",
                flush=True,
            )

            user_stopped = preview_wait(
                cv2,
                video,
                result_monitor,
                session_id=session_id,
                window_text=window_id,
                phase="CAPTURING 5s",
                target_time=t1,
                preview=not args.no_preview,
            )
            if user_stopped:
                print(
                    f"[CAPTURE] stop requested; incomplete {window_id} is NOT published.",
                    flush=True,
                )
                break

            # Final device callbacks can arrive just after T1. Selection below
            # remains strictly tied to [T0, T1].
            settle_deadline = t1 + args.settle_sec
            while time.monotonic() < settle_deadline or not audio.has_until(t1):
                if time.monotonic() > t1 + 1.5:
                    raise LiveInferenceError(
                        f"Audio did not become available through T1 for {window_id}"
                    )
                time.sleep(0.005)

            frames = video.window(t0, t1)
            video_stats = stage02.frame_statistics(frames, t0, t1)
            audio_raw, audio_stats = audio.extract_exact(
                t0, stage02.WINDOW_SECONDS
            )
            audio_model = stage02.resample_to_16k(audio_raw, capture_sr)

            audio_overflows_delta = audio.input_overflows - start_overflows
            audio_callback_errors_delta = (
                audio.callback_errors - start_callback_errors
            )

            capture_status, warnings, errors = stage02.window_acceptance(
                video_stats,
                audio_stats,
                audio_overflows_delta=audio_overflows_delta,
                audio_callback_errors_delta=audio_callback_errors_delta,
            )

            record: dict[str, Any] = {
                "window_id": window_id,
                "capture_status": capture_status,
                "window_start_monotonic": t0,
                "window_end_monotonic": t1,
                "video_frames": video_stats["frames"],
                "video_measured_fps": video_stats["measured_fps_within_window"],
                "video_max_gap_ms": video_stats["max_interval_ms"],
                "audio_samples_capture": audio_stats["samples"],
                "audio_samples_model": int(audio_model.size),
                "audio_rms_dbfs": audio_stats["rms_dbfs"],
                "warnings": warnings,
                "errors": errors,
                "descriptor_published": False,
            }

            window_dir = capture_root / window_id
            window_dir.mkdir(parents=True, exist_ok=False)

            capture_metadata = {
                "schema": stage02.METADATA_SCHEMA,
                "version": VERSION,
                "stage02_capture_core_version": stage02.VERSION,
                "status": capture_status,
                "session_id": session_id,
                "clock_id": clock_id,
                "window_id": window_id,
                "window_seconds": stage02.WINDOW_SECONDS,
                "window_start_monotonic": t0,
                "window_end_monotonic": t1,
                "camera": {
                    **camera_info,
                    **video_stats,
                    "timestamps_monotonic": [x.timestamp for x in frames],
                },
                "audio": {
                    **audio_stats,
                    "capture_sample_rate": capture_sr,
                    "model_sample_rate": stage02.MODEL_AUDIO_SR,
                    "model_samples": int(audio_model.size),
                    "resampled": capture_sr != stage02.MODEL_AUDIO_SR,
                    "input_overflows_in_window": audio_overflows_delta,
                    "callback_errors_in_window": audio_callback_errors_delta,
                },
                "warnings": warnings,
                "errors": errors,
                "emotion_inference_performed_by_capture_code": False,
                "frozen_main_is_only_inference_runtime": True,
            }

            if capture_status == "FAIL":
                stage02.atomic_write_json(
                    window_dir / "capture_metadata.json",
                    capture_metadata,
                )
                capture_records.append(record)
                print(
                    f"[CAPTURE] {window_id} FAILED integrity checks; "
                    f"descriptor withheld: {'; '.join(errors)}",
                    flush=True,
                )
                continue

            require(
                len(audio_model)
                == int(stage02.MODEL_AUDIO_SR * stage02.WINDOW_SECONDS),
                "Live model audio is not exactly 80000 samples",
            )

            audio_path = window_dir / "audio.wav"
            stage02.save_pcm16_wav(
                audio_path,
                audio_model,
                stage02.MODEL_AUDIO_SR,
            )
            wav_info = stage02.read_wav_header(audio_path)
            require(
                wav_info["channels"] == 1
                and wav_info["sample_rate"] == stage02.MODEL_AUDIO_SR
                and wav_info["frames"]
                == int(stage02.MODEL_AUDIO_SR * stage02.WINDOW_SECONDS),
                "Saved live WAV violates the frozen Audio input contract",
            )

            video_path, encoded_video = stage02.write_video_exact_5s(
                cv2,
                window_dir,
                frames,
            )

            descriptor = stage02.descriptor_for_window(
                session_id=session_id,
                clock_id=clock_id,
                window_id=window_id,
                window_start=t0,
                window_end=t1,
                audio_path=audio_path,
                video_path=video_path,
                audio_newest=t1,
                video_newest=float(video_stats["last_frame_monotonic"]),
                audio_capture_rate=capture_sr,
            )
            stage02.validate_main_live_descriptor_shape(descriptor)

            capture_metadata["saved_audio"] = {
                "file": str(audio_path.resolve()),
                "bytes": audio_path.stat().st_size,
                **wav_info,
            }
            capture_metadata["saved_video"] = encoded_video
            capture_metadata["descriptor_main_live_contract_pass"] = True

            stage02.atomic_write_json(
                window_dir / "capture_metadata.json",
                capture_metadata,
            )
            stage02.atomic_write_json(
                window_dir / "descriptor.json",
                descriptor,
            )

            # Descriptor is committed to the inbox LAST, after immutable media
            # and local audit metadata exist.
            inbox_descriptor = inbox_dir / f"{window_id}.json"
            stage02.atomic_write_json(inbox_descriptor, descriptor)
            descriptors_published += 1
            record["descriptor_published"] = True
            record["descriptor_path"] = str(
                (window_dir / "descriptor.json").resolve()
            )
            record["inbox_descriptor"] = str(inbox_descriptor.resolve())
            record["audio_file"] = str(audio_path.resolve())
            record["video_file"] = str(video_path.resolve())
            capture_records.append(record)

            print(
                f"[PUBLISH] {window_id} -> frozen main.py | "
                f"video={video_stats['frames']} frames "
                f"({video_stats['measured_fps_within_window']:.2f} FPS) | "
                f"audio={len(audio_model)} samples | "
                f"RMS={audio_stats['rms_dbfs']:.2f} dBFS",
                flush=True,
            )

            # Do NOT wait for inference here. Camera and microphone remain
            # continuous; the frozen main scheduler processes the just-published
            # window while we continue toward the next real 5-second boundary.

        # Explicit producer EOF. main.py drains all pending windows before exit.
        stage02.atomic_write_json(
            inbox_dir / "stop.json",
            {
                "session_id": session_id,
                "stop": True,
                "created_utc": utc_now(),
                "reason": "USER_STOP" if user_stopped else "PLANNED_CAPTURE_COMPLETE",
            },
        )

        print("\nCapture producer stopped. Draining pending frozen-main inference...", flush=True)

        # Deadline is bounded by the configured per-window budget plus shutdown.
        drain_timeout = max(
            30.0,
            args.live_stale_sec
            + effective_config["runtime"]["shutdown_grace_sec"]
            + 10.0,
        )
        rc = wait_for_main_exit(process, timeout=drain_timeout)
        if rc is None:
            raise LiveInferenceError(
                "Frozen main.py did not terminate after producer EOF and drain timeout."
            )

    except KeyboardInterrupt:
        user_stopped = True
        print("\nKeyboardInterrupt received: stopping capture and draining published windows.", flush=True)
        try:
            stage02.atomic_write_json(
                inbox_dir / "stop.json",
                {
                    "session_id": session_id,
                    "stop": True,
                    "created_utc": utc_now(),
                    "reason": "KEYBOARD_INTERRUPT",
                },
            )
        except Exception:
            pass

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        print(f"\nLIVE INFERENCE ERROR: {fatal_error}", file=sys.stderr, flush=True)
        try:
            stage02.atomic_write_json(
                inbox_dir / "stop.json",
                {
                    "session_id": session_id,
                    "stop": True,
                    "created_utc": utc_now(),
                    "reason": "STAGE03_FATAL_ERROR",
                },
            )
        except Exception:
            pass

    finally:
        if audio is not None:
            audio.stop()
        if video is not None:
            video.stop()
        elif cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Give main a short opportunity to consume stop.json if we reached
        # cleanup through an exception. Then never leave an orphan model runtime.
        if process.poll() is None:
            rc = wait_for_main_exit(
                process,
                timeout=max(5.0, min(args.live_stale_sec + 5.0, 35.0)),
            )
            if rc is None:
                terminate_process(process)

        result_monitor.stop()
        try:
            console.thread.join(timeout=2.0)
        except Exception:
            pass

    main_returncode = process.poll()
    main_summary_path = main_output / "run_summary.json"
    main_summary = read_json(main_summary_path) if main_summary_path.is_file() else None

    results = list(result_monitor.results.values())
    results.sort(key=lambda r: r["window_id"])
    ok_results = [r for r in results if r.get("status") == "OK"]
    no_decisions = [r for r in results if r.get("status") == "NO_DECISION"]
    error_results = [
        r for r in results if r.get("status") not in ("OK", "NO_DECISION")
    ]

    branch_counts = collections.Counter(
        r.get("active_branch") or r.get("status") for r in results
    )
    emotion_counts = collections.Counter(
        r.get("emotion") for r in ok_results if r.get("emotion")
    )

    main_status = main_summary.get("status") if main_summary else None
    main_assets_unchanged = (
        bool(main_summary.get("assets_verified_after_run"))
        and main_summary.get("quality_modules_modified") is False
        and main_summary.get("weights_modified") is False
        if main_summary
        else False
    )

    if fatal_error is not None:
        final_status = "FAIL"
    elif descriptors_published == 0:
        final_status = "STOPPED_NO_PUBLISHED_WINDOW"
    elif main_returncode != 0:
        final_status = "FAIL_MAIN_RUNTIME"
    elif error_results:
        final_status = "COMPLETE_WITH_INFERENCE_ERRORS"
    elif len(results) < descriptors_published:
        final_status = "INCOMPLETE_RESULTS"
    else:
        final_status = "PASS"

    summary = {
        "schema": SUMMARY_SCHEMA,
        "version": VERSION,
        "status": final_status,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "session_id": session_id,
        "clock_id": clock_id,
        "frozen_main": {
            "path": str(main_path.resolve()),
            "version": main_version,
            "sha256": sha256_file(main_path),
            "returncode": main_returncode,
            "run_summary": (
                str(main_summary_path.resolve())
                if main_summary_path.is_file()
                else None
            ),
            "reported_status": main_status,
            "assets_unchanged_verified": main_assets_unchanged,
        },
        "configuration": {
            "original_system_config": str(system_config.resolve()),
            "original_system_config_sha256": sha256_file(system_config),
            "session_runtime_config": str(runtime_config.resolve()),
            "runtime_config_delta": str(runtime_delta.resolve()),
            "release": str(release_path.resolve()),
            "live_stale_sec": args.live_stale_sec,
            "live_deadline_sec": args.live_deadline_sec,
            "main_source_modified": False,
            "quality_formula_changed": False,
            "quality_thresholds_changed": False,
            "fusion_weights_changed": False,
        },
        "capture": {
            "device_config": str(device_config.resolve()),
            "planned_windows": (
                None if args.windows == 0 else args.windows
            ),
            "attempted_windows": len(capture_records),
            "descriptors_published": descriptors_published,
            "user_stopped": user_stopped,
            "audio_stream_startup_seconds": audio_startup_seconds,
            "camera": camera_info,
            "records": capture_records,
        },
        "inference": {
            "results_received": len(results),
            "ok_results": len(ok_results),
            "no_decision_results": len(no_decisions),
            "error_results": len(error_results),
            "branch_counts": dict(branch_counts),
            "emotion_counts": dict(emotion_counts),
            "results": [
                {k: v for k, v in r.items() if k != "raw"}
                for r in results
            ],
        },
        "eeg": {
            "present": False,
            "reason": "EEG_DEVICE_NOT_CONNECTED",
        },
        "robot_action_performed": False,
        "fatal_error": fatal_error,
        "next_stage": (
            "REALTIME_AV_INTERACTION_VALIDATED"
            if final_status == "PASS"
            else "REVIEW_STAGE03_LOGS_BEFORE_NEXT_STAGE"
        ),
    }
    atomic_write_json(session_root / "session_summary.json", summary)

    print()
    print("=" * 104)
    print("STAGE 03 LIVE AV INFERENCE SUMMARY")
    print("=" * 104)
    print(f"Status                  : {final_status}")
    print(f"Published AV windows    : {descriptors_published}")
    print(f"Inference results       : {len(results)}")
    print(f"OK results              : {len(ok_results)}")
    print(f"NO_DECISION             : {len(no_decisions)}")
    print(f"Inference errors        : {len(error_results)}")
    print(f"Routes                  : {dict(branch_counts)}")
    print(f"Predicted emotions      : {dict(emotion_counts)}")
    print(f"EEG                     : UNAVAILABLE")
    print(f"Frozen main status      : {main_status}")
    print(f"Frozen assets unchanged : {main_assets_unchanged}")
    print(f"Main console log        : {main_console_log}")
    print(f"Session summary         : {session_root / 'session_summary.json'}")
    if fatal_error:
        print(f"Fatal error             : {fatal_error}")
    print("=" * 104)

    return 0 if final_status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except LiveInferenceError as exc:
        print(f"\nLIVE INFERENCE ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
