#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
04_realtime_emotion_interface.py
================================

EAV LiveInteraction — Stage 04
Integrated real-time participant-facing emotion interaction interface.

REAL execution path
-------------------
Laptop camera + microphone
        |
        | continuous acquisition
        | same Python time.monotonic() clock
        v
strict synchronized 5-second Audio + Video windows
        |
        | EEG explicitly unavailable
        v
frozen ../main.py Runtime + LiveScheduler
        |
        | result row DIRECTLY IN MEMORY
        v
EmotionStateManager
        |
        | subscriber callback DIRECTLY IN MEMORY
        v
LiveEmotionDisplay
        |
        v
participant sees current emotion immediately on screen

Important
---------
Prediction-result JSON files are NOT used as an interaction transport in this
script.  Audit files are written as a side effect only.  The user-facing path
is entirely in-process memory after the model result exists.

The frozen recognition system is not modified:
    - no model retraining;
    - no probability recalibration;
    - no quality formula changes;
    - no quality-threshold changes;
    - no F4/AF4-B weight changes;
    - no legacy router;
    - no fake EEG.

EEG is represented honestly as unavailable.  Therefore a valid Audio+Video
window is expected to use the existing robust AF4-B route.

Operational timing
------------------
The frozen main.py live scheduler's original sub-second timing values were
engineering placeholders.  This interface changes ONLY the in-memory live
stale/deadline budget for this session (defaults: 12s / 10s), just as Stage 03
did.  The deployment system_config.json file itself is never overwritten.

Default first run
-----------------
    python .\LiveInteraction\04_realtime_emotion_interface.py

This captures 3 consecutive windows so the integrated UI can be verified.

Continuous participant-facing operation:
    python .\LiveInteraction\04_realtime_emotion_interface.py --windows 0

Press Q or ESC in the UI to stop.

Audit outputs
-------------
    MERIT-Realtime Model/runs/realtime_interface_<timestamp>/
        captures/                  # model-compatible AV windows
        audit_results.jsonl        # side-channel audit; NOT UI transport
        state_events.jsonl         # state snapshots; NOT UI transport
        runtime_policy.json
        session_summary.json

Dependencies / sibling modules
------------------------------
    02_capture_synchronized_av_window.py
    emotion_state_manager.py
    live_display.py

Deployment root
---------------
    ../main.py
    ../system_config.json
    ../Quality/
    ../Fusion/
    ../audio/
    ../video/
    ../eeg/
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
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


VERSION = "EAV-REALTIME-EMOTION-INTERFACE.1.0"
SUMMARY_SCHEMA = "eav.live_interaction.realtime_interface.v1"

EXPECTED_MAIN_VERSION = "EAV-MAIN-INTEGRATION.2.1.0-FINAL-TEST"
EXPECTED_STAGE02_PREFIX = "EAV-LIVE-SYNC-CAPTURE."
EXPECTED_STATE_MANAGER_PREFIX = "EAV-EMOTION-STATE-MANAGER."
EXPECTED_DISPLAY_PREFIX = "EAV-LIVE-DISPLAY."

DEFAULT_WINDOWS = 3
DEFAULT_LEAD_SEC = 0.50
DEFAULT_SETTLE_SEC = 0.12
DEFAULT_RETENTION_SEC = 20.0

DEFAULT_LIVE_STALE_SEC = 12.0
DEFAULT_LIVE_DEADLINE_SEC = 10.0
DEFAULT_STATE_TIMEOUT_SEC = 12.0
DEFAULT_CONFIRMATIONS_REQUIRED = 2

EEG_UNIT = "uV"
EEG_UNIT_EVIDENCE = (
    "Historical eq1_feature_contract.json: unit=uV; volts_per_source_unit=1e-6; "
    "USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED"
)


class InterfaceError(RuntimeError):
    """Integrated live-interface error."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise InterfaceError(message)


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


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
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
        json_safe(obj),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"

    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        last_error: BaseException | None = None
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
            raise InterfaceError(
                f"Could not atomically publish JSON: {path}"
            ) from last_error
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


class JsonlAudit:
    """Thread-safe append-only audit. Never used as live transport."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._file = path.open("x", encoding="utf-8", newline="\n")
        self.count = 0

    def append(self, obj: Any) -> None:
        payload = json.dumps(
            json_safe(obj),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            self._file.write(payload + "\n")
            self._file.flush()
            self.count += 1

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


def import_module(path: Path, name: str):
    require(path.is_file(), f"Missing module: {path}")
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


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def discover_release(
    deployment_root: Path,
    configured: str | None,
    explicit: str | None,
) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p.is_file(), f"--release not found: {p}")
        return p

    if configured:
        p = Path(configured).expanduser().resolve()
        if p.is_file():
            return p

    for p in (
        deployment_root / "Quality" / "quality_layer_release_candidate.json",
        deployment_root / "quality_layer_release_candidate.json",
    ):
        if p.is_file():
            return p.resolve()

    candidates = list(
        deployment_root.glob(
            "system_checks/**/quality_layer_release_candidate.json"
        )
    )
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
        "No KEEP_V1 quality_layer_release_candidate.json found. "
        "Pass --release explicitly.",
    )
    hashes = {h for _, h in valid if h}
    require(
        len(hashes) <= 1,
        "Multiple non-identical KEEP_V1 release candidates found. "
        "Pass --release explicitly.",
    )
    valid.sort(key=lambda x: (len(str(x[0])), str(x[0])))
    return valid[0][0]


def prepare_runtime_config_in_memory(
    *,
    main_mod,
    system_config: Path,
    release: Path,
    device: str,
    fusion_device: str,
    stale_sec: float,
    deadline_sec: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load the frozen config through frozen main.py and make only documented
    session-local operational changes in memory.
    """
    require(0.0 < stale_sec <= 30.0, "stale_sec must be in (0,30]")
    require(0.0 <= deadline_sec <= stale_sec, "deadline_sec must be <= stale_sec")

    config = main_mod.load_config(system_config)
    original_runtime = copy.deepcopy(config["runtime"])

    config["quality_layer"]["release"] = str(release.resolve())

    # Live scheduling only.
    config["runtime"]["stale_after_sec"] = float(stale_sec)
    config["runtime"]["deadline_sec"] = float(deadline_sec)

    # Required constructor contract even though EEG is absent from every source.
    config["eeg"]["training_unit"] = EEG_UNIT
    config["eeg"]["unit_evidence"] = EEG_UNIT_EVIDENCE

    # Runtime hardware choice only; no model/weight changes.
    for role in (
        "eeg_emotion",
        "audio_emotion",
        "video_emotion",
        "video_quality",
    ):
        config["modules"][role]["kwargs"]["device"] = device
    config["modules"]["fusion"]["kwargs"]["device"] = fusion_device

    policy = {
        "schema": "eav.live_interaction.runtime_policy.v1",
        "created_utc": utc_now(),
        "original_system_config": str(system_config.resolve()),
        "original_system_config_sha256": sha256_file(system_config),
        "release": str(release.resolve()),
        "live_runtime_changes": {
            "runtime.stale_after_sec": {
                "before": original_runtime.get("stale_after_sec"),
                "after": float(stale_sec),
                "purpose": "LIVE_SCHEDULING_BUDGET_ONLY",
            },
            "runtime.deadline_sec": {
                "before": original_runtime.get("deadline_sec"),
                "after": float(deadline_sec),
                "purpose": "LIVE_SCHEDULING_BUDGET_ONLY",
            },
        },
        "runtime_device": device,
        "fusion_device": fusion_device,
        "eeg_runtime_unit_contract": EEG_UNIT,
        "eeg_source_present": False,
        "main_source_modified": False,
        "system_config_overwritten": False,
        "model_weights_modified": False,
        "quality_formula_modified": False,
        "quality_thresholds_modified": False,
        "fusion_weights_modified": False,
        "legacy_router_enabled": False,
        "interaction_result_transport": "IN_PROCESS_MEMORY",
        "files_are_live_transport": False,
    }
    return config, policy


class CaptureViewState:
    """Small thread-safe runtime view shared with the UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self.current_window_id: str | None = None
        self.phase = "INITIALIZING"
        self.t0: float | None = None
        self.t1: float | None = None
        self.last_capture_fps: float | None = None
        self.message: str | None = None

    def set(
        self,
        *,
        window_id: str | None = None,
        phase: str | None = None,
        t0: float | None = None,
        t1: float | None = None,
        capture_fps: float | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if window_id is not None:
                self.current_window_id = window_id
            if phase is not None:
                self.phase = phase
            if t0 is not None:
                self.t0 = t0
            if t1 is not None:
                self.t1 = t1
            if capture_fps is not None:
                self.last_capture_fps = capture_fps
            if message is not None:
                self.message = message

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "window_id": self.current_window_id,
                "phase": self.phase,
                "t0": self.t0,
                "t1": self.t1,
                "capture_fps": self.last_capture_fps,
                "message": self.message,
            }


class SynchronizedWindowProducer:
    """
    Background producer that converts continuous hardware streams into immutable
    synchronized 5-second AV windows.

    It never runs emotion inference and never touches the state manager.
    Completed descriptors are placed into an in-memory queue.
    """

    def __init__(
        self,
        *,
        stage02,
        cv2,
        video,
        audio,
        capture_sr: int,
        capture_root: Path,
        session_id: str,
        clock_id: str,
        windows: int,
        lead_sec: float,
        settle_sec: float,
        stop_event: threading.Event,
        output_queue: queue.Queue,
        view_state: CaptureViewState,
        capture_audit: JsonlAudit,
    ):
        self.stage02 = stage02
        self.cv2 = cv2
        self.video = video
        self.audio = audio
        self.capture_sr = int(capture_sr)
        self.capture_root = capture_root
        self.session_id = session_id
        self.clock_id = clock_id
        self.windows = int(windows)
        self.lead_sec = float(lead_sec)
        self.settle_sec = float(settle_sec)
        self.stop_event = stop_event
        self.output_queue = output_queue
        self.view_state = view_state
        self.capture_audit = capture_audit

        self.thread = threading.Thread(
            target=self._run,
            name="EAV-SynchronizedWindowProducer",
            daemon=True,
        )
        self.done = threading.Event()
        self.error: str | None = None
        self.attempted = 0
        self.published = 0
        self.failed = 0

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def _wait_until(self, target: float) -> bool:
        while time.monotonic() < target:
            if self.stop_event.is_set():
                return False
            if self.video.thread_error:
                raise InterfaceError(
                    f"Camera acquisition failed: {self.video.thread_error}"
                )
            time.sleep(0.003)
        return True

    def _put_event(self, item: dict[str, Any]) -> None:
        # Completed windows and fatal producer events must still reach the main
        # loop even if the operator has just requested shutdown.  Only a window
        # interrupted before T1 is discarded.
        while True:
            try:
                self.output_queue.put(item, timeout=0.10)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        try:
            first_t0 = time.monotonic() + self.lead_sec
            idx = 0

            while not self.stop_event.is_set():
                if self.windows > 0 and idx >= self.windows:
                    break

                idx += 1
                self.attempted += 1
                window_id = f"window_{idx:06d}"
                t0 = first_t0 + (idx - 1) * self.stage02.WINDOW_SECONDS
                t1 = t0 + self.stage02.WINDOW_SECONDS

                self.view_state.set(
                    window_id=window_id,
                    phase="WAITING FOR WINDOW",
                    t0=t0,
                    t1=t1,
                    message="Continuous camera + microphone active",
                )

                if not self._wait_until(t0):
                    break

                start_overflows = self.audio.input_overflows
                start_callback_errors = self.audio.callback_errors

                self.view_state.set(
                    window_id=window_id,
                    phase="CAPTURING",
                    t0=t0,
                    t1=t1,
                )

                if not self._wait_until(t1):
                    # User stopped before the window completed: never publish a
                    # partial source window.
                    break

                self.view_state.set(
                    window_id=window_id,
                    phase="FINALIZING WINDOW",
                    t0=t0,
                    t1=t1,
                )

                settle_deadline = t1 + self.settle_sec
                while (
                    time.monotonic() < settle_deadline
                    or not self.audio.has_until(t1)
                ):
                    if self.stop_event.is_set():
                        break
                    if time.monotonic() > t1 + 1.5:
                        raise InterfaceError(
                            f"Audio did not become available through T1 "
                            f"for {window_id}"
                        )
                    time.sleep(0.005)

                if self.stop_event.is_set() and not self.audio.has_until(t1):
                    break

                frames = self.video.window(t0, t1)
                video_stats = self.stage02.frame_statistics(
                    frames, t0, t1
                )
                audio_raw, audio_stats = self.audio.extract_exact(
                    t0, self.stage02.WINDOW_SECONDS
                )
                audio_model = self.stage02.resample_to_16k(
                    audio_raw, self.capture_sr
                )

                overflow_delta = (
                    self.audio.input_overflows - start_overflows
                )
                callback_error_delta = (
                    self.audio.callback_errors - start_callback_errors
                )

                status, warnings, errors = self.stage02.window_acceptance(
                    video_stats,
                    audio_stats,
                    audio_overflows_delta=overflow_delta,
                    audio_callback_errors_delta=callback_error_delta,
                )

                window_dir = self.capture_root / window_id
                window_dir.mkdir(parents=True, exist_ok=False)

                compact_capture = {
                    "window_id": window_id,
                    "status": status,
                    "window_start_monotonic": t0,
                    "window_end_monotonic": t1,
                    "video_frames": video_stats["frames"],
                    "video_fps": video_stats["measured_fps_within_window"],
                    "video_max_gap_ms": video_stats["max_interval_ms"],
                    "audio_capture_samples": audio_stats["samples"],
                    "audio_model_samples": int(audio_model.size),
                    "audio_rms_dbfs": audio_stats["rms_dbfs"],
                    "warnings": warnings,
                    "errors": errors,
                }

                self.view_state.set(
                    capture_fps=finite_or_none(
                        video_stats["measured_fps_within_window"]
                    )
                )

                if status == "FAIL":
                    self.failed += 1
                    metadata = {
                        "schema": self.stage02.METADATA_SCHEMA,
                        "version": VERSION,
                        "status": status,
                        "session_id": self.session_id,
                        "clock_id": self.clock_id,
                        **compact_capture,
                        "descriptor_published": False,
                    }
                    atomic_write_json(
                        window_dir / "capture_metadata.json",
                        metadata,
                    )
                    self.capture_audit.append(metadata)
                    self._put_event(
                        {
                            "type": "capture_failure",
                            "window_id": window_id,
                            "errors": errors,
                            "warnings": warnings,
                        }
                    )
                    continue

                expected_audio = int(
                    self.stage02.MODEL_AUDIO_SR
                    * self.stage02.WINDOW_SECONDS
                )
                require(
                    len(audio_model) == expected_audio,
                    f"{window_id}: model audio is not exactly "
                    f"{expected_audio} samples",
                )

                audio_path = window_dir / "audio.wav"
                self.stage02.save_pcm16_wav(
                    audio_path,
                    audio_model,
                    self.stage02.MODEL_AUDIO_SR,
                )
                wav_info = self.stage02.read_wav_header(audio_path)

                video_path, encoded_video = (
                    self.stage02.write_video_exact_5s(
                        self.cv2,
                        window_dir,
                        frames,
                    )
                )

                descriptor = self.stage02.descriptor_for_window(
                    session_id=self.session_id,
                    clock_id=self.clock_id,
                    window_id=window_id,
                    window_start=t0,
                    window_end=t1,
                    audio_path=audio_path,
                    video_path=video_path,
                    audio_newest=t1,
                    video_newest=float(
                        video_stats["last_frame_monotonic"]
                    ),
                    audio_capture_rate=self.capture_sr,
                )
                self.stage02.validate_main_live_descriptor_shape(descriptor)

                metadata = {
                    "schema": self.stage02.METADATA_SCHEMA,
                    "version": VERSION,
                    "stage02_capture_core_version": self.stage02.VERSION,
                    "status": status,
                    "session_id": self.session_id,
                    "clock_id": self.clock_id,
                    **compact_capture,
                    "saved_audio": {
                        "file": str(audio_path.resolve()),
                        "bytes": audio_path.stat().st_size,
                        **wav_info,
                    },
                    "saved_video": encoded_video,
                    "descriptor_main_live_contract_pass": True,
                    "descriptor_published_to_runtime_memory": True,
                    "emotion_result_transport": "IN_PROCESS_MEMORY",
                }

                # Input media and descriptor are retained for audit/model input.
                # Crucially, the descriptor itself is passed through memory to
                # the scheduler; no inbox/result file is polled.
                atomic_write_json(
                    window_dir / "capture_metadata.json",
                    metadata,
                )
                atomic_write_json(
                    window_dir / "descriptor.json",
                    descriptor,
                )
                self.capture_audit.append(metadata)

                self._put_event(
                    {
                        "type": "descriptor",
                        "descriptor": descriptor,
                        "capture": compact_capture,
                    }
                )
                self.published += 1

            self.view_state.set(
                phase="CAPTURE COMPLETE"
                if not self.stop_event.is_set()
                else "STOPPING",
                message="Waiting for remaining inference results",
            )

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._put_event(
                {
                    "type": "producer_error",
                    "error": self.error,
                    "traceback": traceback.format_exc(),
                }
            )
            self.stop_event.set()
        finally:
            self.done.set()


def compact_result(row: Mapping[str, Any]) -> dict[str, Any]:
    fusion = row.get("fusion") if isinstance(row.get("fusion"), Mapping) else {}
    final = fusion.get("final") if isinstance(fusion.get("final"), Mapping) else {}
    return {
        "window_id": row.get("window_id"),
        "status": row.get("status"),
        "active_branch": fusion.get("active_branch"),
        "emotion": final.get("emotion"),
        "confidence": final.get("confidence"),
        "probabilities": final.get("probabilities"),
        "availability": fusion.get("availability"),
        "quality_for_fusion": fusion.get("quality_for_fusion"),
        "elapsed_seconds_fusion": row.get("elapsed_seconds"),
        "elapsed_since_capture_end_seconds": row.get(
            "elapsed_since_capture_end_seconds"
        ),
        "selected_variant": row.get("selected_variant"),
        "mode": row.get("mode"),
        "quality_formula_changed": row.get("quality_formula_changed"),
        "thresholds_changed": row.get("thresholds_changed"),
        "weights_trained": row.get("weights_trained"),
        "legacy_router_executed": row.get("legacy_router_executed"),
        "legacy_q_used_by_candidate": row.get(
            "legacy_q_used_by_candidate"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Integrated real-time camera+microphone emotion interface using "
            "the frozen main Runtime directly in memory."
        )
    )
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_WINDOWS,
        help=(
            f"Number of consecutive 5s windows (default {DEFAULT_WINDOWS}); "
            "use 0 for continuous operation until Q/ESC/Ctrl+C."
        ),
    )
    p.add_argument("--main", default=None)
    p.add_argument("--system-config", default=None)
    p.add_argument("--device-config", default=None)
    p.add_argument("--release", default=None)
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    p.add_argument(
        "--fusion-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    p.add_argument(
        "--live-stale-sec",
        type=float,
        default=DEFAULT_LIVE_STALE_SEC,
    )
    p.add_argument(
        "--live-deadline-sec",
        type=float,
        default=DEFAULT_LIVE_DEADLINE_SEC,
    )
    p.add_argument(
        "--state-timeout-sec",
        type=float,
        default=DEFAULT_STATE_TIMEOUT_SEC,
    )
    p.add_argument(
        "--confirmations-required",
        type=int,
        default=DEFAULT_CONFIRMATIONS_REQUIRED,
    )
    p.add_argument(
        "--lead-sec",
        type=float,
        default=DEFAULT_LEAD_SEC,
    )
    p.add_argument(
        "--settle-sec",
        type=float,
        default=DEFAULT_SETTLE_SEC,
    )
    p.add_argument("--session-id", default=None)
    p.add_argument("--clock-id", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument(
        "--hide-probabilities",
        action="store_true",
        help="Hide the five raw probability bars from the participant UI.",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Validate local module contracts only; do not load models/hardware.",
    )
    return p.parse_args(argv)


def self_test(
    *,
    script_dir: Path,
    deployment_root: Path,
) -> dict[str, Any]:
    paths = {
        "stage02": script_dir / "02_capture_synchronized_av_window.py",
        "state_manager": script_dir / "emotion_state_manager.py",
        "display": script_dir / "live_display.py",
        "main": deployment_root / "main.py",
        "system_config": deployment_root / "system_config.json",
        "device_config": script_dir / "config" / "detected_av_devices.json",
    }

    checks = []
    for name, path in paths.items():
        checks.append(
            {
                "name": f"path_{name}",
                "pass": path.is_file(),
                "path": str(path),
            }
        )

    if all(x["pass"] for x in checks):
        stage02 = import_module(
            paths["stage02"], "_eav_stage04_selftest_stage02"
        )
        state_mod = import_module(
            paths["state_manager"], "_eav_stage04_selftest_state"
        )
        display_mod = import_module(
            paths["display"], "_eav_stage04_selftest_display"
        )
        main_mod = import_module(
            paths["main"], "_eav_stage04_selftest_main"
        )

        checks.extend(
            [
                {
                    "name": "main_version",
                    "pass": getattr(main_mod, "VERSION", None)
                    == EXPECTED_MAIN_VERSION,
                    "value": getattr(main_mod, "VERSION", None),
                },
                {
                    "name": "stage02_version",
                    "pass": str(getattr(stage02, "VERSION", "")).startswith(
                        EXPECTED_STAGE02_PREFIX
                    ),
                    "value": getattr(stage02, "VERSION", None),
                },
                {
                    "name": "state_manager_version",
                    "pass": str(
                        getattr(state_mod, "VERSION", "")
                    ).startswith(EXPECTED_STATE_MANAGER_PREFIX),
                    "value": getattr(state_mod, "VERSION", None),
                },
                {
                    "name": "display_version",
                    "pass": str(
                        getattr(display_mod, "VERSION", "")
                    ).startswith(EXPECTED_DISPLAY_PREFIX),
                    "value": getattr(display_mod, "VERSION", None),
                },
                {
                    "name": "state_manager_internal_self_test",
                    "pass": state_mod.self_test()["status"] == "PASS",
                },
                {
                    "name": "display_internal_self_test",
                    "pass": display_mod.self_test()["status"] == "PASS",
                },
                {
                    "name": "main_exposes_runtime_scheduler",
                    "pass": hasattr(main_mod, "Runtime")
                    and hasattr(main_mod, "LiveScheduler"),
                },
            ]
        )

    passed = sum(bool(x["pass"]) for x in checks)
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "version": VERSION,
        "n_checks": len(checks),
        "n_passed": passed,
        "checks": checks,
        "models_loaded": False,
        "hardware_opened": False,
        "inference_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent

    if args.self_test:
        report = self_test(
            script_dir=script_dir,
            deployment_root=deployment_root,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2

    require(0 <= args.windows <= 10000, "--windows must be 0..10000")
    require(0.10 <= args.lead_sec <= 5.0, "--lead-sec must be 0.10..5.0")
    require(0.02 <= args.settle_sec <= 1.0, "--settle-sec must be 0.02..1.0")
    require(
        1 <= args.confirmations_required <= 10,
        "--confirmations-required must be 1..10",
    )
    require(
        1.0 <= args.state_timeout_sec <= 120.0,
        "--state-timeout-sec must be 1..120",
    )

    stage02_path = (
        script_dir / "02_capture_synchronized_av_window.py"
    )
    state_path = script_dir / "emotion_state_manager.py"
    display_path = script_dir / "live_display.py"

    stage02 = import_module(stage02_path, "_eav_stage04_capture")
    state_mod = import_module(state_path, "_eav_stage04_state")
    display_mod = import_module(display_path, "_eav_stage04_display")

    require(
        str(stage02.VERSION).startswith(EXPECTED_STAGE02_PREFIX),
        f"Unexpected Stage-02 version: {stage02.VERSION}",
    )
    require(
        str(state_mod.VERSION).startswith(EXPECTED_STATE_MANAGER_PREFIX),
        f"Unexpected state-manager version: {state_mod.VERSION}",
    )
    require(
        str(display_mod.VERSION).startswith(EXPECTED_DISPLAY_PREFIX),
        f"Unexpected display version: {display_mod.VERSION}",
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
    device_config_path = (
        Path(args.device_config).expanduser().resolve()
        if args.device_config
        else script_dir / "config" / "detected_av_devices.json"
    )

    require(main_path.is_file(), f"Frozen main.py missing: {main_path}")
    require(system_config.is_file(), f"system_config.json missing: {system_config}")
    require(
        device_config_path.is_file(),
        f"Stage-01 device config missing: {device_config_path}",
    )

    main_mod = import_module(main_path, "_eav_stage04_frozen_main")
    require(
        getattr(main_mod, "VERSION", None) == EXPECTED_MAIN_VERSION,
        f"Wrong main.py version: {getattr(main_mod, 'VERSION', None)!r}",
    )
    require(
        hasattr(main_mod, "Runtime") and hasattr(main_mod, "LiveScheduler"),
        "Frozen main.py does not expose Runtime + LiveScheduler",
    )

    device_cfg = read_json(device_config_path)
    require(
        device_cfg.get("schema")
        == "eav.live_interaction.detected_av_devices.v1",
        "Unexpected Stage-01 device config schema",
    )
    require(
        device_cfg.get("probe_status") in ("PASS", "PASS_WITH_WARNINGS"),
        "Stage-01 device probe has not passed",
    )

    loaded_base = main_mod.load_config(system_config)
    release_path = discover_release(
        deployment_root,
        loaded_base.get("quality_layer", {}).get("release"),
        args.release,
    )

    runtime_config, runtime_policy = prepare_runtime_config_in_memory(
        main_mod=main_mod,
        system_config=system_config,
        release=release_path,
        device=args.device,
        fusion_device=args.fusion_device,
        stale_sec=args.live_stale_sec,
        deadline_sec=args.live_deadline_sec,
    )

    session_id = args.session_id or f"realtime_{tag_now()}"
    clock_id = (
        args.clock_id
        or f"python_monotonic_{session_id}"
    )
    require(session_id.strip() == session_id and session_id, "Invalid session-id")
    require(clock_id.strip() == clock_id and clock_id, "Invalid clock-id")

    session_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else script_dir / "runs" / f"realtime_interface_{tag_now()}"
    )
    require(not session_root.exists(), f"Output already exists: {session_root}")
    session_root.mkdir(parents=True, exist_ok=False)
    capture_root = session_root / "captures"
    capture_root.mkdir()

    runtime_policy.update(
        {
            "interface_version": VERSION,
            "frozen_main": str(main_path.resolve()),
            "frozen_main_version": main_mod.VERSION,
            "frozen_main_sha256": sha256_file(main_path),
            "session_id": session_id,
            "clock_id": clock_id,
            "state_manager": {
                "confirmations_required": args.confirmations_required,
                "state_timeout_sec": args.state_timeout_sec,
            },
        }
    )
    atomic_write_json(
        session_root / "runtime_policy.json",
        runtime_policy,
    )

    result_audit = JsonlAudit(session_root / "audit_results.jsonl")
    state_audit = JsonlAudit(session_root / "state_events.jsonl")
    capture_audit = JsonlAudit(session_root / "capture_events.jsonl")

    print("=" * 108)
    print("EAV LiveInteraction — Stage 04 REAL-TIME EMOTION INTERFACE")
    print("=" * 108)
    print(f"Version                   : {VERSION}")
    print(f"Frozen main               : {main_mod.VERSION}")
    print(f"Frozen main SHA256        : {sha256_file(main_path)}")
    print(f"Session ID                : {session_id}")
    print(f"Clock ID                  : {clock_id}")
    print(
        f"Windows                   : "
        f"{'CONTINUOUS' if args.windows == 0 else args.windows}"
    )
    print(f"Window duration           : {stage02.WINDOW_SECONDS:.1f} s")
    print("Result transport           : IN-PROCESS MEMORY")
    print("JSON result polling        : DISABLED")
    print("EEG                       : UNAVAILABLE")
    print("Audio + Video             : REAL LAPTOP HARDWARE")
    print("Expected fusion route     : AF4-B")
    print(
        f"Stable-state confirmation : "
        f"{args.confirmations_required} consecutive windows"
    )
    print(f"State timeout             : {args.state_timeout_sec:.1f} s")
    print(f"Robot action              : DISABLED")
    print(f"Session output            : {session_root}")
    print("=" * 108)
    print()
    print("Loading the frozen multimodal runtime before opening the live session...")
    print()

    runtime = None
    scheduler = None
    cap = None
    video = None
    audio = None
    producer = None
    display = None
    state_manager = None

    stop_event = threading.Event()
    descriptor_queue: queue.Queue = queue.Queue(maxsize=16)
    view_state = CaptureViewState()

    fatal_error: str | None = None
    started_utc = utc_now()
    result_records: list[dict[str, Any]] = []
    state_records = 0
    scheduler_event_records: list[dict[str, Any]] = []
    submitted_windows = 0
    capture_sr = None
    audio_startup_seconds = None
    camera_info = None
    shutdown_info = None
    assets_after = None
    display_token = None
    audit_token = None

    try:
        # Load all frozen model/runtime assets first.  Real capture does not
        # begin until cold loading is finished.
        runtime = main_mod.Runtime(
            runtime_config,
            fusion_only=False,
            clock_id=clock_id,
            allow_test=False,
        )
        scheduler = main_mod.LiveScheduler(
            runtime,
            session_id=session_id,
            clock_id=clock_id,
        )

        print()
        print("Frozen models loaded. Opening real camera + microphone...")
        print()

        camera_cfg = device_cfg["camera"]
        audio_cfg = device_cfg["audio"]
        capture_sr = int(round(float(audio_cfg["capture_samplerate"])))
        audio_device_index = int(audio_cfg["device_index"])

        cap, camera_info = stage02.open_camera(cv2, camera_cfg)

        retention = max(
            DEFAULT_RETENTION_SEC,
            stage02.WINDOW_SECONDS + args.settle_sec + 5.0,
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
            "No camera frame arrived after capture start",
        )
        require(
            audio.first_block_event.wait(timeout=5.0),
            "No microphone block arrived after capture start",
        )

        state_manager = state_mod.EmotionStateManager(
            confirmations_required=args.confirmations_required,
            state_timeout_sec=args.state_timeout_sec,
        )
        display = display_mod.LiveEmotionDisplay(
            window_name="EAV Real-Time Emotion Interaction",
            fullscreen=args.fullscreen,
            show_raw_probabilities=not args.hide_probabilities,
        )

        # Direct participant-facing in-memory path.
        display_token = state_manager.subscribe(display.on_state)

        def audit_state(snapshot: dict[str, Any]) -> None:
            nonlocal state_records
            state_audit.append(snapshot)
            state_records += 1

        audit_token = state_manager.subscribe(audit_state)

        producer = SynchronizedWindowProducer(
            stage02=stage02,
            cv2=cv2,
            video=video,
            audio=audio,
            capture_sr=capture_sr,
            capture_root=capture_root,
            session_id=session_id,
            clock_id=clock_id,
            windows=args.windows,
            lead_sec=args.lead_sec,
            settle_sec=args.settle_sec,
            stop_event=stop_event,
            output_queue=descriptor_queue,
            view_state=view_state,
            capture_audit=capture_audit,
        )
        producer.start()

        print(
            f"Camera                    : index {camera_info['index']} / "
            f"{camera_info['backend_actual']} / "
            f"{camera_info['reported_width']}x{camera_info['reported_height']}"
        )
        print(
            f"Microphone                : device {audio_device_index} / "
            f"{audio_cfg.get('device_name')}"
        )
        print(f"Audio capture             : {capture_sr} Hz mono")
        print(f"Audio stream startup      : {audio_startup_seconds:.3f} s")
        print()
        print("REAL-TIME INTERFACE STARTED")
        print("The camera view is continuous; emotion state updates as soon as inference returns.")
        print("Press Q or ESC in the UI to stop.")
        print()

        latest_latency: float | None = None
        producer_finished_seen = False
        last_expire_check = 0.0

        while True:
            # ----------------------------------------------------------
            # 1. Receive newly finalized source windows IN MEMORY.
            # ----------------------------------------------------------
            while True:
                try:
                    item = descriptor_queue.get_nowait()
                except queue.Empty:
                    break

                kind = item.get("type")

                if kind == "descriptor":
                    descriptor = item["descriptor"]

                    # Avoid knowingly submitting while workers are still busy.
                    # Normally inference is ~2-3s and the source cadence is 5s,
                    # so this path should be immediately available.
                    busy = any(
                        worker.busy
                        for name, worker in scheduler.workers.items()
                        if descriptor["modalities"][name]["present"]
                    )
                    if busy:
                        # Put it back and let the UI/scheduler progress first.
                        try:
                            descriptor_queue.put_nowait(item)
                        except queue.Full:
                            raise InterfaceError(
                                "Descriptor queue unexpectedly full while "
                                "waiting for frozen inference workers"
                            )
                        break

                    scheduler.submit_window(descriptor)
                    submitted_windows += 1
                    print(
                        f"[SUBMIT] {descriptor['window_id']} -> "
                        f"frozen Runtime/LiveScheduler (memory)",
                        flush=True,
                    )

                elif kind == "capture_failure":
                    msg = (
                        f"{item.get('window_id')} capture failed: "
                        + "; ".join(item.get("errors") or [])
                    )
                    view_state.set(message=msg)
                    print(f"[CAPTURE ERROR] {msg}", flush=True)

                elif kind == "producer_error":
                    raise InterfaceError(
                        f"Window producer failed: {item.get('error')}"
                    )

            # ----------------------------------------------------------
            # 2. Poll the FROZEN main scheduler directly IN MEMORY.
            # ----------------------------------------------------------
            rows = scheduler.poll()
            for event in scheduler.events:
                scheduler_event_records.append(copy.deepcopy(event))
                print(
                    f"[SCHEDULER EVENT] {event}",
                    flush=True,
                )
            scheduler.events.clear()

            for row in rows:
                compact = compact_result(row)
                result_records.append(compact)
                result_audit.append(compact)

                latest_latency = finite_or_none(
                    row.get("elapsed_since_capture_end_seconds")
                )
                display.update_runtime(
                    inference_latency_seconds=latest_latency,
                    message="Model result delivered directly in memory",
                )

                # This single line is the key architecture change:
                # model result -> state manager, no JSON polling.
                state = state_manager.update(row)

                stable = state["stable"]
                raw = state["raw"]
                transition = state["transition"]

                if stable["available"]:
                    stable_text = (
                        f"{stable['emotion']} "
                        f"{100.0 * float(stable['confidence']):.1f}%"
                        if stable["confidence"] is not None
                        else str(stable["emotion"])
                    )
                else:
                    stable_text = "NO_CURRENT_EVIDENCE"

                transition_text = ""
                if transition["active"]:
                    transition_text = (
                        f" | candidate={transition['candidate_emotion']} "
                        f"{transition['consecutive_count']}/"
                        f"{transition['required_count']}"
                    )

                print(
                    f"[RESULT] {row['window_id']} | "
                    f"raw={raw['emotion']} "
                    f"{100.0 * float(raw['confidence']):.1f}% | "
                    f"stable={stable_text}{transition_text} | "
                    f"route={row['fusion'].get('active_branch')} | "
                    f"latency_from_T1={latest_latency}",
                    flush=True,
                )

            # ----------------------------------------------------------
            # 3. Expire stale interaction state without reading files.
            # ----------------------------------------------------------
            now = time.monotonic()
            if now - last_expire_check >= 0.10:
                state_manager.expire(now_monotonic=now)
                last_expire_check = now

            # ----------------------------------------------------------
            # 4. Render the REAL camera + current emotion state continuously.
            # ----------------------------------------------------------
            view = view_state.snapshot()
            remaining = None
            if view["phase"] == "CAPTURING" and view["t1"] is not None:
                remaining = max(0.0, view["t1"] - now)
            elif (
                view["phase"] == "WAITING FOR WINDOW"
                and view["t0"] is not None
            ):
                remaining = max(0.0, view["t0"] - now)

            display.update_runtime(
                session_id=session_id,
                window_id=view["window_id"],
                phase=view["phase"],
                remaining_seconds=remaining,
                capture_fps=view["capture_fps"],
                inference_latency_seconds=latest_latency,
                message=view["message"],
            )

            frame = video.get_latest()
            if frame is not None:
                if display.show(frame, wait_ms=1):
                    print("\nUI stop requested.", flush=True)
                    stop_event.set()

            # ----------------------------------------------------------
            # 5. Exit only after capture stops AND submitted inference drains.
            # ----------------------------------------------------------
            if producer.done.is_set():
                producer_finished_seen = True

            queue_empty = descriptor_queue.empty()
            scheduler_empty = not scheduler.windows

            if (
                producer_finished_seen
                and queue_empty
                and scheduler_empty
            ):
                break

            # If Q/ESC was pressed, producer will stop at the next safe point.
            # Already-submitted complete windows are still allowed to finish.
            if (
                stop_event.is_set()
                and producer.done.is_set()
                and queue_empty
                and scheduler_empty
            ):
                break

            time.sleep(
                max(
                    0.001,
                    min(
                        0.01,
                        float(runtime_config["runtime"]["poll_interval_sec"]),
                    ),
                )
            )

        if producer.error:
            raise InterfaceError(
                f"Capture producer ended with error: {producer.error}"
            )

        # Ensure final state snapshot is audited/available.
        final_state = state_manager.snapshot()

        assets_after = runtime.verify_unchanged()

        if any(
            x.get("event") == "HOST_WINDOW_ERROR"
            for x in scheduler_event_records
        ):
            final_status = "COMPLETE_WITH_SCHEDULER_ERRORS"
        elif any(
            r.get("status") != "OK"
            for r in result_records
        ):
            final_status = "COMPLETE_WITH_INFERENCE_ERRORS"
        elif submitted_windows != len(result_records):
            final_status = "INCOMPLETE_RESULTS"
        elif producer.published == 0:
            final_status = "STOPPED_NO_VALID_WINDOW"
        else:
            final_status = "PASS"

    except KeyboardInterrupt:
        stop_event.set()
        fatal_error = "KeyboardInterrupt"
        final_status = "INTERRUPTED"
        final_state = (
            state_manager.snapshot()
            if state_manager is not None
            else None
        )
        print("\nKeyboardInterrupt: stopping live interface.", flush=True)

    except Exception as exc:
        stop_event.set()
        fatal_error = f"{type(exc).__name__}: {exc}"
        final_status = "FAIL"
        final_state = (
            state_manager.snapshot()
            if state_manager is not None
            else None
        )
        print(
            f"\nREAL-TIME INTERFACE ERROR: {fatal_error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()

    finally:
        stop_event.set()

        if producer is not None:
            producer.join(timeout=5.0)

        if scheduler is not None:
            try:
                shutdown_info = scheduler.close()
            except Exception as exc:
                shutdown_info = {
                    "error": f"{type(exc).__name__}: {exc}"
                }

        if audio is not None:
            try:
                audio.stop()
            except Exception:
                pass

        if video is not None:
            try:
                video.stop()
            except Exception:
                pass
        elif cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if state_manager is not None:
            if display_token is not None:
                try:
                    state_manager.unsubscribe(display_token)
                except Exception:
                    pass
            if audit_token is not None:
                try:
                    state_manager.unsubscribe(audit_token)
                except Exception:
                    pass

        if display is not None:
            try:
                display.close()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        result_audit.close()
        state_audit.close()
        capture_audit.close()

    # ------------------------------------------------------------------
    # Final audit summary. Not used for live interaction.
    # ------------------------------------------------------------------
    branch_counts = collections.Counter(
        r.get("active_branch") or r.get("status")
        for r in result_records
    )
    raw_emotion_counts = collections.Counter(
        r.get("emotion")
        for r in result_records
        if r.get("emotion")
    )

    if runtime is not None and assets_after is None:
        try:
            assets_after = runtime.verify_unchanged()
        except Exception:
            assets_after = None

    summary = {
        "schema": SUMMARY_SCHEMA,
        "version": VERSION,
        "status": final_status,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "session_id": session_id,
        "clock_id": clock_id,

        "architecture": {
            "prediction_result_transport": "IN_PROCESS_MEMORY",
            "state_transport": "IN_PROCESS_SUBSCRIBER_CALLBACK",
            "ui_reads_prediction_json": False,
            "files_used_as_interaction_transport": False,
            "audit_files_written": True,
            "frozen_main_runtime_used_directly": True,
        },

        "frozen_system": {
            "main_path": str(main_path.resolve()),
            "main_version": main_mod.VERSION,
            "main_sha256": sha256_file(main_path),
            "release": str(release_path.resolve()),
            "system_config": str(system_config.resolve()),
            "system_config_sha256": sha256_file(system_config),
            "quality_formula_modified": False,
            "quality_thresholds_modified": False,
            "fusion_weights_modified": False,
            "models_retrained": False,
            "legacy_router_enabled": False,
            "assets_unchanged_verified": assets_after is not None,
            "assets_after": assets_after,
        },

        "capture": {
            "device_config": str(device_config_path.resolve()),
            "camera": camera_info,
            "audio_capture_rate": capture_sr,
            "audio_stream_startup_seconds": audio_startup_seconds,
            "planned_windows": None if args.windows == 0 else args.windows,
            "attempted_windows": producer.attempted if producer else 0,
            "published_windows": producer.published if producer else 0,
            "failed_windows": producer.failed if producer else 0,
            "submitted_windows": submitted_windows,
        },

        "inference": {
            "results_received": len(result_records),
            "branch_counts": dict(branch_counts),
            "raw_emotion_counts": dict(raw_emotion_counts),
            "results": result_records,
            "scheduler_events": scheduler_event_records,
        },

        "emotion_state": {
            "confirmations_required": args.confirmations_required,
            "state_timeout_sec": args.state_timeout_sec,
            "state_snapshots_audited": state_records,
            "final_state": final_state,
        },

        "eeg": {
            "present": False,
            "reason": "EEG_DEVICE_NOT_CONNECTED",
        },

        "robot_action_performed": False,
        "shutdown": shutdown_info,
        "fatal_error": fatal_error,
        "next_stage": (
            "INTERACTION_POLICY_AND_ROBOT_RESPONSE"
            if final_status == "PASS"
            else "REVIEW_REALTIME_INTERFACE_BEFORE_ROBOT_RESPONSE"
        ),
    }

    atomic_write_json(
        session_root / "session_summary.json",
        summary,
    )

    print()
    print("=" * 108)
    print("STAGE 04 REAL-TIME EMOTION INTERFACE SUMMARY")
    print("=" * 108)
    print(f"Status                   : {final_status}")
    print("Result transport         : IN-PROCESS MEMORY")
    print("UI JSON polling          : False")
    print(f"Published AV windows     : {summary['capture']['published_windows']}")
    print(f"Submitted windows        : {submitted_windows}")
    print(f"Inference results        : {len(result_records)}")
    print(f"Routes                   : {dict(branch_counts)}")
    print(f"Raw emotions             : {dict(raw_emotion_counts)}")
    if final_state is not None:
        stable = final_state.get("stable", {})
        print(f"Final stable emotion     : {stable.get('emotion')}")
        print(f"Interaction status       : {final_state.get('interaction_status')}")
    print("EEG                      : UNAVAILABLE")
    print(f"Frozen assets verified   : {assets_after is not None}")
    print(f"Robot action             : False")
    print(f"Session summary          : {session_root / 'session_summary.json'}")
    if fatal_error:
        print(f"Fatal/stop reason        : {fatal_error}")
    print("=" * 108)

    return 0 if final_status == "PASS" else 130 if final_status == "INTERRUPTED" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InterfaceError as exc:
        print(f"INTERFACE ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
