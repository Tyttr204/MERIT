#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
01_probe_av_devices.py
======================

LiveInteraction Stage 01: laptop camera + microphone hardware/timebase audit.

Purpose
-------
This script does NOT run any emotion model, quality model, fusion model, EEG
logic, or robot action. It only verifies that the local computer can provide:

1. a usable camera stream;
2. a usable microphone input stream;
3. stable continuous capture for a short audit period;
4. timestamps from one shared Python ``time.monotonic()`` clock;
5. the device parameters needed by the next LiveInteraction modules.

Outputs
-------
By default the script writes:

    MERIT-Realtime Model/config/detected_av_devices.json

and a history report:

    MERIT-Realtime Model/runs/device_probe_<timestamp>/probe_report.json

No raw audio or raw video content is saved.

Recommended first run
---------------------
    python .\LiveInteraction\01_probe_av_devices.py

Press Q or ESC in the preview window to stop early.

Optional examples
-----------------
    python .\LiveInteraction\01_probe_av_devices.py --duration 20
    python .\LiveInteraction\01_probe_av_devices.py --camera-index 0
    python .\LiveInteraction\01_probe_av_devices.py --audio-device 1
    python .\LiveInteraction\01_probe_av_devices.py --audio-device "Microphone Array"
    python .\LiveInteraction\01_probe_av_devices.py --no-preview
    python .\LiveInteraction\01_probe_av_devices.py --list-only

Dependencies
------------
    numpy
    opencv-python
    sounddevice

The script is intended for Python 3.10+ and is designed primarily for the
Windows environment used by the EAV deployment project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "EAV-LIVE-AV-DEVICE-PROBE.1.1"
SCHEMA = "eav.live_interaction.av_device_probe.v1"

# V1.1: simultaneous-audit timing begins after PortAudio stream startup;
#       resample_required is evaluated against the fixed 16 kHz model contract.

TARGET_MODEL_AUDIO_SR = 16000
TARGET_AUDIO_CHANNELS = 1
TARGET_WINDOW_SECONDS = 5.0

DEFAULT_DURATION = 15.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30.0
DEFAULT_SCAN_MAX = 4


class ProbeError(RuntimeError):
    """Fatal device/configuration problem."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def finite_float(value: Any, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"{name}: numeric value required") from exc
    require(math.isfinite(v), f"{name}: NaN/Inf is not allowed")
    return v


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON safely; tolerate brief Windows destination-file locks."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        json_safe(obj),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"

    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

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
            raise ProbeError(
                f"Cannot publish JSON after 20 attempts: {path}"
            ) from last_error
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def load_dependencies():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise ProbeError(
            "OpenCV is unavailable. Install it in the active environment with:\n"
            "  python -m pip install opencv-python"
        ) from exc

    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise ProbeError(
            "python-sounddevice is unavailable. Install it in the active environment with:\n"
            "  python -m pip install sounddevice"
        ) from exc

    return cv2, sd


def backend_candidates(cv2, requested: str) -> list[tuple[str, int]]:
    requested = requested.lower()

    def has(name: str) -> bool:
        return hasattr(cv2, name)

    mapping: dict[str, list[tuple[str, int]]] = {
        "any": [("ANY", int(cv2.CAP_ANY))],
        "dshow": [("DSHOW", int(cv2.CAP_DSHOW))] if has("CAP_DSHOW") else [],
        "msmf": [("MSMF", int(cv2.CAP_MSMF))] if has("CAP_MSMF") else [],
    }

    if requested != "auto":
        choices = mapping.get(requested, [])
        require(choices, f"Requested camera backend is unavailable: {requested}")
        return choices

    result: list[tuple[str, int]] = []
    if os.name == "nt":
        if has("CAP_DSHOW"):
            result.append(("DSHOW", int(cv2.CAP_DSHOW)))
        if has("CAP_MSMF"):
            result.append(("MSMF", int(cv2.CAP_MSMF)))
    result.append(("ANY", int(cv2.CAP_ANY)))

    # Remove duplicate backend codes while preserving preference order.
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for name, code in result:
        if code not in seen:
            unique.append((name, code))
            seen.add(code)
    return unique


def try_open_camera(
    cv2,
    index: int,
    backend_name: str,
    backend_code: int,
    width: int,
    height: int,
    fps: float,
    fourcc: str | None,
    warmup_frames: int = 5,
) -> tuple[Any | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "index": index,
        "requested_backend": backend_name,
        "opened": False,
        "first_frame_ok": False,
        "error": None,
    }

    cap = None
    try:
        cap = cv2.VideoCapture(index, backend_code)
        if not cap.isOpened():
            info["error"] = "VideoCapture did not open"
            cap.release()
            return None, info

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        if fourcc:
            code = cv2.VideoWriter_fourcc(*fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, code)

        ok = False
        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                break
            time.sleep(0.03)

        info["opened"] = True
        info["first_frame_ok"] = bool(ok and frame is not None and frame.size)

        if not info["first_frame_ok"]:
            info["error"] = "Camera opened but no valid frame was received"
            cap.release()
            return None, info

        try:
            backend_actual = cap.getBackendName()
        except Exception:
            backend_actual = backend_name

        info.update(
            actual_backend=str(backend_actual),
            reported_width=int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            reported_height=int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            reported_fps=float(cap.get(cv2.CAP_PROP_FPS)),
            reported_fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)),
        )
        return cap, info

    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None, info


def discover_camera(
    cv2,
    requested_index: int | None,
    requested_backend: str,
    scan_max: int,
    width: int,
    height: int,
    fps: float,
    fourcc: str | None,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    candidates = backend_candidates(cv2, requested_backend)
    indices = [requested_index] if requested_index is not None else list(range(scan_max + 1))
    attempts: list[dict[str, Any]] = []

    for index in indices:
        require(index is not None and index >= 0, "Camera index must be >= 0")
        for backend_name, backend_code in candidates:
            cap, info = try_open_camera(
                cv2=cv2,
                index=int(index),
                backend_name=backend_name,
                backend_code=backend_code,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
            )
            attempts.append(info)
            if cap is not None:
                return cap, info, attempts

    lines = [
        f"camera index={x['index']} backend={x['requested_backend']}: {x.get('error')}"
        for x in attempts
    ]
    raise ProbeError(
        "No usable camera could be opened.\n"
        "Check Windows Settings > Privacy & security > Camera and close applications "
        "that may own the camera.\nAttempts:\n  " + "\n  ".join(lines)
    )


def list_audio_devices(sd) -> tuple[list[dict[str, Any]], int | None]:
    raw = sd.query_devices()
    hostapis = sd.query_hostapis()

    input_devices: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        max_inputs = int(item["max_input_channels"])
        if max_inputs <= 0:
            continue
        host_index = int(item["hostapi"])
        host_name = (
            str(hostapis[host_index]["name"])
            if 0 <= host_index < len(hostapis)
            else f"hostapi_{host_index}"
        )
        input_devices.append(
            {
                "index": index,
                "name": str(item["name"]),
                "hostapi_index": host_index,
                "hostapi_name": host_name,
                "max_input_channels": max_inputs,
                "default_samplerate": float(item["default_samplerate"]),
                "default_low_input_latency": float(item["default_low_input_latency"]),
                "default_high_input_latency": float(item["default_high_input_latency"]),
            }
        )

    default_input: int | None = None
    try:
        default_value = sd.default.device[0]
        if default_value is not None and int(default_value) >= 0:
            default_input = int(default_value)
    except Exception:
        default_input = None

    return input_devices, default_input


def resolve_audio_device(
    devices: list[dict[str, Any]],
    default_index: int | None,
    requested: str | None,
) -> dict[str, Any]:
    require(devices, "No microphone/input audio device was found")

    if requested is None:
        if default_index is not None:
            hit = next((d for d in devices if d["index"] == default_index), None)
            if hit is not None:
                return hit
        return devices[0]

    text = str(requested).strip()
    if re.fullmatch(r"\d+", text):
        idx = int(text)
        hit = next((d for d in devices if d["index"] == idx), None)
        require(hit is not None, f"Audio input device index {idx} does not exist")
        return hit

    matches = [d for d in devices if text.casefold() in d["name"].casefold()]
    require(matches, f"No input device name contains: {text!r}")
    require(
        len(matches) == 1,
        "Audio device name is ambiguous. Matching devices: "
        + ", ".join(f"{d['index']}:{d['name']}" for d in matches),
    )
    return matches[0]


def check_audio_rate(sd, device_index: int, samplerate: float) -> tuple[bool, str | None]:
    try:
        sd.check_input_settings(
            device=device_index,
            channels=TARGET_AUDIO_CHANNELS,
            dtype="float32",
            samplerate=float(samplerate),
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


@dataclass
class AudioStats:
    callbacks: int = 0
    frames: int = 0
    samples: int = 0
    sum_squares: float = 0.0
    peak_abs: float = 0.0
    clipping_samples: int = 0
    first_callback_monotonic: float | None = None
    last_callback_monotonic: float | None = None
    input_overflows: int = 0
    input_underflows: int = 0
    callback_status_events: int = 0
    callback_errors: int = 0
    status_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)

    def rms(self) -> float | None:
        if self.samples <= 0:
            return None
        return math.sqrt(max(0.0, self.sum_squares / self.samples))

    def rms_dbfs(self) -> float | None:
        value = self.rms()
        if value is None:
            return None
        return 20.0 * math.log10(max(value, 1e-12))


def build_audio_callback(stats: AudioStats):
    def callback(indata, frames: int, time_info, status) -> None:
        try:
            now = time.monotonic()
            if stats.first_callback_monotonic is None:
                stats.first_callback_monotonic = now
            stats.last_callback_monotonic = now
            stats.callbacks += 1
            stats.frames += int(frames)

            if status:
                stats.callback_status_events += 1
                message = str(status)
                if len(stats.status_messages) < 50:
                    stats.status_messages.append(message)
                try:
                    if status.input_overflow:
                        stats.input_overflows += 1
                    if status.input_underflow:
                        stats.input_underflows += 1
                except Exception:
                    pass

            if indata is not None and len(indata):
                # Keep the callback lightweight: only simple numeric diagnostics.
                x = np.asarray(indata[:, 0], dtype=np.float32)
                stats.samples += int(x.size)
                stats.sum_squares += float(np.dot(x, x))
                peak = float(np.max(np.abs(x))) if x.size else 0.0
                if peak > stats.peak_abs:
                    stats.peak_abs = peak
                stats.clipping_samples += int(np.count_nonzero(np.abs(x) >= 0.999))

        except Exception as exc:
            # Exceptions raised directly from a PortAudio callback may not reach
            # the main thread, so record them here and let the audit fail later.
            stats.callback_errors += 1
            if len(stats.error_messages) < 20:
                stats.error_messages.append(f"{type(exc).__name__}: {exc}")

    return callback


def interval_statistics(timestamps: list[float]) -> dict[str, Any]:
    if len(timestamps) < 2:
        return {
            "measured_fps": None,
            "mean_interval_ms": None,
            "median_interval_ms": None,
            "p95_interval_ms": None,
            "jitter_std_ms": None,
            "estimated_dropped_frames": None,
        }

    times = np.asarray(timestamps, dtype=np.float64)
    dt = np.diff(times)
    span = float(times[-1] - times[0])
    fps = (len(times) - 1) / span if span > 0 else None
    median = float(np.median(dt))

    dropped = 0
    if median > 0:
        for delta in dt:
            if delta > 1.5 * median:
                dropped += max(0, int(round(float(delta) / median)) - 1)

    return {
        "measured_fps": fps,
        "mean_interval_ms": float(np.mean(dt) * 1000.0),
        "median_interval_ms": median * 1000.0,
        "p95_interval_ms": float(np.percentile(dt, 95) * 1000.0),
        "jitter_std_ms": float(np.std(dt) * 1000.0),
        "estimated_dropped_frames": int(dropped),
    }


def draw_preview(cv2, frame, lines: list[str]) -> Any:
    canvas = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1

    y = 28
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (14, y),
            font,
            scale,
            (255, 255, 255),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (14, y),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += 26
    return canvas


def run_capture_probe(
    cv2,
    sd,
    cap,
    camera_info: dict[str, Any],
    audio_device: dict[str, Any],
    requested_audio_sr: int,
    duration: float,
    preview: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []

    target_ok, target_error = check_audio_rate(
        sd, audio_device["index"], requested_audio_sr
    )

    capture_sr = float(requested_audio_sr)
    resample_required = False

    if not target_ok:
        native_sr = float(audio_device["default_samplerate"])
        native_ok, native_error = check_audio_rate(
            sd, audio_device["index"], native_sr
        )
        require(
            native_ok,
            "Selected microphone supports neither target "
            f"{requested_audio_sr} Hz nor its default {native_sr:g} Hz.\n"
            f"Target error: {target_error}\nNative error: {native_error}",
        )
        capture_sr = native_sr
        resample_required = abs(capture_sr - TARGET_MODEL_AUDIO_SR) > 0.5
        warnings.append(
            f"MICROPHONE_DOES_NOT_ACCEPT_{requested_audio_sr}_HZ_DIRECTLY;"
            f"_CAPTURE_AT_{capture_sr:g}_HZ_AND_RESAMPLE_LATER"
        )

    stats = AudioStats()
    callback = build_audio_callback(stats)

    frame_times: list[float] = []
    read_attempts = 0
    read_failures = 0
    frame_shapes: set[tuple[int, int, int]] = set()
    brightness_samples: list[float] = []

    preview_active = bool(preview)
    user_stopped = False
    window_name = "EAV LiveInteraction - AV Device Probe"

    print()
    print("Starting simultaneous camera + microphone probe.")
    print("Please speak normally and remain visible to the camera.")
    if preview_active:
        print("Press Q or ESC in the camera window to stop early.")
    print()

    # Stream/device construction may take noticeable time on Windows,
    # especially through MME.  Do NOT include that startup latency in the
    # simultaneous-capture audit interval: audio frame accounting begins only
    # once PortAudio has started the stream.
    stream_setup_start = time.monotonic()
    capture_start: float | None = None
    overall_end: float | None = None

    stream = None
    try:
        stream = sd.InputStream(
            device=audio_device["index"],
            samplerate=capture_sr,
            channels=TARGET_AUDIO_CHANNELS,
            dtype="float32",
            blocksize=0,
            callback=callback,
        )
        stream.start()
        capture_start = time.monotonic()
        stream_startup_seconds = capture_start - stream_setup_start

        while True:
            now = time.monotonic()
            elapsed = now - capture_start
            if elapsed >= duration:
                break

            read_attempts += 1
            ok, frame = cap.read()
            capture_time = time.monotonic()

            if not ok or frame is None or not frame.size:
                read_failures += 1
                time.sleep(0.002)
                continue

            frame_times.append(capture_time)
            frame_shapes.add(tuple(int(x) for x in frame.shape))

            if len(frame_times) % 10 == 0:
                brightness_samples.append(float(np.mean(frame)))

            if preview_active:
                rms_db = stats.rms_dbfs()
                overlay = [
                    f"AV DEVICE PROBE {elapsed:5.1f}/{duration:.1f}s",
                    f"Camera: index={camera_info['index']} backend={camera_info.get('actual_backend')}",
                    f"Frames: {len(frame_times)}  read failures: {read_failures}",
                    f"Mic: {audio_device['index']}  SR={capture_sr:g} Hz mono",
                    f"Audio RMS: {rms_db:.1f} dBFS" if rms_db is not None else "Audio RMS: waiting...",
                    "No raw audio/video is saved",
                ]

                try:
                    shown = draw_preview(cv2, frame, overlay)
                    cv2.imshow(window_name, shown)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        user_stopped = True
                        break
                except Exception as exc:
                    preview_active = False
                    warnings.append(
                        "PREVIEW_DISABLED_AFTER_ERROR:"
                        f"{type(exc).__name__}:{exc}"
                    )

        overall_end = time.monotonic()

    finally:
        if overall_end is None:
            overall_end = time.monotonic()
        if capture_start is None:
            capture_start = overall_end
            stream_startup_seconds = capture_start - stream_setup_start

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        try:
            cap.release()
        except Exception:
            pass

        if preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    elapsed = max(0.0, overall_end - capture_start)
    cam_stats = interval_statistics(frame_times)

    camera_result = {
        **camera_info,
        "probe_duration_seconds": elapsed,
        "read_attempts": read_attempts,
        "frames_captured": len(frame_times),
        "read_failures": read_failures,
        "read_failure_rate": (
            read_failures / read_attempts if read_attempts else None
        ),
        "frame_shapes": [list(x) for x in sorted(frame_shapes)],
        "mean_sampled_brightness_0_255": (
            float(np.mean(brightness_samples))
            if brightness_samples
            else None
        ),
        **cam_stats,
        "first_frame_monotonic": (
            frame_times[0] if frame_times else None
        ),
        "last_frame_monotonic": (
            frame_times[-1] if frame_times else None
        ),
        "preview_requested": preview,
        "preview_active_at_end": preview_active,
    }

    audio_duration = stats.frames / capture_sr if capture_sr > 0 else 0.0
    audio_result = {
        "device_index": audio_device["index"],
        "name": audio_device["name"],
        "hostapi_index": audio_device["hostapi_index"],
        "hostapi_name": audio_device["hostapi_name"],
        "max_input_channels": audio_device["max_input_channels"],
        "device_default_samplerate": audio_device["default_samplerate"],
        "target_samplerate_requested": int(requested_audio_sr),
        "target_samplerate_supported_directly": bool(target_ok),
        "target_samplerate_error": target_error,
        "capture_samplerate": float(capture_sr),
        "capture_channels": TARGET_AUDIO_CHANNELS,
        "capture_dtype": "float32",
        "blocksize": 0,
        "resample_to_model_16000_required": bool(
            abs(float(capture_sr) - TARGET_MODEL_AUDIO_SR) > 0.5
        ),
        "callbacks": stats.callbacks,
        "frames_captured": stats.frames,
        "audio_sample_duration_seconds": audio_duration,
        "coverage_ratio_vs_probe_elapsed": (
            audio_duration / elapsed if elapsed > 0 else None
        ),
        "rms_linear": stats.rms(),
        "rms_dbfs": stats.rms_dbfs(),
        "peak_abs": stats.peak_abs,
        "clipping_samples": stats.clipping_samples,
        "clipping_fraction": (
            stats.clipping_samples / stats.samples if stats.samples else None
        ),
        "input_overflows": stats.input_overflows,
        "input_underflows": stats.input_underflows,
        "callback_status_events": stats.callback_status_events,
        "callback_status_messages": stats.status_messages,
        "callback_errors": stats.callback_errors,
        "callback_error_messages": stats.error_messages,
        "first_callback_monotonic": stats.first_callback_monotonic,
        "last_callback_monotonic": stats.last_callback_monotonic,
    }

    camera_first = camera_result["first_frame_monotonic"]
    audio_first = audio_result["first_callback_monotonic"]
    first_skew = (
        abs(float(camera_first) - float(audio_first))
        if camera_first is not None and audio_first is not None
        else None
    )

    sync_result = {
        "clock_source": "time.monotonic()",
        "same_python_monotonic_clock_used_for_camera_and_audio": True,
        "probe_start_monotonic": capture_start,
        "audio_stream_startup_seconds": stream_startup_seconds,
        "probe_end_monotonic": overall_end,
        "probe_elapsed_seconds": elapsed,
        "camera_first_activity_monotonic": camera_first,
        "audio_first_activity_monotonic": audio_first,
        "first_activity_absolute_skew_seconds": first_skew,
        "note": (
            "This stage verifies a shared host clock only. "
            "Exact 5-second AV window synchronization is implemented in Stage 02."
        ),
    }

    if user_stopped:
        warnings.append("USER_STOPPED_PROBE_BEFORE_REQUESTED_DURATION")

    return camera_result, audio_result, sync_result, warnings


def evaluate_results(
    camera: dict[str, Any],
    audio: dict[str, Any],
    sync: dict[str, Any],
    duration_requested: float,
    inherited_warnings: list[str],
) -> tuple[str, list[str], list[str]]:
    warnings = list(inherited_warnings)
    errors: list[str] = []

    elapsed = finite_float(sync["probe_elapsed_seconds"], "elapsed")
    min_expected_duration = min(duration_requested * 0.85, max(3.0, duration_requested - 1.0))

    if elapsed < min_expected_duration:
        warnings.append(
            f"PROBE_SHORTER_THAN_REQUESTED:{elapsed:.2f}s"
        )

    frames = int(camera.get("frames_captured") or 0)
    measured_fps = camera.get("measured_fps")
    failure_rate = camera.get("read_failure_rate")

    if frames < max(10, int(5 * max(elapsed, 1.0))):
        errors.append("CAMERA_CAPTURE_RATE_TOO_LOW_OR_NO_FRAMES")

    if measured_fps is None or float(measured_fps) < 5.0:
        errors.append("CAMERA_MEASURED_FPS_BELOW_5")

    if failure_rate is not None and float(failure_rate) > 0.10:
        errors.append("CAMERA_READ_FAILURE_RATE_ABOVE_10_PERCENT")

    requested_fps = float(camera.get("reported_fps") or DEFAULT_FPS)
    if measured_fps is not None and requested_fps > 0:
        if float(measured_fps) < 0.80 * requested_fps:
            warnings.append(
                "CAMERA_MEASURED_FPS_SUBSTANTIALLY_BELOW_REPORTED_OR_REQUESTED"
            )

    brightness = camera.get("mean_sampled_brightness_0_255")
    if brightness is not None and float(brightness) < 5.0:
        warnings.append("CAMERA_IMAGE_APPEARS_VERY_DARK")
    if brightness is not None and float(brightness) > 250.0:
        warnings.append("CAMERA_IMAGE_APPEARS_NEAR_SATURATION")

    audio_frames = int(audio.get("frames_captured") or 0)
    if audio_frames <= 0:
        errors.append("NO_AUDIO_FRAMES_CAPTURED")

    coverage = audio.get("coverage_ratio_vs_probe_elapsed")
    if coverage is None or float(coverage) < 0.90:
        errors.append("AUDIO_CAPTURE_COVERAGE_BELOW_90_PERCENT")

    if int(audio.get("callback_errors") or 0) > 0:
        errors.append("AUDIO_CALLBACK_ERRORS_OCCURRED")

    if int(audio.get("input_overflows") or 0) > 0:
        warnings.append(
            f"AUDIO_INPUT_OVERFLOW_COUNT={audio['input_overflows']}"
        )

    rms_db = audio.get("rms_dbfs")
    if rms_db is not None and float(rms_db) < -75.0:
        warnings.append(
            "AUDIO_LEVEL_VERY_LOW;SPEAK_NORMALLY_DURING_NEXT_PROBE"
        )

    clip_fraction = audio.get("clipping_fraction")
    if clip_fraction is not None and float(clip_fraction) > 0.001:
        warnings.append(
            f"AUDIO_CLIPPING_FRACTION={float(clip_fraction):.6f}"
        )

    if not audio.get("target_samplerate_supported_directly", False):
        warnings.append(
            "MODEL_TARGET_16000_HZ_NOT_DIRECTLY_SUPPORTED;"
            "LATER_AUDIO_STREAM_MUST_RESAMPLE"
        )

    if errors:
        return "FAIL", warnings, errors
    if warnings:
        return "PASS_WITH_WARNINGS", warnings, errors
    return "PASS", warnings, errors


def print_device_list(
    camera_attempts: list[dict[str, Any]] | None,
    audio_devices: list[dict[str, Any]],
    default_audio_index: int | None,
) -> None:
    print()
    print("=" * 96)
    print("AUDIO INPUT DEVICES")
    print("=" * 96)
    for d in audio_devices:
        marker = " [DEFAULT]" if d["index"] == default_audio_index else ""
        print(
            f"[{d['index']:>2}] {d['name']}{marker}\n"
            f"     host={d['hostapi_name']} | inputs={d['max_input_channels']} | "
            f"default_sr={d['default_samplerate']:g}"
        )

    if camera_attempts is not None:
        print()
        print("=" * 96)
        print("CAMERA OPEN ATTEMPTS")
        print("=" * 96)
        for x in camera_attempts:
            print(
                f"index={x['index']} backend={x['requested_backend']} "
                f"opened={x['opened']} first_frame={x['first_frame_ok']} "
                f"error={x.get('error')}"
            )
    print("=" * 96)


def print_summary(report: dict[str, Any], config_path: Path, history_path: Path) -> None:
    camera = report["camera"]["selected"]
    audio = report["audio"]["selected"]
    sync = report["clock_and_sync_probe"]

    print()
    print("=" * 96)
    print("EAV LIVEINTERACTION — AV DEVICE PROBE")
    print("=" * 96)
    print(f"Status                  : {report['status']}")
    print(f"Probe duration          : {sync['probe_elapsed_seconds']:.2f} s")
    print()
    print("Camera")
    print(f"  Device index          : {camera['index']}")
    print(f"  Backend               : {camera.get('actual_backend')}")
    print(
        f"  Resolution            : "
        f"{camera.get('reported_width')} x {camera.get('reported_height')}"
    )
    print(f"  Reported FPS          : {camera.get('reported_fps')}")
    print(
        f"  Measured FPS          : "
        f"{camera.get('measured_fps'):.3f}"
        if camera.get("measured_fps") is not None
        else "  Measured FPS          : N/A"
    )
    print(f"  Frames captured       : {camera.get('frames_captured')}")
    print(f"  Read failures         : {camera.get('read_failures')}")
    print(f"  Estimated dropped     : {camera.get('estimated_dropped_frames')}")
    print()
    print("Microphone")
    print(f"  Device index          : {audio['device_index']}")
    print(f"  Device name           : {audio['name']}")
    print(f"  Host API              : {audio['hostapi_name']}")
    print(f"  Default sample rate   : {audio['device_default_samplerate']:g} Hz")
    print(f"  Capture sample rate   : {audio['capture_samplerate']:g} Hz")
    print(
        f"  16 kHz direct support : "
        f"{audio['target_samplerate_supported_directly']}"
    )
    print(
        f"  Resample required     : "
        f"{audio['resample_to_model_16000_required']}"
    )
    print(
        f"  RMS                   : "
        f"{audio['rms_dbfs']:.2f} dBFS"
        if audio.get("rms_dbfs") is not None
        else "  RMS                   : N/A"
    )
    print(f"  Peak                  : {audio.get('peak_abs')}")
    print(f"  Input overflows       : {audio.get('input_overflows')}")
    print()
    print("Clock")
    print(
        f"  Audio stream startup  : "
        f"{sync.get('audio_stream_startup_seconds'):.3f} s"
        if sync.get("audio_stream_startup_seconds") is not None
        else "  Audio stream startup  : N/A"
    )
    print(f"  Source                : {sync['clock_source']}")
    print(
        f"  First AV activity skew: "
        f"{sync['first_activity_absolute_skew_seconds']:.4f} s"
        if sync.get("first_activity_absolute_skew_seconds") is not None
        else "  First AV activity skew: N/A"
    )
    print()
    if report["warnings"]:
        print("Warnings")
        for item in report["warnings"]:
            print(f"  - {item}")
    if report["errors"]:
        print("Errors")
        for item in report["errors"]:
            print(f"  - {item}")
    print()
    print(f"Detected config         : {config_path}")
    print(f"History report          : {history_path}")
    print("Raw audio/video saved   : False")
    print("=" * 96)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the laptop camera and microphone for the EAV LiveInteraction module. "
            "No emotion inference is performed."
        )
    )
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Simultaneous capture audit duration in seconds (default: {DEFAULT_DURATION}).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Explicit camera index. If omitted, indices are scanned.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf", "any"),
        default="auto",
        help="OpenCV camera backend (default: auto; Windows prefers DSHOW then MSMF).",
    )
    parser.add_argument(
        "--scan-camera-max",
        type=int,
        default=DEFAULT_SCAN_MAX,
        help=f"Highest camera index to scan when --camera-index is omitted (default: {DEFAULT_SCAN_MAX}).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--camera-fourcc",
        default=None,
        help="Optional four-character camera format request, e.g. MJPG. Default: unchanged.",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Input device index or unique case-insensitive name substring. Default: system input.",
    )
    parser.add_argument(
        "--audio-samplerate",
        type=int,
        default=TARGET_MODEL_AUDIO_SR,
        help=f"Requested direct capture sample rate (default: {TARGET_MODEL_AUDIO_SR}).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run the same camera/audio probe without an OpenCV preview window.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List input audio devices and probe camera indices; do not run the timed capture audit.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Detected device config JSON. Default: "
            "MERIT-Realtime Model/config/detected_av_devices.json"
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not create runs/device_probe_*/probe_report.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    require(3.0 <= args.duration <= 300.0, "--duration must be between 3 and 300 seconds")
    require(args.width > 0 and args.height > 0, "Camera width/height must be positive")
    require(1.0 <= args.fps <= 240.0, "--fps must be in [1, 240]")
    require(0 <= args.scan_camera_max <= 20, "--scan-camera-max must be in [0, 20]")
    require(args.audio_samplerate >= 8000, "--audio-samplerate must be >= 8000")
    if args.camera_index is not None:
        require(args.camera_index >= 0, "--camera-index must be >= 0")
    if args.camera_fourcc is not None:
        require(
            len(args.camera_fourcc) == 4,
            "--camera-fourcc must contain exactly four characters, e.g. MJPG",
        )

    cv2, sd = load_dependencies()

    script_dir = Path(__file__).resolve().parent
    config_output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else script_dir / "config" / "detected_av_devices.json"
    )
    history_dir = script_dir / "runs" / f"device_probe_{timestamp_tag()}"
    history_path = history_dir / "probe_report.json"

    print("=" * 96)
    print("EAV LiveInteraction — Stage 01 AV device probe")
    print(f"Version                 : {VERSION}")
    print(f"Python                  : {sys.version.split()[0]}")
    print(f"Platform                : {platform.platform()}")
    print(f"OpenCV                  : {getattr(cv2, '__version__', 'unknown')}")
    print(f"sounddevice             : {getattr(sd, '__version__', 'unknown')}")
    print("Emotion models          : NOT LOADED")
    print("EEG                     : NOT USED")
    print("Raw AV recording        : DISABLED")
    print("=" * 96)

    audio_devices, default_audio = list_audio_devices(sd)
    selected_audio = resolve_audio_device(
        audio_devices, default_audio, args.audio_device
    )

    cap, camera_info, camera_attempts = discover_camera(
        cv2=cv2,
        requested_index=args.camera_index,
        requested_backend=args.camera_backend,
        scan_max=args.scan_camera_max,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.camera_fourcc,
    )

    print_device_list(camera_attempts, audio_devices, default_audio)

    if args.list_only:
        try:
            cap.release()
        except Exception:
            pass
        print()
        print("LIST-ONLY COMPLETE. No timed capture audit was performed.")
        return 0

    camera_result = audio_result = sync_result = None
    run_warnings: list[str] = []
    run_errors: list[str] = []
    status = "FAIL"
    started_utc = utc_now()

    try:
        camera_result, audio_result, sync_result, capture_warnings = run_capture_probe(
            cv2=cv2,
            sd=sd,
            cap=cap,
            camera_info=camera_info,
            audio_device=selected_audio,
            requested_audio_sr=args.audio_samplerate,
            duration=args.duration,
            preview=not args.no_preview,
        )

        status, run_warnings, run_errors = evaluate_results(
            camera=camera_result,
            audio=audio_result,
            sync=sync_result,
            duration_requested=args.duration,
            inherited_warnings=capture_warnings,
        )

    except Exception as exc:
        run_errors.append(f"{type(exc).__name__}: {exc}")
        status = "FAIL"
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if camera_result is None:
        camera_result = {
            **camera_info,
            "frames_captured": 0,
            "read_failures": None,
            "measured_fps": None,
        }

    if audio_result is None:
        audio_result = {
            "device_index": selected_audio["index"],
            "name": selected_audio["name"],
            "hostapi_name": selected_audio["hostapi_name"],
            "device_default_samplerate": selected_audio["default_samplerate"],
            "target_samplerate_requested": args.audio_samplerate,
            "target_samplerate_supported_directly": None,
            "capture_samplerate": None,
            "capture_channels": TARGET_AUDIO_CHANNELS,
            "frames_captured": 0,
            "input_overflows": None,
        }

    if sync_result is None:
        info = time.get_clock_info("monotonic")
        sync_result = {
            "clock_source": "time.monotonic()",
            "same_python_monotonic_clock_used_for_camera_and_audio": True,
            "probe_elapsed_seconds": 0.0,
            "first_activity_absolute_skew_seconds": None,
            "clock_info": {
                "implementation": info.implementation,
                "monotonic": info.monotonic,
                "adjustable": info.adjustable,
                "resolution_seconds": info.resolution,
            },
        }
    else:
        info = time.get_clock_info("monotonic")
        sync_result["clock_info"] = {
            "implementation": info.implementation,
            "monotonic": info.monotonic,
            "adjustable": info.adjustable,
            "resolution_seconds": info.resolution,
        }

    recommended = {
        "schema": "eav.live_interaction.detected_av_devices.v1",
        "generated_utc": utc_now(),
        "probe_version": VERSION,
        "window_seconds": TARGET_WINDOW_SECONDS,
        "clock": {
            "source": "time.monotonic()",
            "clock_id_recommendation": "python_time_monotonic_same_host",
        },
        "camera": {
            "index": camera_result.get("index"),
            "backend": camera_result.get("actual_backend")
            or camera_result.get("requested_backend"),
            "reported_width": camera_result.get("reported_width"),
            "reported_height": camera_result.get("reported_height"),
            "reported_fps": camera_result.get("reported_fps"),
            "measured_fps": camera_result.get("measured_fps"),
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "requested_fourcc": args.camera_fourcc,
        },
        "audio": {
            "device_index": audio_result.get("device_index"),
            "device_name": audio_result.get("name"),
            "hostapi_name": audio_result.get("hostapi_name"),
            "capture_samplerate": audio_result.get("capture_samplerate"),
            "channels": TARGET_AUDIO_CHANNELS,
            "dtype": "float32",
            "model_target_samplerate": TARGET_MODEL_AUDIO_SR,
            "resample_required": audio_result.get(
                "resample_to_model_16000_required"
            ),
        },
        "eeg": {
            "present": False,
            "reason": "EEG_DEVICE_NOT_CONNECTED",
        },
        "probe_status": status,
    }

    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "purpose": "LIVE_AV_CAPTURE_HARDWARE_AND_SHARED_CLOCK_AUDIT_ONLY",
        "emotion_inference_performed": False,
        "quality_layer_invoked": False,
        "fusion_invoked": False,
        "robot_action_performed": False,
        "raw_audio_saved": False,
        "raw_video_saved": False,
        "target_future_window_seconds": TARGET_WINDOW_SECONDS,
        "camera": {
            "selected": camera_result,
            "open_attempts": camera_attempts,
        },
        "audio": {
            "selected": audio_result,
            "available_input_devices": audio_devices,
            "system_default_input_device_index": default_audio,
        },
        "clock_and_sync_probe": sync_result,
        "recommended_live_config": recommended,
        "warnings": list(dict.fromkeys(run_warnings)),
        "errors": list(dict.fromkeys(run_errors)),
        "next_stage": (
            "02_capture_synchronized_av_window.py"
            if status != "FAIL"
            else "FIX_DEVICE_OR_PERMISSION_FAILURES_AND_REPEAT_STAGE_01"
        ),
    }

    # detected_av_devices.json is intentionally the latest successful/attempted
    # local hardware state. The timestamped history preserves audit provenance.
    atomic_write_json(config_output, recommended)

    if not args.no_history:
        atomic_write_json(history_path, report)

    print_summary(report, config_output, history_path)

    return 0 if status in ("PASS", "PASS_WITH_WARNINGS") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except ProbeError as exc:
        print(f"\nPROBE ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
