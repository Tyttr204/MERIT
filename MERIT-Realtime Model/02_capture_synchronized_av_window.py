#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
02_capture_synchronized_av_window.py
====================================

EAV LiveInteraction — Stage 02
Real laptop camera + microphone synchronized 5-second window capture.

This stage DOES NOT run:
    - EEG inference
    - Audio emotion inference
    - Video emotion inference
    - Quality control
    - F4 / AF4-B fusion
    - robot actions

It only proves that the already verified laptop camera and microphone can be
kept running continuously and can be cut into the SAME real 5-second windows
on one shared host time axis.

Default input
-------------
    MERIT-Realtime Model/config/detected_av_devices.json

Default output
--------------
    MERIT-Realtime Model/captures/sync_capture_<timestamp>/
        session_summary.json
        window_000001/
            audio.wav
            video.mp4        (or video.avi fallback)
            descriptor.json
            capture_metadata.json

The descriptor follows the existing frozen main.py live-source contract:
    schema = eav.system.source_window.v1
    split  = live
    window_seconds = 5.0

EEG is explicitly unavailable:
    present = false
    reason  = EEG_DEVICE_NOT_CONNECTED

Recommended run
---------------
    python .\LiveInteraction\02_capture_synchronized_av_window.py

Capture several consecutive windows:
    python .\LiveInteraction\02_capture_synchronized_av_window.py --windows 3

Press Q or ESC in the preview window to stop after the current capture wait
loop. An interrupted/incomplete window is never published as a valid
descriptor.

Dependencies
------------
    numpy
    opencv-python
    sounddevice

Optional:
    scipy  (only needed if a future microphone must be resampled to 16 kHz)

Important design choices
------------------------
1. Camera acquisition and microphone acquisition run continuously.
2. Both are mapped onto Python time.monotonic().
3. One decision window is exactly [T0, T0 + 5.0 s].
4. Audio is sliced to exactly 80,000 samples at 16 kHz.
5. Video frames are selected only from the same [T0, T1] physical interval.
6. No missing samples/frames are fabricated to make a failed window pass.
7. A valid descriptor is atomically published only after both media files have
   been completed and verified.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import platform
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Deque, Iterable

import numpy as np


VERSION = "EAV-LIVE-SYNC-CAPTURE.1.0"
SUMMARY_SCHEMA = "eav.live_interaction.sync_capture_summary.v1"
METADATA_SCHEMA = "eav.live_interaction.sync_window_metadata.v1"
SOURCE_SCHEMA = "eav.system.source_window.v1"

WINDOW_SECONDS = 5.0
MODEL_AUDIO_SR = 16000
MODEL_AUDIO_CHANNELS = 1

DEFAULT_WINDOWS = 1
DEFAULT_LEAD_SECONDS = 0.50
DEFAULT_SETTLE_SECONDS = 0.12
DEFAULT_RETENTION_SECONDS = 8.0

# Stage-02 engineering acceptance limits.  These are acquisition-integrity
# checks only; they are NOT emotion/quality model thresholds.
MIN_VIDEO_FPS = 10.0
MIN_VIDEO_FRAMES = int(MIN_VIDEO_FPS * WINDOW_SECONDS)
MAX_BOUNDARY_GAP_SEC = 0.20
WARN_MAX_FRAME_GAP_SEC = 0.15
FAIL_MAX_FRAME_GAP_SEC = 0.35


class CaptureError(RuntimeError):
    """Fatal Stage-02 acquisition or contract error."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise CaptureError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
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
            raise CaptureError(
                f"Could not atomically publish JSON: {path}"
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
        raise CaptureError(
            "OpenCV is unavailable in the active environment."
        ) from exc

    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise CaptureError(
            "python-sounddevice is unavailable in the active environment."
        ) from exc

    return cv2, sd


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"Expected JSON object: {path}")
    return data


def backend_code(cv2, name: str | None) -> tuple[str, int]:
    text = (name or "ANY").upper()
    if "DSHOW" in text and hasattr(cv2, "CAP_DSHOW"):
        return "DSHOW", int(cv2.CAP_DSHOW)
    if "MSMF" in text and hasattr(cv2, "CAP_MSMF"):
        return "MSMF", int(cv2.CAP_MSMF)
    return "ANY", int(cv2.CAP_ANY)


def open_camera(cv2, camera_cfg: dict[str, Any]):
    index = int(camera_cfg.get("index", 0))
    backend_name, backend = backend_code(cv2, camera_cfg.get("backend"))

    cap = cv2.VideoCapture(index, backend)
    require(cap.isOpened(), f"Could not open camera index {index} via {backend_name}")

    width = int(camera_cfg.get("requested_width") or camera_cfg.get("reported_width") or 1280)
    height = int(camera_cfg.get("requested_height") or camera_cfg.get("reported_height") or 720)
    fps = float(camera_cfg.get("requested_fps") or camera_cfg.get("measured_fps") or 20.0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    requested_fourcc = camera_cfg.get("requested_fourcc")
    if isinstance(requested_fourcc, str) and len(requested_fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))

    # Warm up and prove that frames are actually readable.
    first = None
    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            first = frame
            break
        time.sleep(0.03)
    require(first is not None, "Camera opened but produced no valid frame")

    try:
        actual_backend = cap.getBackendName()
    except Exception:
        actual_backend = backend_name

    info = {
        "index": index,
        "backend_requested": backend_name,
        "backend_actual": str(actual_backend),
        "reported_width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "reported_height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    return cap, info


@dataclass
class VideoFrame:
    timestamp: float
    frame: np.ndarray


class VideoCollector:
    """Continuously read camera frames in a dedicated thread."""

    def __init__(
        self,
        cap,
        *,
        retention_seconds: float,
    ):
        self.cap = cap
        self.retention_seconds = float(retention_seconds)
        self.frames: Deque[VideoFrame] = collections.deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.first_frame_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest: np.ndarray | None = None

        self.read_attempts = 0
        self.read_failures = 0
        self.thread_error: str | None = None

    def start(self) -> None:
        require(self.thread is None, "Video collector already started")
        self.thread = threading.Thread(
            target=self._run,
            name="EAV-LiveInteraction-VideoCollector",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.read_attempts += 1
                ok, frame = self.cap.read()
                ts = time.monotonic()

                if not ok or frame is None or not frame.size:
                    self.read_failures += 1
                    time.sleep(0.002)
                    continue

                item = VideoFrame(ts, frame.copy())
                with self.lock:
                    self.frames.append(item)
                    self.latest = item.frame
                    cutoff = ts - self.retention_seconds
                    while self.frames and self.frames[0].timestamp < cutoff:
                        self.frames.popleft()

                self.first_frame_event.set()

        except Exception as exc:
            self.thread_error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()

    def get_latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def window(self, start: float, end: float) -> list[VideoFrame]:
        with self.lock:
            return [
                VideoFrame(x.timestamp, x.frame.copy())
                for x in self.frames
                if start <= x.timestamp <= end
            ]

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        try:
            self.cap.release()
        except Exception:
            pass


@dataclass
class AudioBlock:
    start_index: int
    end_index: int
    predicted_start_monotonic: float
    observed_adc_start_monotonic: float | None
    data: np.ndarray


class AudioCollector:
    """Continuously acquire microphone blocks and maintain a sample-clock mapping."""

    def __init__(
        self,
        sd,
        *,
        device_index: int,
        samplerate: int,
        retention_seconds: float,
    ):
        self.sd = sd
        self.device_index = int(device_index)
        self.samplerate = int(samplerate)
        self.retention_seconds = float(retention_seconds)

        self.lock = threading.Lock()
        self.blocks: Deque[AudioBlock] = collections.deque()
        self.first_block_event = threading.Event()

        self.stream = None
        self.anchor_monotonic: float | None = None
        self.total_samples = 0

        self.callback_count = 0
        self.status_events = 0
        self.input_overflows = 0
        self.input_underflows = 0
        self.callback_errors = 0
        self.status_messages: list[str] = []
        self.error_messages: list[str] = []
        self.timing_residuals: list[float] = []

    @staticmethod
    def _time_field(time_info, name: str) -> float | None:
        try:
            value = getattr(time_info, name)
            value = float(value)
            return value if math.isfinite(value) else None
        except Exception:
            return None

    def _callback(self, indata, frames: int, time_info, status) -> None:
        try:
            now = time.monotonic()
            x = np.asarray(indata[:, 0], dtype=np.float32).copy()

            if status:
                self.status_events += 1
                text = str(status)
                if len(self.status_messages) < 50:
                    self.status_messages.append(text)
                try:
                    if status.input_overflow:
                        self.input_overflows += 1
                    if status.input_underflow:
                        self.input_underflows += 1
                except Exception:
                    pass

            pa_current = self._time_field(time_info, "currentTime")
            pa_adc = self._time_field(time_info, "inputBufferAdcTime")

            observed_adc_start = None
            if pa_current is not None and pa_adc is not None:
                # Map the PortAudio timing domain to the same host monotonic
                # clock used by video.  The mapping is sampled inside the
                # callback to avoid assuming the two APIs expose identical
                # epochs.
                observed_adc_start = pa_adc + (now - pa_current)

            with self.lock:
                start_index = self.total_samples
                end_index = start_index + int(x.shape[0])

                if self.anchor_monotonic is None:
                    if observed_adc_start is not None:
                        self.anchor_monotonic = observed_adc_start
                    else:
                        # Conservative fallback: callback arrival minus block
                        # duration.  This is less precise and will be reported.
                        self.anchor_monotonic = now - (len(x) / self.samplerate)

                predicted_start = (
                    self.anchor_monotonic + start_index / self.samplerate
                )

                if observed_adc_start is not None:
                    self.timing_residuals.append(
                        observed_adc_start - predicted_start
                    )
                    if len(self.timing_residuals) > 10000:
                        del self.timing_residuals[:5000]

                self.blocks.append(
                    AudioBlock(
                        start_index=start_index,
                        end_index=end_index,
                        predicted_start_monotonic=predicted_start,
                        observed_adc_start_monotonic=observed_adc_start,
                        data=x,
                    )
                )
                self.total_samples = end_index

                # Prune old audio while keeping enough history for the current
                # 5-second extraction.
                keep_samples = int(
                    math.ceil(self.retention_seconds * self.samplerate)
                )
                cutoff_index = max(0, self.total_samples - keep_samples)
                while self.blocks and self.blocks[0].end_index < cutoff_index:
                    self.blocks.popleft()

            self.callback_count += 1
            self.first_block_event.set()

        except Exception as exc:
            self.callback_errors += 1
            if len(self.error_messages) < 20:
                self.error_messages.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> float:
        self.sd.check_input_settings(
            device=self.device_index,
            channels=MODEL_AUDIO_CHANNELS,
            dtype="float32",
            samplerate=float(self.samplerate),
        )
        setup_start = time.monotonic()
        self.stream = self.sd.InputStream(
            device=self.device_index,
            samplerate=float(self.samplerate),
            channels=MODEL_AUDIO_CHANNELS,
            dtype="float32",
            blocksize=0,
            callback=self._callback,
        )
        self.stream.start()
        return time.monotonic() - setup_start

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass

    def _target_indices(self, start: float, seconds: float) -> tuple[int, int, float, float]:
        with self.lock:
            require(self.anchor_monotonic is not None, "Audio time anchor not available")
            anchor = float(self.anchor_monotonic)

        n = int(round(seconds * self.samplerate))
        start_index = int(round((start - anchor) * self.samplerate))
        end_index = start_index + n
        actual_start = anchor + start_index / self.samplerate
        actual_end = anchor + end_index / self.samplerate
        return start_index, end_index, actual_start, actual_end

    def has_until(self, end_time: float) -> bool:
        with self.lock:
            if self.anchor_monotonic is None:
                return False
            available_end = (
                self.anchor_monotonic + self.total_samples / self.samplerate
            )
        return available_end >= end_time

    def extract_exact(
        self,
        start: float,
        seconds: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        start_index, end_index, actual_start, actual_end = self._target_indices(
            start, seconds
        )
        require(start_index >= 0, "Requested audio window starts before audio stream anchor")

        n = end_index - start_index
        output = np.empty(n, dtype=np.float32)
        covered = np.zeros(n, dtype=np.bool_)

        with self.lock:
            blocks = list(self.blocks)
            anchor = float(self.anchor_monotonic)
            total_samples = int(self.total_samples)
            residuals = np.asarray(self.timing_residuals, dtype=np.float64)

        require(
            end_index <= total_samples,
            "Requested audio window has not been fully captured yet",
        )

        used_blocks = 0
        observed_starts = []
        for block in blocks:
            left = max(start_index, block.start_index)
            right = min(end_index, block.end_index)
            if right <= left:
                continue

            src_a = left - block.start_index
            src_b = right - block.start_index
            dst_a = left - start_index
            dst_b = right - start_index
            output[dst_a:dst_b] = block.data[src_a:src_b]
            covered[dst_a:dst_b] = True
            used_blocks += 1
            if block.observed_adc_start_monotonic is not None:
                observed_starts.append(block.observed_adc_start_monotonic)

        missing = int(np.count_nonzero(~covered))
        require(
            missing == 0,
            f"Audio window has {missing} missing samples; no padding/fabrication is allowed",
        )

        rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(output))) if output.size else 0.0

        return output, {
            "sample_rate": self.samplerate,
            "samples": int(output.size),
            "requested_start_monotonic": start,
            "requested_end_monotonic": start + seconds,
            "sample_clock_start_monotonic": actual_start,
            "sample_clock_end_monotonic": actual_end,
            "start_alignment_error_seconds": actual_start - start,
            "end_alignment_error_seconds": actual_end - (start + seconds),
            "used_callback_blocks": used_blocks,
            "missing_samples": missing,
            "rms_linear": rms,
            "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
            "peak_abs": peak,
            "clipping_samples": int(np.count_nonzero(np.abs(output) >= 0.999)),
            "anchor_monotonic": anchor,
            "callback_timing_residual_mean_ms": (
                float(np.mean(residuals) * 1000.0) if residuals.size else None
            ),
            "callback_timing_residual_std_ms": (
                float(np.std(residuals) * 1000.0) if residuals.size else None
            ),
            "callback_timing_residual_max_abs_ms": (
                float(np.max(np.abs(residuals)) * 1000.0)
                if residuals.size
                else None
            ),
        }


def save_pcm16_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    require(samples.ndim == 1, "Audio output must be mono")
    require(np.isfinite(samples).all(), "Audio contains NaN/Inf")
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(pcm.tobytes())


def read_wav_header(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        return {
            "channels": int(wf.getnchannels()),
            "sample_width_bytes": int(wf.getsampwidth()),
            "sample_rate": int(wf.getframerate()),
            "frames": int(wf.getnframes()),
            "duration_seconds": wf.getnframes() / wf.getframerate(),
        }


def resample_to_16k(samples: np.ndarray, source_sr: int) -> np.ndarray:
    if int(source_sr) == MODEL_AUDIO_SR:
        return samples.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly  # type: ignore
    except Exception as exc:
        raise CaptureError(
            f"Microphone capture rate is {source_sr} Hz, but scipy is unavailable "
            "for the required resampling to 16000 Hz."
        ) from exc

    g = math.gcd(int(source_sr), MODEL_AUDIO_SR)
    y = resample_poly(
        samples,
        MODEL_AUDIO_SR // g,
        int(source_sr) // g,
    ).astype(np.float32)

    expected = int(WINDOW_SECONDS * MODEL_AUDIO_SR)
    # resample_poly can differ by a sample due to rational endpoint rounding.
    # Trimming a possible extra sample is deterministic. Missing samples are
    # never padded.
    require(
        len(y) >= expected,
        f"Resampling produced too few samples ({len(y)} < {expected}); no padding allowed",
    )
    return y[:expected]


def frame_statistics(frames: list[VideoFrame], start: float, end: float) -> dict[str, Any]:
    require(frames, "No video frames selected")
    ts = np.asarray([f.timestamp for f in frames], dtype=np.float64)
    dt = np.diff(ts)

    first_offset = float(ts[0] - start)
    last_lag = float(end - ts[-1])
    physical_span = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
    measured_fps = (
        (len(ts) - 1) / physical_span
        if len(ts) >= 2 and physical_span > 0
        else None
    )

    return {
        "frames": len(frames),
        "first_frame_monotonic": float(ts[0]),
        "last_frame_monotonic": float(ts[-1]),
        "first_frame_offset_from_window_start_seconds": first_offset,
        "last_frame_lag_from_window_end_seconds": last_lag,
        "physical_frame_span_seconds": physical_span,
        "measured_fps_within_window": measured_fps,
        "mean_interval_ms": float(np.mean(dt) * 1000.0) if dt.size else None,
        "median_interval_ms": float(np.median(dt) * 1000.0) if dt.size else None,
        "p95_interval_ms": float(np.percentile(dt, 95) * 1000.0) if dt.size else None,
        "max_interval_ms": float(np.max(dt) * 1000.0) if dt.size else None,
    }


def write_video_exact_5s(
    cv2,
    directory: Path,
    frames: list[VideoFrame],
) -> tuple[Path, dict[str, Any]]:
    require(frames, "Cannot write empty video window")

    first_frame = frames[0].frame
    require(first_frame.ndim == 3 and first_frame.shape[2] in (3, 4), "Unexpected video frame shape")
    height, width = first_frame.shape[:2]

    for item in frames:
        require(
            item.frame.shape[:2] == (height, width),
            "Camera resolution changed inside one window",
        )

    # Constant-rate container duration is forced to exactly 5 seconds:
    # N frames / (N/5 fps) = 5 seconds.  The frame CONTENT is never fabricated;
    # every encoded frame was physically captured inside the target interval.
    writer_fps = len(frames) / WINDOW_SECONDS
    require(writer_fps >= MIN_VIDEO_FPS, "Video frame rate below Stage-02 minimum")

    attempts = [
        ("video.mp4", "mp4v"),
        ("video.avi", "MJPG"),
    ]

    errors = []
    selected: Path | None = None
    selected_codec = None

    for filename, codec in attempts:
        path = directory / filename
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(path),
                fourcc,
                float(writer_fps),
                (width, height),
            )
            if not writer.isOpened():
                errors.append(f"{codec}: VideoWriter did not open")
                try:
                    writer.release()
                except Exception:
                    pass
                continue

            for item in frames:
                frame = item.frame
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                writer.write(frame)
            writer.release()

            if path.is_file() and path.stat().st_size > 0:
                selected = path
                selected_codec = codec
                break
            errors.append(f"{codec}: output file missing/empty")
        except Exception as exc:
            errors.append(f"{codec}: {type(exc).__name__}: {exc}")
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    require(
        selected is not None,
        "Could not encode video window. Attempts: " + " | ".join(errors),
    )

    # Re-open and validate the completed immutable file.
    cap = cv2.VideoCapture(str(selected))
    require(cap.isOpened(), f"Encoded video cannot be reopened: {selected}")
    fps_read = float(cap.get(cv2.CAP_PROP_FPS))
    count_prop = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width_read = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height_read = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()

    require(decoded == len(frames), f"Video decode count mismatch: {decoded} != {len(frames)}")
    require(fps_read > 0, "Encoded video reports invalid FPS")

    duration = decoded / fps_read
    require(
        abs(duration - WINDOW_SECONDS) <= max(1.0 / fps_read + 0.01, 0.07),
        f"Encoded video is not a 5-second window: {duration:.6f}s",
    )

    return selected, {
        "codec": selected_codec,
        "file": str(selected.resolve()),
        "bytes": selected.stat().st_size,
        "writer_fps": writer_fps,
        "reported_fps_after_reopen": fps_read,
        "frame_count_property": count_prop,
        "decoded_frames": decoded,
        "duration_seconds_by_frame_count": duration,
        "width": width_read,
        "height": height_read,
        "duration_contract_pass": True,
    }


def wait_with_preview(
    cv2,
    video: VideoCollector,
    *,
    target_end: float,
    session_id: str,
    window_number: int,
    total_windows: int,
    preview: bool,
) -> bool:
    """Wait until target_end, showing latest camera frame. Returns user-stop."""
    window_name = "EAV LiveInteraction - Stage 02 synchronized capture"
    user_stop = False

    while True:
        now = time.monotonic()
        if now >= target_end:
            break

        if video.thread_error:
            raise CaptureError(f"Video acquisition thread failed: {video.thread_error}")

        if preview:
            frame = video.get_latest()
            if frame is not None:
                remaining = max(0.0, target_end - now)
                lines = [
                    "EAV LiveInteraction - Stage 02",
                    f"Session: {session_id}",
                    f"Window: {window_number}/{total_windows}",
                    f"Remaining: {remaining:4.1f} s",
                    "Camera + microphone are being captured continuously",
                    "No emotion inference in Stage 02",
                    "Q / ESC: stop after current wait",
                ]
                canvas = frame.copy()
                y = 28
                for line in lines:
                    cv2.putText(
                        canvas, line, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas, line, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 0, 0), 1, cv2.LINE_AA,
                    )
                    y += 26

                try:
                    cv2.imshow(window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        user_stop = True
                        break
                except Exception:
                    # Preview is not part of the capture contract.
                    preview = False

        time.sleep(0.003)

    return user_stop


def window_acceptance(
    video_stats: dict[str, Any],
    audio_stats: dict[str, Any],
    *,
    audio_overflows_delta: int,
    audio_callback_errors_delta: int,
) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    if int(video_stats["frames"]) < MIN_VIDEO_FRAMES:
        errors.append(
            f"VIDEO_TOO_FEW_FRAMES:{video_stats['frames']}<{MIN_VIDEO_FRAMES}"
        )

    first_offset = float(video_stats["first_frame_offset_from_window_start_seconds"])
    last_lag = float(video_stats["last_frame_lag_from_window_end_seconds"])

    if first_offset > MAX_BOUNDARY_GAP_SEC:
        errors.append(f"VIDEO_START_BOUNDARY_GAP:{first_offset:.6f}s")
    if last_lag > MAX_BOUNDARY_GAP_SEC:
        errors.append(f"VIDEO_END_BOUNDARY_GAP:{last_lag:.6f}s")

    max_gap_ms = video_stats.get("max_interval_ms")
    if max_gap_ms is not None:
        max_gap = float(max_gap_ms) / 1000.0
        if max_gap > FAIL_MAX_FRAME_GAP_SEC:
            errors.append(f"VIDEO_INTERNAL_GAP_TOO_LARGE:{max_gap:.6f}s")
        elif max_gap > WARN_MAX_FRAME_GAP_SEC:
            warnings.append(f"VIDEO_INTERNAL_GAP_WARNING:{max_gap:.6f}s")

    if int(audio_stats.get("missing_samples", 0)) != 0:
        errors.append("AUDIO_HAS_MISSING_SAMPLES")

    if int(audio_stats.get("samples", 0)) <= 0:
        errors.append("AUDIO_EMPTY")

    if abs(float(audio_stats.get("start_alignment_error_seconds", 99.0))) > 1.0 / MODEL_AUDIO_SR + 1e-6:
        errors.append("AUDIO_START_ALIGNMENT_ERROR")

    if abs(float(audio_stats.get("end_alignment_error_seconds", 99.0))) > 1.0 / MODEL_AUDIO_SR + 1e-6:
        errors.append("AUDIO_END_ALIGNMENT_ERROR")

    if audio_overflows_delta > 0:
        errors.append(f"AUDIO_INPUT_OVERFLOW_COUNT={audio_overflows_delta}")

    if audio_callback_errors_delta > 0:
        errors.append(f"AUDIO_CALLBACK_ERROR_COUNT={audio_callback_errors_delta}")

    if float(audio_stats.get("rms_dbfs", -999.0)) < -75.0:
        warnings.append("AUDIO_LEVEL_VERY_LOW")

    if errors:
        return "FAIL", warnings, errors
    if warnings:
        return "PASS_WITH_WARNINGS", warnings, errors
    return "PASS", warnings, errors


def descriptor_for_window(
    *,
    session_id: str,
    clock_id: str,
    window_id: str,
    window_start: float,
    window_end: float,
    audio_path: Path,
    video_path: Path,
    audio_newest: float,
    video_newest: float,
    audio_capture_rate: int,
) -> dict[str, Any]:
    return {
        "schema": SOURCE_SCHEMA,
        "window_id": window_id,
        "window_seconds": WINDOW_SECONDS,
        "task_condition": "Speaking",
        "split": "live",
        "session_id": session_id,
        "clock_id": clock_id,
        "window_end_monotonic": window_end,
        "identity": {
            "capture_source": "laptop_builtin_av",
            "window_idx_0based": int(window_id.rsplit("_", 1)[-1]) - 1,
        },
        "modalities": {
            "eeg": {
                "present": False,
                "reason": "EEG_DEVICE_NOT_CONNECTED",
            },
            "audio": {
                "present": True,
                "path": str(audio_path.resolve()),
                "kind": "audio_window",
                "sample_rate": MODEL_AUDIO_SR,
                "capture_sample_rate": int(audio_capture_rate),
                "channels": 1,
                "continuous": True,
                "capture_ok": True,
                "timing": {
                    "clock_id": clock_id,
                    "window_start_monotonic": window_start,
                    "window_end_monotonic": window_end,
                    "newest_sample_monotonic": audio_newest,
                },
            },
            "video": {
                "present": True,
                "path": str(video_path.resolve()),
                "kind": "video_window",
                "continuous": True,
                "capture_ok": True,
                "timing": {
                    "clock_id": clock_id,
                    "window_start_monotonic": window_start,
                    "window_end_monotonic": window_end,
                    "newest_sample_monotonic": video_newest,
                },
            },
        },
    }


def validate_main_live_descriptor_shape(d: dict[str, Any]) -> None:
    """Local structural mirror of the frozen main.py live-source contract."""
    require(d.get("schema") == SOURCE_SCHEMA, "Descriptor schema mismatch")
    require(d.get("window_seconds") == 5.0, "Descriptor window_seconds must be 5.0")
    require(str(d.get("task_condition", "")).lower() == "speaking", "Descriptor must be Speaking")
    require(d.get("split") == "live", "Descriptor split must be live")
    require(isinstance(d.get("session_id"), str) and d["session_id"].strip(), "Missing session_id")
    require(isinstance(d.get("clock_id"), str) and d["clock_id"].strip(), "Missing clock_id")
    require(
        set(d.get("modalities", {})) == {"eeg", "audio", "video"},
        "Descriptor must contain exactly EEG/Audio/Video",
    )

    end = float(d["window_end_monotonic"])
    for name in ("audio", "video"):
        item = d["modalities"][name]
        require(item.get("present") is True, f"{name} must be present")
        require(bool(item.get("path")), f"{name} path missing")
        timing = item.get("timing")
        require(isinstance(timing, dict), f"{name} timing missing")
        require(timing.get("clock_id") == d["clock_id"], f"{name} clock mismatch")
        require(
            float(timing["window_start_monotonic"]) == end - WINDOW_SECONDS,
            f"{name} start span mismatch",
        )
        require(
            float(timing["window_end_monotonic"]) == end,
            f"{name} end span mismatch",
        )
        newest = float(timing["newest_sample_monotonic"])
        require(
            end - WINDOW_SECONDS <= newest <= end + 0.02,
            f"{name} newest sample outside main.py contract",
        )

    require(d["modalities"]["eeg"].get("present") is False, "EEG must be explicitly unavailable")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Capture one or more real synchronized 5-second laptop Audio+Video "
            "windows. No emotion inference is performed."
        )
    )
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Stage-01 detected device config. Default: "
            "MERIT-Realtime Model/config/detected_av_devices.json"
        ),
    )
    p.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_WINDOWS,
        help=f"Number of consecutive 5-second windows (default: {DEFAULT_WINDOWS}).",
    )
    p.add_argument(
        "--lead-sec",
        type=float,
        default=DEFAULT_LEAD_SECONDS,
        help=f"Future lead before first T0 after streams are ready (default: {DEFAULT_LEAD_SECONDS}).",
    )
    p.add_argument(
        "--settle-sec",
        type=float,
        default=DEFAULT_SETTLE_SECONDS,
        help=f"Wait after T1 so the final callbacks/frames arrive (default: {DEFAULT_SETTLE_SECONDS}).",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Explicit session output directory. Must not already exist.",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="Explicit session id. Default: generated sync_<timestamp>.",
    )
    p.add_argument(
        "--clock-id",
        default=None,
        help="Explicit host clock id. Default: python_monotonic_<session-id>.",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live camera preview.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require(1 <= args.windows <= 100, "--windows must be in [1, 100]")
    require(0.10 <= args.lead_sec <= 5.0, "--lead-sec must be in [0.10, 5.0]")
    require(0.02 <= args.settle_sec <= 1.0, "--settle-sec must be in [0.02, 1.0]")

    cv2, sd = load_dependencies()

    script_dir = Path(__file__).resolve().parent
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else script_dir / "config" / "detected_av_devices.json"
    )
    cfg = read_json(config_path)

    require(
        cfg.get("schema") == "eav.live_interaction.detected_av_devices.v1",
        "Unexpected Stage-01 device-config schema",
    )
    require(
        cfg.get("probe_status") in ("PASS", "PASS_WITH_WARNINGS"),
        "Stage-01 device probe has not passed",
    )

    camera_cfg = cfg.get("camera")
    audio_cfg = cfg.get("audio")
    require(isinstance(camera_cfg, dict), "Missing camera configuration")
    require(isinstance(audio_cfg, dict), "Missing audio configuration")

    audio_device_index = int(audio_cfg["device_index"])
    capture_sr = int(round(float(audio_cfg["capture_samplerate"])))
    require(capture_sr >= 8000, "Invalid microphone capture sample rate")

    session_id = args.session_id or f"sync_{tag_now()}"
    require(session_id.strip() == session_id and session_id, "Invalid session-id")
    clock_id = args.clock_id or f"python_monotonic_{session_id}"
    require(clock_id.strip() == clock_id and clock_id, "Invalid clock-id")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else script_dir / "captures" / f"sync_capture_{tag_now()}"
    )
    require(not output_root.exists(), f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    print("=" * 100)
    print("EAV LiveInteraction — Stage 02 synchronized AV window capture")
    print("=" * 100)
    print(f"Version                 : {VERSION}")
    print(f"Python                  : {sys.version.split()[0]}")
    print(f"Platform                : {platform.platform()}")
    print(f"Device config           : {config_path}")
    print(f"Session ID              : {session_id}")
    print(f"Clock ID                : {clock_id}")
    print(f"Windows planned         : {args.windows}")
    print(f"Window seconds          : {WINDOW_SECONDS:.1f}")
    print(f"Audio model contract    : {MODEL_AUDIO_SR} Hz mono / 80000 samples")
    print("EEG                     : UNAVAILABLE")
    print("Emotion inference       : DISABLED")
    print("Quality/Fusion          : DISABLED")
    print(f"Output                  : {output_root}")
    print("=" * 100)

    cap = None
    video: VideoCollector | None = None
    audio: AudioCollector | None = None
    summary_windows: list[dict[str, Any]] = []
    session_status = "FAIL"
    fatal_error: str | None = None
    audio_startup_seconds = None
    user_stopped = False

    try:
        cap, camera_info = open_camera(cv2, camera_cfg)
        retention = max(
            DEFAULT_RETENTION_SECONDS,
            WINDOW_SECONDS + args.settle_sec + 1.5,
        )

        video = VideoCollector(cap, retention_seconds=retention)
        audio = AudioCollector(
            sd,
            device_index=audio_device_index,
            samplerate=capture_sr,
            retention_seconds=retention,
        )

        video.start()
        audio_startup_seconds = audio.start()

        require(
            video.first_frame_event.wait(timeout=5.0),
            "No camera frame arrived after collector start",
        )
        require(
            audio.first_block_event.wait(timeout=5.0),
            "No microphone block arrived after collector start",
        )

        # Ensure the first requested T0 lies safely after both streams have
        # established their clocks and buffers.
        first_start = time.monotonic() + float(args.lead_sec)

        print()
        print(f"Camera                  : index {camera_info['index']} / {camera_info['backend_actual']}")
        print(
            f"Camera format           : {camera_info['reported_width']} x "
            f"{camera_info['reported_height']} / reported {camera_info['reported_fps']:.3f} FPS"
        )
        print(
            f"Microphone              : device {audio_device_index} / "
            f"{audio_cfg.get('device_name')}"
        )
        print(f"Capture sample rate     : {capture_sr} Hz")
        print(f"Audio stream startup    : {audio_startup_seconds:.3f} s")
        print()
        print("Continuous capture started.")
        if not args.no_preview:
            print("Press Q or ESC to stop after the current wait.")
        print()

        for idx in range(args.windows):
            window_number = idx + 1
            window_id = f"window_{window_number:06d}"
            t0 = first_start + idx * WINDOW_SECONDS
            t1 = t0 + WINDOW_SECONDS

            # Snapshot counters at window start for integrity deltas.
            start_overflows = audio.input_overflows
            start_callback_errors = audio.callback_errors

            # Wait until T0 if needed.
            while time.monotonic() < t0:
                if video.thread_error:
                    raise CaptureError(f"Video thread failed: {video.thread_error}")
                if not args.no_preview:
                    frame = video.get_latest()
                    if frame is not None:
                        remaining = t0 - time.monotonic()
                        canvas = frame.copy()
                        text = f"Next synchronized window starts in {max(0.0, remaining):.1f}s"
                        cv2.putText(
                            canvas, text, (14, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                            (255, 255, 255), 3, cv2.LINE_AA,
                        )
                        cv2.putText(
                            canvas, text, (14, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                            (0, 0, 0), 1, cv2.LINE_AA,
                        )
                        try:
                            cv2.imshow(
                                "EAV LiveInteraction - Stage 02 synchronized capture",
                                canvas,
                            )
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord("q"), ord("Q"), 27):
                                user_stopped = True
                                break
                        except Exception:
                            pass
                time.sleep(0.003)

            if user_stopped:
                break

            print(
                f"[{window_number}/{args.windows}] CAPTURING {window_id} | "
                f"T0={t0:.6f}  T1={t1:.6f}"
            )

            user_stopped = wait_with_preview(
                cv2,
                video,
                target_end=t1,
                session_id=session_id,
                window_number=window_number,
                total_windows=args.windows,
                preview=not args.no_preview,
            )
            if user_stopped:
                print("User requested stop. Current window is not published.")
                break

            # Let final callback/frame delivery settle. This does not extend the
            # target source span; selection remains strictly [T0, T1].
            deadline = t1 + args.settle_sec
            while time.monotonic() < deadline or not audio.has_until(t1):
                if time.monotonic() > t1 + 1.5:
                    raise CaptureError(
                        f"Audio did not become available through T1 for {window_id}"
                    )
                time.sleep(0.005)

            frames = video.window(t0, t1)
            video_stats = frame_statistics(frames, t0, t1)

            audio_raw, audio_stats_capture = audio.extract_exact(
                t0, WINDOW_SECONDS
            )
            audio_model = resample_to_16k(audio_raw, capture_sr)

            require(
                len(audio_model) == int(MODEL_AUDIO_SR * WINDOW_SECONDS),
                "Model audio window is not exactly 80000 samples",
            )

            audio_overflows_delta = audio.input_overflows - start_overflows
            callback_errors_delta = audio.callback_errors - start_callback_errors

            status, warnings, errors = window_acceptance(
                video_stats,
                audio_stats_capture,
                audio_overflows_delta=audio_overflows_delta,
                audio_callback_errors_delta=callback_errors_delta,
            )

            window_dir = output_root / window_id
            window_dir.mkdir(parents=True, exist_ok=False)

            metadata = {
                "schema": METADATA_SCHEMA,
                "version": VERSION,
                "status": status,
                "session_id": session_id,
                "clock_id": clock_id,
                "window_id": window_id,
                "window_seconds": WINDOW_SECONDS,
                "window_start_monotonic": t0,
                "window_end_monotonic": t1,
                "camera": {
                    **camera_info,
                    **video_stats,
                    "timestamps_monotonic": [x.timestamp for x in frames],
                    "collector_read_attempts_total": video.read_attempts,
                    "collector_read_failures_total": video.read_failures,
                },
                "audio": {
                    **audio_stats_capture,
                    "capture_sample_rate": capture_sr,
                    "model_sample_rate": MODEL_AUDIO_SR,
                    "model_samples": int(audio_model.size),
                    "resampled": capture_sr != MODEL_AUDIO_SR,
                    "input_overflows_in_window": audio_overflows_delta,
                    "callback_errors_in_window": callback_errors_delta,
                    "callback_status_events_total": audio.status_events,
                    "callback_status_messages": audio.status_messages,
                },
                "warnings": warnings,
                "errors": errors,
                "emotion_inference_performed": False,
                "quality_layer_invoked": False,
                "fusion_invoked": False,
            }

            # A failed capture may keep metadata for diagnosis, but must never
            # masquerade as a valid live source descriptor.
            if status == "FAIL":
                atomic_write_json(window_dir / "capture_metadata.json", metadata)
                summary_windows.append(
                    {
                        "window_id": window_id,
                        "status": status,
                        "warnings": warnings,
                        "errors": errors,
                        "window_dir": str(window_dir),
                    }
                )
                print(f"  -> FAIL | {'; '.join(errors)}")
                continue

            audio_path = window_dir / "audio.wav"
            save_pcm16_wav(audio_path, audio_model, MODEL_AUDIO_SR)
            wav_info = read_wav_header(audio_path)
            require(
                wav_info["channels"] == 1
                and wav_info["sample_rate"] == MODEL_AUDIO_SR
                and wav_info["frames"] == MODEL_AUDIO_SR * int(WINDOW_SECONDS),
                "Saved WAV does not satisfy 16 kHz mono 5-second contract",
            )

            video_path, encoded_video = write_video_exact_5s(
                cv2, window_dir, frames
            )

            # Audio sample-clock end is aligned to T1 to within <= 1 source
            # sample. For the main live timing packet use the decision window
            # endpoint itself after this check has passed.
            audio_newest = t1
            video_newest = float(video_stats["last_frame_monotonic"])

            descriptor = descriptor_for_window(
                session_id=session_id,
                clock_id=clock_id,
                window_id=window_id,
                window_start=t0,
                window_end=t1,
                audio_path=audio_path,
                video_path=video_path,
                audio_newest=audio_newest,
                video_newest=video_newest,
                audio_capture_rate=capture_sr,
            )
            validate_main_live_descriptor_shape(descriptor)

            metadata["saved_audio"] = {
                "file": str(audio_path.resolve()),
                "bytes": audio_path.stat().st_size,
                **wav_info,
            }
            metadata["saved_video"] = encoded_video
            metadata["descriptor_main_live_contract_pass"] = True

            # Commit media first, then metadata, descriptor LAST. A later live
            # consumer that sees descriptor.json can therefore trust that both
            # referenced immutable files already exist and are complete.
            atomic_write_json(window_dir / "capture_metadata.json", metadata)
            atomic_write_json(window_dir / "descriptor.json", descriptor)

            summary_windows.append(
                {
                    "window_id": window_id,
                    "status": status,
                    "warnings": warnings,
                    "errors": errors,
                    "window_start_monotonic": t0,
                    "window_end_monotonic": t1,
                    "video_frames": video_stats["frames"],
                    "video_measured_fps": video_stats["measured_fps_within_window"],
                    "video_max_gap_ms": video_stats["max_interval_ms"],
                    "audio_capture_samples": audio_stats_capture["samples"],
                    "audio_model_samples": int(audio_model.size),
                    "audio_rms_dbfs": audio_stats_capture["rms_dbfs"],
                    "descriptor": str((window_dir / "descriptor.json").resolve()),
                    "audio_file": str(audio_path.resolve()),
                    "video_file": str(video_path.resolve()),
                }
            )

            print(
                f"  -> {status} | video={video_stats['frames']} frames "
                f"({video_stats['measured_fps_within_window']:.2f} FPS) | "
                f"audio={audio_model.size} samples | "
                f"RMS={audio_stats_capture['rms_dbfs']:.2f} dBFS"
            )
            if warnings:
                for warning in warnings:
                    print(f"     warning: {warning}")

        published = [
            x for x in summary_windows
            if x["status"] in ("PASS", "PASS_WITH_WARNINGS")
        ]
        failed = [x for x in summary_windows if x["status"] == "FAIL"]

        if user_stopped and not published:
            session_status = "STOPPED_NO_VALID_WINDOW"
        elif failed:
            session_status = "COMPLETE_WITH_FAILED_WINDOWS"
        elif len(published) == args.windows:
            session_status = "PASS"
        elif published:
            session_status = "STOPPED_WITH_VALID_WINDOWS"
        else:
            session_status = "FAIL"

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        session_status = "FAIL"
        print(f"\nCAPTURE ERROR: {fatal_error}", file=sys.stderr)

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

    valid_count = sum(
        x["status"] in ("PASS", "PASS_WITH_WARNINGS")
        for x in summary_windows
    )
    failed_count = sum(x["status"] == "FAIL" for x in summary_windows)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "version": VERSION,
        "status": session_status,
        "created_utc": utc_now(),
        "session_id": session_id,
        "clock_id": clock_id,
        "device_config": str(config_path),
        "window_seconds": WINDOW_SECONDS,
        "planned_windows": args.windows,
        "attempted_windows": len(summary_windows),
        "valid_windows": valid_count,
        "failed_windows": failed_count,
        "user_stopped": user_stopped,
        "fatal_error": fatal_error,
        "audio_stream_startup_seconds": audio_startup_seconds,
        "shared_clock": "time.monotonic()",
        "eeg": {
            "present": False,
            "reason": "EEG_DEVICE_NOT_CONNECTED",
        },
        "emotion_inference_performed": False,
        "quality_layer_invoked": False,
        "fusion_invoked": False,
        "robot_action_performed": False,
        "main_system_modified": False,
        "windows": summary_windows,
        "next_stage": (
            "03_live_av_inference.py"
            if session_status in ("PASS", "STOPPED_WITH_VALID_WINDOWS")
            else "REVIEW_STAGE_02_CAPTURE_INTEGRITY"
        ),
    }

    atomic_write_json(output_root / "session_summary.json", summary)

    print()
    print("=" * 100)
    print("STAGE 02 SYNCHRONIZED AV CAPTURE SUMMARY")
    print("=" * 100)
    print(f"Status                  : {session_status}")
    print(f"Planned windows         : {args.windows}")
    print(f"Attempted windows       : {len(summary_windows)}")
    print(f"Valid windows           : {valid_count}")
    print(f"Failed windows          : {failed_count}")
    print(f"Window duration         : {WINDOW_SECONDS:.1f} s")
    print(f"Audio contract          : {MODEL_AUDIO_SR} Hz mono / 80000 samples")
    print(f"EEG                     : UNAVAILABLE")
    print(f"Emotion inference       : False")
    print(f"Output                  : {output_root}")
    print(f"Summary                 : {output_root / 'session_summary.json'}")
    if fatal_error:
        print(f"Fatal error             : {fatal_error}")
    print("=" * 100)

    return 0 if session_status in ("PASS", "STOPPED_WITH_VALID_WINDOWS") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except CaptureError as exc:
        print(f"\nCAPTURE ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
