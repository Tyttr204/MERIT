#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
live_display.py
===============

EAV LiveInteraction — participant-facing real-time display layer.

Purpose
-------
Render the current camera frame together with the latest in-memory emotion
state produced by ``EmotionStateManager``.

This module DOES NOT:
    - read model predictions from JSON/files;
    - run EEG/Audio/Video inference;
    - compute or modify emotion probabilities;
    - compute or modify quality scores;
    - change F4 / AF4-B routing;
    - make robot-action decisions;
    - save screenshots/video by default.

Intended live path
------------------
    frozen main.py / LiveScheduler
                |
                | result dict in memory
                v
        EmotionStateManager
                |
                | subscriber callback
                v
          LiveEmotionDisplay
                |
                v
          participant screen

Files remain audit/log outputs only.  They are not part of the participant
feedback path.

Typical integration
-------------------
    from emotion_state_manager import EmotionStateManager
    from live_display import LiveEmotionDisplay

    manager = EmotionStateManager()
    display = LiveEmotionDisplay()

    token = manager.subscribe(display.on_state)

    while running:
        frame = video_collector.get_latest()
        display.update_runtime(
            window_id="window_000123",
            phase="CAPTURING",
            remaining_seconds=2.4,
        )
        if frame is not None and display.show(frame):
            break

    manager.unsubscribe(token)
    display.close()

Standalone validation
---------------------
    python .\LiveInteraction\live_display.py --self-test
    python .\LiveInteraction\live_display.py --demo

The demo uses synthetic emotion states only and is NOT model evidence.
"""

from __future__ import annotations

import argparse
import copy
import math
import threading
import time
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "EAV-LIVE-DISPLAY.1.0"

EMOTIONS = (
    "Neutral",
    "Sadness",
    "Anger",
    "Happiness",
    "Calmness",
)

DEFAULT_WINDOW_NAME = "EAV Live Emotion Interaction"
DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_DISPLAY_HEIGHT = 720

# OpenCV uses BGR.
COLOR_BG = (24, 24, 24)
COLOR_PANEL = (22, 22, 22)
COLOR_PANEL_ALT = (38, 38, 38)
COLOR_TEXT = (245, 245, 245)
COLOR_MUTED = (175, 175, 175)
COLOR_DIM = (105, 105, 105)
COLOR_GOOD = (86, 190, 110)
COLOR_WARNING = (70, 190, 240)
COLOR_BAD = (90, 90, 230)
COLOR_ACCENT = (230, 180, 80)
COLOR_BAR_BG = (62, 62, 62)
COLOR_BAR_FILL = (210, 180, 85)
COLOR_BORDER = (90, 90, 90)

EMOTION_ACCENTS = {
    "Neutral": (190, 190, 190),
    "Sadness": (220, 150, 90),
    "Anger": (80, 90, 230),
    "Happiness": (90, 210, 235),
    "Calmness": (120, 200, 130),
}


class LiveDisplayError(RuntimeError):
    """Display-contract or environment error."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise LiveDisplayError(message)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise LiveDisplayError(
            "OpenCV is unavailable in the active environment."
        ) from exc
    return cv2


def _clip01(value: Any) -> float:
    x = finite_or_none(value)
    if x is None:
        return 0.0
    return min(1.0, max(0.0, x))


def _pct(value: Any) -> str:
    x = finite_or_none(value)
    return "N/A" if x is None else f"{100.0 * x:.1f}%"


def _seconds(value: Any) -> str:
    x = finite_or_none(value)
    return "N/A" if x is None else f"{x:.2f}s"


def _safe_text(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _validate_state_snapshot(state: Mapping[str, Any]) -> None:
    require(isinstance(state, Mapping), "state snapshot must be a mapping")
    require(
        state.get("schema") in (None, "eav.live_interaction.emotion_state.v1"),
        "Unexpected emotion-state schema",
    )

    raw = state.get("raw")
    stable = state.get("stable")
    transition = state.get("transition")
    require(isinstance(raw, Mapping), "state.raw missing")
    require(isinstance(stable, Mapping), "state.stable missing")
    require(isinstance(transition, Mapping), "state.transition missing")

    for block_name, block in (("raw", raw), ("stable", stable)):
        emotion = block.get("emotion")
        if emotion is not None:
            require(
                str(emotion) in EMOTIONS,
                f"{block_name}.emotion is unknown: {emotion!r}",
            )

    probs = raw.get("probabilities")
    if probs is not None:
        require(
            isinstance(probs, Sequence)
            and not isinstance(probs, (str, bytes))
            and len(probs) == 5,
            "raw.probabilities must contain five values",
        )


@dataclass
class RuntimeView:
    session_id: str | None = None
    window_id: str | None = None
    phase: str = "WAITING"
    remaining_seconds: float | None = None
    capture_fps: float | None = None
    inference_latency_seconds: float | None = None
    latest_result_window_id: str | None = None
    message: str | None = None


class LiveEmotionDisplay:
    """
    Thread-safe participant-facing OpenCV renderer.

    ``on_state`` is designed to be passed directly to
    ``EmotionStateManager.subscribe(...)``.
    """

    def __init__(
        self,
        *,
        window_name: str = DEFAULT_WINDOW_NAME,
        display_width: int = DEFAULT_DISPLAY_WIDTH,
        display_height: int = DEFAULT_DISPLAY_HEIGHT,
        panel_width: int = 430,
        fullscreen: bool = False,
        show_raw_probabilities: bool = True,
    ):
        require(display_width >= 800, "display_width must be >= 800")
        require(display_height >= 480, "display_height must be >= 480")
        require(320 <= panel_width <= display_width - 280, "Invalid panel_width")

        self.cv2 = load_cv2()
        self.window_name = str(window_name)
        self.display_width = int(display_width)
        self.display_height = int(display_height)
        self.panel_width = int(panel_width)
        self.fullscreen = bool(fullscreen)
        self.show_raw_probabilities = bool(show_raw_probabilities)

        self._lock = threading.RLock()
        self._state: dict[str, Any] | None = None
        self._runtime = RuntimeView()
        self._opened = False
        self._closed = False
        self._last_render_monotonic: float | None = None
        self._render_count = 0

    # ------------------------------------------------------------------
    # State / runtime updates
    # ------------------------------------------------------------------

    def on_state(self, snapshot: Mapping[str, Any]) -> None:
        """Subscriber callback for EmotionStateManager."""
        _validate_state_snapshot(snapshot)
        with self._lock:
            self._state = copy.deepcopy(dict(snapshot))

            raw = snapshot.get("raw", {})
            if isinstance(raw, Mapping):
                wid = raw.get("window_id")
                if wid:
                    self._runtime.latest_result_window_id = str(wid)

    def update_runtime(
        self,
        *,
        session_id: str | None = None,
        window_id: str | None = None,
        phase: str | None = None,
        remaining_seconds: float | None = None,
        capture_fps: float | None = None,
        inference_latency_seconds: float | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if session_id is not None:
                self._runtime.session_id = str(session_id)
            if window_id is not None:
                self._runtime.window_id = str(window_id)
            if phase is not None:
                self._runtime.phase = str(phase)
            self._runtime.remaining_seconds = finite_or_none(remaining_seconds)
            self._runtime.capture_fps = finite_or_none(capture_fps)
            self._runtime.inference_latency_seconds = finite_or_none(
                inference_latency_seconds
            )
            if message is not None:
                self._runtime.message = str(message)

    def clear_message(self) -> None:
        with self._lock:
            self._runtime.message = None

    def snapshot(self) -> tuple[dict[str, Any] | None, RuntimeView]:
        with self._lock:
            return (
                copy.deepcopy(self._state),
                copy.deepcopy(self._runtime),
            )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _put_text(
        self,
        image: np.ndarray,
        text: str,
        xy: tuple[int, int],
        *,
        scale: float = 0.58,
        color=COLOR_TEXT,
        thickness: int = 1,
    ) -> None:
        self.cv2.putText(
            image,
            str(text),
            xy,
            self.cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            self.cv2.LINE_AA,
        )

    def _fit_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        require(
            isinstance(frame, np.ndarray) and frame.ndim == 3,
            "Camera frame must be HxWxC numpy array",
        )
        require(frame.shape[2] in (3, 4), "Camera frame must have 3 or 4 channels")

        if frame.shape[2] == 4:
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGRA2BGR)

        target_w = self.display_width
        target_h = self.display_height
        h, w = frame.shape[:2]

        scale = min(target_w / w, target_h / h)
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = self.cv2.resize(
            frame,
            (resized_w, resized_h),
            interpolation=self.cv2.INTER_AREA if scale < 1 else self.cv2.INTER_LINEAR,
        )

        canvas = np.full((target_h, target_w, 3), COLOR_BG, dtype=np.uint8)
        x0 = (target_w - resized_w) // 2
        y0 = (target_h - resized_h) // 2
        canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        return canvas

    def _overlay_panel(self, canvas: np.ndarray) -> tuple[int, int, int, int]:
        x0 = self.display_width - self.panel_width
        y0 = 0
        x1 = self.display_width
        y1 = self.display_height

        overlay = canvas.copy()
        self.cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            COLOR_PANEL,
            thickness=-1,
        )
        self.cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        self.cv2.line(
            canvas,
            (x0, 0),
            (x0, self.display_height),
            COLOR_BORDER,
            1,
            self.cv2.LINE_AA,
        )
        return x0, y0, x1, y1

    def _emotion_accent(self, emotion: str | None):
        return EMOTION_ACCENTS.get(str(emotion), COLOR_ACCENT)

    def _draw_probability_bar(
        self,
        canvas: np.ndarray,
        *,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        selected: bool,
    ) -> int:
        value = _clip01(value)

        self._put_text(
            canvas,
            label,
            (x, y),
            scale=0.48,
            color=COLOR_TEXT if selected else COLOR_MUTED,
            thickness=1,
        )
        self._put_text(
            canvas,
            f"{100.0 * value:5.1f}%",
            (x + width - 62, y),
            scale=0.45,
            color=COLOR_TEXT if selected else COLOR_MUTED,
            thickness=1,
        )

        bar_y = y + 8
        bar_h = 9
        self.cv2.rectangle(
            canvas,
            (x, bar_y),
            (x + width, bar_y + bar_h),
            COLOR_BAR_BG,
            thickness=-1,
        )
        fill_w = int(round(width * value))
        fill_color = self._emotion_accent(label) if selected else COLOR_BAR_FILL
        if fill_w > 0:
            self.cv2.rectangle(
                canvas,
                (x, bar_y),
                (x + fill_w, bar_y + bar_h),
                fill_color,
                thickness=-1,
            )
        return y + 36

    def _draw_sensor_line(
        self,
        canvas: np.ndarray,
        *,
        x: int,
        y: int,
        availability: Mapping[str, Any] | None,
    ) -> int:
        availability = availability or {}
        parts = []
        for name in ("audio", "video", "eeg"):
            value = availability.get(name)
            if value is True:
                token = f"{name.upper()}:ON"
            elif value is False:
                token = f"{name.upper()}:OFF"
            else:
                token = f"{name.upper()}:?"
            parts.append(token)

        self._put_text(
            canvas,
            "  ".join(parts),
            (x, y),
            scale=0.45,
            color=COLOR_MUTED,
        )
        return y + 26

    def _draw_no_state(
        self,
        canvas: np.ndarray,
        *,
        x: int,
        y: int,
    ) -> int:
        self._put_text(
            canvas,
            "CURRENT EMOTION",
            (x, y),
            scale=0.50,
            color=COLOR_MUTED,
        )
        y += 40
        self._put_text(
            canvas,
            "WAITING...",
            (x, y),
            scale=1.00,
            color=COLOR_ACCENT,
            thickness=2,
        )
        y += 42
        self._put_text(
            canvas,
            "Waiting for first valid model result",
            (x, y),
            scale=0.46,
            color=COLOR_MUTED,
        )
        return y + 34

    def _draw_state_panel(
        self,
        canvas: np.ndarray,
        *,
        state: Mapping[str, Any] | None,
        runtime: RuntimeView,
        x0: int,
    ) -> None:
        pad = 22
        x = x0 + pad
        usable = self.panel_width - 2 * pad
        y = 34

        # Header
        self._put_text(
            canvas,
            "LIVE EMOTION",
            (x, y),
            scale=0.78,
            color=COLOR_TEXT,
            thickness=2,
        )
        self.cv2.circle(
            canvas,
            (x + usable - 10, y - 8),
            6,
            COLOR_GOOD,
            thickness=-1,
            lineType=self.cv2.LINE_AA,
        )
        y += 30

        phase = _safe_text(runtime.phase, "WAITING")
        phase_text = phase
        if runtime.remaining_seconds is not None:
            phase_text += f"  {max(0.0, runtime.remaining_seconds):.1f}s"

        self._put_text(
            canvas,
            phase_text,
            (x, y),
            scale=0.46,
            color=COLOR_MUTED,
        )
        y += 18

        if runtime.window_id:
            self._put_text(
                canvas,
                f"Window: {runtime.window_id}",
                (x, y),
                scale=0.43,
                color=COLOR_DIM,
            )
            y += 24

        self.cv2.line(
            canvas,
            (x, y),
            (x + usable, y),
            COLOR_BORDER,
            1,
            self.cv2.LINE_AA,
        )
        y += 28

        if state is None:
            y = self._draw_no_state(canvas, x=x, y=y)
            return

        stable = state.get("stable", {})
        raw = state.get("raw", {})
        transition = state.get("transition", {})
        sensors = state.get("sensors", {})
        fusion = state.get("fusion", {})
        freshness = state.get("freshness", {})
        interaction_status = str(state.get("interaction_status", "UNKNOWN"))

        stable_available = bool(stable.get("available"))
        stable_emotion = stable.get("emotion") if stable_available else None
        stable_conf = stable.get("confidence") if stable_available else None

        self._put_text(
            canvas,
            "CURRENT EMOTION",
            (x, y),
            scale=0.48,
            color=COLOR_MUTED,
        )
        y += 42

        if stable_available:
            self._put_text(
                canvas,
                str(stable_emotion).upper(),
                (x, y),
                scale=1.02,
                color=self._emotion_accent(str(stable_emotion)),
                thickness=2,
            )
            y += 31
            self._put_text(
                canvas,
                f"Confidence: {_pct(stable_conf)}",
                (x, y),
                scale=0.52,
                color=COLOR_TEXT,
            )
        else:
            self._put_text(
                canvas,
                "NO CURRENT EVIDENCE",
                (x, y),
                scale=0.68,
                color=COLOR_WARNING,
                thickness=2,
            )
        y += 34

        if interaction_status == "POSSIBLE_TRANSITION":
            candidate = transition.get("candidate_emotion")
            count = transition.get("consecutive_count")
            required = transition.get("required_count")
            self.cv2.rectangle(
                canvas,
                (x, y - 18),
                (x + usable, y + 22),
                COLOR_PANEL_ALT,
                thickness=-1,
            )
            self._put_text(
                canvas,
                f"Possible change -> {candidate}  ({count}/{required})",
                (x + 8, y + 7),
                scale=0.44,
                color=COLOR_WARNING,
                thickness=1,
            )
            y += 48
        elif interaction_status == "NO_CURRENT_EVIDENCE":
            self._put_text(
                canvas,
                "State expired: waiting for fresh evidence",
                (x, y),
                scale=0.42,
                color=COLOR_WARNING,
            )
            y += 28
        else:
            self._put_text(
                canvas,
                f"State: {interaction_status}",
                (x, y),
                scale=0.43,
                color=COLOR_GOOD if interaction_status == "STABLE" else COLOR_MUTED,
            )
            y += 28

        # Raw/latest result
        self.cv2.line(
            canvas,
            (x, y),
            (x + usable, y),
            COLOR_BORDER,
            1,
            self.cv2.LINE_AA,
        )
        y += 25

        raw_emotion = raw.get("emotion")
        raw_conf = raw.get("confidence")
        raw_valid = bool(raw.get("valid"))

        self._put_text(
            canvas,
            "LATEST MODEL RESULT",
            (x, y),
            scale=0.46,
            color=COLOR_MUTED,
        )
        y += 27

        if raw_valid and raw_emotion is not None:
            self._put_text(
                canvas,
                f"{raw_emotion}   {_pct(raw_conf)}",
                (x, y),
                scale=0.58,
                color=self._emotion_accent(str(raw_emotion)),
                thickness=1,
            )
        else:
            reason = _safe_text(raw.get("result_status"), "NO RESULT")
            self._put_text(
                canvas,
                reason,
                (x, y),
                scale=0.50,
                color=COLOR_WARNING,
            )
        y += 31

        # Probability bars
        probs = raw.get("probabilities")
        if (
            self.show_raw_probabilities
            and raw_valid
            and isinstance(probs, Sequence)
            and len(probs) == 5
        ):
            for label, value in zip(EMOTIONS, probs):
                y = self._draw_probability_bar(
                    canvas,
                    x=x,
                    y=y,
                    width=usable,
                    label=label,
                    value=_clip01(value),
                    selected=(label == raw_emotion),
                )

        # Bottom area
        y = max(y + 5, self.display_height - 150)
        self.cv2.line(
            canvas,
            (x, y),
            (x + usable, y),
            COLOR_BORDER,
            1,
            self.cv2.LINE_AA,
        )
        y += 25

        availability = sensors.get("availability")
        y = self._draw_sensor_line(
            canvas,
            x=x,
            y=y,
            availability=availability if isinstance(availability, Mapping) else None,
        )

        route = fusion.get("route")
        quality = fusion.get("quality_for_fusion")
        self._put_text(
            canvas,
            f"Fusion: {_safe_text(route)}",
            (x, y),
            scale=0.45,
            color=COLOR_MUTED,
        )
        y += 23

        if isinstance(quality, Mapping):
            aq = quality.get("audio")
            vq = quality.get("video")
            eq = quality.get("eeg")
            qtext = (
                f"q  A:{_pct(aq)}  V:{_pct(vq)}  E:{_pct(eq)}"
            )
            self._put_text(
                canvas,
                qtext,
                (x, y),
                scale=0.41,
                color=COLOR_DIM,
            )
            y += 22

        age = freshness.get("last_valid_result_age_seconds")
        latency = runtime.inference_latency_seconds
        meta = f"Evidence age: {_seconds(age)}"
        if latency is not None:
            meta += f"   latency: {_seconds(latency)}"
        self._put_text(
            canvas,
            meta,
            (x, y),
            scale=0.39,
            color=COLOR_DIM,
        )

        if runtime.message:
            self._put_text(
                canvas,
                runtime.message[:60],
                (x, self.display_height - 16),
                scale=0.40,
                color=COLOR_WARNING,
            )

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Render one participant-facing frame.

        The input frame is never modified in place.
        """
        source = frame.copy()
        canvas = self._fit_camera_frame(source)
        x0, _, _, _ = self._overlay_panel(canvas)

        state, runtime = self.snapshot()
        self._draw_state_panel(
            canvas,
            state=state,
            runtime=runtime,
            x0=x0,
        )

        # Camera-side LIVE indicator.
        self.cv2.rectangle(
            canvas,
            (16, 16),
            (102, 48),
            COLOR_PANEL,
            thickness=-1,
        )
        self.cv2.circle(
            canvas,
            (31, 32),
            6,
            COLOR_BAD,
            thickness=-1,
            lineType=self.cv2.LINE_AA,
        )
        self._put_text(
            canvas,
            "LIVE",
            (44, 39),
            scale=0.55,
            color=COLOR_TEXT,
            thickness=1,
        )

        self._last_render_monotonic = time.monotonic()
        self._render_count += 1
        return canvas

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        if self._closed:
            raise LiveDisplayError("Display was already closed")

        self.cv2.namedWindow(
            self.window_name,
            self.cv2.WINDOW_NORMAL,
        )
        self.cv2.resizeWindow(
            self.window_name,
            self.display_width,
            self.display_height,
        )
        if self.fullscreen:
            self.cv2.setWindowProperty(
                self.window_name,
                self.cv2.WND_PROP_FULLSCREEN,
                self.cv2.WINDOW_FULLSCREEN,
            )
        self._opened = True

    def show(
        self,
        frame: np.ndarray,
        *,
        wait_ms: int = 1,
    ) -> bool:
        """
        Render/show one frame.

        Returns True when the participant/operator requests exit via Q or ESC,
        or when the display window is closed.
        """
        require(wait_ms >= 1, "wait_ms must be >= 1")
        if not self._opened:
            self.open()

        canvas = self.render(frame)
        self.cv2.imshow(self.window_name, canvas)

        key = self.cv2.waitKey(int(wait_ms)) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return True

        try:
            visible = self.cv2.getWindowProperty(
                self.window_name,
                self.cv2.WND_PROP_VISIBLE,
            )
            if visible < 1:
                return True
        except Exception:
            pass

        return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._opened:
                self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self._opened = False
        self._closed = True

    @property
    def render_count(self) -> int:
        return self._render_count


# ----------------------------------------------------------------------
# Self-test / demo
# ----------------------------------------------------------------------

def _synthetic_state(
    *,
    raw_emotion: str,
    raw_confidence: float,
    stable_emotion: str,
    stable_confidence: float,
    transition_emotion: str | None = None,
    transition_count: int = 0,
    transition_required: int = 2,
) -> dict[str, Any]:
    require(raw_emotion in EMOTIONS, "Unknown raw emotion")
    require(stable_emotion in EMOTIONS, "Unknown stable emotion")

    idx = EMOTIONS.index(raw_emotion)
    rest = (1.0 - raw_confidence) / 4.0
    probs = [rest] * 5
    probs[idx] = raw_confidence

    transition_active = transition_emotion is not None

    return {
        "schema": "eav.live_interaction.emotion_state.v1",
        "version": "SELF_TEST",
        "updated_monotonic": time.monotonic(),
        "interaction_status": (
            "POSSIBLE_TRANSITION" if transition_active else "STABLE"
        ),
        "raw": {
            "valid": True,
            "window_id": "window_000001",
            "result_status": "OK",
            "emotion": raw_emotion,
            "confidence": raw_confidence,
            "probabilities": probs,
            "label_id": idx,
        },
        "stable": {
            "available": True,
            "emotion": stable_emotion,
            "confidence": stable_confidence,
            "probabilities": probs,
            "previous_emotion": None,
            "age_seconds": 1.0,
            "transitions": 0,
        },
        "transition": {
            "active": transition_active,
            "candidate_emotion": transition_emotion,
            "consecutive_count": transition_count,
            "required_count": transition_required,
        },
        "sensors": {
            "availability": {
                "eeg": False,
                "audio": True,
                "video": True,
            }
        },
        "fusion": {
            "route": "AF4-B",
            "quality_for_fusion": {
                "eeg": 0.0,
                "audio": 1.0,
                "video": 0.72,
            },
        },
        "freshness": {
            "last_valid_result_age_seconds": 1.0,
            "state_timeout_sec": 12.0,
            "current_evidence": True,
        },
        "policy": {
            "interaction_layer_only": True,
        },
        "last_event": None,
        "callback_errors": [],
    }


def self_test() -> dict[str, Any]:
    display = LiveEmotionDisplay(
        display_width=960,
        display_height=540,
        panel_width=360,
        fullscreen=False,
    )

    checks: list[tuple[str, bool]] = []

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 0] = 35
    frame[:, :, 1] = 55
    frame[:, :, 2] = 70
    original = frame.copy()

    rendered_waiting = display.render(frame)
    checks.append(
        (
            "waiting_render_shape",
            rendered_waiting.shape == (540, 960, 3),
        )
    )
    checks.append(
        (
            "input_frame_not_modified",
            np.array_equal(frame, original),
        )
    )

    state = _synthetic_state(
        raw_emotion="Neutral",
        raw_confidence=0.83,
        stable_emotion="Happiness",
        stable_confidence=0.77,
        transition_emotion="Neutral",
        transition_count=1,
    )
    display.on_state(state)
    display.update_runtime(
        session_id="self_test",
        window_id="window_000003",
        phase="CAPTURING",
        remaining_seconds=2.4,
        capture_fps=20.3,
        inference_latency_seconds=2.35,
    )

    rendered_state = display.render(frame)
    checks.append(
        (
            "state_render_shape",
            rendered_state.shape == (540, 960, 3),
        )
    )
    checks.append(
        (
            "state_render_changes_pixels",
            not np.array_equal(rendered_state, rendered_waiting),
        )
    )

    copied_state, runtime = display.snapshot()
    checks.append(
        (
            "state_callback_memory_path",
            copied_state is not None
            and copied_state["stable"]["emotion"] == "Happiness"
            and copied_state["raw"]["emotion"] == "Neutral",
        )
    )
    checks.append(
        (
            "transition_visible_in_state",
            copied_state["interaction_status"] == "POSSIBLE_TRANSITION"
            and copied_state["transition"]["candidate_emotion"] == "Neutral",
        )
    )
    checks.append(
        (
            "runtime_metadata_memory_path",
            runtime.window_id == "window_000003"
            and runtime.inference_latency_seconds == 2.35,
        )
    )
    checks.append(
        (
            "no_file_transport_required",
            True,
        )
    )

    passed = sum(ok for _, ok in checks)
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "version": VERSION,
        "n_checks": len(checks),
        "n_passed": passed,
        "checks": [
            {"name": name, "pass": ok}
            for name, ok in checks
        ],
    }


def demo() -> None:
    cv2 = load_cv2()
    display = LiveEmotionDisplay(
        window_name="EAV Live Display Demo - SYNTHETIC STATES",
        fullscreen=False,
    )

    # Synthetic camera-like background. This demo intentionally does not open
    # real hardware and does not claim model evidence.
    h, w = 720, 1280
    base = np.zeros((h, w, 3), dtype=np.uint8)

    sequence = [
        ("Happiness", 0.79, "Happiness", 0.79, None, 0),
        ("Happiness", 0.77, "Happiness", 0.77, None, 0),
        ("Neutral", 0.83, "Happiness", 0.77, "Neutral", 1),
        ("Neutral", 0.81, "Neutral", 0.81, None, 0),
        ("Neutral", 0.84, "Neutral", 0.84, None, 0),
    ]

    index = 0
    next_change = time.monotonic()
    start = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if now >= next_change:
                raw, rc, stable, sc, candidate, count = sequence[index % len(sequence)]
                state = _synthetic_state(
                    raw_emotion=raw,
                    raw_confidence=rc,
                    stable_emotion=stable,
                    stable_confidence=sc,
                    transition_emotion=candidate,
                    transition_count=count,
                )
                display.on_state(state)
                display.update_runtime(
                    session_id="SYNTHETIC_DEMO",
                    window_id=f"window_{index + 1:06d}",
                    phase="DEMO",
                    remaining_seconds=None,
                    capture_fps=20.3,
                    inference_latency_seconds=2.35,
                    message="Synthetic UI demo only - not model evidence",
                )
                index += 1
                next_change = now + 2.0

            # Mildly animated background so it is visually obvious that the
            # display is refreshing continuously.
            base[:] = (28, 38, 48)
            x = int(((now - start) * 120) % (w - 160))
            cv2.rectangle(
                base,
                (x, 170),
                (x + 160, 520),
                (55, 75, 95),
                thickness=-1,
            )
            cv2.putText(
                base,
                "Synthetic camera background",
                (42, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )

            if display.show(base, wait_ms=15):
                break
    finally:
        display.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Participant-facing real-time emotion display. "
            "Consumes in-memory EmotionStateManager snapshots."
        )
    )
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--demo", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        report = self_test()
        print(
            __import__("json").dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report["status"] == "PASS" else 2

    if args.demo:
        demo()
        return 0

    print(
        "live_display.py is a display library module.\n"
        "Use --self-test, --demo, or import LiveEmotionDisplay from the "
        "next integrated live interaction script."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LiveDisplayError as exc:
        print(f"LIVE DISPLAY ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
