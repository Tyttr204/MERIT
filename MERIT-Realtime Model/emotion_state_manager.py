#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
emotion_state_manager.py
========================

EAV LiveInteraction — interaction-layer emotion state manager.

Purpose
-------
Convert a stream of already-computed emotion-recognition results into a stable,
directly consumable interaction state for a live UI or future robot policy.

This module DOES NOT:
    - run EEG / Audio / Video models;
    - alter model probabilities;
    - alter quality scores;
    - alter quality thresholds;
    - alter F4 / AF4-B routing;
    - retrain or calibrate anything;
    - read prediction results from files.

The intended live path is:

    frozen main.py / LiveScheduler
                |
                | result dict in memory
                v
        EmotionStateManager.update(...)
                |
                +--> subscriber callback -> live display
                +--> subscriber callback -> interaction policy
                +--> optional external logger

Files are therefore audit outputs only; they are not the interaction transport.

Default state rule
------------------
1. The first valid emotion result establishes the stable state immediately.
2. If the next raw emotion equals the stable state, it re-confirms that state.
3. If it differs, it becomes a transition candidate.
4. A new emotion must appear in TWO consecutive valid windows before the stable
   state changes.
5. Invalid / NO_DECISION / error results never fabricate a new emotion and never
   count as evidence for a transition.
6. If no valid evidence arrives for a configurable time, the stable state
   expires to NO_CURRENT_EVIDENCE instead of being shown indefinitely.

This is interaction-layer hysteresis only.  The raw model output remains
available unchanged in every snapshot.

Typical use
-----------
    from emotion_state_manager import EmotionStateManager

    manager = EmotionStateManager(confirmations_required=2)

    def on_state(snapshot):
        print(snapshot["stable"]["emotion"])

    manager.subscribe(on_state)

    # row is returned directly by the frozen live runtime:
    manager.update(row)

Command-line validation
-----------------------
    python .\LiveInteraction\emotion_state_manager.py --self-test
    python .\LiveInteraction\emotion_state_manager.py --demo
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import threading
import time
import uuid
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


VERSION = "EAV-EMOTION-STATE-MANAGER.1.0"
STATE_SCHEMA = "eav.live_interaction.emotion_state.v1"
EVENT_SCHEMA = "eav.live_interaction.emotion_state_event.v1"

EMOTIONS = (
    "Neutral",
    "Sadness",
    "Anger",
    "Happiness",
    "Calmness",
)

DEFAULT_CONFIRMATIONS_REQUIRED = 2
DEFAULT_STATE_TIMEOUT_SEC = 12.0
DEFAULT_MAX_SEEN_WINDOWS = 10000

# Frozen runtime emits OK for a valid final prediction.
DEFAULT_VALID_RESULT_STATUSES = frozenset({"OK"})


class EmotionStateError(RuntimeError):
    """Bad input contract or invalid state-manager configuration."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise EmotionStateError(message)


def finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise EmotionStateError(f"{name}: finite number required") from exc
    require(math.isfinite(out), f"{name}: NaN/Inf is not allowed")
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _window_id(row: Mapping[str, Any]) -> str:
    value = row.get("window_id")
    if not value and isinstance(row.get("source_window"), Mapping):
        value = row["source_window"].get("window_id")
    require(isinstance(value, str) and value.strip(), "Result has no valid window_id")
    return value


def _source_window_end(row: Mapping[str, Any]) -> float | None:
    value = row.get("window_end_monotonic")
    if value is None and isinstance(row.get("source_window"), Mapping):
        value = row["source_window"].get("window_end_monotonic")
    if value is None:
        return None
    return finite(value, "window_end_monotonic")


def _fusion(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("fusion")
    return value if isinstance(value, Mapping) else row


def _final_packet(row: Mapping[str, Any]) -> Mapping[str, Any]:
    f = _fusion(row)
    final = f.get("final")
    return final if isinstance(final, Mapping) else row


def _result_status(row: Mapping[str, Any]) -> str:
    value = row.get("status")
    if value is None:
        value = _fusion(row).get("status")
    return str(value) if value is not None else "UNKNOWN"


def _route(row: Mapping[str, Any]) -> str | None:
    value = row.get("active_branch")
    if value is None:
        value = _fusion(row).get("active_branch")
    return None if value is None else str(value)


def _availability(row: Mapping[str, Any]) -> dict[str, bool] | None:
    value = row.get("availability")
    if value is None:
        value = _fusion(row).get("availability")
    if not isinstance(value, Mapping):
        return None
    out: dict[str, bool] = {}
    for name in ("eeg", "audio", "video"):
        if name in value:
            out[name] = bool(value[name])
    return out or None


def _quality(row: Mapping[str, Any]) -> dict[str, float] | None:
    value = row.get("quality_for_fusion")
    if value is None:
        value = _fusion(row).get("quality_for_fusion")
    if not isinstance(value, Mapping):
        return None
    out: dict[str, float] = {}
    for name in ("eeg", "audio", "video"):
        if name in value and value[name] is not None:
            out[name] = finite(value[name], f"quality_for_fusion.{name}")
    return out or None


def _probabilities_and_emotion(
    row: Mapping[str, Any],
) -> tuple[list[float], str, float, int]:
    final = _final_packet(row)

    probs = final.get("probabilities")
    require(
        isinstance(probs, Sequence) and not isinstance(probs, (str, bytes)),
        "Valid result has no probability vector",
    )
    require(len(probs) == len(EMOTIONS), "Expected exactly five class probabilities")

    p = [finite(x, f"probabilities[{i}]") for i, x in enumerate(probs)]
    require(all(-1e-8 <= x <= 1.0 + 1e-8 for x in p), "Probability outside [0,1]")
    total = sum(p)
    require(
        abs(total - 1.0) <= 0.02,
        f"Probability sum is implausible ({total:.8f}); probabilities are not rewritten",
    )

    argmax = max(range(len(p)), key=p.__getitem__)
    expected_emotion = EMOTIONS[argmax]

    emotion = final.get("emotion")
    if emotion is None:
        emotion = expected_emotion
    emotion = str(emotion)
    require(emotion in EMOTIONS, f"Unknown emotion: {emotion!r}")
    require(
        emotion == expected_emotion,
        f"Emotion/probability argmax mismatch: {emotion!r} != {expected_emotion!r}",
    )

    confidence_value = final.get("confidence")
    confidence = p[argmax] if confidence_value is None else finite(
        confidence_value, "confidence"
    )
    require(-1e-8 <= confidence <= 1.0 + 1e-8, "Confidence outside [0,1]")
    require(
        abs(confidence - p[argmax]) <= 0.02,
        "Confidence does not match top-1 probability",
    )

    label_value = final.get("label_id")
    label_id = argmax if label_value is None else int(label_value)
    require(label_id == argmax, "label_id does not match probability argmax")

    return p, emotion, confidence, label_id


@dataclass
class RawEmotion:
    valid: bool = False
    window_id: str | None = None
    source_window_end_monotonic: float | None = None
    received_monotonic: float | None = None
    result_status: str = "NO_RESULT"
    emotion: str | None = None
    confidence: float | None = None
    probabilities: list[float] | None = None
    label_id: int | None = None
    route: str | None = None
    availability: dict[str, bool] | None = None
    quality_for_fusion: dict[str, float] | None = None
    reason: str | None = None


@dataclass
class StableEmotion:
    available: bool = False
    emotion: str | None = None
    confidence: float | None = None
    probabilities: list[float] | None = None

    established_monotonic: float | None = None
    established_from_window_id: str | None = None

    last_confirmed_monotonic: float | None = None
    last_confirmed_window_id: str | None = None

    previous_emotion: str | None = None
    transitions: int = 0


@dataclass
class TransitionCandidate:
    active: bool = False
    emotion: str | None = None
    consecutive_count: int = 0
    required_count: int = DEFAULT_CONFIRMATIONS_REQUIRED
    first_window_id: str | None = None
    latest_window_id: str | None = None
    latest_confidence: float | None = None
    latest_probabilities: list[float] | None = None


Subscriber = Callable[[dict[str, Any]], None]


class EmotionStateManager:
    """
    Thread-safe interaction-layer state machine.

    update(row)
        Feed one frozen-runtime result directly in memory.

    snapshot()
        Return the current full state.

    expire()
        Explicitly apply the no-new-evidence timeout.

    subscribe(callback)
        Receive every changed snapshot immediately.  This is the intended
        connection for live display and interaction-policy modules.
    """

    def __init__(
        self,
        *,
        confirmations_required: int = DEFAULT_CONFIRMATIONS_REQUIRED,
        state_timeout_sec: float | None = DEFAULT_STATE_TIMEOUT_SEC,
        valid_result_statuses: Sequence[str] = tuple(DEFAULT_VALID_RESULT_STATUSES),
        max_seen_windows: int = DEFAULT_MAX_SEEN_WINDOWS,
    ):
        require(
            isinstance(confirmations_required, int) and confirmations_required >= 1,
            "confirmations_required must be an integer >= 1",
        )
        if state_timeout_sec is not None:
            state_timeout_sec = finite(state_timeout_sec, "state_timeout_sec")
            require(state_timeout_sec > 0.0, "state_timeout_sec must be > 0 or None")

        require(
            isinstance(max_seen_windows, int) and max_seen_windows >= 10,
            "max_seen_windows must be >= 10",
        )

        statuses = {str(x) for x in valid_result_statuses}
        require(statuses, "At least one valid result status is required")

        self.confirmations_required = confirmations_required
        self.state_timeout_sec = state_timeout_sec
        self.valid_result_statuses = frozenset(statuses)
        self.max_seen_windows = max_seen_windows

        self._lock = threading.RLock()
        self._raw = RawEmotion()
        self._stable = StableEmotion()
        self._candidate = TransitionCandidate(
            required_count=confirmations_required
        )

        self._last_valid_received_monotonic: float | None = None
        self._last_source_window_end: float | None = None
        self._seen_windows: set[str] = set()
        self._seen_order: list[str] = []

        self._subscribers: dict[str, Subscriber] = {}
        self._event_seq = 0
        self._last_event: dict[str, Any] | None = None
        self._callback_errors: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, callback: Subscriber) -> str:
        require(callable(callback), "subscriber must be callable")
        token = uuid.uuid4().hex
        with self._lock:
            self._subscribers[token] = callback
        return token

    def unsubscribe(self, token: str) -> bool:
        with self._lock:
            return self._subscribers.pop(str(token), None) is not None

    def reset(self, *, notify: bool = True) -> dict[str, Any]:
        with self._lock:
            self._raw = RawEmotion()
            self._stable = StableEmotion()
            self._candidate = TransitionCandidate(
                required_count=self.confirmations_required
            )
            self._last_valid_received_monotonic = None
            self._last_source_window_end = None
            self._seen_windows.clear()
            self._seen_order.clear()
            event = self._new_event_locked(
                "STATE_RESET",
                window_id=None,
                details={},
            )
            snapshot = self._snapshot_locked(time.monotonic())
        if notify:
            self._notify(snapshot)
        return snapshot

    def update(
        self,
        row: Mapping[str, Any],
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """
        Consume one result directly from the frozen runtime.

        Both forms below are accepted:
          * the full main.py result row containing ``fusion.final``;
          * the simplified Stage-03 result shape with top-level emotion fields.

        Invalid/no-decision/error results are represented in ``raw`` but never
        create or confirm an emotion transition.
        """
        require(isinstance(row, Mapping), "update() requires a mapping result")
        now = time.monotonic() if now_monotonic is None else finite(
            now_monotonic, "now_monotonic"
        )
        wid = _window_id(row)
        source_end = _source_window_end(row)
        status = _result_status(row)

        with self._lock:
            if wid in self._seen_windows:
                event = self._new_event_locked(
                    "DUPLICATE_RESULT_IGNORED",
                    window_id=wid,
                    details={"result_status": status},
                )
                snapshot = self._snapshot_locked(now)
                # Duplicate is intentionally idempotent: state does not change.
                snapshot["last_event"] = copy.deepcopy(event)
                return snapshot

            if (
                source_end is not None
                and self._last_source_window_end is not None
                and source_end <= self._last_source_window_end
            ):
                event = self._new_event_locked(
                    "OUT_OF_ORDER_RESULT_IGNORED",
                    window_id=wid,
                    details={
                        "source_window_end_monotonic": source_end,
                        "last_source_window_end_monotonic": self._last_source_window_end,
                    },
                )
                self._remember_window_locked(wid)
                snapshot = self._snapshot_locked(now)
                snapshot["last_event"] = copy.deepcopy(event)
                return snapshot

            self._remember_window_locked(wid)
            if source_end is not None:
                self._last_source_window_end = source_end

            if status not in self.valid_result_statuses:
                self._raw = RawEmotion(
                    valid=False,
                    window_id=wid,
                    source_window_end_monotonic=source_end,
                    received_monotonic=now,
                    result_status=status,
                    route=_route(row),
                    availability=_availability(row),
                    quality_for_fusion=_quality(row),
                    reason="RESULT_STATUS_NOT_VALID_EVIDENCE",
                )
                event = self._new_event_locked(
                    "INVALID_RESULT_NO_STATE_UPDATE",
                    window_id=wid,
                    details={"result_status": status},
                )
                # An invalid result breaks a candidate's "consecutive valid
                # windows" sequence.  It does not erase the already-established
                # stable state.
                self._clear_candidate_locked()
                self._apply_expiry_locked(now)
                snapshot = self._snapshot_locked(now)
            else:
                probs, emotion, confidence, label_id = _probabilities_and_emotion(row)
                self._raw = RawEmotion(
                    valid=True,
                    window_id=wid,
                    source_window_end_monotonic=source_end,
                    received_monotonic=now,
                    result_status=status,
                    emotion=emotion,
                    confidence=confidence,
                    probabilities=list(probs),
                    label_id=label_id,
                    route=_route(row),
                    availability=_availability(row),
                    quality_for_fusion=_quality(row),
                )
                self._last_valid_received_monotonic = now

                event = self._consume_valid_locked(
                    emotion=emotion,
                    confidence=confidence,
                    probabilities=probs,
                    window_id=wid,
                    now=now,
                )
                snapshot = self._snapshot_locked(now)

        self._notify(snapshot)
        return snapshot

    def expire(
        self,
        *,
        now_monotonic: float | None = None,
        notify: bool = True,
    ) -> dict[str, Any]:
        """
        Apply freshness expiry even when no new model result arrives.

        A UI loop can call this periodically (e.g. 5–10 Hz).  It does not poll
        any files or run any model.
        """
        now = time.monotonic() if now_monotonic is None else finite(
            now_monotonic, "now_monotonic"
        )
        with self._lock:
            changed = self._apply_expiry_locked(now)
            snapshot = self._snapshot_locked(now)
        if changed and notify:
            self._notify(snapshot)
        return snapshot

    def snapshot(
        self,
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now_monotonic is None else finite(
            now_monotonic, "now_monotonic"
        )
        with self._lock:
            self._apply_expiry_locked(now)
            return self._snapshot_locked(now)

    # ------------------------------------------------------------------
    # Core state machine
    # ------------------------------------------------------------------

    def _consume_valid_locked(
        self,
        *,
        emotion: str,
        confidence: float,
        probabilities: Sequence[float],
        window_id: str,
        now: float,
    ) -> dict[str, Any]:
        if not self._stable.available:
            self._stable = StableEmotion(
                available=True,
                emotion=emotion,
                confidence=confidence,
                probabilities=list(probabilities),
                established_monotonic=now,
                established_from_window_id=window_id,
                last_confirmed_monotonic=now,
                last_confirmed_window_id=window_id,
                previous_emotion=self._stable.previous_emotion,
                transitions=self._stable.transitions,
            )
            self._clear_candidate_locked()
            return self._new_event_locked(
                "STABLE_STATE_INITIALIZED",
                window_id=window_id,
                details={
                    "emotion": emotion,
                    "confidence": confidence,
                },
            )

        if emotion == self._stable.emotion:
            # Same raw class confirms the existing stable state and refreshes
            # the displayed confidence/probabilities with the newest unmodified
            # model output.
            self._stable.confidence = confidence
            self._stable.probabilities = list(probabilities)
            self._stable.last_confirmed_monotonic = now
            self._stable.last_confirmed_window_id = window_id

            had_candidate = self._candidate.active
            candidate_emotion = self._candidate.emotion
            self._clear_candidate_locked()

            return self._new_event_locked(
                "STABLE_STATE_RECONFIRMED",
                window_id=window_id,
                details={
                    "emotion": emotion,
                    "confidence": confidence,
                    "cancelled_candidate": candidate_emotion if had_candidate else None,
                },
            )

        # Raw emotion differs from stable state.
        if self.confirmations_required == 1:
            return self._switch_stable_locked(
                emotion=emotion,
                confidence=confidence,
                probabilities=probabilities,
                window_id=window_id,
                now=now,
                confirmations=1,
            )

        if self._candidate.active and self._candidate.emotion == emotion:
            self._candidate.consecutive_count += 1
            self._candidate.latest_window_id = window_id
            self._candidate.latest_confidence = confidence
            self._candidate.latest_probabilities = list(probabilities)
        else:
            self._candidate = TransitionCandidate(
                active=True,
                emotion=emotion,
                consecutive_count=1,
                required_count=self.confirmations_required,
                first_window_id=window_id,
                latest_window_id=window_id,
                latest_confidence=confidence,
                latest_probabilities=list(probabilities),
            )

        if self._candidate.consecutive_count >= self.confirmations_required:
            return self._switch_stable_locked(
                emotion=emotion,
                confidence=confidence,
                probabilities=probabilities,
                window_id=window_id,
                now=now,
                confirmations=self._candidate.consecutive_count,
            )

        return self._new_event_locked(
            "TRANSITION_CANDIDATE_UPDATED",
            window_id=window_id,
            details={
                "stable_emotion": self._stable.emotion,
                "candidate_emotion": emotion,
                "consecutive_count": self._candidate.consecutive_count,
                "required_count": self.confirmations_required,
                "latest_confidence": confidence,
            },
        )

    def _switch_stable_locked(
        self,
        *,
        emotion: str,
        confidence: float,
        probabilities: Sequence[float],
        window_id: str,
        now: float,
        confirmations: int,
    ) -> dict[str, Any]:
        previous = self._stable.emotion
        transitions = self._stable.transitions + 1

        self._stable = StableEmotion(
            available=True,
            emotion=emotion,
            confidence=confidence,
            probabilities=list(probabilities),
            established_monotonic=now,
            established_from_window_id=window_id,
            last_confirmed_monotonic=now,
            last_confirmed_window_id=window_id,
            previous_emotion=previous,
            transitions=transitions,
        )
        self._clear_candidate_locked()

        return self._new_event_locked(
            "STABLE_STATE_CHANGED",
            window_id=window_id,
            details={
                "from": previous,
                "to": emotion,
                "confidence": confidence,
                "confirmations": confirmations,
                "transition_number": transitions,
            },
        )

    def _apply_expiry_locked(self, now: float) -> bool:
        if (
            self.state_timeout_sec is None
            or not self._stable.available
            or self._last_valid_received_monotonic is None
        ):
            return False

        age = now - self._last_valid_received_monotonic
        if age <= self.state_timeout_sec:
            return False

        previous = self._stable.emotion
        transitions = self._stable.transitions
        self._stable = StableEmotion(
            available=False,
            previous_emotion=previous,
            transitions=transitions,
        )
        self._clear_candidate_locked()
        self._new_event_locked(
            "STABLE_STATE_EXPIRED",
            window_id=self._raw.window_id,
            details={
                "previous_emotion": previous,
                "age_seconds": age,
                "state_timeout_sec": self.state_timeout_sec,
            },
        )
        return True

    def _clear_candidate_locked(self) -> None:
        self._candidate = TransitionCandidate(
            required_count=self.confirmations_required
        )

    def _remember_window_locked(self, window_id: str) -> None:
        self._seen_windows.add(window_id)
        self._seen_order.append(window_id)
        if len(self._seen_order) > self.max_seen_windows:
            excess = len(self._seen_order) - self.max_seen_windows
            old = self._seen_order[:excess]
            del self._seen_order[:excess]
            for wid in old:
                self._seen_windows.discard(wid)

    def _new_event_locked(
        self,
        event_type: str,
        *,
        window_id: str | None,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._event_seq += 1
        event = {
            "schema": EVENT_SCHEMA,
            "version": VERSION,
            "sequence": self._event_seq,
            "event": event_type,
            "window_id": window_id,
            "created_monotonic": time.monotonic(),
            "details": json_safe(dict(details)),
        }
        self._last_event = event
        return event

    # ------------------------------------------------------------------
    # Snapshot / notification
    # ------------------------------------------------------------------

    def _snapshot_locked(self, now: float) -> dict[str, Any]:
        last_valid_age = (
            now - self._last_valid_received_monotonic
            if self._last_valid_received_monotonic is not None
            else None
        )

        if not self._stable.available:
            interaction_status = (
                "WAITING_FOR_FIRST_VALID_RESULT"
                if self._last_valid_received_monotonic is None
                else "NO_CURRENT_EVIDENCE"
            )
        elif self._candidate.active:
            interaction_status = "POSSIBLE_TRANSITION"
        else:
            interaction_status = "STABLE"

        stable_age = (
            now - self._stable.established_monotonic
            if self._stable.available
            and self._stable.established_monotonic is not None
            else None
        )

        return {
            "schema": STATE_SCHEMA,
            "version": VERSION,
            "updated_monotonic": now,
            "interaction_status": interaction_status,

            "raw": {
                "valid": self._raw.valid,
                "window_id": self._raw.window_id,
                "source_window_end_monotonic": self._raw.source_window_end_monotonic,
                "received_monotonic": self._raw.received_monotonic,
                "result_status": self._raw.result_status,
                "emotion": self._raw.emotion,
                "confidence": self._raw.confidence,
                "probabilities": copy.deepcopy(self._raw.probabilities),
                "label_id": self._raw.label_id,
                "reason": self._raw.reason,
            },

            "stable": {
                "available": self._stable.available,
                "emotion": self._stable.emotion,
                "confidence": self._stable.confidence,
                "probabilities": copy.deepcopy(self._stable.probabilities),
                "previous_emotion": self._stable.previous_emotion,
                "established_from_window_id": self._stable.established_from_window_id,
                "last_confirmed_window_id": self._stable.last_confirmed_window_id,
                "age_seconds": stable_age,
                "transitions": self._stable.transitions,
            },

            "transition": {
                "active": self._candidate.active,
                "candidate_emotion": self._candidate.emotion,
                "consecutive_count": self._candidate.consecutive_count,
                "required_count": self._candidate.required_count,
                "first_window_id": self._candidate.first_window_id,
                "latest_window_id": self._candidate.latest_window_id,
                "latest_confidence": self._candidate.latest_confidence,
                "latest_probabilities": copy.deepcopy(
                    self._candidate.latest_probabilities
                ),
            },

            "sensors": {
                "availability": copy.deepcopy(self._raw.availability),
            },

            "fusion": {
                "route": self._raw.route,
                "quality_for_fusion": copy.deepcopy(
                    self._raw.quality_for_fusion
                ),
            },

            "freshness": {
                "last_valid_result_age_seconds": last_valid_age,
                "state_timeout_sec": self.state_timeout_sec,
                "current_evidence": (
                    self._stable.available
                    and (
                        self.state_timeout_sec is None
                        or last_valid_age is not None
                        and last_valid_age <= self.state_timeout_sec
                    )
                ),
            },

            "policy": {
                "confirmations_required": self.confirmations_required,
                "raw_probabilities_modified": False,
                "quality_modified": False,
                "fusion_modified": False,
                "interaction_layer_only": True,
            },

            "last_event": copy.deepcopy(self._last_event),
            "callback_errors": copy.deepcopy(self._callback_errors[-20:]),
        }

    def _notify(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.items())

        for token, callback in subscribers:
            try:
                callback(copy.deepcopy(snapshot))
            except Exception as exc:
                with self._lock:
                    self._callback_errors.append(
                        {
                            "subscriber_token": token,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if len(self._callback_errors) > 100:
                        del self._callback_errors[:-100]


# ----------------------------------------------------------------------
# Helpers for development / next integration stage
# ----------------------------------------------------------------------

def make_demo_result(
    window_id: str,
    emotion: str,
    confidence: float,
    *,
    source_end: float,
    route: str = "AF4-B",
) -> dict[str, Any]:
    """
    Synthetic helper for this module's self-test/demo only.
    It is never intended as sensor or model evidence.
    """
    require(emotion in EMOTIONS, "Unknown demo emotion")
    require(0.20 <= confidence <= 1.0, "Demo confidence must be >= 0.20")

    idx = EMOTIONS.index(emotion)
    rest = (1.0 - confidence) / 4.0
    probs = [rest] * 5
    probs[idx] = confidence

    return {
        "window_id": window_id,
        "window_end_monotonic": source_end,
        "status": "OK",
        "active_branch": route,
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probs,
        "label_id": idx,
        "availability": {
            "eeg": False,
            "audio": True,
            "video": True,
        },
        "quality_for_fusion": {
            "eeg": 0.0,
            "audio": 1.0,
            "video": 0.7,
        },
    }


def self_test() -> dict[str, Any]:
    base = 1000.0
    manager = EmotionStateManager(
        confirmations_required=2,
        state_timeout_sec=12.0,
    )

    checks: list[tuple[str, bool]] = []

    s1 = manager.update(
        make_demo_result(
            "window_000001",
            "Happiness",
            0.80,
            source_end=base + 5,
        ),
        now_monotonic=base + 7,
    )
    checks.append(
        (
            "first_valid_initializes",
            s1["stable"]["emotion"] == "Happiness"
            and s1["interaction_status"] == "STABLE",
        )
    )

    s2 = manager.update(
        make_demo_result(
            "window_000002",
            "Happiness",
            0.76,
            source_end=base + 10,
        ),
        now_monotonic=base + 12,
    )
    checks.append(
        (
            "same_emotion_reconfirms",
            s2["stable"]["emotion"] == "Happiness"
            and not s2["transition"]["active"],
        )
    )

    s3 = manager.update(
        make_demo_result(
            "window_000003",
            "Neutral",
            0.83,
            source_end=base + 15,
        ),
        now_monotonic=base + 17,
    )
    checks.append(
        (
            "single_difference_is_candidate",
            s3["stable"]["emotion"] == "Happiness"
            and s3["transition"]["candidate_emotion"] == "Neutral"
            and s3["transition"]["consecutive_count"] == 1
            and s3["interaction_status"] == "POSSIBLE_TRANSITION",
        )
    )

    s4 = manager.update(
        make_demo_result(
            "window_000004",
            "Neutral",
            0.81,
            source_end=base + 20,
        ),
        now_monotonic=base + 22,
    )
    checks.append(
        (
            "second_consecutive_switches",
            s4["stable"]["emotion"] == "Neutral"
            and not s4["transition"]["active"]
            and s4["stable"]["previous_emotion"] == "Happiness"
            and s4["stable"]["transitions"] == 1,
        )
    )

    no_decision = {
        "window_id": "window_000005",
        "window_end_monotonic": base + 25,
        "status": "NO_DECISION",
        "active_branch": "NO_DECISION",
        "availability": {
            "eeg": False,
            "audio": False,
            "video": False,
        },
    }
    s5 = manager.update(no_decision, now_monotonic=base + 27)
    checks.append(
        (
            "no_decision_does_not_fabricate_state",
            s5["raw"]["valid"] is False
            and s5["stable"]["emotion"] == "Neutral",
        )
    )

    # A new candidate after invalid evidence starts at 1, not 2.
    s6 = manager.update(
        make_demo_result(
            "window_000006",
            "Happiness",
            0.75,
            source_end=base + 30,
        ),
        now_monotonic=base + 32,
    )
    checks.append(
        (
            "candidate_sequence_restarts",
            s6["stable"]["emotion"] == "Neutral"
            and s6["transition"]["candidate_emotion"] == "Happiness"
            and s6["transition"]["consecutive_count"] == 1,
        )
    )

    duplicate = manager.update(
        make_demo_result(
            "window_000006",
            "Happiness",
            0.75,
            source_end=base + 30,
        ),
        now_monotonic=base + 33,
    )
    checks.append(
        (
            "duplicate_is_idempotent",
            duplicate["stable"]["emotion"] == "Neutral"
            and duplicate["transition"]["consecutive_count"] == 1,
        )
    )

    expired = manager.expire(
        now_monotonic=base + 32 + 12.01,
        notify=False,
    )
    checks.append(
        (
            "state_expires_without_fresh_evidence",
            expired["stable"]["available"] is False
            and expired["interaction_status"] == "NO_CURRENT_EVIDENCE",
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
    manager = EmotionStateManager(
        confirmations_required=2,
        state_timeout_sec=12.0,
    )

    def printer(state: dict[str, Any]) -> None:
        raw = state["raw"]
        stable = state["stable"]
        trans = state["transition"]
        print(
            f"RAW={raw['emotion']} "
            f"({raw['confidence'] if raw['confidence'] is not None else 'N/A'}) | "
            f"STABLE={stable['emotion']} | "
            f"STATUS={state['interaction_status']} | "
            f"CANDIDATE={trans['candidate_emotion']} "
            f"{trans['consecutive_count']}/{trans['required_count']}"
        )

    manager.subscribe(printer)

    t = time.monotonic()
    sequence = [
        ("Happiness", 0.79),
        ("Happiness", 0.77),
        ("Neutral", 0.83),
        ("Neutral", 0.81),
        ("Neutral", 0.84),
    ]

    for i, (emotion, confidence) in enumerate(sequence, 1):
        row = make_demo_result(
            f"window_{i:06d}",
            emotion,
            confidence,
            source_end=t + i * 5,
        )
        manager.update(row, now_monotonic=t + i * 5 + 2.3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interaction-layer emotion state manager. "
            "No model inference and no file polling."
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
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2

    if args.demo:
        demo()
        return 0

    print(
        "emotion_state_manager.py is a library module.\n"
        "Use --self-test, --demo, or import EmotionStateManager from the "
        "next live interaction script."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EmotionStateError as exc:
        print(f"STATE MANAGER ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
