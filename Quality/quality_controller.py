#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EAV unified distribution-quality controller -- candidate replay/shadow only.

Python 3.10+, standard library only. Place beside the UNMODIFIED reviewed
``distribution_quality.py`` and ``distribution_quality_params.json``.

This is an adapter/controller, NOT another quality model or a fusion adapter.
It reads the current window's reports, validates their provenance declarations,
extracts exactly the scorer's 11 raw features, and calls score_window ONCE.
It NEVER reads packet.quality, q_eeg/q_audio/q_video, quality_state, classifier
confidence, emotion labels, or corruption-family labels to make a decision.
It never changes availability on the basis of B, never builds q_new, never
invokes a detector, fusion model, device driver, training job, or robot action.

API (construct once at system startup):
    controller = QualityController("Quality/distribution_quality_params.json")
    result = controller.assess_window(source_window=descriptor, reports=reports)

Two accepted report contracts (can be mixed across modalities):
A) main.py EAV-MAIN-INTEGRATION.1.0 reports from Runtime.process():
   modality, window_id, status, algorithm_error, error, reason, source, quality,
   emotion, packet. The packet carries session/window/available/evidence_id.
   Its legacy 'quality' and 'probabilities' fields may be OMITTED entirely.
B) Native, packet-free eav.quality_controller.report.v1:
   schema, modality, session_id, window_id, window (scorer identity), available,
   status, algorithm_error, error, reason, source, evidence_id, quality;
   emotion is optional.

Availability is a producer declaration that is cross-checked, not inferred
from a quality value. A missing/broken report is NOT evidence of sensor loss.
A main.py ERROR/MODULE_ERROR_EXCLUDED report is a control error even if its
placeholder packet says available=False. NO_DECISION requires three validated
unavailable reports and a valid request/window identity.

Window identity:
* EAV replay: descriptor.identity.{pair_key,window_idx_0based} together with
  matching descriptor modality window_index determine the original trial-relative
  [5*w,5*(w+1)] span. The generated clock label explicitly says trial_relative;
  it is NOT a measured wall/monotonic timestamp. No arbitrary 0..5 fallback.
* Other replay: set descriptor.quality_window BEFORE producing the reports, with
  session_id/window_id/clock_id/start_seconds/end_seconds for the actual source.
  Recorded monotonic capture descriptors are also accepted in replay; timing
  consistency is checked, but their age relative to the current clock is not.
* Shadow: use the current host/source monotonic clock, descriptor.clock_id and
  window_end_monotonic and per-source timing. Caller must pass trusted NOW and
  clock_id; NEVER take 'now' from an input JSON. All source spans must be aligned
  exactly (1e-8 duration tolerance only); no timestamp relabelling/resampling.
  This controller is deliberately stricter than the fusion synchronizer's 0.1s
  alignment tolerance. Misaligned source windows must be fixed upstream.

Only DECLARED source/evidence hashes are compared. No raw files are opened or
rehashed, and no underlying sensor content/clock truth is independently proven.
main.py V1 detector dictionaries do not independently bind every feature to a
source hash: the trusted producing Runtime.process supplies that binding.
The native format can carry a full window identity but has the same trust limit.
Do not attach a new identity to an old report to make a validation error vanish.

Replay is stateless and deterministic except processing_seconds. Shadow rejects
stale/future data, but does not provide an exactly-once dispatcher. A scheduler
must prevent duplicate publication. No previous scores or decisions are cached.

CLI:
    python Quality/quality_controller.py --preflight-only --output report.json
    python Quality/quality_controller.py --self-test
    python Quality/quality_controller.py --write-example example.json
    python Quality/quality_controller.py --write-example native.json --example-format native
    python Quality/quality_controller.py --input-json example.json --output result.json

Input JSON: {"schema": INPUT_SCHEMA, "source_window": ..., "reports": ...}
Batch: {"schema": BATCH_SCHEMA, "samples": [<input>, ...]}
An old main.py log containing only heads/fusion and no descriptor is NOT enough
for this interface. Supply the original descriptor; never reconstruct it by guess.

Exit codes: 0 valid request (also ROBUST_FUSION/NO_DECISION); 2 contract/load/I/O
error; 3 failed tests. Outputs use exclusive creation, never overwrite. Inference
errors returned by this controller must block the candidate path; never silently
use a baseline answer and count it as a successful new-policy prediction.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
import math
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tempfile
import threading
import time
from typing import Any

VERSION = "EAV-QUALITY-CONTROLLER.1.0"
INPUT_SCHEMA = "eav.quality_controller.input.v1"
BATCH_SCHEMA = "eav.quality_controller.batch.v1"
OUTPUT_SCHEMA = "eav.quality_controller.result.v1"
NATIVE_REPORT_SCHEMA = "eav.quality_controller.report.v1"
SOURCE_SCHEMA = "eav.system.source_window.v1"
MAIN_PACKET_SCHEMA = "eav.af4c.modality.packet.v1"
MODALITIES = ("eeg", "audio", "video")
REVIEWED_SCORER_SHA256 = "23dd2d8074ce2adb548e3c8d71ac0ee2fb92b6cb88d24337aae600d2607c1d28"
REVIEWED_SCORER_VERSION = "EAV-DISTRIBUTION-QUALITY-RUNTIME.1.0"
REVIEWED_BUNDLE_SHA256 = "b9fd67533622c0de733ec7e5c0f4b5a80cb9819b74b07f57700b623c5a79d503"
TEST_SUBJECTS = frozenset(("subject03", "subject05", "subject20", "subject31", "subject35", "subject39"))
ERROR_STATES = frozenset(("ERROR", "FAIL", "FAILED", "QUALITY_CONTROL_ERROR", "INFERENCE_ERROR", "MODEL_ERROR"))
MAX_JSON_BYTES = 32 * 1024 * 1024
_MISSING = object()
_IMPORT_LOCK = threading.RLock()


class QualityControllerError(ValueError):
    """Invalid controller configuration or source/report contract."""


def _need(ok: Any, message: str) -> None:
    if not ok:
        raise QualityControllerError(message)


def _obj(value: Any, label: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{label}: expected an object/mapping")
    return value


def _text(value: Any, label: str) -> str:
    _need(isinstance(value, str) and value == value.strip() and 0 < len(value) <= 2048,
          f"{label}: expected a nonempty trimmed string (<=2048 chars)")
    return value


def _real(value: Any, label: str) -> float:
    _need(isinstance(value, Real) and not isinstance(value, bool),
          f"{label}: expected a finite real scalar, not {type(value).__name__}")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise QualityControllerError(f"{label}: not representable as float64") from exc
    _need(math.isfinite(result), f"{label}: NaN/Inf/overflow is forbidden")
    return result


def _bool(value: Any, label: str) -> bool:
    # main.py accepts bool or integer 0/1. Normalize this boundary ONLY. The
    # distribution scorer still receives strict built-in bools, never strings.
    if type(value) is bool:
        return value
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(value)
    raise QualityControllerError(f"{label}: expected bool or integer 0/1; no truthiness coercion")


def _index(value: Any, label: str) -> int:
    n = _real(value, label)
    _need(n == int(n) and 0 <= n <= 1000000000, f"{label}: nonnegative integer required")
    return int(n)


def _sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None,
          f"{label}: expected a SHA256 hex digest")
    return value.lower()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_path(a: str, b: str) -> bool:
    # Compare declarations lexically, including Windows paths on a Linux test
    # host. Do not resolve/read source files and do not guess a path mapping.
    windows = bool(re.match(r"^[A-Za-z]:", a) or re.match(r"^[A-Za-z]:", b) or "\\" in a or "\\" in b)
    cls = PureWindowsPath if windows else PurePosixPath
    return cls(a) == cls(b)


def _error(code: str, message: str, modality: str | None = None) -> dict[str, Any]:
    result = {"code": code, "message": message}
    if modality is not None:
        result["modality"] = modality
    return result


def _load_scorer() -> Any:
    """Load the exact sibling, not a module with the same name on sys.path."""
    path = Path(__file__).resolve().with_name("distribution_quality.py")
    _need(path.is_file(), f"Missing scorer: {path}. Keep both Python scripts in Quality/.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    _need(digest == REVIEWED_SCORER_SHA256,
          f"Scorer SHA256 mismatch: {digest}; expected {REVIEWED_SCORER_SHA256}. "
          "Use the unmodified reviewed distribution_quality.py, not a reformatted copy.")
    name = "_eav_qc_scorer_" + hashlib.sha256(str(path).encode()).hexdigest()[:24]
    with _IMPORT_LOCK:
        if name in sys.modules:
            module = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            _need(spec is not None and spec.loader is not None, "Cannot create scorer module spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module  # dataclasses needs the registered module
            try:
                # Execute exactly the bytes that were hashed, not a second read.
                exec(compile(raw, str(path), "exec"), module.__dict__)
            except BaseException:
                sys.modules.pop(name, None)
                raise
    _need(module.VERSION == REVIEWED_SCORER_VERSION, "Unsupported scorer API version")
    return module


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in items:
        _need(key not in out, f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _bad_constant(token: str) -> None:
    raise QualityControllerError(f"Nonstandard JSON numeric constant: {token}")


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser()
    _need(p.is_file(), f"JSON file not found: {p}")
    _need(p.stat().st_size <= MAX_JSON_BYTES, f"JSON exceeds {MAX_JSON_BYTES} bytes")
    raw = p.read_bytes()
    _need(len(raw) <= MAX_JSON_BYTES, "JSON grew beyond size limit while reading")
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs, parse_constant=_bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QualityControllerError(f"Invalid UTF-8 JSON: {p}: {exc}") from exc


def write_json(path: str | Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


class QualityController:
    """Stateless controller using only the reviewed distribution scorer.

    Parameters are loaded/checked once. API input/report failures return a
    structured QUALITY_CONTROL_ERROR; construction failures raise. Caller owns
    source capture, truthful metadata, immutable completed reports, publication
    deduplication, and any later fusion integration.
    """

    def __init__(self, params_path: str | Path | None = None, *,
                 expected_sha256: str = REVIEWED_BUNDLE_SHA256,
                 max_age_seconds: float = 1.0,
                 future_tolerance_seconds: float = 0.02,
                 newest_sample_tolerance_seconds: float = 0.10,
                 allow_test: bool = False) -> None:
        self._dq = _load_scorer()
        self._scorer = self._dq.DistributionQualityScorer(params_path, expected_sha256=expected_sha256)
        self.max_age_seconds = _real(max_age_seconds, "max_age_seconds")
        self.future_tolerance_seconds = _real(future_tolerance_seconds, "future_tolerance_seconds")
        self.newest_sample_tolerance_seconds = _real(newest_sample_tolerance_seconds, "newest_sample_tolerance_seconds")
        _need(0 <= self.max_age_seconds <= 30, "max_age_seconds must be in [0,30]")
        _need(0 <= self.future_tolerance_seconds <= .1, "future_tolerance_seconds must be in [0,.1]")
        _need(0 <= self.newest_sample_tolerance_seconds <= .5,
              "newest_sample_tolerance_seconds must be in [0,.5]")
        self.allow_test = _bool(allow_test, "allow_test")
        self._identity = {
            "controller_version": VERSION,
            "controller_script_sha256": _file_sha(Path(__file__)),
            "scorer": self._scorer.identity,
            "timing_policy": {"max_age_seconds": self.max_age_seconds,
                "future_tolerance_seconds": self.future_tolerance_seconds,
                "newest_sample_tolerance_seconds": self.newest_sample_tolerance_seconds,
                "source_span_alignment": "exact; no resampling or relabelling",
                "limits_are_engineering_settings_not_learned": True},
            "allow_test": self.allow_test,
            "legacy_q_used": False,
            "fusion_quality_mapping_implemented": False,
        }

    @property
    def identity(self) -> dict[str, Any]:
        return deepcopy(self._identity)

    @property
    def feature_order(self) -> dict[str, list[str]]:
        return self._scorer.feature_order

    @property
    def thresholds(self) -> dict[str, float]:
        return self._scorer.thresholds

    def _window(self, descriptor: Any, mode: str, now: Any, clock: Any) -> tuple[Mapping, dict, str]:
        d = _obj(descriptor, "source_window")
        _need(d.get("schema") == SOURCE_SCHEMA, f"source_window.schema must be {SOURCE_SCHEMA}")
        sid, wid = _text(d.get("session_id"), "session_id"), _text(d.get("window_id"), "window_id")
        _need(_real(d.get("window_seconds"), "window_seconds") == 5., "Only 5-second source windows are supported")
        _need(_text(d.get("task_condition"), "task_condition").lower() == "speaking", "Current EAV policy is Speaking-only")
        _need(mode in ("replay", "shadow"), "Only mode='replay' or mode='shadow'; no production/live control")
        split = d.get("split", "replay" if mode == "replay" else "live")
        _need(split in ("train", "val", "test", "replay", "live", "synthetic"), "Invalid source split")
        identity = _obj(d.get("identity", {}), "source_window.identity")
        subject = identity.get("subject")
        if subject is not None:
            subject = _text(subject, "subject").lower()
            match = re.fullmatch(r"subject0*(\d+)", subject)
            if match:
                subject = f"subject{int(match[1]):02d}"
        _need(self.allow_test or (split != "test" and subject not in TEST_SUBJECTS),
              "TEST scoring requires explicit allow_test=True / --allow-test; never used for fitting")
        items = _obj(d.get("modalities"), "source_window.modalities")
        _need(set(items) == set(MODALITIES), "Exactly eeg/audio/video descriptors required; explicit present=false for absence")
        for m in MODALITIES:
            item = _obj(items[m], f"descriptor.{m}")
            _bool(item.get("present"), f"descriptor.{m}.present")
            for field in ("capture_ok", "continuous"):
                if field in item:
                    _bool(item[field], f"descriptor.{m}.{field}")
        if mode == "shadow":
            cid = _text(d.get("clock_id"), "source_window.clock_id")
            _need(_text(clock, "trusted caller clock_id") == cid, "Caller/source clock IDs disagree")
            current = _real(now, "trusted caller now_seconds")
            end = _real(d.get("window_end_monotonic"), "window_end_monotonic")
            _need(end - current <= self.future_tolerance_seconds, "Source decision window is in the future")
            window = {"session_id": sid, "window_id": wid, "clock_id": cid,
                      "start_seconds": end - 5., "end_seconds": end}
            binding = "declared_same_host_monotonic_capture_span"
        elif "quality_window" in d:
            window = self._dq.WindowIdentity.from_mapping(d["quality_window"]).to_dict()
            binding = "explicit_replay_source_span"
        elif "window_end_monotonic" in d or "clock_id" in d:
            cid = _text(d.get("clock_id"), "recorded replay clock_id")
            end = _real(d.get("window_end_monotonic"), "recorded replay window_end_monotonic")
            window = {"session_id": sid, "window_id": wid, "clock_id": cid,
                      "start_seconds": end - 5., "end_seconds": end}
            binding = "recorded_monotonic_capture_span; REPLAY_no_wall_clock_freshness"
        else:
            # No timestamps are invented for arbitrary old replay logs.
            pair = _text(identity.get("pair_key"), "identity.pair_key (or provide quality_window)")
            w = _index(identity.get("window_idx_0based"), "identity.window_idx_0based (or provide quality_window)")
            _need(w in range(4), "EAV trial window index must be 0..3")
            for m in MODALITIES:
                item = items[m]
                if _bool(item["present"], m + ".present"):
                    _need(_index(item.get("window_index"), m + ".window_index") == w,
                          f"{m}: descriptor window index differs from paired EAV window")
            window = {"session_id": sid, "window_id": wid,
                      "clock_id": "eav_trial_relative:" + pair,
                      "start_seconds": float(w * 5), "end_seconds": float((w + 1) * 5)}
            binding = "EAV_pair_and_window_index; trial_relative_NOT_wall_clock"
        window = self._dq.WindowIdentity.from_mapping(window).to_dict()
        _text(window["clock_id"], "window.clock_id")
        if "window_idx_0based" in identity:
            wi = _index(identity["window_idx_0based"], "identity.window_idx_0based")
            for m in MODALITIES:
                if "window_index" in items[m]:
                    _need(_index(items[m]["window_index"], m + ".window_index") == wi,
                          f"{m}: paired EAV index and descriptor index disagree")
        if "quality_window" in d and "window_end_monotonic" in d:
            _need(window["end_seconds"] == _real(d["window_end_monotonic"], "window_end_monotonic")
                  and window["clock_id"] == d.get("clock_id"), "Explicit and recorded source spans disagree")
        _need(window["session_id"] == sid and window["window_id"] == wid, "quality_window identity contradicts descriptor")
        if mode == "shadow" and "quality_window" in d:
            _need(self._dq.WindowIdentity.from_mapping(d["quality_window"]).to_dict() == window,
                  "quality_window differs from declared capture span")
        return d, window, binding

    def _check_part(self, part: Any, label: str, m: str, window: Mapping) -> None:
        if part is None:
            return
        part = _obj(part, label)
        error = part.get("error")
        _need(error is None or error == "", f"{label}: producer error: {str(error)[:400]}")
        status = part.get("status")
        if status is not None:
            _need(isinstance(status, str), f"{label}.status: expected string")
            _need(status.upper() not in ERROR_STATES, f"{label}: producer status={status}")
        if "algorithm_error" in part:
            _need(not _bool(part["algorithm_error"], label + ".algorithm_error"), f"{label}: algorithm_error")
        for key in ("session_id", "window_id"):
            if key in part:
                _need(part[key] == window[key], f"{label}: mismatched {key}")
        if "modality" in part:
            _need(part["modality"] == m, f"{label}: mismatched modality")
        if "window_seconds" in part:
            _need(_real(part["window_seconds"], label + ".window_seconds") == 5., f"{label}: not a 5s report")
        if "quality_window" in part:
            _need(self._dq.WindowIdentity.from_mapping(part["quality_window"]).to_dict() == window,
                  f"{label}: wrong quality_window")

    def _source(self, m: str, item: Mapping, report: Mapping, packet: Mapping | None) -> dict:
        source = _obj(report.get("source"), f"{m}.source")
        source_name = _text(source.get("source"), f"{m}.source.source")
        if "samples" in item:
            _need(not item.get("path"), f"{m}: samples and path are ambiguous")
            _need(source_name == "adapter_array" and m != "video", f"{m}: incorrect in-memory source binding")
        else:
            declared = _text(item.get("path"), f"descriptor.{m}.path")
            _need(_same_path(source_name, declared), f"{m}: report source path differs from source descriptor")
        for key in ("window_index", "trial_index"):
            if key in source:
                _need(key in item and _index(source[key], "source." + key) == _index(item[key], "descriptor." + key),
                      f"{m}: source/descriptor {key} mismatch")
        if "variable" in source and "variable" in item:
            actual, declared = source["variable"], item["variable"]
            _need(actual == declared or {actual, declared} == {"seg", "seg1"}, f"{m}: EEG variable mismatch")
        source_digest = source.get("source_sha256", source.get("sha256"))
        source_digest = _sha(source_digest, f"{m}.source file/array SHA256")
        if "source_sha256" in source and "sha256" in source:
            _need(_sha(source["sha256"], m + ".source.sha256") == source_digest,
                  f"{m}: contradictory source file digest fields")
        for key in ("session_id", "window_id"):
            if key in source:
                expected = report.get(key) if packet is None else packet.get(key)
                _need(source[key] == expected, f"{m}: source {key} mismatch")
        window_digest = source.get("window_sha256")
        if window_digest is None:
            _need(source_name == "adapter_array", f"{m}: missing window_sha256 from file reader")
            window_digest = source_digest
        window_digest = _sha(window_digest, f"{m}.window_sha256")
        evidence = report.get("evidence_id") if packet is None else packet.get("evidence_id")
        _need(_sha(evidence, f"{m}.evidence_id") == window_digest, f"{m}: packet/report evidence ID differs from source window hash")
        for name, observed in (("source_sha256", source_digest), ("sha256", source_digest), ("window_sha256", window_digest)):
            if name in item:
                _need(_sha(item[name], f"descriptor.{m}.{name}") == observed, f"{m}: declared {name} mismatch")
        result = {"source": source_name, "source_sha256": source_digest, "window_sha256": window_digest,
                  "evidence_id": window_digest, "declarations_consistent": True, "raw_bytes_rehashed": False}
        for key in ("window_index", "trial_index"):
            if key in item:
                result[key] = _index(item[key], m + "." + key)
        return result

    def _timing(self, m: str, item: Mapping, packet: Mapping | None, window: Mapping,
                available: bool, reason: str, mode: str, now: Any) -> dict:
        if not _bool(item["present"], m + ".present"):
            return {"checked": False, "freshness_checked": False, "reason": "NO_SOURCE"}
        has_timing = "timing" in item or (packet is not None and "timing" in packet)
        if mode == "replay" and not has_timing:
            return {"checked": False, "freshness_checked": False, "reason": "OFFLINE_REPLAY_NO_CAPTURE_CLOCK"}
        t = _obj(item.get("timing"), f"descriptor.{m}.timing")
        cid = _text(t.get("clock_id"), m + ".timing.clock_id")
        start = _real(t.get("window_start_monotonic"), m + ".capture_start")
        end = _real(t.get("window_end_monotonic"), m + ".capture_end")
        newest = _real(t.get("newest_sample_monotonic"), m + ".newest_sample")
        _need(cid == window["clock_id"], f"{m}: capture clock mismatch")
        _need(start == window["start_seconds"] and end == window["end_seconds"],
              f"{m}: capture span mismatch; controller does not relabel/resample source windows")
        _need(start <= newest <= end + self.future_tolerance_seconds, f"{m}: invalid newest sample time")
        if packet is not None:
            pt = _obj(packet.get("timing"), m + ".packet.timing")
            for key in ("clock_id", "window_start_monotonic", "window_end_monotonic", "newest_sample_monotonic"):
                _need(pt.get(key) == t[key], f"{m}: packet/descriptor timing mismatch: {key}")
        if mode == "replay":
            if available:
                _need(end - newest <= self.newest_sample_tolerance_seconds,
                      f"{m}: recorded newest sample does not cover window end")
            return {"checked": True, "freshness_checked": False, "clock_id": cid,
                    "start_seconds": start, "end_seconds": end,
                    "newest_sample_seconds": newest, "age_seconds": None,
                    "reason": "RECORDED_TIMING_CHECKED_WITHOUT_CURRENT_AGE"}
        current = _real(now, "now_seconds")
        _need(newest <= current + self.future_tolerance_seconds, f"{m}: newest sample is in future")
        age = current - newest
        if available:
            _need(end - newest <= self.newest_sample_tolerance_seconds, f"{m}: newest sample does not cover window end")
            _need(age <= self.max_age_seconds, f"{m}: available report has stale newest sample")
        if reason == "STALE_BEFORE_SOURCE_READ":
            _need(age > self.max_age_seconds and not available, f"{m}: unsupported stale-source declaration")
        return {"checked": True, "freshness_checked": True, "clock_id": cid, "start_seconds": start,
                "end_seconds": end, "newest_sample_seconds": newest, "age_seconds": age}

    def _metrics(self, m: str, quality: Mapping) -> dict[str, float]:
        # Select only exact required features. Do not copy/read unused legacy q,
        # class probabilities, labels, or old quality-state mappings.
        values = {}
        for path in self.feature_order[m]:
            flat = quality.get(path, _MISSING)
            nested: Any = quality
            for component in path.split("."):
                if not isinstance(nested, Mapping) or component not in nested:
                    nested = _MISSING
                    break
                nested = nested[component]
            _need(not (flat is not _MISSING and nested is not _MISSING),
                  f"{m}.{path}: ambiguous flat AND nested representations")
            raw = flat if flat is not _MISSING else nested
            _need(raw is not _MISSING, f"{m}: missing required feature {path}")
            values[path] = _real(raw, f"{m}.{path}")
        return values

    def _adapt(self, m: str, report: Any, d: Mapping, window: Mapping,
               mode: str, now: Any) -> tuple[dict, dict]:
        r = _obj(report, f"reports.{m}")
        item = d["modalities"][m]
        # ERROR tests precede use of the placeholder availability.
        _need("algorithm_error" in r, f"{m}: missing algorithm_error declaration")
        _need(not _bool(r["algorithm_error"], m + ".algorithm_error"),
              f"{m}: algorithm_error=True; cannot interpret error placeholder as sensor loss")
        _need("error" in r, f"{m}: missing explicit error field")
        _need(r["error"] is None or r["error"] == "", f"{m}: producer error: {str(r['error'])[:400]}")
        status = r.get("status")
        _need(status in ("OK", "UNAVAILABLE"), f"{m}: unexpected report status={status!r}")
        reason = _text(r.get("reason"), m + ".reason")
        _need(reason != "MODULE_ERROR_EXCLUDED", f"{m}: excluded module error is NOT source absence")
        _need(r.get("modality") == m and r.get("window_id") == window["window_id"], f"{m}: report identity mismatch")
        if "session_id" in r:
            _need(r["session_id"] == window["session_id"], f"{m}: report session mismatch")
        if "window_seconds" in r:
            _need(_real(r["window_seconds"], m + ".window_seconds") == 5., f"{m}: report duration is not 5s")
        if "window" in r:
            _need(self._dq.WindowIdentity.from_mapping(r["window"]).to_dict() == window,
                  f"{m}: report source window mismatch")
        native = r.get("schema") == NATIVE_REPORT_SCHEMA
        packet = None
        if native:
            _need("packet" not in r, f"{m}: native report must not also contain a legacy packet")
            _need(r.get("session_id") == window["session_id"], f"{m}: native session_id is missing/wrong")
            _need(self._dq.WindowIdentity.from_mapping(r.get("window")).to_dict() == window,
                  f"{m}: native window span/clock/identity mismatch")
            available = _bool(r.get("available"), m + ".available")
        else:
            _need("schema" not in r, f"{m}: unsupported report schema {r.get('schema')!r}")
            packet = _obj(r.get("packet"), m + ".packet")
            _need(packet.get("schema") == MAIN_PACKET_SCHEMA, f"{m}: unsupported main.py packet schema")
            for key, expected in (("modality", m), ("session_id", window["session_id"]), ("window_id", window["window_id"])):
                _need(packet.get(key) == expected, f"{m}: packet {key} mismatch")
            _need(_real(packet.get("window_seconds"), m + ".packet.window_seconds") == 5., f"{m}: packet not 5s")
            if "window" in packet:
                _need(self._dq.WindowIdentity.from_mapping(packet["window"]).to_dict() == window, f"{m}: packet source window mismatch")
            available = _bool(packet.get("available"), m + ".packet.available")
            if "available" in r:
                _need(_bool(r["available"], m + ".available") == available, f"{m}: root and packet availability disagree")
        _need((status == "OK") == available, f"{m}: status and effective availability disagree")
        for field in ("quality", "emotion"):
            self._check_part(r.get(field), m + "." + field, m, window)
        quality, emotion = r.get("quality"), r.get("emotion")
        declared_unavailable = []
        for label, part in (("quality", quality), ("emotion", emotion)):
            if isinstance(part, Mapping) and m + "_available" in part:
                flag = _bool(part[m + "_available"], m + "." + label + ".available")
                _need(not available or flag, f"{m}: effective availability contradicts {label} availability")
                if not flag:
                    declared_unavailable.append(label)
        present = _bool(item["present"], m + ".present")
        capture_ok = _bool(item.get("capture_ok", True), m + ".capture_ok")
        continuous = _bool(item.get("continuous", True), m + ".continuous")
        _need(not available or (present and capture_ok and continuous), f"{m}: available despite missing/broken source")
        timing = self._timing(m, item, packet, window, available, reason, mode, now)
        source = None
        if r.get("source") is not None:
            _need(present, f"{m}: source report exists for descriptor present=false")
            source = self._source(m, item, r, packet)
        if available:
            _need(source is not None, f"{m}: available report needs a source identity/evidence hash")
            metrics = self._metrics(m, _obj(quality, m + ".quality"))
            availability_basis = "producer_available_with_consistent_source_and_report_contracts"
        else:
            metrics = None
            if not present:
                availability_basis = "descriptor_present_false"
            elif not capture_ok or not continuous:
                availability_basis = "descriptor_capture_failure_or_noncontiguous"
            elif reason == "STALE_BEFORE_SOURCE_READ" and mode == "shadow" and timing["checked"]:
                availability_basis = "current_clock_confirms_stale_source"
            else:
                _need(source is not None and bool(declared_unavailable),
                      f"{m}: unavailable lacks capture/observability evidence; cannot infer absence from q or report failure")
                availability_basis = "explicit_" + "_and_".join(declared_unavailable) + "_unavailable"
        elapsed = r.get("elapsed_seconds")
        if elapsed is not None:
            elapsed = _real(elapsed, m + ".elapsed_seconds")
            _need(elapsed >= 0, f"{m}: negative elapsed_seconds")
        adapted = {"modality": m, "window": dict(window), "metrics": metrics}
        audit = {"modality": m, "report_contract": "native_v1" if native else "main_v1",
                 "available": available, "reported_status": status, "availability_reason": reason,
                 "availability_basis": availability_basis, "source": source, "timing": timing,
                 "producer_processing_seconds": elapsed,
                 "raw_quality_feature_count": 0 if metrics is None else len(metrics),
                 "legacy_q_used": False}
        return adapted, audit

    def assess_window(self, *, source_window: Any, reports: Any, mode: str = "replay",
                      now_seconds: float | None = None, clock_id: str | None = None) -> dict[str, Any]:
        """Adapt Runtime.process reports or native reports; never modify inputs.

        Shadow callers supply now_seconds from the ACTUAL matching clock at
        assessment time. Replay deliberately ignores wall-clock age. On error,
        uncertain availability stays None rather than being rewritten to False.
        """
        start = time.perf_counter()
        out: dict[str, Any] = {
            "schema": OUTPUT_SCHEMA, "controller_version": VERSION,
            "status": "QUALITY_CONTROL_ERROR", "candidate_route": "QUALITY_CONTROL_ERROR",
            "mode": mode if isinstance(mode, str) else None, "candidate_only": True,
            "window": None, "window_identity_verified": False, "freshness_checked": False,
            "availability": {m: None for m in MODALITIES}, "availability_all_validated": False,
            "modalities": {}, "adapter_audit": {}, "errors": [], "reasons": [],
            "any_quality_alarm": None, "quality_for_fusion": None, "legacy_q_used": False,
            "active_router_modified": False, "fusion_executed": False,
            "production_approved": False, "live_robot_control_authorized": False,
            "robot_action_performed": False, "scorer_invoked": False,
            "scorer_input": None, "identity": self.identity,
            "source_bytes_independently_verified": False,
            "provenance_limit": "Declared identities/hashes only; detector/source pairing relies on trusted producer. No raw files reread.",
        }
        try:
            d, window, binding = self._window(source_window, mode, now_seconds, clock_id)
            out["window"], out["window_binding"] = window, binding
            rs = _obj(reports, "reports")
            _need(set(rs) == set(MODALITIES), "reports must contain exactly eeg/audio/video; missing report is a control error")
        except Exception as exc:
            code = "REQUEST_OR_WINDOW_CONTRACT_ERROR" if isinstance(exc, (ValueError, TypeError, OverflowError)) else "INTERNAL_CONTROLLER_ERROR"
            out["errors"].append(_error(code, f"{type(exc).__name__}: {exc}"))
            return self._finish(out, start)
        packets = {}
        for m in MODALITIES:
            try:
                packet, audit = self._adapt(m, rs[m], d, window, mode, now_seconds)
                packets[m], out["adapter_audit"][m] = packet, audit
                out["availability"][m] = audit["available"]
            except Exception as exc:
                code = "REPORT_CONTRACT_ERROR" if isinstance(exc, (ValueError, TypeError, OverflowError)) else "INTERNAL_CONTROLLER_ERROR"
                out["errors"].append(_error(code, f"{type(exc).__name__}: {exc}", m))
                out["adapter_audit"][m] = {"modality": m, "available": None,
                    "error": str(exc), "legacy_q_used": False}
        if out["errors"]:
            # Do not call the scorer with manufactured availability to work around
            # its all-missing priority. The controller has not established a mask.
            return self._finish(out, start)
        out["availability_all_validated"] = True
        out["window_identity_verified"] = True
        sample = {"schema": self._dq.INPUT_SCHEMA, "window": window,
                  "availability": dict(out["availability"]), "modalities": packets}
        out["scorer_input"] = sample
        # No current evidence -> valid NO_DECISION even if capture is old. Future
        # capture/contradictory identities are still rejected above. The scorer
        # itself does not need to read unavailable metrics.
        out["scorer_invoked"] = True
        try:
            result = self._scorer.score_window(sample, mode=mode, now_seconds=now_seconds,
                clock_id=clock_id, max_age_seconds=self.max_age_seconds,
                future_tolerance_seconds=self.future_tolerance_seconds)
        except Exception as exc:
            out["errors"].append(_error("SCORER_EXECUTION_ERROR", f"{type(exc).__name__}: {exc}"))
            return self._finish(out, start)
        out["distribution_result"] = result
        for key in ("status", "candidate_route", "modalities", "any_quality_alarm", "freshness_checked", "errors", "reasons"):
            out[key] = deepcopy(result[key])
        # For valid all-missing packets controller identity validation was already
        # performed even though the scorer correctly skips evidence evaluation.
        out["window_identity_verified"] = out["window_identity_verified"] and (
            result["window_identity_verified"] or result["candidate_route"] == "NO_DECISION")
        return self._finish(out, start)

    def _finish(self, out: dict, start: float) -> dict:
        if out["errors"]:
            out["status"] = out["candidate_route"] = "QUALITY_CONTROL_ERROR"
            out["reasons"] = list(dict.fromkeys(["QUALITY_CONTROL_ERROR"] + [e["code"] for e in out["errors"]]))
        if not out["modalities"]:
            for m in MODALITIES:
                out["modalities"][m] = {"modality": m, "available": out["availability"][m],
                    "status": "QUALITY_CONTROL_ERROR" if out["availability"][m] is None else "NOT_SCORED",
                    "B": None, "T": self.thresholds[m], "B_over_T": None, "exceeds_threshold": None,
                    "raw_features": {}, "evidence": {}, "quality_for_fusion": None}
        route = out["candidate_route"]
        out["downstream"] = {
            "quality_decision_valid": route in ("HEALTHY_FUSION", "ROBUST_FUSION", "NO_DECISION"),
            "withhold_candidate_emotion": route in ("QUALITY_CONTROL_ERROR", "NO_DECISION"),
            "recommended_branch": {"HEALTHY_FUSION": "F4", "ROBUST_FUSION": "AF4-B"}.get(route),
            "fusion_input_ready": False, "quality_for_fusion": None,
            "note": "Recommendation only. New quality-to-fusion adapter not implemented; never pass B as old q."}
        out["processing_seconds"] = time.perf_counter() - start
        return out

    def assess_input(self, payload: Any, *, mode: str = "replay",
                     now_seconds: float | None = None, clock_id: str | None = None) -> dict:
        """One explicit controller JSON envelope, not a legacy fusion-only log."""
        data = _obj(payload, "controller input")
        _need(data.get("schema") == INPUT_SCHEMA, f"Input schema must be {INPUT_SCHEMA}")
        return self.assess_window(source_window=data.get("source_window"), reports=data.get("reports"),
            mode=mode, now_seconds=now_seconds, clock_id=clock_id)

    def make_example(self, *, report_format: str = "main-v1") -> dict:
        """Synthetic contract fixture only; source paths/hashes are not real data."""
        _need(report_format in ("main-v1", "native"), "example format must be main-v1 or native")
        base = self._scorer.make_example()
        sid, wid, pair, w = "SYNTHETIC_CONTROLLER_EXAMPLE", "synthetic_window_1", "synthetic_trial", 1
        d = {"schema": SOURCE_SCHEMA, "session_id": sid, "window_id": wid, "window_seconds": 5.,
             "task_condition": "Speaking", "split": "synthetic",
             "identity": {"pair_key": pair, "window_idx_0based": w}, "modalities": {}}
        window = {"session_id": sid, "window_id": wid, "clock_id": "eav_trial_relative:" + pair,
                  "start_seconds": 5., "end_seconds": 10.}
        reports = {}
        for m in MODALITIES:
            path = f"SYNTHETIC_NOT_REAL/{m}_trial.bin"
            sh = hashlib.sha256((m + "synthetic_source").encode()).hexdigest()
            wh = hashlib.sha256((m + "synthetic_window").encode()).hexdigest()
            d["modalities"][m] = {"present": True, "path": path, "window_index": w}
            source = {"source": path, "sha256": sh, "window_sha256": wh, "window_index": w}
            quality = deepcopy(base["modalities"][m]["metrics"])
            quality.update({"status": "OK", "error": None, m + "_available": True})
            r = {"modality": m, "window_id": wid, "status": "OK", "reason": "CURRENT_CLASSIFIER_AND_QUALITY",
                 "algorithm_error": False, "error": None, "source": source, "quality": quality,
                 "emotion": {m + "_available": True, "status": "OK", "error": None}, "elapsed_seconds": 0.}
            if report_format == "main-v1":
                r["packet"] = {"schema": MAIN_PACKET_SCHEMA, "modality": m, "session_id": sid,
                               "window_id": wid, "window_seconds": 5., "available": True,
                               "evidence_id": wh, "reason": r["reason"]}
            else:
                r.update(schema=NATIVE_REPORT_SCHEMA, session_id=sid, window=dict(window), available=True, evidence_id=wh)
            reports[m] = r
        return {"schema": INPUT_SCHEMA, "real_measurements": False,
                "note": "Synthetic nominal metrics and dummy source hashes; not actual clean trials or raw-sensor validation.",
                "source_window": d, "reports": reports}

    def preflight(self) -> dict:
        scoring = self._scorer.preflight()
        testing = run_self_tests(self)
        return {"status": "PASS" if scoring["status"] == testing["status"] == "PASS" else "FAIL",
                "controller_version": VERSION, "identity": self.identity,
                "thresholds": self.thresholds, "feature_order": self.feature_order,
                "controller_tests": testing, "scorer_preflight": scoring,
                "legacy_q_used": False, "real_detector_reports_executed": False,
                "raw_sensor_data_used": False, "main_py_modified": False,
                "fusion_adapter_validated": False, "production_approved": False,
                "live_robot_control_authorized": False}


def _without_timing(result: dict) -> dict:
    copy = deepcopy(result)
    copy.pop("processing_seconds", None)
    return copy


def run_self_tests(controller: QualityController) -> dict:
    """Behavior tests with real bundle + synthetic reports (not accuracy tests)."""
    c = controller
    checks = []

    def check(name: str, fn: Any) -> None:
        try:
            value = fn()
            if value is False:
                raise AssertionError(name)
            checks.append({"name": name, "status": "PASS"})
        except Exception as exc:
            checks.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})

    def route(data: dict, **kw: Any) -> str:
        return c.assess_input(data, **kw)["candidate_route"]

    def mutated(fn: Any, expected: str = "QUALITY_CONTROL_ERROR", **kw: Any) -> bool:
        data = c.make_example()
        fn(data)
        return route(data, **kw) == expected

    def absent(data: dict, m: str) -> None:
        data["source_window"]["modalities"][m]["present"] = False
        r = data["reports"][m]
        r.update(status="UNAVAILABLE", reason="NO_CURRENT_SOURCE", source=None, quality=None, emotion=None)
        if "packet" in r:
            r["packet"].update(available=False, reason="NO_CURRENT_SOURCE")
            r["packet"].pop("evidence_id", None)
        else:
            r["available"] = False
            r.pop("evidence_id", None)

    def shadow_example() -> dict:
        data = c.make_example()
        d = data["source_window"]
        d.update(clock_id="synthetic_monotonic", window_end_monotonic=100., split="live")
        for m in MODALITIES:
            tm = {"clock_id": "synthetic_monotonic", "window_start_monotonic": 95.,
                  "window_end_monotonic": 100., "newest_sample_monotonic": 100.}
            d["modalities"][m]["timing"] = dict(tm)
            data["reports"][m]["packet"]["timing"] = dict(tm)
        return data

    nominal = c.make_example()
    check("main_v1_nominal", lambda: route(nominal) == "HEALTHY_FUSION")
    check("native_without_packets_or_old_q", lambda: route(c.make_example(report_format="native")) == "HEALTHY_FUSION")
    check("no_input_mutation", lambda: (c.assess_input(nominal), nominal == c.make_example())[1])
    check("repeat_replay_no_cache", lambda: _without_timing(c.assess_input(nominal)) == _without_timing(c.assess_input(nominal)))
    check("thresholds_come_from_scorer", lambda: all(c.assess_input(nominal)["modalities"][m]["T"] == c.thresholds[m] for m in MODALITIES))
    check("scorer_once_and_direct_parity", lambda: c.assess_input(nominal)["distribution_result"] == c._scorer.score_window(c.assess_input(nominal)["scorer_input"]))
    for bits in itertools.product((False, True), repeat=3):
        def combo(bits=bits):
            data = c.make_example()
            for m, value in zip(MODALITIES, bits):
                if not value:
                    absent(data, m)
            expected = "NO_DECISION" if not any(bits) else "HEALTHY_FUSION" if all(bits) else "ROBUST_FUSION"
            result = c.assess_input(data)
            return result["candidate_route"] == expected and result["availability"] == dict(zip(MODALITIES, bits))
        check("availability_" + "".join(str(int(b)) for b in bits), combo)
    for m in MODALITIES:
        check(m + "_algorithm_error_not_absence", lambda m=m: mutated(lambda x: x["reports"][m].update(algorithm_error=True)))
        check(m + "_failed_status_rejected", lambda m=m: mutated(lambda x: x["reports"][m].update(status="ERROR")))
        check(m + "_error_field_rejected", lambda m=m: mutated(lambda x: x["reports"][m].update(error="failed")))
        check(m + "_missing_report_rejected", lambda m=m: mutated(lambda x: x["reports"].pop(m)))
        check(m + "_wrong_report_window", lambda m=m: mutated(lambda x: x["reports"][m].update(window_id="old")))
        check(m + "_wrong_packet_session", lambda m=m: mutated(lambda x: x["reports"][m]["packet"].update(session_id="old_session")))
        check(m + "_wrong_evidence_hash", lambda m=m: mutated(lambda x: x["reports"][m]["packet"].update(evidence_id="0" * 64)))
        check(m + "_wrong_source_path", lambda m=m: mutated(lambda x: x["reports"][m]["source"].update(source="wrong/path")))
        check(m + "_wrong_source_index", lambda m=m: mutated(lambda x: x["reports"][m]["source"].update(window_index=3)))
        check(m + "_missing_raw_report", lambda m=m: mutated(lambda x: x["reports"][m].update(quality=None)))
        check(m + "_nested_quality_error", lambda m=m: mutated(lambda x: x["reports"][m]["quality"].update(error="detector failed")))
        check(m + "_nested_emotion_error", lambda m=m: mutated(lambda x: x["reports"][m]["emotion"].update(status="ERROR")))
        check(m + "_false_quality_availability", lambda m=m: mutated(lambda x: x["reports"][m]["quality"].update({m + "_available": False})))
        check(m + "_available_without_source", lambda m=m: mutated(lambda x: x["reports"][m].update(source=None)))
        for value, label in (("false", "string"), (2, "out_of_range"), (0.0, "float")):
            check(m + "_bad_available_" + label, lambda m=m, value=value: mutated(lambda x: x["reports"][m]["packet"].update(available=value)))
    def fail_all():
        data = c.make_example()
        for m in MODALITIES:
            absent(data, m)
            data["reports"][m].update(status="ERROR", algorithm_error=True, error="worker crash")
        result = c.assess_input(data)
        return result["candidate_route"] == "QUALITY_CONTROL_ERROR" and not result["scorer_invoked"]
    check("three_algorithm_errors_never_NO_DECISION", fail_all)
    check("bad_request_even_when_all_missing", lambda: mutated(lambda x: ([absent(x,m) for m in MODALITIES], x["source_window"].update(schema="wrong"))))
    check("unsupported_mode_even_all_missing", lambda: mutated(lambda x: [absent(x,m) for m in MODALITIES], mode="live"))
    check("missing_window_span_not_fabricated", lambda: mutated(lambda x: x["source_window"].pop("identity")))
    check("descriptor_index_mismatch", lambda: mutated(lambda x: x["source_window"]["modalities"]["audio"].update(window_index=0)))
    check("missing_algorithm_error_flag", lambda: mutated(lambda x: x["reports"]["audio"].pop("algorithm_error")))
    check("unknown_modality_rejected", lambda: mutated(lambda x: x["reports"].update(text={})))
    check("test_data_requires_opt_in", lambda: mutated(lambda x: x["source_window"].update(split="test")) if not c.allow_test else True)
    for m in MODALITIES:
        for path in c.feature_order[m]:
            def invalid_feature(value: Any, m=m, path=path):
                data = c.make_example()
                q = data["reports"][m]["quality"]
                parts = path.split(".")
                for part in parts[:-1]:
                    q = q[part]
                q[parts[-1]] = value
                r = c.assess_input(data)
                return r["candidate_route"] == "QUALITY_CONTROL_ERROR" and not r["scorer_invoked"]
            for value, label in ((float("nan"), "NaN"), (float("inf"), "Inf"), ("0.5", "numeric_string"), (True, "bool"), ([1], "array")):
                check(f"{m}.{path}_{label}", lambda value=value, f=invalid_feature: f(value))
    def raw_change(m: str, parts: tuple, value: float):
        data = c.make_example()
        q = data["reports"][m]["quality"]
        for key in parts[:-1]:
            q = q[key]
        q[parts[-1]] = value
        r = c.assess_input(data)
        return r["candidate_route"] == "ROBUST_FUSION" and r["availability"][m] is True
    check("audio_low_rms_is_anomaly_not_absence", lambda: raw_change("audio", ("signal_metrics", "rms_dbfs"), -160.))
    check("video_low_technical_is_anomaly_not_absence", lambda: raw_change("video", ("physical_quality", "technical_raw"), -1.))
    check("eeg_line_noise_is_anomaly_not_absence", lambda: raw_change("eeg", ("features", "line_fraction_mean"), 1.))
    def old_q_invariance():
        first = c.assess_input(c.make_example())
        data = c.make_example()
        for m in MODALITIES:
            r = data["reports"][m]
            r["packet"]["quality"] = {"ignored": "not a quality number"}
            r["packet"]["probabilities"] = "IGNORED_BY_QUALITY_CONTROLLER"
            r["quality"]["q_" + m] = float("nan")
            r["quality"]["quality_state"] = "DEGRADED"
            r["emotion"]["confidence"] = -999
        return _without_timing(first) == _without_timing(c.assess_input(data))
    check("old_q_old_state_and_confidence_never_used", old_q_invariance)
    def label_invariance():
        data = c.make_example()
        data["source_window"]["reference_label"] = "not-consumed"
        data["source_window"]["corruption_family"] = "not-consumed"
        return _without_timing(c.assess_input(data)) == _without_timing(c.assess_input(c.make_example()))
    check("emotion_and_corruption_labels_not_used", label_invariance)
    def observability_absence():
        data = c.make_example()
        r = data["reports"]["video"]
        r.update(status="UNAVAILABLE", reason="NO_FACE")
        r["packet"]["available"] = False
        r["quality"]["video_available"] = False
        r["emotion"]["video_available"] = False
        r["quality"]["physical_quality"]["technical_raw"] = None
        return route(data) == "ROBUST_FUSION"
    check("valid_no_face_skips_DOVER_metrics", observability_absence)
    def unexplained_absence():
        data = c.make_example()
        data["reports"]["audio"].update(status="UNAVAILABLE", reason="UNEXPLAINED")
        data["reports"]["audio"]["packet"]["available"] = False
        return route(data) == "QUALITY_CONTROL_ERROR"
    check("unexplained_absence_rejected", unexplained_absence)
    shadow = shadow_example()
    shadow_kw = {"mode": "shadow", "now_seconds": 100. + min(.1, c.max_age_seconds * .5),
                 "clock_id": "synthetic_monotonic"}
    check("shadow_fresh", lambda: route(shadow, **shadow_kw) == "HEALTHY_FUSION")
    check("shadow_stale", lambda: route(shadow, **{**shadow_kw, "now_seconds": 101. + c.max_age_seconds}) == "QUALITY_CONTROL_ERROR")
    check("shadow_future", lambda: route(shadow, **{**shadow_kw, "now_seconds": 99.}) == "QUALITY_CONTROL_ERROR")
    check("shadow_wrong_clock", lambda: route(shadow, **{**shadow_kw, "clock_id": "other"}) == "QUALITY_CONTROL_ERROR")
    check("shadow_requires_now", lambda: route(shadow, mode="shadow", clock_id="synthetic_monotonic") == "QUALITY_CONTROL_ERROR")
    def shifted():
        data = shadow_example()
        for key in ("window_start_monotonic", "window_end_monotonic", "newest_sample_monotonic"):
            data["source_window"]["modalities"]["audio"]["timing"][key] += .01
            data["reports"]["audio"]["packet"]["timing"][key] += .01
        return route(data, **shadow_kw) == "QUALITY_CONTROL_ERROR"
    check("shadow_offset_not_silently_relabelled", shifted)
    def packet_time_mismatch():
        data = shadow_example()
        data["reports"]["audio"]["packet"]["timing"]["newest_sample_monotonic"] = 99.99
        return route(data, **shadow_kw) == "QUALITY_CONTROL_ERROR"
    check("shadow_packet_timestamp_mismatch", packet_time_mismatch)
    check("strict_json_serializable", lambda: bool(json.dumps(c.assess_input(nominal), allow_nan=False)))
    check("no_fusion_mapping_or_action", lambda: c.assess_input(nominal)["quality_for_fusion"] is None and not c.assess_input(nominal)["fusion_executed"])
    def io_check():
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "测试.json"
            write_json(path, nominal)
            if read_json(path) != nominal:
                return False
            try:
                write_json(path, {})
            except FileExistsError:
                return True
        return False
    check("unicode_JSON_roundtrip_no_overwrite", io_check)
    failed = sum(x["status"] == "FAIL" for x in checks)
    return {"status": "FAIL" if failed else "PASS", "tests": len(checks),
            "passed": len(checks) - failed, "failed": failed, "checks": checks,
            "real_parameter_bundle_used": True, "input_reports": "synthetic main_v1/native_v1 fixtures",
            "raw_detector_runs": False, "hardware_tested": False, "emotion_accuracy_evaluated": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=VERSION)
    a = p.add_mutually_exclusive_group()
    a.add_argument("--preflight-only", "--preflight", action="store_true", dest="preflight_only")
    a.add_argument("--self-test", action="store_true")
    a.add_argument("--write-example", metavar="PATH")
    a.add_argument("--input-json", metavar="PATH")
    p.add_argument("--params", help="Default: distribution_quality_params.json beside the scorer")
    p.add_argument("--expected-sha256", default=REVIEWED_BUNDLE_SHA256, help="Explicit trust pin for a compatible reviewed parameter bundle")
    p.add_argument("--output", help="New UTF-8 JSON; never overwrites")
    p.add_argument("--example-format", choices=("main-v1", "native"), default="main-v1")
    p.add_argument("--mode", choices=("replay", "shadow"), default="replay")
    p.add_argument("--clock-id", help="Shadow: trusted same-host monotonic clock ID; do not take it from input as verification")
    p.add_argument("--max-age-seconds", type=float, default=1.)
    p.add_argument("--future-tolerance-seconds", type=float, default=.02)
    p.add_argument("--newest-sample-tolerance-seconds", type=float, default=.10)
    p.add_argument("--allow-test", action="store_true", help="Explicit scoring-only opt-in; no threshold fitting")
    args = p.parse_args(argv)
    args.help_only = not any((args.preflight_only, args.self_test, args.write_example, args.input_json))
    if args.help_only:
        p.print_help()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = None
    try:
        args = parse_args(argv)
        if args.help_only:
            return 0
        _need(not (args.write_example and args.output), "--write-example owns its output path; do not also use --output")
        target = args.write_example or args.output
        if target:
            _need(not Path(target).expanduser().exists(), f"Output exists; choose a NEW filename: {target}")
        if args.mode == "shadow":
            _need(bool(args.input_json), "--mode shadow is only for an explicit input")
            _text(args.clock_id, "--clock-id")
        c = QualityController(args.params, expected_sha256=args.expected_sha256,
            max_age_seconds=args.max_age_seconds, future_tolerance_seconds=args.future_tolerance_seconds,
            newest_sample_tolerance_seconds=args.newest_sample_tolerance_seconds, allow_test=args.allow_test)
        if args.write_example:
            write_json(args.write_example, c.make_example(report_format=args.example_format))
            print(json.dumps({"status": "PASS", "output": str(Path(args.write_example).resolve()),
                              "real_measurements": False, "exit_code": 0}, ensure_ascii=False, indent=2))
            return 0
        if args.preflight_only:
            result = c.preflight()
        elif args.self_test:
            result = run_self_tests(c)
        else:
            payload = read_json(args.input_json)
            def assess(sample: Any) -> dict:
                # Capture current time here, never from stored JSON arguments.
                return c.assess_input(sample, mode=args.mode,
                    now_seconds=time.monotonic() if args.mode == "shadow" else None, clock_id=args.clock_id)
            if isinstance(payload, Mapping) and payload.get("schema") == BATCH_SCHEMA:
                samples = payload.get("samples")
                _need(isinstance(samples, list) and 0 < len(samples) <= 10000, "Batch requires 1..10000 samples")
                results = []
                for index, sample in enumerate(samples):
                    try:
                        results.append(assess(sample))
                    except QualityControllerError as exc:
                        results.append({"schema": OUTPUT_SCHEMA, "status": "QUALITY_CONTROL_ERROR",
                            "candidate_route": "QUALITY_CONTROL_ERROR", "batch_index": index,
                            "errors": [_error("INPUT_ENVELOPE_ERROR", str(exc))], "quality_for_fusion": None,
                            "candidate_only": True, "fusion_executed": False, "legacy_q_used": False,
                            "production_approved": False, "live_robot_control_authorized": False})
                result = {"schema": BATCH_SCHEMA + ".result", "controller_version": VERSION,
                          "status": "QUALITY_CONTROL_ERROR" if any(r["status"] == "QUALITY_CONTROL_ERROR" for r in results) else "OK",
                          "sample_count": len(results), "results": results, "legacy_q_used": False,
                          "candidate_only": True, "fusion_executed": False,
                          "production_approved": False, "live_robot_control_authorized": False}
            else:
                result = assess(payload)
        code = 3 if result["status"] == "FAIL" else 2 if result["status"] == "QUALITY_CONTROL_ERROR" else 0
        if args.output:
            write_json(args.output, result)
            summary = {"status": result["status"], "output": str(Path(args.output).resolve()), "exit_code": code}
            if "candidate_route" in result:
                summary["candidate_route"] = result["candidate_route"]
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except Exception as exc:
        # Never turn a software exception into a successful missing-sensor event.
        print(json.dumps({"status": "QUALITY_CONTROL_ERROR", "controller_version": VERSION,
            "candidate_route": "QUALITY_CONTROL_ERROR", "error_type": type(exc).__name__,
            "message": str(exc), "production_approved": False,
            "live_robot_control_authorized": False, "exit_code": 2}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
