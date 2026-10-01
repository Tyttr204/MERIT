#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic EAV distribution-quality runtime (candidate replay/shadow only).

This module consumes EXISTING quality-detector metrics, not EEG/audio/video
waveforms. It never fits a distribution, changes a threshold, loads an emotion
model, computes AF4-B quality inputs, or commands a robot.

Supported policy: eav.distribution_quality_policy.bundle.v1, exported on
2026-09-26: EEG hybrid_v2_gaussian_line; Audio/Video hybrid; alpha=0.05.
Python 3.10+; standard library only. All arithmetic uses Python float (binary64).

Placement:
    final_deployment/Quality/distribution_quality.py
    final_deployment/Quality/distribution_quality_params.json
The actual project directory may be named "最终部署". --params can point elsewhere.

CLI (run from 最终部署):
    python Quality/distribution_quality.py --self-test
    python Quality/distribution_quality.py --preflight-only
    python Quality/distribution_quality.py --preflight-only --verify-sources
    python Quality/distribution_quality.py --write-example quality_example.json
    python Quality/distribution_quality.py --input-json quality_example.json
    python Quality/distribution_quality.py --parity-json golden_full_reference.json

Python API:
    scorer = DistributionQualityScorer("Quality/distribution_quality_params.json")
    result = scorer.score_modality("audio", audio_quality, available=True)
    result = scorer.score_window(sample)  # replay, with aligned packet metadata

score_modality is ARITHMETIC ONLY: it cannot validate synchronization/freshness.
score_window checks explicit session/window/clock/span/modality metadata. Shadow
mode additionally requires caller-supplied current time and matching clock ID:
    scorer.score_window(sample, mode="shadow", now_seconds=now, clock_id=clock_id)
The caller must obtain that time from the actual source clock, not stale input.
Effective availability is an EXTERNAL, already validated bool for each modality.
This module cannot prove that a caller's availability/clock declarations are true.

Input schema (see --write-example for all 11 exact feature paths):
    {"schema": INPUT_SCHEMA,
     "window": {"session_id": "s1", "window_id": "w1", "clock_id": "replay",
                "start_seconds": 0.0, "end_seconds": 5.0},
     "availability": {"eeg": true, "audio": true, "video": true},
     "modalities": {
       "audio": {"modality": "audio", "window": <same identity>,
                 "metrics": {"dnsmos": {"OVRL_raw": 3.0},
                             "signal_metrics": {"rms_dbfs": -37.0}}}, ...}}
Metrics may use nested keys OR exact flat dotted keys. Duplicate encodings of
one required feature are rejected. Extra detector metrics are ignored. Numeric
strings, booleans, complex numbers, arrays, missing values and NaN/Inf are NOT
accepted as quality scalars. Missing evidence is never replaced with zero.

Route priority:
    all three externally unavailable -> NO_DECISION
    otherwise any quality-contract error -> QUALITY_CONTROL_ERROR
    otherwise any unavailable OR any B > T -> ROBUST_FUSION
    otherwise -> HEALTHY_FUSION
All routes are CANDIDATE recommendations, NOT changes to the frozen tau=.80 router.
B==T is nominal; no epsilon is added to the threshold. Unavailable B is None,
not zero. All feature evidence is capped at 12 BEFORE fixed-denominator RMS.

Integrity:
The default SHA256 pin is the exact supplied 46,809-byte bundle. Its embedded
policy fingerprint is recorded, NOT independently recomputed: the exporter's
fingerprint serialization algorithm is not included in that JSON. File-byte
integrity is instead verified against the separately embedded reviewed SHA256.
An optional .json.sha256/.sha256 sidecar is an ADDITIONAL consistency check.
--expected-sha256 is an explicit external trust pin for a compatible bundle; it
is NOT an option to skip verification or authorize production. Source-file hashes
can be checked with --verify-sources, optionally --source-root for relocated runs.
This is not a cryptographic signature or a hardware/accuracy certification.

Golden parity JSON:
    {"schema": PARITY_SCHEMA, "bundle_sha256": <the exact bundle digest>,
     "policy_fingerprint_sha256": <embedded fingerprint>,
     "reference_scope": "full_6_VAL_subjects_clean_5s_windows",
     "expected_results_origin": "name/version of independent offline scorer",
     "cases": [{"sample": <input above>, "expected": {
         "candidate_route": "HEALTHY_FUSION", "modalities": {
            "eeg": {"B": <number>, "exceeds_threshold": false,
                    "evidence": {<ALL exact feature paths>: <numbers>}}, ...}}}]}
Only complete, all-available, error-free arithmetic cases are accepted by this
parity gate. Do NOT compare full-reference runtime scores against five-subject
LOSO scores: those use different references. Golden results must be produced
independently; copying this runtime's outputs is not an independent parity test.

Exit codes: 0 success (including valid ROBUST_FUSION/NO_DECISION); 2 contract or
asset error; 3 failed self-test/parity. Outputs never overwrite existing files.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PureWindowsPath
import re
from statistics import NormalDist, median
import sys
import tempfile
from typing import Any

VERSION = "EAV-DISTRIBUTION-QUALITY-RUNTIME.1.0"
BUNDLE_SCHEMA = "eav.distribution_quality_policy.bundle.v1"
INPUT_SCHEMA = "eav.distribution_quality.input.v1"
OUTPUT_SCHEMA = "eav.distribution_quality.result.v1"
PARITY_SCHEMA = "eav.distribution_quality.parity.v1"
REFERENCE_SCOPE = "full_6_VAL_subjects_clean_5s_windows"
REVIEWED_BUNDLE_SHA256 = "b9fd67533622c0de733ec7e5c0f4b5a80cb9819b74b07f57700b623c5a79d503"
REVIEWED_POLICY_FINGERPRINT = "b085e33d4e1fb6c02c637b3c36a3530353e773299d30cea007f7a5e119c3f8b4"
MODALITIES = ("eeg", "audio", "video")
EVIDENCE_CAP = 12.0
TAIL_EPS = 1e-12
WINDOW_SECONDS = 5.0
ABS_TOL = 1e-10
REL_TOL = 1e-9
_NORMAL = NormalDist()
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_MISSING = object()

# Exact reviewed feature schema: feature_path, transform, direction, feature_type.
FEATURE_SPEC = {
    "eeg": (
        ("features.hf55_90_fraction_mean", "empirical", "higher_is_worse", "continuous"),
        ("features.line_fraction_mean", "mean_std_z", "higher_is_worse", "continuous"),
        ("features.pyprep_bad_fraction", "empirical", "higher_is_worse", "guard"),
        ("features.raw_flat_channel_fraction", "empirical", "higher_is_worse", "guard"),
        ("features.raw_hold_fraction_mean", "empirical", "higher_is_worse", "guard"),
        ("features.rms_uv_median", "empirical", "two_sided", "continuous"),
        ("features.slow02_1_fraction_mean", "empirical", "higher_is_worse", "continuous"),
    ),
    "audio": (
        ("dnsmos.OVRL_raw", "empirical", "lower_is_worse", "continuous"),
        ("signal_metrics.rms_dbfs", "mean_std_z", "lower_is_worse", "continuous"),
    ),
    "video": (
        ("face_observability.raw_detection_rate", "empirical", "lower_is_worse", "guard"),
        ("physical_quality.technical_raw", "mean_std_z", "lower_is_worse", "continuous"),
    ),
}


class QualityContractError(ValueError):
    """Invalid input, unsupported policy or failed asset integrity check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualityContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping/object")
    return value


def _real(value: Any, name: str) -> float:
    # bool subclasses int; reject before accepting Real (also rejects numpy.bool_).
    _require(not isinstance(value, bool) and isinstance(value, Real),
             f"{name}: expected a finite real scalar, not {type(value).__name__}")
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise QualityContractError(f"{name}: not representable as float64") from exc
    _require(math.isfinite(number), f"{name}: NaN/Inf or float64 overflow is forbidden")
    return number


def _text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value


def _digest(value: Any, name: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{name} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _bad_json_constant(value: str) -> None:
    raise QualityContractError(f"Nonstandard/nonfinite JSON constant: {value}")


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_unique_object, parse_constant=_bad_json_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise QualityContractError(f"Invalid UTF-8 JSON in {path}: {exc}") from exc


def write_json(path: str | Path, value: Any) -> None:
    """Serialize BEFORE opening, and use exclusive creation; never overwrite."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _get_feature(metrics: Mapping[str, Any], path: str) -> Any:
    flat = metrics.get(path, _MISSING)
    nested: Any = metrics
    for key in path.split("."):
        if not isinstance(nested, Mapping) or key not in nested:
            nested = _MISSING
            break
        nested = nested[key]
    if flat is not _MISSING and nested is not _MISSING:
        raise QualityContractError(f"{path}: ambiguous flat AND nested representations")
    result = flat if flat is not _MISSING else nested
    _require(result is not _MISSING, f"Missing required feature: {path}")
    return result


@dataclass(frozen=True)
class FeatureReference:
    path: str
    transform: str
    direction: str
    feature_type: str
    n: int
    values: tuple[float, ...] = ()
    mu: float | None = None
    sigma: float | None = None
    nominal_value: float = 0.0


def _evidence(x: float, ref: FeatureReference) -> tuple[float, dict[str, Any]]:
    """No fitting, interpolation, reference sorting or deduplication at runtime."""
    details: dict[str, Any] = {"transform": ref.transform, "direction": ref.direction,
                               "feature_type": ref.feature_type, "n_reference": ref.n}
    if ref.transform == "mean_std_z":
        if ref.mu is None or ref.sigma is None or ref.sigma <= 0:
            raise QualityContractError("Invalid standardized reference")
        # Finite x can produce an infinite intermediate z. The frozen cap still
        # applies; record overflow without emitting non-JSON Infinity/NaN.
        z = (x - ref.mu) / ref.sigma
        if ref.direction == "higher_is_worse":
            raw = max(0.0, z)
        elif ref.direction == "lower_is_worse":
            raw = max(0.0, -z)
        elif ref.direction == "two_sided":
            raw = abs(z)
        else:
            raise QualityContractError("Unsupported direction")
        details.update(mu=ref.mu, sigma_sample=ref.sigma,
                       z=z if math.isfinite(z) else None,
                       intermediate_overflow=not math.isfinite(z))
    elif ref.transform == "empirical":
        lower_count = bisect_right(ref.values, x)       # reference <= x
        upper_count = ref.n - bisect_left(ref.values, x)  # reference >= x
        p_lower = (1.0 + lower_count) / (ref.n + 1.0)
        p_upper = (1.0 + upper_count) / (ref.n + 1.0)
        if ref.direction == "two_sided":
            p = max(TAIL_EPS, min(1.0, 2.0 * min(p_lower, p_upper)))
            inv_argument = max(p / 2.0, TAIL_EPS)
        elif ref.direction in ("lower_is_worse", "higher_is_worse"):
            p = p_lower if ref.direction == "lower_is_worse" else p_upper
            inv_argument = min(1.0 - TAIL_EPS, max(TAIL_EPS, p))
        else:
            raise QualityContractError("Unsupported direction")
        # isf(p) = -ppf(p): avoid subtracting tiny p from 1.
        raw = max(0.0, -_NORMAL.inv_cdf(inv_argument))
        details.update(lower_count=lower_count, upper_count=upper_count,
                       p_lower=p_lower, p_upper=p_upper, tail_probability=p,
                       inverse_normal_argument=inv_argument)
    else:
        raise QualityContractError(f"Unsupported transform: {ref.transform}")
    e = min(EVIDENCE_CAP, raw)
    details.update(evidence=e, cap_applied=raw > EVIDENCE_CAP,
                   uncapped_evidence=raw if math.isfinite(raw) else None)
    return e, details


def _rms(evidence: Sequence[float], required_count: int) -> float:
    _require(len(evidence) == required_count and required_count > 0,
             "All required features, including zero guards, must be present")
    _require(all(math.isfinite(e) and 0.0 <= e <= EVIDENCE_CAP for e in evidence),
             "Invalid evidence before RMS")
    return math.sqrt(sum(e * e for e in evidence) / required_count)


def _route(availability: Mapping[str, bool], alarm: bool, has_error: bool) -> str:
    if not any(availability[m] for m in MODALITIES):
        return "NO_DECISION"
    if has_error:
        return "QUALITY_CONTROL_ERROR"
    if not all(availability[m] for m in MODALITIES) or alarm:
        return "ROBUST_FUSION"
    return "HEALTHY_FUSION"


@dataclass(frozen=True)
class WindowIdentity:
    session_id: str
    window_id: str
    clock_id: str
    start_seconds: float
    end_seconds: float

    @classmethod
    def from_mapping(cls, value: Any) -> WindowIdentity:
        obj = _mapping(value, "window")
        start = _real(obj.get("start_seconds"), "window.start_seconds")
        end = _real(obj.get("end_seconds"), "window.end_seconds")
        _require(math.isclose(end - start, WINDOW_SECONDS, rel_tol=0.0, abs_tol=1e-8),
                 "Quality metrics must describe exactly one 5-second source window")
        return cls(_text(obj.get("session_id"), "window.session_id"),
                   _text(obj.get("window_id"), "window.window_id"),
                   _text(obj.get("clock_id"), "window.clock_id"), start, end)

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "window_id": self.window_id,
                "clock_id": self.clock_id, "start_seconds": self.start_seconds,
                "end_seconds": self.end_seconds}


@dataclass(frozen=True)
class ModalityPolicy:
    modality: str
    recipe: str
    threshold: float
    features: tuple[FeatureReference, ...]


def _validate_policy(bundle: Mapping[str, Any]) -> dict[str, ModalityPolicy]:
    """Strictly support this reviewed recipe; refuse silent schema migrations."""
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "Unsupported bundle schema")
    _require(bundle.get("exporter_version") == "EAV-DISTRIBUTION-QUALITY-EXPORT.1.0",
             "Unsupported exporter version")
    _require(bundle.get("artifact_state") == "EXPORTED_CANDIDATE_NOT_DEPLOYABLE",
             "Only the exported candidate artifact is supported")
    act = _mapping(bundle.get("activation"), "activation")
    for key in ("production_approved", "live_robot_control_authorized",
                "active_configuration_modified", "existing_fusion_weights_modified",
                "AF4B_new_quality_mapping_selected"):
        _require(act.get(key) is False, f"activation.{key} must be false")
    _require(act.get("default_mode") == "candidate_replay_or_shadow_only",
             "Only replay/shadow activation is supported")
    recipe = _mapping(bundle.get("recipe"), "recipe")
    _require(recipe.get("modality_order") == list(MODALITIES), "Wrong modality order")
    _require(recipe.get("alpha_target") == 0.05 and recipe.get("eeg_line_policy") == "gaussian_line"
             and recipe.get("audio_policy") == recipe.get("video_policy") == "hybrid",
             "Unsupported recipe; this runtime must not choose a new policy")
    _require(recipe.get("production_selected") is False, "Not a production runtime")
    scoring = _mapping(bundle.get("scoring"), "scoring")
    aggregation = _mapping(scoring.get("aggregation"), "aggregation")
    _require(aggregation.get("name") == "RMS" and
             aggregation.get("formula") == "sqrt(sum(e_k**2)/K)" and
             aggregation.get("counts") == {m: len(FEATURE_SPEC[m]) for m in MODALITIES},
             "Unsupported aggregation")
    _require(aggregation.get("drop_missing_features_and_shrink_denominator") is False and
             aggregation.get("zero_valued_guards_included_in_denominator") is True and
             aggregation.get("feature_order") == "lexicographically sorted feature_path",
             "Fixed lexicographic feature set/RMS denominator is mandatory")
    _require(_real(scoring.get("evidence_cap"), "evidence_cap") == EVIDENCE_CAP and
             _real(scoring.get("runtime_window_seconds"), "window_seconds") == WINDOW_SECONDS,
             "Evidence cap/window duration mismatch")
    _require(scoring.get("comparison") == "strict B>T; equality is nominal for quality routing",
             "Strict B>T is required")
    _require(scoring.get("do_not_pass_anomaly_as_AF4B_quality") is True,
             "Anomaly and AF4-B quality must remain separate")
    _require(scoring.get("mean_std_z") == {
        "formula": "z=(x-mu)/sigma_sample", "higher_is_worse": "max(0,z)",
        "lower_is_worse": "max(0,-z)", "raw_normality_assumed_by_scoring": False,
        "std_ddof": 1, "two_sided": "abs(z)"}, "Standardized-evidence contract changed")
    emp = _mapping(scoring.get("empirical"), "scoring.empirical")
    for key, expected in {
        "lower_tail": "p=(1+count(reference<=x))/(n+1)",
        "upper_tail": "p=(1+count(reference>=x))/(n+1)",
        "one_sided": "e=max(0,normal_isf(clip(p,1e-12,1-1e-12)))",
        "two_sided": "p2=clip(min(1,2*min(p_lower,p_upper)),1e-12,1); e=max(0,normal_isf(max(p2/2,1e-12)))",
        "reference_values": "sorted, duplicates retained; no deduplication, smoothing, interpolation, or extrapolation",
        "finite_support_tail_saturation_retained": True,
    }.items():
        _require(emp.get(key) == expected, f"Empirical scoring contract changed: {key}")
    _require(scoring.get("system_route_priority") == [
        "all three unavailable -> NO_DECISION", "quality-contract error -> QUALITY_CONTROL_ERROR",
        "any unavailable or any B>T -> ROBUST_FUSION", "otherwise -> HEALTHY_FUSION"],
        "System route priority changed")
    tol = _mapping(scoring.get("reference_verification_tolerances"), "tolerances")
    _require(tol.get("absolute") == ABS_TOL and tol.get("relative") == REL_TOL,
             "Reference verification tolerance mismatch")
    limitations = _mapping(bundle.get("calibration_limitations"), "calibration_limitations")
    _require(limitations.get("split") == "VAL_ONLY" and limitations.get("TEST_used") is False,
             "Only the existing VAL-only development bundle is supported")
    _require(limitations.get("subjects") == ["subject08", "subject09", "subject10",
                                            "subject13", "subject14", "subject33"],
             "Unexpected reference subject set")
    _digest(bundle.get("policy_fingerprint_sha256"), "policy_fingerprint_sha256")
    modalities = _mapping(bundle.get("modalities"), "modalities")
    _require(set(modalities) == set(MODALITIES), "Bundle must contain exactly EEG/Audio/Video")
    policies = {}
    for m in MODALITIES:
        obj = _mapping(modalities[m], m)
        specs = FEATURE_SPEC[m]
        paths = [s[0] for s in specs]
        _require(obj.get("feature_order") == paths == sorted(paths), f"{m}: feature order mismatch")
        _require(type(obj.get("required_feature_count")) is int and
                 obj["required_feature_count"] == len(specs), f"{m}: feature count mismatch")
        expected_recipe = "hybrid_v2_gaussian_line" if m == "eeg" else "hybrid"
        _require(obj.get("recipe") == expected_recipe, f"{m}: recipe mismatch")
        refs = _mapping(obj.get("reference_models"), f"{m}.reference_models")
        _require(set(refs) == set(paths), f"{m}: unexpected/missing reference features")
        parsed = []
        for path, transform, direction, feature_type in specs:
            r = _mapping(refs[path], f"{m}.{path}")
            _require((r.get("transform"), r.get("direction"), r.get("feature_type")) ==
                     (transform, direction, feature_type), f"{m}.{path}: reference semantics mismatch")
            n = r.get("n_reference")
            _require(type(n) is int and n == 120 and r.get("reference_scope") == REFERENCE_SCOPE,
                     f"{m}.{path}: expected 120 full-six-subject clean references")
            rmin = _real(r.get("reference_min"), path + ".reference_min")
            rmax = _real(r.get("reference_max"), path + ".reference_max")
            med = _real(r.get("reference_median"), path + ".reference_median")
            _require(rmin <= med <= rmax, f"{path}: inconsistent reference range")
            if transform == "empirical":
                arr = r.get("clean_sorted_values")
                _require(isinstance(arr, list) and len(arr) == n, f"{path}: wrong reference length")
                values = tuple(_real(x, f"{path}.reference[{i}]") for i, x in enumerate(arr))
                _require(all(a <= b for a, b in zip(values, values[1:])),
                         f"{path}: references must already be sorted; refusing to repair")
                _require(type(r.get("unique_values")) is int and r["unique_values"] == len(set(values)),
                         f"{path}: duplicate/unique reference count mismatch")
                _require(_close(values[0], rmin) and _close(values[-1], rmax) and
                         _close(float(median(values)), med), f"{path}: reference summary mismatch")
                parsed.append(FeatureReference(path, transform, direction, feature_type,
                                               n, values=values, nominal_value=med))
            else:
                mu = _real(r.get("mu"), path + ".mu")
                sigma = _real(r.get("sigma_sample"), path + ".sigma_sample")
                _require(sigma > 0 and r.get("std_ddof") == 1 and rmin <= mu <= rmax,
                         f"{path}: invalid mean/sample standard deviation")
                parsed.append(FeatureReference(path, transform, direction, feature_type,
                                               n, mu=mu, sigma=sigma, nominal_value=mu))
        t = _mapping(obj.get("threshold"), f"{m}.threshold")
        threshold = _real(t.get("value"), f"{m}.threshold.value")
        _require(0.0 < threshold <= EVIDENCE_CAP and t.get("operator") == ">",
                 f"{m}: invalid threshold/comparator")
        for key, value in {"alpha_target": 0.05, "n_calibration": 120,
                           "order_statistic_k_1based": 115,
                           "calibration_score_reference_subjects": 5,
                           "calibration_score_reference_windows_per_feature": 100}.items():
            _require(t.get(key) == value, f"{m}: threshold provenance mismatch: {key}")
        _require(t.get("production_frozen") is False and t.get("used_for_nested_outer_evaluation") is False,
                 f"{m}: threshold must remain an unfrozen runtime candidate")
        source_row = _mapping(t.get("source_row"), f"{m}.threshold.source_row")
        _require(_close(_real(source_row.get("threshold_anomaly"), "source threshold"), threshold),
                 f"{m}: threshold/source-row mismatch")
        policies[m] = ModalityPolicy(m, expected_recipe, threshold, tuple(parsed))
    provenance = _mapping(bundle.get("provenance"), "provenance")
    sources = _mapping(provenance.get("sources"), "provenance.sources")
    _require(bool(sources), "Missing source lineage")
    for role, value in sources.items():
        source = _mapping(value, f"source.{role}")
        _digest(source.get("sha256"), f"source.{role}.sha256")
        _require(type(source.get("bytes")) is int and source["bytes"] > 0,
                 f"source.{role}: invalid byte count")
        _text(source.get("path_at_export"), f"source.{role}.path")
    return policies


def _check_sidecars(path: Path, expected: str) -> list[str]:
    """Accept one standard sha256sum line (digest [*]filename) or a bare digest."""
    checked = []
    candidates = (Path(str(path) + ".sha256"), path.with_suffix(".sha256"))
    for sidecar in dict.fromkeys(candidates):
        if not sidecar.is_file():
            continue
        lines = [line.strip() for line in sidecar.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        _require(len(lines) == 1, f"{sidecar}: expected one sha256sum line")
        parts = lines[0].split(maxsplit=1)
        actual = _digest(parts[0], str(sidecar))
        if len(parts) == 2:
            target = parts[1].lstrip("*")
            _require(Path(target).name == path.name or PureWindowsPath(target).name == path.name,
                     f"{sidecar}: sidecar targets a different file")
        _require(actual == expected, f"{sidecar}: SHA256 mismatch")
        checked.append(str(sidecar.resolve()))
    return checked


def _error(code: str, message: str, *, modality: str | None = None,
           feature: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if modality is not None:
        result["modality"] = modality
    if feature is not None:
        result["feature"] = feature
    return result


class DistributionQualityScorer:
    """Immutable scoring references with explicit candidate-only output contracts."""

    def __init__(self, params_path: str | Path | None = None, *,
                 expected_sha256: str = REVIEWED_BUNDLE_SHA256) -> None:
        path = (Path(params_path).expanduser() if params_path is not None else
                Path(__file__).resolve().with_name("distribution_quality_params.json"))
        _require(path.is_file(), f"Bundle not found: {path}. Place the exported JSON beside this "
                 "script, or provide --params / params_path explicitly.")
        expected = _digest(expected_sha256, "expected_sha256")
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        _require(observed == expected, f"Bundle SHA256 mismatch. Expected {expected}; observed "
                 f"{observed}. Do not edit/re-save policy parameters to bypass this check.")
        # Parse the SAME bytes we hashed (no hash-then-reopen race).
        try:
            bundle = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                                parse_constant=_bad_json_constant)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise QualityContractError(f"Invalid bundle JSON: {exc}") from exc
        _mapping(bundle, "bundle")
        self._policies = _validate_policy(bundle)
        if expected == REVIEWED_BUNDLE_SHA256:
            _require(bundle["policy_fingerprint_sha256"] == REVIEWED_POLICY_FINGERPRINT,
                     "Reviewed policy fingerprint label mismatch")
        sidecars = _check_sidecars(path, observed)
        self._bundle = deepcopy(bundle)
        self._params_path = path.resolve()
        self._identity = {
            "runtime_version": VERSION, "bundle_schema": BUNDLE_SCHEMA,
            "runtime_script_sha256": sha256_file(Path(__file__)),
            "python_version": sys.version.split()[0],
            "numeric_backend": "stdlib NormalDist; binary64",
            "bundle_path": str(self._params_path), "bundle_bytes": len(raw),
            "bundle_sha256": observed, "bundle_sha256_verified": True,
            "bundle_pin_source": ("reviewed_bundle_pin" if expected == REVIEWED_BUNDLE_SHA256
                                  else "explicit_caller_supplied_sha256"),
            "policy_fingerprint_sha256": bundle["policy_fingerprint_sha256"],
            "exporter_policy_fingerprint_recomputed": False,
            "fingerprint_note": "Exporter fingerprint is a label; integrity verified by pinned file bytes.",
            "sidecars_verified": sidecars,
            "artifact_state": bundle["artifact_state"], "created_utc": bundle["created_utc"],
            "reference_scope": REFERENCE_SCOPE, "reference_windows_per_feature": 120,
            "production_approved": False, "live_robot_control_authorized": False,
        }

    @property
    def identity(self) -> dict[str, Any]:
        return deepcopy(self._identity)

    @property
    def thresholds(self) -> dict[str, float]:
        return {m: self._policies[m].threshold for m in MODALITIES}

    @property
    def feature_order(self) -> dict[str, list[str]]:
        return {m: [r.path for r in self._policies[m].features] for m in MODALITIES}

    def score_modality(self, modality: str, metrics: Any, *, available: bool) -> dict[str, Any]:
        """Arithmetic-only entry point; no window/freshness or system routing claims.

        Detector output `metrics` can contain many extra fields; only the exact
        frozen features are read. A false availability skips all metrics. Input
        failures are returned explicitly, without changing the availability mask.
        An unknown modality name raises QualityContractError (a caller/API error).
        """
        _require(isinstance(modality, str) and modality in MODALITIES,
                 f"modality must be one of {MODALITIES}")
        policy = self._policies[modality]
        result: dict[str, Any] = {
            "modality": modality, "scope": "ARITHMETIC_ONLY", "recipe": policy.recipe,
            "available": available if type(available) is bool else None,
            "status": "QUALITY_CONTROL_ERROR", "B": None, "T": policy.threshold,
            "B_over_T": None, "exceeds_threshold": None, "comparison": ">",
            "required_feature_count": len(policy.features), "scored_feature_count": 0,
            "feature_order": [r.path for r in policy.features],
            "raw_features": {}, "evidence": {}, "evidence_details": {},
            "quality_for_fusion": None, "errors": [], "reasons": [],
        }
        if type(available) is not bool:
            result["errors"].append(_error("INVALID_AVAILABILITY",
                "Effective availability must be an external validated bool, not a truthy value",
                modality=modality))
            result["reasons"] = ["INVALID_AVAILABILITY"]
            return result
        if not available:
            result.update(status="UNAVAILABLE", reasons=["EXTERNALLY_UNAVAILABLE"])
            return result
        if not isinstance(metrics, Mapping):
            result["errors"].append(_error("INVALID_METRICS", "Available modality needs a metrics mapping",
                                           modality=modality))
        else:
            for ref in policy.features:
                try:
                    x = _real(_get_feature(metrics, ref.path), f"{modality}.{ref.path}")
                    e, details = _evidence(x, ref)
                    result["raw_features"][ref.path] = x
                    result["evidence"][ref.path] = e
                    result["evidence_details"][ref.path] = details
                except QualityContractError as exc:
                    result["errors"].append(_error("INVALID_FEATURE", str(exc),
                                                   modality=modality, feature=ref.path))
        result["scored_feature_count"] = len(result["evidence"])
        if result["errors"]:
            result["reasons"] = ["QUALITY_EVIDENCE_CONTRACT_ERROR"]
            # Partial evidence is diagnostic only. Never shrink K or impute zero.
            return result
        score = _rms([result["evidence"][r.path] for r in policy.features], len(policy.features))
        alarm = score > policy.threshold  # deliberately no tolerance/rounding
        result.update(B=score, B_over_T=score / policy.threshold,
                      exceeds_threshold=alarm, status="ANOMALOUS" if alarm else "NOMINAL",
                      reasons=["B_GT_T" if alarm else "B_LE_T"])
        return result

    def score_window(self, sample: Any, *, mode: str = "replay",
                     now_seconds: float | None = None, clock_id: str | None = None,
                     max_age_seconds: float = 1.0,
                     future_tolerance_seconds: float = 0.02) -> dict[str, Any]:
        """Validate aligned packets and return a candidate system route.

        Shadow mode needs a trusted caller clock to check age of the source span.
        No state/cache, network, sensors, emotion/fusion models or robot writes.
        Re-evaluation of the same replay input is deterministic and permitted.
        """
        out: dict[str, Any] = {
            "schema": OUTPUT_SCHEMA, "runtime_version": VERSION,
            "mode": mode if isinstance(mode, str) else None,
            "candidate_only": True, "production_approved": False,
            "live_robot_control_authorized": False, "active_router_modified": False,
            "quality_for_fusion": None,
            "status": "QUALITY_CONTROL_ERROR", "candidate_route": "QUALITY_CONTROL_ERROR",
            "window": None, "availability": None, "any_quality_alarm": None,
            "window_identity_verified": False, "freshness_checked": False,
            "modalities": {}, "reasons": [], "errors": [],
            "bundle_sha256": self._identity["bundle_sha256"],
            "policy_fingerprint_sha256": self._identity["policy_fingerprint_sha256"],
        }
        # Availability must be established before the NO_DECISION priority applies.
        try:
            obj = _mapping(sample, "sample")
            availability_obj = _mapping(obj.get("availability"), "availability")
            _require(set(availability_obj) == set(MODALITIES),
                     "availability must specify exactly eeg, audio, video")
            _require(all(type(availability_obj[m]) is bool for m in MODALITIES),
                     "availability values must be bool; strings and 0/1 are not accepted")
            availability = {m: availability_obj[m] for m in MODALITIES}
            out["availability"] = availability
        except QualityContractError as exc:
            out["errors"] = [_error("INVALID_AVAILABILITY_CONTRACT", str(exc))]
            out["reasons"] = ["INVALID_AVAILABILITY_CONTRACT"]
            return out
        try:
            _require(obj.get("schema") == INPUT_SCHEMA, f"sample.schema must be {INPUT_SCHEMA}")
            _require(isinstance(mode, str) and mode in ("replay", "shadow"),
                     "Only replay or shadow mode is allowed")
        except QualityContractError as exc:
            out["errors"].append(_error("INVALID_REQUEST_CONTRACT", str(exc)))
        if not any(availability.values()):
            # Source packets and quality evidence are irrelevant when ALL are absent.
            out["modalities"] = {m: self.score_modality(m, None, available=False) for m in MODALITIES}
            out.update(status="NO_DECISION", candidate_route="NO_DECISION",
                       any_quality_alarm=False, reasons=["ALL_MODALITIES_UNAVAILABLE"])
            return out
        window = None
        try:
            window = WindowIdentity.from_mapping(obj.get("window"))
            out["window"] = window.to_dict()
            if mode == "shadow":
                now = _real(now_seconds, "shadow.now_seconds")
                _require(_text(clock_id, "shadow.clock_id") == window.clock_id,
                         "Current time and source window must use the same clock")
                max_age = _real(max_age_seconds, "max_age_seconds")
                future_tol = _real(future_tolerance_seconds, "future_tolerance_seconds")
                _require(max_age >= 0 and future_tol >= 0, "Freshness tolerances must be nonnegative")
                _require(window.end_seconds - now <= future_tol, "Source window is in the future")
                _require(now - window.end_seconds <= max_age, "Source window is stale")
                out["freshness_checked"] = True
        except QualityContractError as exc:
            out["errors"].append(_error("WINDOW_CONTRACT_ERROR", str(exc)))
        packets = obj.get("modalities")
        if not isinstance(packets, Mapping):
            out["errors"].append(_error("INVALID_PACKETS", "sample.modalities must be a mapping"))
            packets = {}
        elif set(packets) - set(MODALITIES):
            out["errors"].append(_error("UNKNOWN_MODALITY", "Unexpected key in sample.modalities"))
        for m in MODALITIES:
            if not availability[m]:
                out["modalities"][m] = self.score_modality(m, None, available=False)
                continue
            try:
                packet = _mapping(packets.get(m), f"modalities.{m}")
                _require(packet.get("modality") == m, f"{m}: packet modality identity mismatch")
                packet_window = WindowIdentity.from_mapping(packet.get("window"))
                _require(window is not None and packet_window == window,
                         f"{m}: session/window/clock/source-span mismatch")
                result = self.score_modality(m, packet.get("metrics"), available=True)
                result["scope"] = "IDENTITY_CHECKED_WINDOW"
            except QualityContractError as exc:
                result = self.score_modality(m, {}, available=True)
                result["errors"] = [_error("PACKET_IDENTITY_ERROR", str(exc), modality=m)]
                result["reasons"] = ["PACKET_IDENTITY_ERROR"]
            out["modalities"][m] = result
            out["errors"].extend(result["errors"])
        has_error = bool(out["errors"])
        alarm = any(r["exceeds_threshold"] is True for r in out["modalities"].values())
        out["any_quality_alarm"] = (True if alarm else None) if has_error else alarm
        out["window_identity_verified"] = window is not None and not any(
            e["code"] in ("WINDOW_CONTRACT_ERROR", "PACKET_IDENTITY_ERROR", "INVALID_PACKETS",
                          "UNKNOWN_MODALITY", "INVALID_REQUEST_CONTRACT") for e in out["errors"])
        route = _route(availability, alarm, has_error)
        reasons = [f"{m}:UNAVAILABLE" for m in MODALITIES if not availability[m]]
        reasons += [f"{m}:B_GT_T" for m in MODALITIES
                    if out["modalities"][m]["exceeds_threshold"] is True]
        if has_error:
            reasons.insert(0, "QUALITY_CONTROL_ERROR")
        if not reasons:
            reasons = ["ALL_AVAILABLE_AND_ALL_B_LE_T"]
        out.update(status="QUALITY_CONTROL_ERROR" if has_error else "OK",
                   candidate_route=route, reasons=reasons)
        return out

    def verify_sources(self, source_root: str | Path | None = None) -> dict[str, Any]:
        """Check source BYTES only. Does not rerun models/fitting or claim parity."""
        provenance = self._bundle["provenance"]
        original_root = PureWindowsPath(provenance["logical_source_run"])
        entries = []
        for role, record in provenance["sources"].items():
            path = Path(record["path_at_export"])
            if source_root is not None:
                try:
                    relative = PureWindowsPath(record["path_at_export"]).relative_to(original_root)
                except ValueError as exc:
                    raise QualityContractError(f"Source {role} is outside logical_source_run") from exc
                _require(not any(p in ("..", ".") for p in relative.parts), "Unsafe relative source path")
                path = Path(source_root).expanduser().joinpath(*relative.parts)
            item: dict[str, Any] = {"role": role, "path": str(path),
                                    "expected_sha256": record["sha256"], "expected_bytes": record["bytes"]}
            try:
                item["observed_bytes"] = path.stat().st_size
                item["observed_sha256"] = sha256_file(path)
                item["status"] = "PASS" if (item["observed_bytes"] == record["bytes"] and
                    item["observed_sha256"] == record["sha256"]) else "MISMATCH"
            except OSError as exc:
                item.update(status="MISSING_OR_UNREADABLE", error=str(exc))
            entries.append(item)
        return {"status": "PASS" if all(e["status"] == "PASS" for e in entries) else "FAIL",
                "scope": "source_file_byte_hashes_only", "arithmetic_replay_performed": False,
                "files": entries}

    def preflight(self, *, verify_sources: bool = False,
                  source_root: str | Path | None = None) -> dict[str, Any]:
        tests = run_self_tests(self)
        sources = self.verify_sources(source_root) if verify_sources else {
            "status": "NOT_RUN", "arithmetic_replay_performed": False}
        return {
            "status": "PASS" if tests["status"] == "PASS" and sources["status"] != "FAIL" else "FAIL",
            "identity": self.identity, "thresholds": self.thresholds,
            "feature_order": self.feature_order, "self_tests": tests,
            "source_verification": sources,
            "empirical_reference_validation": "sorted/counts/duplicates/min/max/median checked",
            "mean_std_reference_validation": "parameters structurally checked; no raw reference arrays in bundle",
            "full_reference_offline_runtime_parity": "NOT_RUN; use independent --parity-json",
            "threshold_transfer_validated": False, "system_OR_alarm_rate_evaluated": False,
            "AF4B_adapter_validated": False, "new_emotion_accuracy_evaluated": False,
            "live_robot_control_authorized": False,
            "calibration_limitations": deepcopy(self._bundle["calibration_limitations"]),
        }

    def make_example(self) -> dict[str, Any]:
        """Artificial nominal metrics, NOT a paired clean trial or accuracy test."""
        window = WindowIdentity("SYNTHETIC_EXAMPLE", "window_0001", "synthetic_replay", 0.0, 5.0).to_dict()
        packets = {}
        for m in MODALITIES:
            metrics: dict[str, Any] = {}
            for ref in self._policies[m].features:
                cursor = metrics
                parts = ref.path.split(".")
                for name in parts[:-1]:
                    cursor = cursor.setdefault(name, {})
                cursor[parts[-1]] = ref.nominal_value
            packets[m] = {"modality": m, "window": deepcopy(window), "metrics": metrics}
        return {"schema": INPUT_SCHEMA,
                "note": "Synthetic smoke test only; not actual clean-window measurements.",
                "window": window, "availability": dict.fromkeys(MODALITIES, True),
                "modalities": packets}


# Readable alias for callers that prefer the module's conceptual name.
DistributionQuality = DistributionQualityScorer


def _synthetic_scorer() -> DistributionQualityScorer:
    """Internal fixture; never a fallback when loading a real bundle fails."""
    scorer = object.__new__(DistributionQualityScorer)
    policies = {}
    for m in MODALITIES:
        features = []
        for path, transform, direction, kind in FEATURE_SPEC[m]:
            if transform == "mean_std_z":
                ref = FeatureReference(path, transform, direction, kind, 120,
                                       mu=0.0, sigma=1.0, nominal_value=0.0)
            else:
                if "raw_flat_channel" in path or "raw_hold_fraction" in path:
                    values = (0.0,) * 120
                elif "raw_detection_rate" in path:
                    values = (1.0,) * 120
                else:
                    values = tuple((i + 1) / 120.0 for i in range(120))
                ref = FeatureReference(path, transform, direction, kind, 120,
                                       values=values, nominal_value=float(median(values)))
            features.append(ref)
        policies[m] = ModalityPolicy(m, "synthetic_self_test_only", 1.0, tuple(features))
    scorer._policies = policies
    scorer._identity = {"bundle_sha256": "SYNTHETIC_NOT_A_BUNDLE",
                        "policy_fingerprint_sha256": "SYNTHETIC_NOT_A_POLICY"}
    return scorer


def run_self_tests(scorer: DistributionQualityScorer | None = None) -> dict[str, Any]:
    """Deterministic engineering tests; not a clean-data false-alarm benchmark."""
    checks = []

    def expect(condition: bool, message: str = "Unexpected result") -> None:
        if not condition:
            raise AssertionError(message)

    def expect_close(a: float, b: float) -> None:
        expect(_close(a, b), f"{a!r} != {b!r}")

    def rejects(function: Any) -> None:
        try:
            function()
        except QualityContractError:
            return
        raise AssertionError("Expected a QualityContractError")

    def check(name: str, function: Any) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as exc:  # Test harness only: preserve all failure diagnostics.
            checks.append({"name": name, "status": "FAIL",
                           "error": f"{type(exc).__name__}: {exc}"})

    for direction, value, expected in (("higher_is_worse", 2.0, 2.0),
                                       ("higher_is_worse", -2.0, 0.0),
                                       ("lower_is_worse", -2.0, 2.0),
                                       ("lower_is_worse", 2.0, 0.0),
                                       ("two_sided", -2.0, 2.0)):
        ref = FeatureReference("fixture", "mean_std_z", direction, "continuous", 120,
                               mu=0.0, sigma=1.0)
        check(f"standardized_{direction}_{value}",
              lambda ref=ref, value=value, expected=expected: expect_close(_evidence(value, ref)[0], expected))
    cap_ref = FeatureReference("fixture", "mean_std_z", "higher_is_worse", "continuous",
                               120, mu=0.0, sigma=0.001)
    check("cap_all_standardized_evidence", lambda: expect(_evidence(100.0, cap_ref)[0] == 12.0))
    check("finite_input_intermediate_overflow_is_capped",
          lambda: expect(_evidence(sys.float_info.max, cap_ref)[0] == 12.0))
    constant = FeatureReference("guard", "empirical", "higher_is_worse", "guard", 120,
                                values=(0.0,) * 120)
    check("constant_guard_equal_reference_is_zero", lambda: expect(_evidence(0.0, constant)[0] == 0.0))
    check("constant_guard_nonzero_is_anomalous", lambda: expect(_evidence(0.001, constant)[0] > 0.0))
    check("empirical_finite_tail_saturation_retained",
          lambda: expect(_evidence(0.001, constant)[0] == _evidence(1e100, constant)[0]))
    tied = FeatureReference("tie", "empirical", "two_sided", "guard", 4, values=(-1.0, 0.0, 0.0, 1.0))
    check("empirical_ties_inclusive_both_tails", lambda: expect(
        _evidence(0.0, tied)[1]["lower_count"] == _evidence(0.0, tied)[1]["upper_count"] == 3))
    three = FeatureReference("ref3", "empirical", "higher_is_worse", "continuous", 3,
                             values=(-1.0, 0.0, 1.0))
    check("empirical_add_one_known_normal_quartile",
          lambda: expect_close(_evidence(2.0, three)[0], 0.6744897501960817))
    two = FeatureReference("ref3", "empirical", "two_sided", "continuous", 3,
                           values=(-1.0, 0.0, 1.0))
    check("empirical_two_sided_both_extremes",
          lambda: expect_close(_evidence(-2.0, two)[0], _evidence(2.0, two)[0]))
    check("RMS_includes_six_zero_guards", lambda: expect_close(_rms([6.0] + [0.0] * 6, 7), 6.0 / math.sqrt(7)))
    check("RMS_missing_feature_rejected", lambda: rejects(lambda: _rms([1.0], 7)))
    for value in (True, "1.0", None, float("nan"), float("inf"), complex(1, 0), [1.0]):
        check(f"numeric_contract_rejects_{type(value).__name__}_{str(value)}",
              lambda value=value: rejects(lambda: _real(value, "fixture")))
    check("duplicate_JSON_key_rejected",
          lambda: rejects(lambda: json.loads('{"a": 1, "a": 2}', object_pairs_hook=_unique_object)))
    check("JSON_NaN_rejected", lambda: rejects(lambda: json.loads('[NaN]', parse_constant=_bad_json_constant)))
    for bits in range(8):
        availability = {m: bool(bits & (1 << i)) for i, m in enumerate(MODALITIES)}
        for error in (False, True):
            for alarm in (False, True):
                wanted = ("NO_DECISION" if bits == 0 else "QUALITY_CONTROL_ERROR" if error else
                          "ROBUST_FUSION" if bits != 7 or alarm else "HEALTHY_FUSION")
                check(f"route_mask{bits}_error{int(error)}_alarm{int(alarm)}",
                      lambda a=availability, e=error, b=alarm, w=wanted: expect(_route(a, b, e) == w))
    active = scorer if scorer is not None else _synthetic_scorer()
    base = active.make_example()
    original = deepcopy(base)
    nominal = active.score_window(base)
    check("nominal_full_window", lambda: expect(nominal["candidate_route"] == "HEALTHY_FUSION"))
    check("input_not_mutated", lambda: expect(base == original))
    check("deterministic_repeat", lambda: expect(active.score_window(base) == nominal))
    check("no_implicit_AF4B_quality_mapping", lambda: expect(nominal["quality_for_fusion"] is None and
        all(v["quality_for_fusion"] is None for v in nominal["modalities"].values())))
    check("candidate_not_production", lambda: expect(nominal["candidate_only"] is True and
        nominal["live_robot_control_authorized"] is False and nominal["active_router_modified"] is False))
    for m in MODALITIES:
        threshold = active.thresholds[m]
        check(f"{m}_threshold_equality_nominal", lambda t=threshold: expect(not (t > t)))
        check(f"{m}_next_float_above_threshold_alarm",
              lambda t=threshold: expect(math.nextafter(t, math.inf) > t))
        check(f"{m}_next_float_below_threshold_nominal",
              lambda t=threshold: expect(not (math.nextafter(t, -math.inf) > t)))
        ref = next(r for r in active._policies[m].features if r.transform == "mean_std_z")
        high = deepcopy(base)
        high["modalities"][m]["metrics"] = {
            r.path: r.nominal_value for r in active._policies[m].features}
        high["modalities"][m]["metrics"][ref.path] = ref.mu + (
            100.0 if ref.direction == "higher_is_worse" else -100.0) * ref.sigma
        check(f"{m}_large_directional_anomaly_routes_robust",
              lambda high=high: expect(active.score_window(high)["candidate_route"] == "ROBUST_FUSION"))
        flat = {r.path: r.nominal_value for r in active._policies[m].features}
        check(f"{m}_flat_and_nested_features_match", lambda m=m, flat=flat: expect(
            active.score_modality(m, flat, available=True)["evidence"] ==
            active.score_modality(m, base["modalities"][m]["metrics"], available=True)["evidence"]))
    broken = deepcopy(base)
    del broken["modalities"]["audio"]["metrics"]["dnsmos"]["OVRL_raw"]
    broken_result = active.score_window(broken)
    check("missing_feature_QC_error_not_healthy", lambda: expect(
        broken_result["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    check("missing_feature_preserves_availability", lambda: expect(
        broken_result["availability"]["audio"] is True and broken_result["modalities"]["audio"]["B"] is None))
    broken["availability"]["eeg"] = False
    check("QC_error_precedes_partial_unavailability", lambda: expect(
        active.score_window(broken)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    for value in (float("nan"), float("inf"), "3.0", True):
        bad = deepcopy(base)
        bad["modalities"]["audio"]["metrics"]["dnsmos"]["OVRL_raw"] = value
        check(f"bad_metric_{type(value).__name__}_{str(value)}_QC_error", lambda bad=bad: expect(
            active.score_window(bad)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
        check(f"bad_metric_{type(value).__name__}_{str(value)}_JSON_safe",
              lambda bad=bad: json.dumps(active.score_window(bad), allow_nan=False))
    ambiguous = deepcopy(base)
    ambiguous["modalities"]["audio"]["metrics"]["dnsmos.OVRL_raw"] = 3.0
    check("ambiguous_feature_encoding_rejected", lambda: expect(
        active.score_window(ambiguous)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    unavailable = deepcopy(base)
    unavailable["availability"]["eeg"] = False
    unavailable["modalities"]["eeg"] = {"invalid": float("nan")}
    check("unavailable_metrics_ignored_without_becoming_healthy", lambda: expect(
        active.score_window(unavailable)["candidate_route"] == "ROBUST_FUSION"))
    absent = {"schema": INPUT_SCHEMA, "availability": dict.fromkeys(MODALITIES, False)}
    check("all_unavailable_no_packets_no_decision", lambda: expect(
        active.score_window(absent)["candidate_route"] == "NO_DECISION"))
    absent["schema"] = "wrong_schema"
    check("all_unavailable_priority_over_contract_error", lambda: expect(
        active.score_window(absent)["candidate_route"] == "NO_DECISION"))
    badmask = deepcopy(base)
    badmask["availability"]["eeg"] = 1
    check("availability_integer_is_not_bool", lambda: expect(
        active.score_window(badmask)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    for field, value in (("session_id", "other_session"), ("window_id", "stale_window"),
                         ("clock_id", "different_clock")):
        bad = deepcopy(base)
        bad["modalities"]["video"]["window"][field] = value
        check(f"packet_{field}_mismatch_rejected", lambda bad=bad: expect(
            active.score_window(bad)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    wrongspan = deepcopy(base)
    wrongspan["modalities"]["audio"]["window"].update(start_seconds=1.0, end_seconds=6.0)
    check("same_duration_different_source_span_rejected", lambda: expect(
        active.score_window(wrongspan)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    wrongduration = deepcopy(base)
    wrongduration["window"]["end_seconds"] = 20.0
    check("twenty_second_window_not_silently_rescaled", lambda: expect(
        active.score_window(wrongduration)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    wrongmodality = deepcopy(base)
    wrongmodality["modalities"]["audio"]["modality"] = "video"
    check("packet_modality_mismatch_rejected", lambda: expect(
        active.score_window(wrongmodality)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    check("shadow_requires_explicit_clock", lambda: expect(
        active.score_window(base, mode="shadow")["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    clock = base["window"]["clock_id"]
    check("shadow_fresh_window_passes", lambda: expect(active.score_window(
        base, mode="shadow", now_seconds=5.1, clock_id=clock)["candidate_route"] == "HEALTHY_FUSION"))
    check("shadow_stale_window_rejected", lambda: expect(active.score_window(
        base, mode="shadow", now_seconds=7.0, clock_id=clock)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    check("shadow_future_window_rejected", lambda: expect(active.score_window(
        base, mode="shadow", now_seconds=4.0, clock_id=clock)["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    check("shadow_wrong_clock_rejected", lambda: expect(active.score_window(
        base, mode="shadow", now_seconds=5.1, clock_id="other")["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    check("production_mode_rejected", lambda: expect(
        active.score_window(base, mode="production")["candidate_route"] == "QUALITY_CONTROL_ERROR"))
    annotated = deepcopy(base)
    annotated["corruption_family"] = "DO_NOT_USE_FOR_SCORING"
    annotated["emotion_label"] = "DO_NOT_USE_FOR_SCORING"
    check("fault_and_emotion_labels_not_used", lambda: expect(active.score_window(annotated) == nominal))
    check("output_serializable_without_NaN", lambda: json.dumps(nominal, allow_nan=False))
    if scorer is not None:
        def mutated_bundle_refused() -> None:
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / "distribution_quality_params.json"
                target.write_bytes(scorer._params_path.read_bytes() + b" ")
                rejects(lambda: DistributionQualityScorer(target, expected_sha256=scorer.identity["bundle_sha256"]))
        check("bundle_byte_tampering_rejected", mutated_bundle_refused)
        malformed = deepcopy(scorer._bundle)
        malformed["modalities"]["eeg"]["feature_order"].reverse()
        check("feature_order_change_rejected", lambda: rejects(lambda: _validate_policy(malformed)))
        malformed_refs = deepcopy(scorer._bundle)
        malformed_refs["modalities"]["audio"]["reference_models"]["dnsmos.OVRL_raw"]["clean_sorted_values"].reverse()
        check("unsorted_empirical_reference_rejected", lambda: rejects(lambda: _validate_policy(malformed_refs)))
    failed = sum(c["status"] == "FAIL" for c in checks)
    return {"status": "FAIL" if failed else "PASS", "tests": len(checks),
            "passed": len(checks) - failed, "failed": failed,
            "real_bundle_used": scorer is not None, "raw_sensor_data_used": False,
            "independent_offline_parity_proven": False, "checks": checks}


def run_parity(scorer: DistributionQualityScorer, golden: Any) -> dict[str, Any]:
    """Compare independent full-reference goldens, never LOSO scores."""
    obj = _mapping(golden, "parity")
    _require(obj.get("schema") == PARITY_SCHEMA, f"Expected {PARITY_SCHEMA}")
    _require(obj.get("bundle_sha256") == scorer.identity["bundle_sha256"],
             "Golden reference bundle hash differs from this runtime's bundle")
    _require(obj.get("policy_fingerprint_sha256") == scorer.identity["policy_fingerprint_sha256"],
             "Golden policy fingerprint label differs")
    _require(obj.get("reference_scope") == REFERENCE_SCOPE,
             "Parity needs the SAME full-six-subject references, not LOSO references")
    origin = _text(obj.get("expected_results_origin"), "expected_results_origin")
    cases = obj.get("cases")
    _require(isinstance(cases, list) and len(cases) > 0, "Parity needs a nonempty cases list")
    mismatches = []
    comparisons = 0
    for i, case in enumerate(cases):
        case = _mapping(case, f"cases[{i}]")
        actual = scorer.score_window(case.get("sample"))
        _require(actual["status"] == "OK" and all(actual["availability"].values()),
                 f"Parity case {i} must contain valid, all-available aligned inputs")
        expected = _mapping(case.get("expected"), f"cases[{i}].expected")
        _require(expected.get("candidate_route") in ("HEALTHY_FUSION", "ROBUST_FUSION"),
                 "All-available arithmetic goldens need HEALTHY_FUSION or ROBUST_FUSION")
        expected_modalities = _mapping(expected.get("modalities"), "expected.modalities")
        _require(set(expected_modalities) == set(MODALITIES), "Golden needs all three modalities")
        comparisons += 1
        if expected.get("candidate_route") != actual["candidate_route"]:
            mismatches.append({"case": i, "field": "candidate_route",
                               "expected": expected.get("candidate_route"), "actual": actual["candidate_route"]})
        for m in MODALITIES:
            e = _mapping(expected_modalities[m], f"expected.{m}")
            a = actual["modalities"][m]
            exp_evidence = _mapping(e.get("evidence"), f"expected.{m}.evidence")
            _require(set(exp_evidence) == set(scorer.feature_order[m]), f"{m}: golden evidence set mismatch")
            _require(type(e.get("exceeds_threshold")) is bool, f"{m}: golden alarm must be bool")
            numeric_pairs = [("B", _real(e.get("B"), f"expected.{m}.B"), a["B"])]
            numeric_pairs.extend((f"evidence.{p}", _real(exp_evidence[p], f"expected.{p}"), a["evidence"][p])
                                 for p in scorer.feature_order[m])
            if "T" in e:
                numeric_pairs.append(("T", _real(e["T"], f"expected.{m}.T"), a["T"]))
            for field, wanted, observed in numeric_pairs:
                comparisons += 1
                if not _close(wanted, observed):
                    mismatches.append({"case": i, "modality": m, "field": field,
                                       "expected": wanted, "actual": observed, "abs_error": abs(wanted-observed)})
            comparisons += 1
            if e["exceeds_threshold"] != a["exceeds_threshold"]:
                mismatches.append({"case": i, "modality": m, "field": "exceeds_threshold",
                                   "expected": e["exceeds_threshold"], "actual": a["exceeds_threshold"]})
    return {"status": "FAIL" if mismatches else "PASS", "cases": len(cases),
            "comparisons": comparisons, "mismatches": mismatches,
            "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
            "boolean_route_comparison": "exact; no threshold epsilon",
            "expected_results_origin": origin,
            "reference_scope": REFERENCE_SCOPE, "identity": scorer.identity,
            "threshold_transfer_validated": False, "production_approved": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--params", type=Path, help="Exact exported bundle; defaults beside this script")
    parser.add_argument("--expected-sha256", default=REVIEWED_BUNDLE_SHA256,
                        help="Trusted bundle byte digest; not a verification bypass")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--self-test", action="store_true", help="No params needed; --params also tests real bundle")
    action.add_argument("--preflight-only", "--preflight", dest="preflight", action="store_true",
                        help="Validate real bundle plus self-tests (default action)")
    action.add_argument("--write-example", type=Path, metavar="PATH", help="Write a synthetic nested-metrics sample")
    action.add_argument("--input-json", type=Path, metavar="PATH", help="Single sample, sample list, or {'samples': [...]} ")
    action.add_argument("--parity-json", type=Path, metavar="PATH", help="Independent full-reference golden cases")
    parser.add_argument("--verify-sources", action="store_true", help="Preflight: check every provenance source's bytes")
    parser.add_argument("--source-root", type=Path, help="Relocated logical_source_run root for source hash checks")
    parser.add_argument("--mode", choices=("replay", "shadow"), default="replay")
    parser.add_argument("--now-seconds", type=float, help="Shadow current time from the caller's actual source clock")
    parser.add_argument("--clock-id", help="Clock identity for --now-seconds")
    parser.add_argument("--max-age-seconds", type=float, default=1.0)
    parser.add_argument("--future-tolerance-seconds", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="Write full JSON report without overwriting; otherwise stdout")
    args = parser.parse_args(argv)
    if args.source_root is not None and not args.verify_sources:
        parser.error("--source-root requires --verify-sources")
    if args.verify_sources and any((args.self_test, args.write_example, args.input_json, args.parity_json)):
        parser.error("--verify-sources is a preflight option")
    if args.write_example and args.output:
        parser.error("--write-example already specifies its destination; omit --output")
    if args.mode == "shadow" and not args.input_json:
        parser.error("--mode shadow is only used with --input-json")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args(argv)
    try:
        if args.output is not None:
            _require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
        if args.write_example is not None:
            _require(not args.write_example.exists(), f"Refusing to overwrite example: {args.write_example}")
        if args.self_test and args.params is None:
            report = run_self_tests()
            exit_code = 0 if report["status"] == "PASS" else 3
        else:
            scorer = DistributionQualityScorer(args.params, expected_sha256=args.expected_sha256)
            if args.write_example:
                write_json(args.write_example, scorer.make_example())
                report = {"status": "PASS", "synthetic_example_written": str(args.write_example.resolve()),
                          "real_measurements": False, "production_approved": False}
                exit_code = 0
            elif args.self_test:
                report = run_self_tests(scorer)
                exit_code = 0 if report["status"] == "PASS" else 3
            elif args.parity_json:
                report = run_parity(scorer, read_json(args.parity_json))
                exit_code = 0 if report["status"] == "PASS" else 3
            elif args.input_json:
                incoming = read_json(args.input_json)
                batch = isinstance(incoming, list) or (isinstance(incoming, Mapping) and "samples" in incoming)
                if isinstance(incoming, Mapping) and "samples" in incoming:
                    _require("availability" not in incoming, "Ambiguous single-sample/batch input")
                    samples = incoming["samples"]
                elif isinstance(incoming, list):
                    samples = incoming
                else:
                    samples = [incoming]
                _require(isinstance(samples, list) and bool(samples), "Expected one sample or a nonempty list")
                results = [scorer.score_window(s, mode=args.mode, now_seconds=args.now_seconds,
                            clock_id=args.clock_id, max_age_seconds=args.max_age_seconds,
                            future_tolerance_seconds=args.future_tolerance_seconds) for s in samples]
                has_error = any(bool(r["errors"]) for r in results)
                report = ({"status": "QUALITY_CONTROL_ERROR" if has_error else "OK",
                           "samples": len(results), "candidate_route_counts": dict(Counter(
                               r["candidate_route"] for r in results)), "results": results}
                          if batch else results[0])
                exit_code = 2 if has_error else 0
            else:
                report = scorer.preflight(verify_sources=args.verify_sources, source_root=args.source_root)
                exit_code = (0 if report["status"] == "PASS" else
                             3 if report["self_tests"]["status"] == "FAIL" else 2)
        if args.output is not None:
            write_json(args.output, report)
            print(json.dumps({"status": report.get("status"), "output": str(args.output.resolve()),
                              "exit_code": exit_code}, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return exit_code
    except (QualityContractError, OSError) as exc:
        report = {"status": "QUALITY_CONTROL_ERROR", "runtime_version": VERSION,
                  "error_type": type(exc).__name__, "message": str(exc),
                  "production_approved": False, "live_robot_control_authorized": False}
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
