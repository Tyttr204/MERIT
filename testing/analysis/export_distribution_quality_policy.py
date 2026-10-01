#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export the reviewed EAV distribution-quality CANDIDATE, never activate it.

Python 3.11+, standard library only. No Torch, pandas, SciPy, model loading,
training, raw-signal inference, downloads, or modification of deployment files.

Reviewed recipe (fixed, not a new policy search):
  EEG   : Hybrid-v2 gaussian_line; six other empirical features; RMS over 7.
  Audio : Hybrid OVRL empirical + level mean/std; RMS over 2.
  Video : DOVER mean/std + empirical face-detection guard; RMS over BOTH 2.
  Target alpha: 0.05; strict B > T; evidence cap: 12, INCLUDING mean/std evidence.

Sources:
  run_summary.json
  hybrid_quality_score_*/hybrid_distribution_quality_score_candidate_artifact.json
  threshold_calibration_*/distribution_threshold_candidate_artifact.json
  eeg_quality_hybrid_v2_*/eeg_quality_hybrid_v2_candidate_artifact.json

Default --verification full additionally requires:
  <run>/quality_numeric_features.csv
  <run>/quality_distribution_samples.csv
  <threshold artifact dir>/outer_window_anomaly_scores.csv
  <EEG-v2 artifact dir>/eeg_v2_outer_window_scores.csv

FULL verification checks raw-file digests, reference geometry, full-reference
parameters, all 360 selected clean LOSO scores, and the three 115th order
statistics. It does NOT rerun 3,840 model cases or validate a production router.

--verification artifacts-only explicitly permits exporting uploaded artifact
copies when raw CSVs are unavailable. The resulting bundle is marked partially
verified and blocked for activation. It is NEVER an implicit fallback.

The exported references use all six VAL subjects (120 windows/feature), whereas
the retained thresholds were calibrated on subject-held-out scores using five
reference subjects (100 windows/feature). This transfer is explicitly flagged:
full-reference deployment does NOT inherit nested-LOSO false-alarm guarantees.
No threshold is silently recalibrated on the full-reference in-sample scores.

From the deployment root:
  python -X utf8 .\\testing\\analysis\\export_distribution_quality_policy.py --self-test
  python -X utf8 .\\testing\\analysis\\export_distribution_quality_policy.py --check-inputs --run-dir .\\system_checks\\run_...
  python -X utf8 .\\testing\\analysis\\export_distribution_quality_policy.py --run-dir .\\system_checks\\run_...

Default output: <run>/quality_policy_export_<UTC timestamp with microseconds>/
  distribution_quality_params.json
  verification_cases.json
  export_audit.json
  README_export.md
  export_manifest.json

The last file is the completion marker. Output directories must be NEW.
Nothing is installed into Quality/, no active configuration is edited, and
no production/live/robot-control permission is granted by a successful export.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import ntpath
import os
import re
import statistics
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = "EAV-DISTRIBUTION-QUALITY-EXPORT.1.0"
SCHEMA = "eav.distribution_quality_policy.bundle.v1"
STATE = "EXPORTED_CANDIDATE_NOT_DEPLOYABLE"
MODALITIES = ("eeg", "audio", "video")
ALPHA = 0.05
N_CLEAN = 120
N_SUBJECTS = 6
CAP = 12.0
ATOL = 1e-10
RTOL = 1e-9
NORMAL = statistics.NormalDist()
LINE = "features.line_fraction_mean"
# feature_path -> (type, direction, source-hybrid transform, exported transform)
SPECS = {
    "eeg": {
        "features.hf55_90_fraction_mean": ("continuous", "higher_is_worse", "empirical", "empirical"),
        LINE: ("continuous", "higher_is_worse", "empirical", "mean_std_z"),
        "features.pyprep_bad_fraction": ("guard", "higher_is_worse", "empirical", "empirical"),
        "features.raw_flat_channel_fraction": ("guard", "higher_is_worse", "empirical", "empirical"),
        "features.raw_hold_fraction_mean": ("guard", "higher_is_worse", "empirical", "empirical"),
        "features.rms_uv_median": ("continuous", "two_sided", "empirical", "empirical"),
        "features.slow02_1_fraction_mean": ("continuous", "higher_is_worse", "empirical", "empirical"),
    },
    "audio": {
        "dnsmos.OVRL_raw": ("continuous", "lower_is_worse", "empirical", "empirical"),
        "signal_metrics.rms_dbfs": ("continuous", "lower_is_worse", "gaussian", "mean_std_z"),
    },
    "video": {
        "face_observability.raw_detection_rate": ("guard", "lower_is_worse", "empirical", "empirical"),
        "physical_quality.technical_raw": ("continuous", "lower_is_worse", "gaussian", "mean_std_z"),
    },
}
# Pin the analysis implementation, not a particular experiment's numerical results.
# Unknown versions must be reviewed instead of silently assuming identical maths.
SUPPORTED = {
    "hybrid": (
        "eav.hybrid_distribution_quality_score_candidate.v1",
        "EAV-HYBRID-DIST-QUALITY.1.0",
        "cecf35c2917f45eb2bbd1ec3baee9be9619833f020bade2ce8742a22f0e538b5",
    ),
    "threshold": (
        "eav.distribution_quality_threshold_candidate.v1",
        "EAV-DIST-QUALITY-THRESHOLDS.1.0",
        "e6b69c51114cf1636fffd647961417fdc6a7fb2614015957f3797acefca25de0",
    ),
    "eeg_v2": (
        "eav.eeg_quality_hybrid_v2_candidate.v1",
        "EAV-EEG-QUALITY-HYBRID-V2.1.0",
        "9138bf00cb7414f158c6e3953bc0e94930d28cc4a6b3f5c399fb2fe075fea021",
    ),
}
ARTIFACT_FILES = {
    "hybrid": ("hybrid_quality_score_*", "hybrid_distribution_quality_score_candidate_artifact.json"),
    "threshold": ("threshold_calibration_*", "distribution_threshold_candidate_artifact.json"),
    "eeg_v2": ("eeg_quality_hybrid_v2_*", "eeg_quality_hybrid_v2_candidate_artifact.json"),
}
SUMMARY_FILES = {
    "hybrid": "hybrid_quality_score_summary.json",
    "threshold": "threshold_calibration_summary.json",
    "eeg_v2": "eeg_quality_hybrid_v2_summary.json",
}
DATA_FILES = {
    "quality_numeric_features": "quality_numeric_features.csv",
    "quality_distribution_samples": "quality_distribution_samples.csv",
}
BUNDLE_FILES = ("distribution_quality_params.json", "verification_cases.json", "export_audit.json", "README_export.md")


class ExportError(RuntimeError):
    """Input/lineage/contract failure; no deployable policy may be inferred."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ExportError(message)


def num(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label}: expected a JSON number, got {value!r}")
    result = float(value)
    require(math.isfinite(result), f"{label}: non-finite number")
    return result


def csv_num(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{label}: invalid numeric field {value!r}") from exc
    require(math.isfinite(result), f"{label}: non-finite numeric field")
    return result


def integer(value: Any, label: str) -> int:
    x = num(value, label)
    require(x.is_integer(), f"{label}: expected an integer")
    return int(x)


def boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "1.0"}:
            return True
        if value.strip().lower() in {"false", "0", "0.0"}:
            return False
    raise ExportError(f"{label}: expected an explicit boolean, got {value!r}")


def close(actual: Any, expected: Any, label: str) -> None:
    a, b = num(actual, label), num(expected, label)
    require(math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL), f"{label}: {a!r} != {b!r}")


def require_false_fields(block: Any, names: Sequence[str], label: str) -> None:
    require(isinstance(block, dict), f"{label}: expected object")
    for name in names:
        require(block.get(name) is False, f"{label}.{name}: must be explicitly false")


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    out = {}
    for key, value in pairs:
        require(key not in out, f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def reject_constant(value: str) -> None:
    raise ExportError(f"Invalid JSON constant: {value}")


def loads(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (ValueError, TypeError) as exc:
        raise ExportError(f"Invalid JSON: {exc}") from exc


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON file: {path}")
    require(path.stat().st_size < 50 * 1024 * 1024, f"Unexpectedly large JSON: {path}")
    try:
        value = loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise ExportError(f"Cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def json_bytes(value: Any) -> bytes:
    # Never sanitize invalid parameters into null. Missing audit fields are
    # explicitly None; all score/threshold/parameter values must be finite.
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(path: Path) -> str:
    require(path.is_file(), f"Missing input file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_text(value: Any, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
            f"{label}: missing or malformed SHA256")
    return value.lower()


def logical_path(value: Any) -> str:
    require(isinstance(value, str) and bool(value.strip()), "Missing source_run_dir")
    if "\\" in value or re.match(r"^[a-zA-Z]:", value):
        return ntpath.normcase(ntpath.normpath(value))
    return os.path.normpath(value)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    require(len(values) >= 2, "Need at least two values for sample standard deviation")
    mu = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((x - mu) ** 2 for x in values) / (len(values) - 1))
    require(math.isfinite(sd) and sd > 0, "Zero/non-finite sample standard deviation")
    return mu, sd


def order_index(n: int) -> int:
    # Exact arithmetic for the FIXED alpha=0.05 recipe.
    return ((n + 1) * 95 + 99) // 100


def validate_run(run: Mapping[str, Any]) -> None:
    require(run.get("status") == "PASS_OFFLINE_PROCESSING", "Source run did not pass")
    require(run.get("formal_balanced_val") is True, "Source run is not formal balanced VAL")
    require(run.get("suite") == "robustness", "Source run is not robustness")
    require_false_fields(run, ("test_data_used", "threshold_retuned", "frozen_model_training_performed"), "run")
    require(integer(run.get("attempted_cases"), "run.attempted_cases") == 3840, "Expected 3840 completed cases")
    require(integer(run.get("planned_cases"), "run.planned_cases") == 3840, "Expected 3840 planned cases")
    for key in ("failed_cases", "failed_hard_checks", "events_with_errors"):
        require(integer(run.get(key), "run." + key) == 0, "Source run contains failures: " + key)
    require(run.get("fatal_error") is None, "Source run has fatal_error")
    integrity = run.get("integrity", {})
    require(integrity.get("status") == "PASS" and integrity.get("changed_or_missing") == [],
            "Source run reports changed/missing protected assets")
    require(run.get("quality_distribution_logging", {}).get("status") == "OBSERVATIONAL_EXPORT_COMPLETE",
            "Source quality logging is incomplete")


def validate_artifact(role: str, obj: Mapping[str, Any]) -> None:
    schema, version, script_hash = SUPPORTED[role]
    require(obj.get("schema") == schema, f"{role}: unsupported schema")
    require(obj.get("version") == version, f"{role}: unsupported analysis version; review it before export")
    require(obj.get("script_sha256") == script_hash, f"{role}: analysis implementation hash is not the reviewed version")
    require(obj.get("artifact_state") == "CANDIDATE_NOT_DEPLOYABLE", f"{role}: unexpected artifact state")
    require(obj.get("source_split") == "VAL_ONLY" and obj.get("test_data_used") is False,
            f"{role}: must be VAL-only with no TEST")
    require(obj.get("runtime_unit") == "5s_window", f"{role}: runtime unit must remain 5s")
    require_false_fields(obj.get("existing_deployment"),
                         ("tau_0_80_changed", "quality_calibrators_changed", "fusion_weights_changed", "emotion_models_changed"), role)
    selection = obj.get("selection", {})
    require(selection and all(value is False for value in selection.values()), f"{role}: previous production/selection state changed")
    if role == "hybrid":
        agg = obj.get("aggregation_candidate", {})
        require(agg.get("name") == "RMS" and agg.get("formula") == "sqrt(mean(evidence^2))",
                "Hybrid RMS definition changed")
        require(agg.get("production_frozen") is False, "Hybrid already frozen")
        require(obj.get("validation_protocol", {}).get("subject_loso_reference_fit") is True, "Hybrid lacks LOSO")
    elif role == "eeg_v2":
        agg = obj.get("aggregation", {})
        require(agg.get("name") == "RMS" and agg.get("changed_from_hybrid_v1") is False,
                "EEG-v2 aggregation changed")
        require(obj.get("scope") == "EEG_QUALITY_POLICY_ONLY", "EEG-v2 scope changed")
        close(obj.get("primary_development_alpha"), ALPHA, "EEG primary alpha")
    else:
        require(obj.get("aggregation") == "RMS", "Threshold-stage aggregation changed")
        rule = obj.get("threshold_rule", {})
        require(rule.get("strict_exceedance") is True, "Threshold comparison must be strict >")
        require(rule.get("exact_coverage_guarantee_claimed") is False, "Unsupported coverage claim")


def find_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "main.py").is_file() and (p / "system_checks").is_dir():
            return p
    raise ExportError("Cannot find deployment root. Supply --run-dir or --deployment-root.")


def select_latest(candidates: list[tuple[str, Path]], label: str) -> Path:
    require(candidates, f"No complete, compatible {label} found. Supply its explicit path.")
    candidates.sort(key=lambda x: (x[0], str(x[1])))
    newest = candidates[-1][0]
    tied = [p for timestamp, p in candidates if timestamp == newest]
    require(len({sha(p) for p in tied}) == 1, f"Ambiguous {label} with the same timestamp; specify a path")
    return candidates[-1][1]


def discover_artifact(run_dir: Path, role: str, linked_hybrid_hash: str | None = None) -> Path:
    pattern, filename = ARTIFACT_FILES[role]
    candidates = []
    for directory in run_dir.glob(pattern):
        p = directory / filename
        summary_path = directory / SUMMARY_FILES[role]
        if not p.is_file() or not summary_path.is_file():
            continue
        try:
            obj, summary = read_json(p), read_json(summary_path)
            validate_artifact(role, obj)
            require(summary.get("status") == "PASS", "incomplete analysis")
            require(summary.get("version") == obj["version"], "summary version mismatch")
            if linked_hybrid_hash is not None:
                key = "hybrid_artifact" if role == "threshold" else "source_hybrid_artifact"
                require(obj.get("source_hashes", {}).get(key) == linked_hybrid_hash, "wrong hybrid lineage")
            timestamp = str(obj["created_local"])
            datetime.fromisoformat(timestamp)
        except (ExportError, KeyError, ValueError):
            continue
        candidates.append((timestamp, p.resolve()))
    return select_latest(candidates, role + " artifact")


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Path]]:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        root = Path(args.deployment_root).expanduser().resolve() if args.deployment_root else find_root(Path.cwd())
        runs = []
        for d in (root / "system_checks").glob("run_*"):
            p = d / "run_summary.json"
            if not p.is_file():
                continue
            try:
                validate_run(read_json(p))
                discover_artifact(d, "eeg_v2")
            except ExportError:
                continue
            # Run directory NAME is immutable; directory mtime changes when
            # subsequent analysis outputs are created, so do not use its mtime.
            runs.append((d.name, p.resolve()))
        run_dir = select_latest(runs, "formal run").parent
    require(run_dir.is_dir(), f"Run directory missing: {run_dir}")
    paths = {"run_summary": Path(args.run_summary).expanduser().resolve() if args.run_summary else run_dir / "run_summary.json"}
    paths["eeg_v2"] = Path(args.eeg_v2_artifact).expanduser().resolve() if args.eeg_v2_artifact else discover_artifact(run_dir, "eeg_v2")
    if args.hybrid_artifact:
        paths["hybrid"] = Path(args.hybrid_artifact).expanduser().resolve()
    else:
        # Prefer the EXACT hybrid linked by the chosen EEG-v2 artifact.
        eeg = read_json(paths["eeg_v2"])
        expected = eeg.get("source_hashes", {}).get("source_hybrid_artifact")
        matches = []
        pattern, filename = ARTIFACT_FILES["hybrid"]
        for d in run_dir.glob(pattern):
            p = d / filename
            if p.is_file() and sha(p) == expected:
                matches.append((str(read_json(p).get("created_local", "")), p.resolve()))
        paths["hybrid"] = select_latest(matches, "exact linked hybrid")
    hh = sha(paths["hybrid"])
    paths["threshold"] = Path(args.threshold_artifact).expanduser().resolve() if args.threshold_artifact else discover_artifact(run_dir, "threshold", hh)
    paths.update({key: run_dir / name for key, name in DATA_FILES.items()})
    paths["audio_video_oof"] = paths["threshold"].parent / "outer_window_anomaly_scores.csv"
    paths["eeg_oof"] = paths["eeg_v2"].parent / "eeg_v2_outer_window_scores.csv"
    return run_dir, paths


def validate_chain(paths: Mapping[str, Path]) -> tuple[dict, list[str], dict]:
    docs = {role: read_json(paths[role]) for role in ("run_summary", "hybrid", "threshold", "eeg_v2")}
    validate_run(docs["run_summary"])
    for role in ("hybrid", "threshold", "eeg_v2"):
        validate_artifact(role, docs[role])
    h, t, e = docs["hybrid"], docs["threshold"], docs["eeg_v2"]
    declared_roots = {logical_path(x.get("source_run_dir")) for x in (h, t, e)}
    require(len(declared_roots) == 1, "Artifacts declare different logical source runs")
    hhash = sha(paths["hybrid"])
    require(t["source_hashes"].get("hybrid_artifact") == hhash, "Threshold artifact links a different hybrid (SHA256)")
    require(e["source_hashes"].get("source_hybrid_artifact") == hhash, "EEG-v2 artifact links a different hybrid (SHA256)")
    data_hashes = {}
    for key in ("run_summary", *DATA_FILES):
        values = {digest_text(x.get("source_hashes", {}).get(key), key) for x in (h, t, e)}
        require(len(values) == 1, f"Source digest disagreement for {key}")
        data_hashes[key] = values.pop()
    require(sha(paths["run_summary"]) == data_hashes["run_summary"], "run_summary bytes do not match source lineage")
    protocols = [h.get("validation_protocol", {}), t.get("nested_subject_protocol", {}), e.get("nested_subject_protocol", {})]
    subject_lists = [protocols[0].get("subjects"), protocols[1].get("outer_subjects"), protocols[2].get("subjects")]
    for subjects in subject_lists:
        require(isinstance(subjects, list) and len(subjects) == len(set(subjects)) == N_SUBJECTS,
                "Expected six distinct explicitly recorded VAL subjects")
        require(all(isinstance(s, str) and re.fullmatch(r"subject\d+", s) for s in subjects), "Invalid subject IDs")
    require(set(subject_lists[0]) == set(subject_lists[1]) == set(subject_lists[2]), "VAL subject sets differ")
    require(protocols[0].get("clean_windows_per_train_fold_feature") == 100, "Hybrid fold reference geometry changed")
    for p in protocols[1:]:
        require(p.get("outer_train_subjects_per_fold") == 5 and p.get("inner_reference_subjects_per_fold") == 4,
                "Nested subject geometry changed")
        require(p.get("outer_subject_used_in_threshold_calibration") is False, "Outer subject entered calibration")
    for key, meta in e.get("fixed_non_line_eeg_policy", {}).items():
        require(key in SPECS["eeg"] and key != LINE, "Unexpected EEG non-line feature")
        expected_type, direction, transform, _ = SPECS["eeg"][key]
        require(meta == {"direction": direction, "feature_type": expected_type, "transform": transform}, "EEG non-line policy drift: " + key)
    require(set(e.get("fixed_non_line_eeg_policy", {})) == set(SPECS["eeg"]) - {LINE}, "Missing fixed EEG non-line feature")
    return docs, sorted(subject_lists[0]), data_hashes


def validate_empirical(block: Mapping[str, Any], label: str) -> list[float]:
    values = block.get("clean_sorted_values")
    require(isinstance(values, list) and len(values) == N_CLEAN, label + ": expected 120 reference values")
    xs = [num(x, label) for x in values]
    require(xs == sorted(xs), label + ": empirical values must be sorted; duplicate values are retained")
    require(integer(block.get("unique_values"), label) == len(set(xs)), label + ": unique count mismatch")
    for key, value in (("min", xs[0]), ("max", xs[-1]), ("median", statistics.median(xs)), ("mean", math.fsum(xs) / len(xs))):
        close(block.get(key), value, label + "." + key)
    return xs


def reference_models(hybrid: Mapping[str, Any]) -> dict:
    source = hybrid.get("full_6_subject_VAL_reference_candidate")
    require(isinstance(source, dict) and set(source) == set(MODALITIES), "Incomplete full-VAL references")
    models = {}
    for modality in MODALITIES:
        require(set(source[modality]) == set(SPECS[modality]), modality + ": feature set changed")
        models[modality] = {}
        for feature in sorted(SPECS[modality]):
            kind, direction, old_transform, transform = SPECS[modality][feature]
            b = source[modality][feature]
            require(b.get("feature_type") == kind and b.get("direction") == direction and b.get("transform") == old_transform,
                    "Source feature semantics changed: " + feature)
            require(integer(b.get("n"), feature) == N_CLEAN, feature + ": wrong reference n")
            require(b.get("fit_scope") == "all_6_VAL_subjects_clean_reference", feature + ": wrong reference scope")
            policy_section = "continuous" if kind == "continuous" else "guards"
            require(hybrid.get("hybrid_policy", {}).get(policy_section, {}).get(modality, {}).get(feature) == old_transform,
                    "Hybrid policy/reference mismatch: " + feature)
            dst = {"feature_type": kind, "direction": direction, "transform": transform,
                   "n_reference": N_CLEAN, "reference_scope": "full_6_VAL_subjects_clean_5s_windows",
                   "source_json_pointer": f"/full_6_subject_VAL_reference_candidate/{modality}/{feature}",
                   "reference_min": num(b.get("min"), feature), "reference_max": num(b.get("max"), feature),
                   "reference_median": num(b.get("median"), feature), "raw_feature_gaussian_population_claimed": False}
            if old_transform == "empirical":
                xs = validate_empirical(b, feature)
                if feature == LINE:
                    mu, sd = mean_sd(xs)
                    dst.update(mu=mu, sigma_sample=sd, std_ddof=1,
                               parameter_derivation="mean/sample_sd recomputed from existing full-VAL clean line values; not new threshold fitting",
                               reference_values_sha256=hashlib.sha256(json_bytes(xs)).hexdigest())
                else:
                    dst.update(clean_sorted_values=xs, unique_values=len(set(xs)),
                               parameter_derivation="copied empirical reference values without deduplication/interpolation")
            else:
                mu, sd = num(b.get("mu"), feature), num(b.get("sigma_sample"), feature)
                require(sd > 0, feature + ": nonpositive sigma")
                close(mu, b.get("mean"), feature + ".mean")
                dst.update(mu=mu, sigma_sample=sd, std_ddof=1, parameter_derivation="copied full-VAL mean/sample_sd")
            models[modality][feature] = dst
    return models


def select_thresholds(docs: Mapping[str, dict]) -> dict:
    result = {}
    for m in MODALITIES:
        role = "eeg_v2" if m == "eeg" else "threshold"
        rows = docs[role].get("full_val_crossfit_threshold_candidates")
        require(isinstance(rows, list), role + ": missing crossfit thresholds")
        matches = []
        for row in rows:
            require(isinstance(row, dict), "Threshold row must be an object")
            selected = (row.get("policy") == "gaussian_line") if m == "eeg" else (row.get("modality") == m and row.get("representation") == "hybrid")
            if selected and num(row.get("target_alpha"), "target_alpha") == ALPHA:
                matches.append(row)
        require(len(matches) == 1, f"{m}: expected exactly one alpha=0.05 threshold for the reviewed recipe")
        row = matches[0]
        require(integer(row.get("n_calibration"), m) == N_CLEAN, m + ": wrong calibration n")
        require(integer(row.get("order_statistic_k_1based"), m) == order_index(N_CLEAN), m + ": wrong order statistic")
        require(row.get("used_for_nested_outer_evaluation") is False, m + ": full-VAL threshold used for evaluation")
        frozen_key = "production_frozen" if m == "eeg" else "deployment_frozen"
        require(row.get(frozen_key) is False, m + ": source threshold already frozen")
        value = num(row.get("threshold_anomaly"), m)
        require(0 <= value <= CAP, m + ": threshold outside the scoring support")
        rate = num(row.get("calibration_exceedance_rate"), m)
        require(0 <= rate <= (N_CLEAN - order_index(N_CLEAN)) / N_CLEAN + ATOL,
                m + ": inconsistent strict-exceedance calibration rate")
        if m != "eeg":
            close(row.get("q_threshold_gaussian_kernel"), math.exp(-0.5 * value * value), m + ".q_kernel")
            close(row.get("q_threshold_halfnormal_survival"), math.erfc(value / math.sqrt(2)), m + ".q_halfnormal")
        result[m] = {"value": value, "operator": ">", "alpha_target": ALPHA,
                     "n_calibration": N_CLEAN, "order_statistic_k_1based": order_index(N_CLEAN),
                     "observed_crossfit_calibration_exceedance_rate": rate,
                     "source_artifact_role": role, "source_recipe": "gaussian_line" if m == "eeg" else "hybrid",
                     "source_row": copy.deepcopy(row), "production_frozen": False,
                     "calibration_score_reference_subjects": 5, "calibration_score_reference_windows_per_feature": 100,
                     "used_for_nested_outer_evaluation": False}
    return result


def evidence(value: float, block: Mapping[str, Any]) -> float:
    x = num(value, "runtime feature")
    direction = block["direction"]
    if block["transform"] == "mean_std_z":
        z = (x - block["mu"]) / block["sigma_sample"]
        e = abs(z) if direction == "two_sided" else max(0.0, -z if direction == "lower_is_worse" else z)
    else:
        xs = block["clean_sorted_values"]
        n = len(xs)
        if direction == "two_sided":
            lo = (1 + sum(v <= x for v in xs)) / (n + 1)
            hi = (1 + sum(v >= x for v in xs)) / (n + 1)
            p2 = max(1e-12, min(1.0, 2 * min(lo, hi)))
            tail = max(p2 / 2, 1e-12)
        else:
            count = sum(v <= x for v in xs) if direction == "lower_is_worse" else sum(v >= x for v in xs)
            tail = min(1 - 1e-12, max(1e-12, (1 + count) / (n + 1)))
        # norm.isf(tail) == -norm.ppf(tail); avoid 1-tail cancellation.
        e = max(0.0, -NORMAL.inv_cdf(tail))
    require(math.isfinite(e), "Non-finite anomaly evidence")
    return min(CAP, e)


def score_reference(values: Mapping[str, Any], blocks: Mapping[str, dict]) -> tuple[float, dict]:
    require(set(values) == set(blocks), "Missing or unexpected quality features; do not change the RMS denominator")
    es = {key: evidence(num(values[key], key), blocks[key]) for key in sorted(blocks)}
    b = math.sqrt(math.fsum(x * x for x in es.values()) / len(es))
    return b, es


def decide_from_scores(available: Mapping[str, bool], scores: Mapping[str, Any], thresholds: Mapping[str, dict]) -> str:
    require(set(available) == set(MODALITIES), "Availability requires all three modalities")
    require(all(type(v) is bool for v in available.values()), "Availability must contain booleans")
    if not any(available.values()):
        return "NO_DECISION"
    # Even when another modality is unavailable, invalid available-modality
    # scores remain errors instead of silently becoming an ordinary degradation.
    exceeded = []
    for m in MODALITIES:
        if available[m]:
            b = num(scores.get(m), m + ".anomaly")
            require(b >= 0, "Anomaly must be nonnegative")
            exceeded.append(b > thresholds[m]["value"])
    return "ROBUST_FUSION" if not all(available.values()) or any(exceeded) else "HEALTHY_FUSION"


def csv_records(path: Path, required: set[str]):
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        require(len(names) == len(set(names)), f"Duplicate CSV columns: {path}")
        require(required.issubset(names), f"{path.name}: missing columns {sorted(required - set(names))}")
        for rownum, row in enumerate(reader, 2):
            require(None not in row, f"{path.name}:{rownum}: malformed CSV row")
            yield rownum, row


def row_id(row: Mapping[str, Any], label: str) -> tuple[str, str, str]:
    out = tuple(str(row.get(k) or "").strip() for k in ("subject", "pair_key", "source_window_id"))
    require(all(out), label + ": blank subject/pair/window identity")
    return out  # type: ignore[return-value]


def geometry(ids: set[tuple[str, str, str]], subjects: Sequence[str], label: str) -> None:
    require(len(ids) == N_CLEAN and {key[0] for key in ids} == set(subjects), label + ": expected 120 windows over the six VAL subjects")
    require(len({key[2] for key in ids}) == N_CLEAN, label + ": ambiguous source_window_id reuse")
    per_trial = Counter((s, p) for s, p, _ in ids)
    require(len(per_trial) == 30 and set(per_trial.values()) == {4}, label + ": expected 30 complete four-window trials")
    require(set(Counter(s for s, _ in per_trial).values()) == {5}, label + ": expected five trials per subject")


def read_clean_data(paths: Mapping[str, Path], subjects: Sequence[str]) -> dict:
    ids_by_m = {m: set() for m in MODALITIES}
    required = {"subject", "pair_key", "source_window_id", "condition", "modality", "available", "q"}
    for line, row in csv_records(paths["quality_distribution_samples"], required):
        if row["condition"] != "reference":
            continue  # matched references and degradation rows are NOT calibration data
        m = row["modality"]
        require(m in MODALITIES, "Unknown clean modality")
        ident = row_id(row, f"sample line {line}")
        require(ident not in ids_by_m[m], f"Duplicate clean sample identity at line {line}")
        require(boolean(row["available"], "reference availability"), "Clean reference contains unavailable evidence")
        q = csv_num(row["q"], "reference q")
        require(0 <= q <= 1, "Invalid legacy quality range")
        ids_by_m[m].add(ident)
    for m, ids in ids_by_m.items():
        geometry(ids, subjects, m)
    require(ids_by_m["eeg"] == ids_by_m["audio"] == ids_by_m["video"], "Clean modality window identities do not align")
    clean = {m: {ident: {} for ident in ids_by_m[m]} for m in MODALITIES}
    reqf = {"subject", "pair_key", "source_window_id", "condition", "modality", "available", "feature_path", "value"}
    for line, row in csv_records(paths["quality_numeric_features"], reqf):
        m, f = row["modality"], row["feature_path"]
        if row["condition"] != "reference" or m not in SPECS or f not in SPECS[m]:
            continue
        ident = row_id(row, f"feature line {line}")
        require(ident in clean[m], f"Reference feature does not match sample identity at line {line}")
        require(boolean(row["available"], "feature availability"), "Reference feature unexpectedly unavailable")
        require(f not in clean[m][ident], f"Duplicate clean feature row: {m}/{ident}/{f}; no averaging allowed")
        clean[m][ident][f] = csv_num(row["value"], f)
    for m, windows in clean.items():
        for ident, vector in windows.items():
            require(set(vector) == set(SPECS[m]), f"Missing required quality feature: {m}/{ident}")
    return clean


def fit_subset(blocks: Mapping[str, dict], vectors: Sequence[Mapping[str, float]]) -> dict:
    fitted = copy.deepcopy(blocks)
    for feature, b in fitted.items():
        values = [num(v[feature], feature) for v in vectors]
        if b["transform"] == "empirical":
            b["clean_sorted_values"] = sorted(values)
            b["unique_values"] = len(set(values))
        else:
            b["mu"], b["sigma_sample"] = mean_sd(values)
        b["n_reference"] = len(values)
    return fitted


def validate_full_parameters(models: dict, clean: dict) -> None:
    for m, blocks in models.items():
        independent = fit_subset(blocks, list(clean[m].values()))
        for feature, b in blocks.items():
            r = independent[feature]
            if b["transform"] == "empirical":
                require(len(b["clean_sorted_values"]) == len(r["clean_sorted_values"]), "Reference count mismatch")
                for x, y in zip(b["clean_sorted_values"], r["clean_sorted_values"]):
                    close(x, y, m + "/" + feature + " empirical raw-reference comparison")
            else:
                close(b["mu"], r["mu"], m + "/" + feature + ".mu")
                close(b["sigma_sample"], r["sigma_sample"], m + "/" + feature + ".sigma_sample")


def read_oof(path: Path, modalities: Sequence[str], eeg_only: bool) -> dict:
    required = {"subject", "pair_key", "source_window_id", "condition", "status", "available", "anomaly_score"}
    required.update({"policy"} if eeg_only else {"representation", "modality"})
    out = {m: {} for m in modalities}
    for line, row in csv_records(path, required):
        if row["condition"] != "reference":
            continue
        if eeg_only:
            if row["policy"] != "gaussian_line":
                continue
            m = "eeg"
        else:
            m = row["modality"]
            if row["representation"] != "hybrid" or m not in modalities:
                continue
        ident = row_id(row, f"OOF line {line}")
        require(ident not in out[m], f"Duplicate selected OOF score at {path.name}:{line}")
        require(row["status"] == "OK" and boolean(row["available"], "OOF availability"), "Invalid selected clean OOF score")
        b = csv_num(row["anomaly_score"], "OOF anomaly")
        require(0 <= b <= CAP, "OOF anomaly outside scoring support")
        out[m][ident] = b
    return out


def full_verify(paths: dict, hashes: dict, subjects: list[str], models: dict, thresholds: dict) -> dict:
    for key in DATA_FILES:
        require(paths[key].is_file(), f"Full verification requires {paths[key]}. Use explicit --verification artifacts-only ONLY for an incomplete archive.")
        require(sha(paths[key]) == hashes[key], f"Raw data SHA256 mismatch: {key}")
    clean = read_clean_data(paths, subjects)
    validate_full_parameters(models, clean)
    saved = read_oof(paths["audio_video_oof"], ("audio", "video"), False)
    saved.update(read_oof(paths["eeg_oof"], ("eeg",), True))
    result = {}
    for m in MODALITIES:
        require(set(saved[m]) == set(clean[m]), m + ": OOF score and raw clean identities differ")
        max_error = 0.0
        for held in subjects:
            training = [v for ident, v in clean[m].items() if ident[0] != held]
            require(len(training) == 100, "LOSO replay requires 100 reference windows")
            refs = fit_subset(models[m], training)
            for ident in sorted(clean[m]):
                if ident[0] != held:
                    continue
                b, _ = score_reference(clean[m][ident], refs)
                expected = saved[m][ident]
                close(b, expected, m + ": independently replayed clean LOSO score " + ident[2])
                max_error = max(max_error, abs(b - expected))
        xs = sorted(saved[m].values())
        reconstructed = xs[order_index(len(xs)) - 1]
        close(reconstructed, thresholds[m]["value"], m + ": rederived 115th order statistic")
        exceedance = sum(v > thresholds[m]["value"] for v in xs) / len(xs)
        close(exceedance, thresholds[m]["observed_crossfit_calibration_exceedance_rate"], m + ": strict exceedance rate")
        fixed_scores = [score_reference(v, models[m])[0] for v in clean[m].values()]
        result[m] = {"n_clean_loso_scores_recomputed": len(xs), "max_absolute_replay_error": max_error,
                     "threshold_order_statistic_verified": True, "rederived_threshold": reconstructed,
                     "crossfit_exceedance": exceedance,
                     "full_reference_in_sample_exceedance_diagnostic_only": sum(v > thresholds[m]["value"] for v in fixed_scores) / len(fixed_scores),
                     "full_reference_in_sample_is_independent_validation": False}
    return result


def scoring_contract() -> dict:
    return {
        "runtime_window_seconds": 5.0,
        "numeric_dtype_semantics": "float64; finite real scalars only; no numeric strings/booleans at scoring boundary",
        "source_unit_conversion": "none; use exactly the existing quality-detector output semantics",
        "mean_std_z": {"std_ddof": 1, "formula": "z=(x-mu)/sigma_sample",
                       "higher_is_worse": "max(0,z)", "lower_is_worse": "max(0,-z)", "two_sided": "abs(z)",
                       "raw_normality_assumed_by_scoring": False},
        "empirical": {"reference_values": "sorted, duplicates retained; no deduplication, smoothing, interpolation, or extrapolation",
                      "upper_tail": "p=(1+count(reference>=x))/(n+1)",
                      "lower_tail": "p=(1+count(reference<=x))/(n+1)",
                      "one_sided": "e=max(0,normal_isf(clip(p,1e-12,1-1e-12)))",
                      "two_sided": "p2=clip(min(1,2*min(p_lower,p_upper)),1e-12,1); e=max(0,normal_isf(max(p2/2,1e-12)))",
                      "finite_support_tail_saturation_retained": True,
                      "gaussian_raw_population_claimed": False},
        "evidence_cap": CAP,
        "cap_applies_to": "ALL feature evidence including EEG mean/std line evidence; standardized evidence is NOT globally unbounded",
        "aggregation": {"name": "RMS", "formula": "sqrt(sum(e_k**2)/K)",
                        "feature_order": "lexicographically sorted feature_path",
                        "counts": {m: len(SPECS[m]) for m in MODALITIES},
                        "zero_valued_guards_included_in_denominator": True,
                        "drop_missing_features_and_shrink_denominator": False},
        "availability": "external validated effective availability; never inferred from B or overwritten by good B",
        "error_policy": "missing/nonfinite/mismatched quality evidence -> explicit QUALITY_CONTROL_ERROR; not healthy and not a sensor-unavailable event",
        "window_identity": "adapter must enforce same session, source span, modality, and 5s window; no stale score reuse",
        "comparison": "strict B>T; equality is nominal for quality routing",
        "system_route_priority": ["all three unavailable -> NO_DECISION", "quality-contract error -> QUALITY_CONTROL_ERROR", "any unavailable or any B>T -> ROBUST_FUSION", "otherwise -> HEALTHY_FUSION"],
        "legacy_fusion_quality": "not changed by this exporter; a separate adapter and paired validation are required",
        "do_not_pass_anomaly_as_AF4B_quality": True,
        "reference_verification_tolerances": {"absolute": ATOL, "relative": RTOL},
    }


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    # Timestamp and provenance are deliberately excluded from behavioural identity.
    payload = {key: policy[key] for key in ("schema", "recipe", "scoring", "modalities")}
    return hashlib.sha256(json_bytes(payload)).hexdigest()


def assemble_policy(docs: dict, subjects: list[str], models: dict, thresholds: dict, source_records: dict, verification: str) -> dict:
    modalities = {}
    for m in MODALITIES:
        modalities[m] = {"recipe": "hybrid_v2_gaussian_line" if m == "eeg" else "hybrid",
                         "required_feature_count": len(models[m]), "feature_order": sorted(models[m]),
                         "reference_models": models[m], "threshold": thresholds[m]}
    policy = {
        "schema": SCHEMA, "exporter_version": VERSION, "artifact_state": STATE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "recipe": {"selected_for_this_development_export": True, "alpha_target": ALPHA,
                   "eeg_line_policy": "gaussian_line", "audio_policy": "hybrid", "video_policy": "hybrid",
                   "modality_order": list(MODALITIES), "production_selected": False},
        "scoring": scoring_contract(), "modalities": modalities,
        "activation": {"production_approved": False, "live_robot_control_authorized": False,
                       "default_mode": "candidate_replay_or_shadow_only", "active_configuration_modified": False,
                       "existing_fusion_weights_modified": False, "AF4B_new_quality_mapping_selected": False},
        "calibration_limitations": {
            "subjects": subjects, "split": "VAL_ONLY", "TEST_used": False,
            "reference_export": "full six-subject parameters, 120 clean 5s windows per feature",
            "threshold_origin": "115th order statistic of 120 subject-LOSO scores, each scored against five reference subjects (100 windows)",
            "parameter_reference_size_and_calibration_reference_size_match": False,
            "full_reference_runtime_threshold_transfer_validated": False,
            "nested_LOSO_performance_is_full_reference_runtime_performance": False,
            "window_independence_claimed": False, "exact_5_percent_false_alarm_guarantee": False,
            "system_any_modality_alarm_rate_calibrated": False,
            "development_recipe_was_chosen_using_prior_VAL_results": True,
            "nested_reference_fitting_does_not_remove_prior_VAL_policy_selection_bias": True,
            "new_emotion_accuracy_improvement_proven": False,
            "quality_scores_are_probabilities_of_correct_emotion": False,
            "required_next_checks": ["runtime arithmetic parity using identical references", "paired full-reference/threshold transfer check", "system-wide OR routing rate and NO_DECISION", "AF4B quality-input adapter compatibility", "frozen-candidate system A/B and independent evaluation before production"],
        },
        "provenance": {"logical_source_run": docs["hybrid"]["source_run_dir"], "sources": source_records,
                       "source_implementation_hashes": {r: docs[r]["script_sha256"] for r in SUPPORTED},
                       "raw_inputs_independently_checked": verification == "full",
                       "verification_level": "FULL_SOURCE_AND_CLEAN_REPLAY_VERIFIED" if verification == "full" else "ARTIFACT_CHAIN_ONLY_PARTIAL_VERIFICATION"},
    }
    policy["policy_fingerprint_sha256"] = policy_fingerprint(policy)
    return policy


def golden_cases(policy: dict) -> dict:
    feature_cases, modality_cases, routes = [], [], []
    for m in MODALITIES:
        blocks = policy["modalities"][m]["reference_models"]
        base = {f: b["reference_median"] for f, b in blocks.items()}
        for f, b in blocks.items():
            low, high = b["reference_min"], b["reference_max"]
            # Numerical probes only; not simulated physical EAV signals.
            probes = [low, b["reference_median"], high, math.nextafter(low, -math.inf), math.nextafter(high, math.inf)]
            for i, value in enumerate(probes):
                feature_cases.append({"id": f"{m}/{f}/{i}", "modality": m, "feature_path": f,
                                      "value": value, "expected_evidence": evidence(value, b)})
        bval, es = score_reference(base, blocks)
        modality_cases.append({"modality": m, "raw_feature_values": base, "expected_evidence": es, "expected_B": bval})
    ts = {m: policy["modalities"][m]["threshold"] for m in MODALITIES}
    for m in MODALITIES:
        for label, value in (("below", math.nextafter(ts[m]["value"], -math.inf)), ("equal", ts[m]["value"]), ("above", math.nextafter(ts[m]["value"], math.inf))):
            bs = {x: 0.0 for x in MODALITIES}; bs[m] = value
            av = {x: True for x in MODALITIES}
            routes.append({"id": m + "/threshold_" + label, "available": av, "B": bs,
                           "expected_route": decide_from_scores(av, bs, ts)})
    for bits in range(8):
        av = {m: bool(bits & (1 << i)) for i, m in enumerate(MODALITIES)}
        bs = {m: 0.0 if av[m] else None for m in MODALITIES}
        routes.append({"id": "availability_mask_" + str(bits), "available": av, "B": bs,
                       "expected_route": decide_from_scores(av, bs, ts)})
    return {"schema": "eav.distribution_quality.verification_cases.v1", "policy_fingerprint_sha256": policy["policy_fingerprint_sha256"],
            "synthetic_arithmetic_probes_not_EAV_inference": True, "feature_cases": feature_cases,
            "modality_cases": modality_cases, "route_cases": routes,
            "required_negative_tests": ["missing required feature raises error", "nonfinite feature raises error", "unknown feature is rejected", "availability is not manufactured from the quality score"]}


def verify_cases(policy: dict, cases: dict) -> int:
    require(cases.get("policy_fingerprint_sha256") == policy_fingerprint(policy), "Golden cases belong to a different policy")
    n = 0
    for c in cases["feature_cases"]:
        b = policy["modalities"][c["modality"]]["reference_models"][c["feature_path"]]
        close(evidence(c["value"], b), c["expected_evidence"], "Feature verification case " + c["id"]); n += 1
    for c in cases["modality_cases"]:
        b, es = score_reference(c["raw_feature_values"], policy["modalities"][c["modality"]]["reference_models"])
        close(b, c["expected_B"], "Modality verification case")
        require(set(es) == set(c["expected_evidence"]), "Golden evidence set changed")
        for f in es:
            close(es[f], c["expected_evidence"][f], "Golden modality evidence")
        n += 1
    ts = {m: policy["modalities"][m]["threshold"] for m in MODALITIES}
    for c in cases["route_cases"]:
        require(decide_from_scores(c["available"], c["B"], ts) == c["expected_route"], "Routing verification case failed: " + c["id"]); n += 1
    return n


def build_export(paths: dict, verification: str) -> tuple[dict, dict, dict]:
    tracked = {key: sha(paths[key]) for key in ("run_summary", "hybrid", "threshold", "eeg_v2")}
    docs, subjects, data_hashes = validate_chain(paths)
    models = reference_models(docs["hybrid"])
    thresholds = select_thresholds(docs)
    raw_report = None
    if verification == "full":
        for key in (*DATA_FILES, "audio_video_oof", "eeg_oof"):
            tracked[key] = sha(paths[key])
        raw_report = full_verify(paths, data_hashes, subjects, models, thresholds)
    sources = {key: {"path_at_export": str(paths[key]), "sha256": value, "bytes": paths[key].stat().st_size,
                     "bytes_verified": True} for key, value in tracked.items()}
    if verification != "full":
        for key in DATA_FILES:
            sources[key] = {"sha256_declared_by_sources": data_hashes[key], "bytes_verified": False,
                            "reason": "explicit artifacts-only mode; no source CSV/LOSO replay verification"}
    policy = assemble_policy(docs, subjects, models, thresholds, sources, verification)
    cases = golden_cases(policy)
    n_cases = verify_cases(policy, cases)
    # Detect files modified while validation/export was in progress.
    for key, before in tracked.items():
        require(sha(paths[key]) == before, "Input changed during export: " + key)
    audit = {"schema": "eav.quality_policy.export_audit.v1", "status": "EXPORT_CHECKS_PASS" if verification == "full" else "PARTIAL_VERIFICATION_ONLY",
             "verification_mode": verification, "policy_fingerprint_sha256": policy["policy_fingerprint_sha256"],
             "source_hash_consistency": True, "source_files_unchanged": True,
             "reference_feature_counts": {m: len(SPECS[m]) for m in MODALITIES},
             "clean_data_geometry_verified": verification == "full", "clean_LOSO_scores_independently_recomputed": verification == "full",
             "threshold_order_statistics_independently_verified": verification == "full",
             "clean_replay_results": raw_report, "exporter_arithmetic_case_count": n_cases,
             "arithmetic_cases_are_independent_runtime_implementation_tests": False,
             "trained_models_loaded": False, "emotion_inference_performed": False,
             "robot_actions_performed": False, "runtime_integration_validated": False,
             "thresholds_reselected_or_retuned": False, "TEST_data_used": False,
             "EEG_full_reference_mean_sd_rederived_from_existing_clean_values": True,
             "historical_nested_evaluation_not_reexecuted": True,
             "production_ready": False,
             "exporter": {"version": VERSION, "script_sha256": sha(Path(__file__).resolve()), "python": sys.version.split()[0]},
             "warnings": ["Candidate export only; production remains disabled.",
                          "Exported full6 reference is different from the full5 LOSO scoring references used to calibrate retained thresholds.",
                          "Do not infer exact 5% individual/system alarm guarantees or improved emotion accuracy.",
                          "Video has two RMS inputs, including the zero-valued face guard; EEG has seven inputs.",
                          "Mean/std line evidence is capped at 12, not unbounded.",
                          "AF4B still expects its separate [0,1] quality input; this export does not adapt it."]}
    if verification != "full":
        audit["warnings"].append("Raw CSV bytes, window geometry, and saved clean LOSO scores were NOT independently checked.")
    return policy, cases, audit


def readme(policy: dict, audit: dict) -> str:
    thresholds = "\n".join(f"- {m}: T = {policy['modalities'][m]['threshold']['value']!r}" for m in MODALITIES)
    return f"""# Distribution-quality policy candidate\n\nVersion: {VERSION}\n\nState: **{STATE}**\nVerification: **{audit['status']}**\n\n## Scope\nOnly the reviewed alpha=0.05 recipe is exported. EEG changes only its line\nevidence to sample-mean/sample-SD standardization. Audio and Video use their\nHybrid recipes. No existing runtime, config, detector or checkpoint is changed.\n\n## Thresholds (retained full precision)\n{thresholds}\n\nThe thresholds are 115th order statistics of 120 clean subject-LOSO scores.\nThey are not outer-fold averages or ordinary interpolated percentiles.\n\n## Critical binding\nThe reference models in this bundle use all 6 VAL subjects. The scores used to\ncalibrate the retained thresholds used 5-subject reference models. The runtime\ntransfer of these thresholds to the full-reference model remains UNVALIDATED.\nDo not equate a successful export with a proven full-reference false-alarm rate.\n\nEvery feature, direction, sample-SD convention, empirical tie/count rule, cap=12,\nand RMS denominator must be preserved. Video has TWO inputs, not DOVER alone.\nNever average away missing features or substitute missing evidence with zero.\n\n## Safety of the interface\nAvailability is supplied by the existing validated pipeline. All unavailable\nmeans NO_DECISION; an invalid quality input is an explicit error, not healthy.\nRMS anomalies are not probabilities of correct emotion. AF4B quality inputs\nare NOT replaced by B or by an unvalidated q mapping.\n\n## Files\n- distribution_quality_params.json: self-contained candidate scoring contract.\n- verification_cases.json: deterministic numerical probes for future runtime parity.\n- export_audit.json: exact verification scope and pending checks.\n- export_manifest.json: checksums and completion marker.\n\nUse --verify-export <this directory> to validate bundle hashes and arithmetic.\nThis verifies internal integrity, not authenticity against an external authority.\nThe exporter does not install or activate the bundle. Do not automatically load\nthe latest candidate in production; explicitly pin a reviewed bundle/version.\n"""


def write_bundle(out_dir: Path, policy: dict, cases: dict, audit: dict) -> None:
    require(not out_dir.exists(), f"Output exists; refusing overwrite: {out_dir}")
    payloads = {"distribution_quality_params.json": json_bytes(policy), "verification_cases.json": json_bytes(cases),
                "export_audit.json": json_bytes(audit), "README_export.md": readme(policy, audit).encode("utf-8")}
    manifest = {"schema": "eav.quality_policy.export_manifest.v1", "status": "COMPLETE_CANDIDATE_EXPORT",
                "artifact_state": STATE, "policy_fingerprint_sha256": policy["policy_fingerprint_sha256"],
                "verification_status": audit["status"], "production_ready": False,
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in payloads.items()}}
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Atomic reservation prevents concurrent runs from using an existing directory.
    out_dir.mkdir(exist_ok=False)
    for name, data in payloads.items():
        with (out_dir / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    # Last-written completion marker; an interrupted directory is not complete.
    with (out_dir / "export_manifest.json").open("xb") as stream:
        stream.write(json_bytes(manifest))
        stream.flush()
        os.fsync(stream.fileno())


def verify_export(out_dir: Path) -> dict:
    manifest = read_json(out_dir / "export_manifest.json")
    require(manifest.get("status") == "COMPLETE_CANDIDATE_EXPORT" and manifest.get("artifact_state") == STATE,
            "Missing/incomplete/unsupported export manifest")
    require(manifest.get("production_ready") is False, "Manifest production state modified")
    files = manifest.get("files", {})
    require(set(files) == set(BUNDLE_FILES), "Unexpected bundle file set")
    for name, entry in files.items():
        path = out_dir / name
        require(path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "Bundle file integrity mismatch: " + name)
    policy = read_json(out_dir / "distribution_quality_params.json")
    require(policy.get("schema") == SCHEMA and policy.get("artifact_state") == STATE, "Policy state/schema mismatch")
    require(policy.get("activation", {}).get("production_approved") is False, "Production activation must remain disabled")
    require(policy_fingerprint(policy) == policy.get("policy_fingerprint_sha256") == manifest.get("policy_fingerprint_sha256"), "Policy fingerprint mismatch")
    n = verify_cases(policy, read_json(out_dir / "verification_cases.json"))
    return {"status": "BUNDLE_INTEGRITY_AND_ARITHMETIC_PASS", "verification_cases": n,
            "policy_fingerprint_sha256": policy["policy_fingerprint_sha256"],
            "source_verification_status": manifest["verification_status"], "source_files_reopened": False,
            "production_ready": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true", help="Synthetic tests; no EAV/model files")
    mode.add_argument("--check-inputs", action="store_true", help="All selected validation, but no output directory")
    mode.add_argument("--verify-export", metavar="BUNDLE_DIR", help="Check an existing bundle only")
    p.add_argument("--deployment-root")
    p.add_argument("--run-dir")
    p.add_argument("--run-summary", help="Explicit run summary; useful for relocated upload copies")
    p.add_argument("--hybrid-artifact")
    p.add_argument("--threshold-artifact")
    p.add_argument("--eeg-v2-artifact")
    p.add_argument("--output-dir", help="New staging bundle directory; never an existing deployment directory")
    p.add_argument("--verification", choices=("full", "artifacts-only"), default="full")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.verify_export:
        print(json_bytes(verify_export(Path(args.verify_export).expanduser().resolve())).decode("utf-8"))
        return 0
    run_dir, paths = resolve_inputs(args)
    print("[VALIDATE] " + str(run_dir), flush=True)
    print("[MODE] " + args.verification + "; no runtime activation", flush=True)
    policy, cases, audit = build_export(paths, args.verification)
    if args.check_inputs:
        status, output = "INPUT_CHECKS_PASS" if args.verification == "full" else "INPUT_CHECKS_PARTIAL_VERIFICATION", None
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        output = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / ("quality_policy_export_" + stamp)
        write_bundle(output, policy, cases, audit)
        verify_export(output)
        status = "EXPORTED_CANDIDATE_VERIFIED" if args.verification == "full" else "EXPORTED_CANDIDATE_PARTIALLY_VERIFIED"
    print(json_bytes({"status": status, "version": VERSION, "output_dir": str(output) if output else None,
                      "thresholds": {m: policy["modalities"][m]["threshold"]["value"] for m in MODALITIES},
                      "feature_counts": {m: len(SPECS[m]) for m in MODALITIES},
                      "source_verification_status": audit["status"], "arithmetic_cases": audit["exporter_arithmetic_case_count"],
                      "production_ready": False, "existing_configuration_changed": False,
                      "full_reference_threshold_transfer_validated": False}).decode("utf-8"))
    return 0


# ---------------------------------------------------------------------------
# Synthetic regression tests, including a full raw-CSV -> clean LOSO -> export
# path. These fixtures are invented test data and never load the user's EAV.
# ---------------------------------------------------------------------------

def _write_fixture_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _put(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(obj))


def _fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    subjects = ["subject08", "subject09", "subject10", "subject13", "subject14", "subject33"]
    feature_rows, sample_rows = [], []
    raw = {m: {} for m in MODALITIES}
    for si, subject in enumerate(subjects):
        for w in range(20):
            i = 20 * si + w
            pair = f"{subject}_trial{w // 4:03d}"
            wid = f"{pair}_w{w % 4 + 1:02d}"
            ident = (subject, pair, wid)
            common = {"subject": subject, "pair_key": pair, "source_window_id": wid,
                      "condition": "reference", "available": True}
            values = {
                "eeg": {
                    LINE: 0.001 + 0.00001 * i + 0.0000003 * ((si + 1) * (w + 7) % 7),
                    "features.hf55_90_fraction_mean": 0.08 + 0.001 * i,
                    "features.slow02_1_fraction_mean": 0.15 + 0.005 * (i % 100),
                    "features.rms_uv_median": 20.0 + 0.13 * i,
                    "features.pyprep_bad_fraction": (i % 5) / 30,
                    "features.raw_flat_channel_fraction": 0.0,
                    "features.raw_hold_fraction_mean": 0.0,
                },
                "audio": {"dnsmos.OVRL_raw": 1.2 + 0.02 * i, "signal_metrics.rms_dbfs": -45.0 + 0.1 * i},
                "video": {"physical_quality.technical_raw": -0.06 + 0.0003 * i,
                          "face_observability.raw_detection_rate": 1.0},
            }
            for m in MODALITIES:
                raw[m][ident] = values[m]
                sample_rows.append({**common, "modality": m, "q": 0.9})
                for f, value in values[m].items():
                    feature_rows.append({**common, "modality": m, "family": "", "severity": "",
                                         "feature_path": f, "value": value})
    # One deliberately extreme NON-reference row with the SAME window identity.
    # It must not enter any nominal parameter or threshold computation.
    extreme = dict(feature_rows[0]); extreme.update(condition="eeg_global_line_severe", family="eeg_global_line", severity="severe", value=1e9)
    feature_rows.append(extreme)
    paths = {"run_summary": root / "run_summary.json",
             **{k: root / v for k, v in DATA_FILES.items()},
             "hybrid": root / "hybrid_quality_score_20260925_000000" / ARTIFACT_FILES["hybrid"][1],
             "threshold": root / "threshold_calibration_20260925_000100" / ARTIFACT_FILES["threshold"][1],
             "eeg_v2": root / "eeg_quality_hybrid_v2_20260925_000200" / ARTIFACT_FILES["eeg_v2"][1]}
    paths["audio_video_oof"] = paths["threshold"].parent / "outer_window_anomaly_scores.csv"
    paths["eeg_oof"] = paths["eeg_v2"].parent / "eeg_v2_outer_window_scores.csv"
    _write_fixture_csv(paths["quality_distribution_samples"], list(sample_rows[0]), sample_rows)
    _write_fixture_csv(paths["quality_numeric_features"], list(feature_rows[0]), feature_rows)
    run = {"status": "PASS_OFFLINE_PROCESSING", "formal_balanced_val": True, "suite": "robustness",
           "version": "SYNTHETIC_EXPORTER_TEST_FIXTURE_NOT_EAV", "test_data_used": False,
           "threshold_retuned": False, "frozen_model_training_performed": False,
           "attempted_cases": 3840, "planned_cases": 3840, "failed_cases": 0,
           "failed_hard_checks": 0, "events_with_errors": 0, "fatal_error": None,
           "integrity": {"status": "PASS", "changed_or_missing": []},
           "quality_distribution_logging": {"status": "OBSERVATIONAL_EXPORT_COMPLETE"}}
    _put(paths["run_summary"], run)
    source_hashes = {k: sha(paths[k]) for k in ("run_summary", *DATA_FILES)}
    objects = {}
    for role, (schema, version, script_hash) in SUPPORTED.items():
        objects[role] = {"schema": schema, "version": version, "script_sha256": script_hash,
                         "artifact_state": "CANDIDATE_NOT_DEPLOYABLE", "source_split": "VAL_ONLY", "test_data_used": False,
                         "runtime_unit": "5s_window", "source_run_dir": str(root), "source_hashes": dict(source_hashes),
                         "created_local": {"hybrid": "2026-09-25T00:00:00", "threshold": "2026-09-25T00:01:00", "eeg_v2": "2026-09-25T00:02:00"}[role],
                         "existing_deployment": {k: False for k in ("tau_0_80_changed", "quality_calibrators_changed", "fusion_weights_changed", "emotion_models_changed")},
                         "selection": {"router_selected": False, "quality_threshold_selected": False}}
    h = objects["hybrid"]
    h["aggregation_candidate"] = {"name": "RMS", "formula": "sqrt(mean(evidence^2))", "production_frozen": False}
    h["validation_protocol"] = {"subjects": subjects, "subject_loso_reference_fit": True, "clean_windows_per_train_fold_feature": 100}
    h["hybrid_policy"] = {"continuous": {}, "guards": {}}
    h["full_6_subject_VAL_reference_candidate"] = {}
    for m in MODALITIES:
        h["full_6_subject_VAL_reference_candidate"][m] = {}
        for f, (kind, direction, old_transform, _) in SPECS[m].items():
            xs = sorted(v[f] for v in raw[m].values())
            b = {"n": N_CLEAN, "min": min(xs), "max": max(xs), "mean": math.fsum(xs) / N_CLEAN,
                 "median": statistics.median(xs), "feature_type": kind, "direction": direction,
                 "transform": old_transform, "fit_scope": "all_6_VAL_subjects_clean_reference"}
            if old_transform == "empirical":
                b.update(clean_sorted_values=xs, unique_values=len(set(xs)))
            else:
                b["mu"], b["sigma_sample"] = mean_sd(xs)
            h["full_6_subject_VAL_reference_candidate"][m][f] = b
            section = "continuous" if kind == "continuous" else "guards"
            h["hybrid_policy"][section].setdefault(m, {})[f] = old_transform
    _put(paths["hybrid"], h)
    t, e = objects["threshold"], objects["eeg_v2"]
    for obj, key in ((t, "hybrid_artifact"), (e, "source_hybrid_artifact")):
        obj["source_hashes"][key] = sha(paths["hybrid"])
        obj["source_hybrid_artifact"] = str(paths["hybrid"])
        obj["nested_subject_protocol"] = {"outer_train_subjects_per_fold": 5, "inner_reference_subjects_per_fold": 4,
                                          "outer_subject_used_in_threshold_calibration": False}
    t["nested_subject_protocol"]["outer_subjects"] = subjects
    e["nested_subject_protocol"]["subjects"] = subjects
    t["aggregation"] = "RMS"
    t["threshold_rule"] = {"strict_exceedance": True, "exact_coverage_guarantee_claimed": False}
    e["aggregation"] = {"name": "RMS", "changed_from_hybrid_v1": False}
    e["scope"] = "EEG_QUALITY_POLICY_ONLY"
    e["primary_development_alpha"] = ALPHA
    e["fixed_non_line_eeg_policy"] = {f: {"feature_type": k, "direction": d, "transform": old}
                                        for f, (k, d, old, _) in SPECS["eeg"].items() if f != LINE}
    models = reference_models(h)
    av_rows, eeg_rows = [], []
    saved = {m: {} for m in MODALITIES}
    for m in MODALITIES:
        for held in subjects:
            refs = fit_subset(models[m], [v for ident, v in raw[m].items() if ident[0] != held])
            for ident, vector in raw[m].items():
                if ident[0] != held:
                    continue
                b, _ = score_reference(vector, refs)
                saved[m][ident] = b
                s, pair, wid = ident
                row = {"subject": s, "pair_key": pair, "source_window_id": wid, "condition": "reference",
                       "status": "OK", "available": True, "anomaly_score": b}
                if m == "eeg":
                    row["policy"] = "gaussian_line"; eeg_rows.append(row)
                else:
                    row.update(modality=m, representation="hybrid"); av_rows.append(row)
    _write_fixture_csv(paths["audio_video_oof"], list(av_rows[0]), av_rows)
    _write_fixture_csv(paths["eeg_oof"], list(eeg_rows[0]), eeg_rows)
    t["full_val_crossfit_threshold_candidates"], e["full_val_crossfit_threshold_candidates"] = [], []
    for m in MODALITIES:
        xs = sorted(saved[m].values()); threshold = xs[114]
        row = {"target_alpha": ALPHA, "n_calibration": 120, "order_statistic_k_1based": 115,
               "threshold_anomaly": threshold, "calibration_exceedance_rate": sum(x > threshold for x in xs) / 120,
               "used_for_nested_outer_evaluation": False}
        if m == "eeg":
            row.update(policy="gaussian_line", production_frozen=False)
            e["full_val_crossfit_threshold_candidates"].append(row)
        else:
            row.update(modality=m, representation="hybrid", deployment_frozen=False,
                       q_threshold_gaussian_kernel=math.exp(-0.5 * threshold * threshold),
                       q_threshold_halfnormal_survival=math.erfc(threshold / math.sqrt(2)))
            t["full_val_crossfit_threshold_candidates"].append(row)
    for role in ("threshold", "eeg_v2"):
        _put(paths[role], objects[role])
    for role in SUPPORTED:
        _put(paths[role].parent / SUMMARY_FILES[role], {"status": "PASS", "version": objects[role]["version"]})
    return paths


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="eav_export_tests_")
        cls.root = Path(cls.tmp.name) / "部署候选"
        cls.paths = _fixture(cls.root)
        cls.policy, cls.cases, cls.audit = build_export(cls.paths, "full")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_full_synthetic_raw_to_bundle(self):
        self.assertEqual(self.audit["status"], "EXPORT_CHECKS_PASS")
        self.assertEqual(sum(v["n_clean_loso_scores_recomputed"] for v in self.audit["clean_replay_results"].values()), 360)
        out = Path(self.tmp.name) / "full_export"
        write_bundle(out, self.policy, self.cases, self.audit)
        self.assertEqual(verify_export(out)["status"], "BUNDLE_INTEGRITY_AND_ARITHMETIC_PASS")

    def test_exact_recipe_counts(self):
        self.assertEqual(self.audit["reference_feature_counts"], {"eeg": 7, "audio": 2, "video": 2})

    def test_video_zero_guard_stays_in_denominator(self):
        blocks = self.policy["modalities"]["video"]["reference_models"]
        d = blocks["physical_quality.technical_raw"]
        values = {"face_observability.raw_detection_rate": 1.0,
                  "physical_quality.technical_raw": d["mu"] - 2 * d["sigma_sample"]}
        b, _ = score_reference(values, blocks)
        self.assertAlmostEqual(b, math.sqrt(2), places=12)

    def test_cap_preserved(self):
        b = {"transform": "mean_std_z", "direction": "higher_is_worse", "mu": 0.0, "sigma_sample": 1.0}
        self.assertEqual(evidence(1000.0, b), 12.0)

    def test_empirical_ties_kept(self):
        b = {"transform": "empirical", "direction": "higher_is_worse", "clean_sorted_values": [0.0] * 120}
        self.assertEqual(evidence(0.0, b), 0.0)
        self.assertAlmostEqual(evidence(1.0, b), -NORMAL.inv_cdf(1 / 121), places=12)

    def test_empirical_two_sided_formula(self):
        b = {"transform": "empirical", "direction": "two_sided", "clean_sorted_values": [float(i) for i in range(120)]}
        self.assertAlmostEqual(evidence(-1.0, b), -NORMAL.inv_cdf(1 / 121), places=12)

    def test_empirical_lower_direction(self):
        b = {"transform": "empirical", "direction": "lower_is_worse", "clean_sorted_values": [1.0] * 120}
        self.assertEqual(evidence(1.0, b), 0.0)
        self.assertGreater(evidence(0.0, b), 0.0)

    def test_order_rule(self):
        self.assertEqual(order_index(100), 96)
        self.assertEqual(order_index(120), 115)

    def test_strict_threshold_equality(self):
        ts = {m: self.policy["modalities"][m]["threshold"] for m in MODALITIES}
        av = {m: True for m in MODALITIES}
        b = {m: ts[m]["value"] for m in MODALITIES}
        self.assertEqual(decide_from_scores(av, b, ts), "HEALTHY_FUSION")
        b["eeg"] = math.nextafter(b["eeg"], math.inf)
        self.assertEqual(decide_from_scores(av, b, ts), "ROBUST_FUSION")

    def test_no_decision_priority(self):
        ts = {m: self.policy["modalities"][m]["threshold"] for m in MODALITIES}
        self.assertEqual(decide_from_scores({m: False for m in MODALITIES}, {}, ts), "NO_DECISION")

    def test_missing_feature_error(self):
        with self.assertRaises(ExportError):
            score_reference({}, self.policy["modalities"]["audio"]["reference_models"])

    def test_nonfinite_not_sanitized(self):
        with self.assertRaises(ExportError):
            num(float("nan"), "parameter")
        with self.assertRaises(ExportError):
            num(float("inf"), "parameter")

    def test_bool_not_numeric(self):
        with self.assertRaises(ExportError):
            num(True, "parameter")

    def test_duplicate_json_rejected(self):
        with self.assertRaises(ExportError):
            loads('{"a":1,"a":2}')

    def test_nan_json_rejected(self):
        with self.assertRaises(ExportError):
            loads('{"a":NaN}')

    def test_standardized_line_not_empirical(self):
        b = self.policy["modalities"]["eeg"]["reference_models"][LINE]
        self.assertEqual(b["transform"], "mean_std_z")
        self.assertNotIn("clean_sorted_values", b)
        self.assertEqual(b["std_ddof"], 1)

    def test_non_line_eeg_unchanged(self):
        blocks = self.policy["modalities"]["eeg"]["reference_models"]
        self.assertTrue(all(b["transform"] == "empirical" for f, b in blocks.items() if f != LINE))

    def test_no_activation_and_transfer_caveat(self):
        self.assertFalse(self.policy["activation"]["production_approved"])
        self.assertFalse(self.policy["calibration_limitations"]["full_reference_runtime_threshold_transfer_validated"])
        self.assertFalse(self.policy["calibration_limitations"]["system_any_modality_alarm_rate_calibrated"])

    def test_output_no_overwrite(self):
        out = Path(self.tmp.name) / "reserved"
        out.mkdir()
        with self.assertRaises(ExportError):
            write_bundle(out, self.policy, self.cases, self.audit)

    def test_bundle_tamper_detected(self):
        out = Path(self.tmp.name) / "tamper"
        write_bundle(out, self.policy, self.cases, self.audit)
        p = out / "distribution_quality_params.json"
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaises(ExportError):
            verify_export(out)

    def test_artifact_only_explicit_scope(self):
        _, _, audit = build_export(self.paths, "artifacts-only")
        self.assertEqual(audit["status"], "PARTIAL_VERIFICATION_ONLY")
        self.assertFalse(audit["clean_LOSO_scores_independently_recomputed"])

    def test_wrong_hybrid_lineage_rejected(self):
        p = Path(self.tmp.name) / "wrong_eeg.json"
        obj = read_json(self.paths["eeg_v2"])
        obj["source_hashes"]["source_hybrid_artifact"] = "a" * 64
        _put(p, obj)
        with self.assertRaises(ExportError):
            build_export({**self.paths, "eeg_v2": p}, "artifacts-only")

    def test_unknown_implementation_rejected(self):
        obj = read_json(self.paths["eeg_v2"])
        obj["script_sha256"] = "b" * 64
        with self.assertRaises(ExportError):
            validate_artifact("eeg_v2", obj)

    def test_raw_csv_missing_full_mode_fails(self):
        with self.assertRaises(ExportError):
            build_export({**self.paths, "quality_numeric_features": Path(self.tmp.name) / "missing.csv"}, "full")

    def test_corrupted_oof_replay_detected(self):
        p = Path(self.tmp.name) / "bad_oof.csv"
        with self.paths["eeg_oof"].open(encoding="utf-8-sig", newline="") as s:
            r = csv.DictReader(s); rows = list(r); columns = r.fieldnames
        rows[0]["anomaly_score"] = 9.0
        _write_fixture_csv(p, columns, rows)
        with self.assertRaises(ExportError):
            build_export({**self.paths, "eeg_oof": p}, "full")

    def test_duplicate_sample_identity_rejected(self):
        p = Path(self.tmp.name) / "duplicate_sample.csv"
        with self.paths["quality_distribution_samples"].open(encoding="utf-8-sig", newline="") as s:
            r = csv.DictReader(s); rows = list(r); columns = r.fieldnames
        rows.append(rows[0])
        _write_fixture_csv(p, columns, rows)
        with self.assertRaises(ExportError):
            read_clean_data({**self.paths, "quality_distribution_samples": p}, ["subject08", "subject09", "subject10", "subject13", "subject14", "subject33"])

    def test_incomplete_discovery_skipped(self):
        d = self.root / "eeg_quality_hybrid_v2_99999999_999999"
        d.mkdir(exist_ok=True)
        _put(d / ARTIFACT_FILES["eeg_v2"][1], read_json(self.paths["eeg_v2"]))
        # No completion summary: an interrupted output may not be auto-selected.
        self.assertEqual(discover_artifact(self.root, "eeg_v2"), self.paths["eeg_v2"])

    def test_scalar_verification_cases(self):
        self.assertEqual(verify_cases(self.policy, self.cases), 75)


def self_test() -> int:
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExportTests))
    print(stream.getvalue())
    print(json_bytes({"status": "PASS" if result.wasSuccessful() else "FAILED", "version": VERSION,
                      "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                      "scope": "SYNTHETIC_DATA_AND_TEMPORARY_FILES_ONLY",
                      "real_EAV_used": False, "trained_models_loaded": False,
                      "existing_configuration_modified": False}).decode("utf-8"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExport interrupted; incomplete outputs have no completion manifest.", file=sys.stderr)
        raise SystemExit(130)
    except (ExportError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"\nPOLICY EXPORT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
