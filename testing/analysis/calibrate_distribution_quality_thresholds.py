#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAV Distribution-Aware Quality Threshold Calibration v1.0
=========================================================

Purpose
-------
Calibrate statistically interpretable anomaly thresholds for the distribution-
aware quality score, using NESTED subject-wise leave-one-subject-out (LOSO)
validation on the formal balanced VAL robustness run.

This script is the threshold-calibration stage that follows:
  1) raw quality distribution audit;
  2) feature-specific Gaussian / empirical anomaly modelling;
  3) RMS modality-level anomaly aggregation;
  4) subject-LOSO hybrid quality-score validation.

Core question
-------------
Given a modality anomaly score B_m >= 0, how large must B_m become before the
runtime should enter the Degraded route?

Instead of reusing the old q < 0.80 rule, this script calibrates thresholds from
CLEAN signal behaviour at explicit target clean false-degradation budgets:

    alpha in {0.01, 0.05, 0.10}

For each modality/policy and each alpha, the script estimates a threshold:

    B_m > T_m(alpha)  -> quality-triggered Degraded

Availability remains a separate hard state:

    unavailable -> Degraded regardless of B_m

Nested subject-wise calibration
-------------------------------
Outer fold:
    hold out one of the six VAL subjects for evaluation.

Inner calibration inside the remaining five subjects:
    for each of those five subjects:
        fit feature reference distributions on the OTHER four subjects;
        score the held inner subject's clean reference windows;
    pool the five inner out-of-fold clean score sets (100 clean windows total);
    derive an upper anomaly threshold for each target alpha.

Then:
    fit the outer held subject's feature reference distributions on all five
    outer-training subjects;
    score the outer held subject;
    apply thresholds calibrated WITHOUT using the outer subject.

This means the subject being evaluated is excluded from BOTH:
    - its feature reference distribution;
    - its threshold calibration data.

Important statistical note
--------------------------
The threshold is a conservative finite-sample UPPER ORDER-STATISTIC threshold:

    k = ceil((n + 1) * (1 - alpha))
    T = k-th smallest calibration anomaly score

with detection rule B > T.

The 100 calibration windows are clustered within trials and subjects, so this
is an engineering calibration rule, NOT a claim of exact conformal coverage or
independent-window statistical guarantees.  Outer-subject performance is
reported explicitly to expose between-subject variation.

Policies compared
-----------------
HYBRID:
  - uses the source Hybrid candidate's feature-specific transforms:
      Audio:
        DNSMOS OVRL_raw -> empirical
        RMS dBFS        -> Gaussian
      EEG continuous    -> empirical
      Video DOVER       -> Gaussian
      bounded/discrete guards -> empirical
  - modality aggregation = RMS

ALL-GAUSSIAN CONTINUOUS BASELINE:
  - all continuous features -> Gaussian directional z anomaly
  - guards remain empirical
  - same RMS aggregation
  - same nested folds

This lets threshold calibration answer:
  - Audio: does all-Gaussian continuous remain better than Hybrid at the SAME
    clean false-degradation budget?
  - EEG: does Hybrid retain its advantage at fixed clean false-degradation?
  - Video: are the two representations effectively equivalent?

Primary evaluation
------------------
Router unit: 5-s window.

For each outer held subject / modality / representation / target alpha:
  - actual clean false-degradation rate;
  - reference/mild/medium/severe detection by corruption family;
  - pooled severity detection;
  - threshold stability across outer folds;
  - equivalent q thresholds for:
        q_G = exp(-0.5 * B^2)
        q_H = 2 * Phi(-B)

No q mapping is required by the router.  Equivalent q thresholds are exported
only as alternate representations of the same anomaly operating point.

This script DOES NOT
--------------------
- use TEST subjects;
- retrain emotion models;
- retrain fusion models;
- modify quality calibrators;
- modify the deployed tau=0.80 router;
- select a final alpha;
- select a final Audio/EEG policy;
- deploy thresholds;
- use corruption family as a score input;
- interpret candidate q as probability of emotion-classification correctness.

Recommended location
--------------------
最终部署/testing/analysis/calibrate_distribution_quality_thresholds.py

Recommended command from 最终部署
---------------------------------
python -X utf8 .\\testing\\analysis\\calibrate_distribution_quality_thresholds.py

Automatic discovery
-------------------
If paths are omitted, the script selects:
  - latest completed formal balanced VAL robustness run;
  - latest valid hybrid_quality_score_* candidate artifact in that run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "EAV-DIST-QUALITY-THRESHOLDS.1.0"

EXPECTED_SUBJECTS = 6
EXPECTED_TRIALS_PER_SUBJECT = 5
EXPECTED_WINDOWS_PER_TRIAL = 4
EXPECTED_CLEAN_WINDOWS_PER_SUBJECT = (
    EXPECTED_TRIALS_PER_SUBJECT * EXPECTED_WINDOWS_PER_TRIAL
)  # 20
EXPECTED_INNER_CALIBRATION_WINDOWS = (
    (EXPECTED_SUBJECTS - 1) * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 100
EXPECTED_INNER_REFERENCE_WINDOWS_PER_FEATURE = (
    (EXPECTED_SUBJECTS - 2) * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 80
EXPECTED_OUTER_REFERENCE_WINDOWS_PER_FEATURE = (
    (EXPECTED_SUBJECTS - 1) * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 100
EXPECTED_FULL_CROSSFIT_CLEAN_WINDOWS = (
    EXPECTED_SUBJECTS * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 120

TARGET_ALPHAS = (0.01, 0.05, 0.10)
REPRESENTATIONS = ("hybrid", "all_gaussian_baseline")
MAPPINGS = ("gaussian_kernel", "halfnormal_survival")
EVIDENCE_CAP = 12.0
AGGREGATION = "RMS"

FAMILY_SPECS: Dict[str, Dict[str, Any]] = {
    "audio_attenuation": {
        "modality": "audio",
        "reference_condition": "audio_matched_reference",
        "conditions": {
            "mild": "audio_attenuation_mild",
            "medium": "audio_attenuation_medium",
            "severe": "audio_attenuation_severe",
        },
    },
    "audio_white_noise": {
        "modality": "audio",
        "reference_condition": "audio_matched_reference",
        "conditions": {
            "mild": "audio_white_noise_mild",
            "medium": "audio_white_noise_medium",
            "severe": "audio_white_noise_severe",
        },
    },
    "video_blur": {
        "modality": "video",
        "reference_condition": "video_matched_reference",
        "conditions": {
            "mild": "video_blur_mild",
            "medium": "video_blur_medium",
            "severe": "video_blur_severe",
        },
    },
    "video_brightness": {
        "modality": "video",
        "reference_condition": "video_matched_reference",
        "conditions": {
            "mild": "video_brightness_mild",
            "medium": "video_brightness_medium",
            "severe": "video_brightness_severe",
        },
    },
    "eeg_global_line": {
        "modality": "eeg",
        "reference_condition": "reference",
        "conditions": {
            "mild": "eeg_global_line_mild",
            "medium": "eeg_global_line_medium",
            "severe": "eeg_global_line_severe",
        },
    },
    "eeg_channel_flatline": {
        "modality": "eeg",
        "reference_condition": "reference",
        "conditions": {
            "mild": "eeg_channel_flatline_mild",
            "medium": "eeg_channel_flatline_medium",
            "severe": "eeg_channel_flatline_severe",
        },
    },
}


class ThresholdCalibrationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ThresholdCalibrationError(message)


def json_safe(obj: Any) -> Any:
    """Recursively convert objects to strict JSON-safe values."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]

    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass

    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def json_text(obj: Any) -> str:
    return json.dumps(
        json_safe(obj),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp_{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json_text(obj) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp_{os.getpid()}")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(obj, dict), f"JSON must be an object: {path}")
    return obj


def explicit_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))

    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return bool(int(value))

    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False

    raise ThresholdCalibrationError(f"{name}: invalid boolean {value!r}")


def finite(values: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return np.asarray(out, dtype=np.float64)


def import_stats():
    try:
        from scipy import stats
    except Exception as exc:
        raise ThresholdCalibrationError("scipy is required.") from exc
    return stats


def discover_deployment_root(start: Path) -> Path:
    start = start.resolve()

    for p in [start, *start.parents]:
        if (p / "main.py").is_file() and (p / "system_checks").is_dir():
            return p

    raise ThresholdCalibrationError(
        "Cannot locate 最终部署 root. Run from 最终部署 or pass --deployment-root."
    )


def discover_latest_formal_run(root: Path) -> Path:
    system_checks = root / "system_checks"
    require(system_checks.is_dir(), f"Missing system_checks: {system_checks}")

    candidates: List[Tuple[float, Path]] = []

    for d in system_checks.iterdir():
        if not d.is_dir():
            continue

        p = d / "run_summary.json"
        if not p.is_file():
            continue

        try:
            s = read_json(p)
        except Exception:
            continue

        if (
            s.get("status") == "PASS_OFFLINE_PROCESSING"
            and s.get("formal_balanced_val") is True
            and s.get("suite") == "robustness"
            and s.get("test_data_used") is False
        ):
            candidates.append((d.stat().st_mtime, d.resolve()))

    require(candidates, "No completed formal balanced VAL robustness run found.")

    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[-1][1]


def discover_latest_hybrid_artifact(run_dir: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []

    for d in run_dir.glob("hybrid_quality_score_*"):
        if not d.is_dir():
            continue

        p = d / "hybrid_distribution_quality_score_candidate_artifact.json"
        if not p.is_file():
            continue

        try:
            a = read_json(p)
        except Exception:
            continue

        if (
            a.get("schema")
            == "eav.hybrid_distribution_quality_score_candidate.v1"
            and a.get("artifact_state") == "CANDIDATE_NOT_DEPLOYABLE"
            and a.get("source_split") == "VAL_ONLY"
            and a.get("test_data_used") is False
        ):
            candidates.append((p.stat().st_mtime, p.resolve()))

    require(
        candidates,
        "No valid hybrid_quality_score candidate artifact found under source run.",
    )

    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[-1][1]


def validate_run(run_dir: Path) -> dict:
    s = read_json(run_dir / "run_summary.json")

    require(s.get("status") == "PASS_OFFLINE_PROCESSING", "Source run is not PASS.")
    require(s.get("formal_balanced_val") is True, "Source is not formal balanced VAL.")
    require(s.get("suite") == "robustness", "Source is not robustness suite.")
    require(s.get("test_data_used") is False, "TEST data used; refusing calibration.")
    require(s.get("threshold_retuned") is False, "Source run retuned threshold.")
    require(
        s.get("frozen_model_training_performed") is False,
        "Source run reports model training.",
    )

    return s


def validate_hybrid_artifact(path: Path, run_dir: Path) -> dict:
    a = read_json(path)

    require(
        a.get("schema") == "eav.hybrid_distribution_quality_score_candidate.v1",
        "Unexpected hybrid artifact schema.",
    )
    require(
        a.get("artifact_state") == "CANDIDATE_NOT_DEPLOYABLE",
        "Hybrid artifact should still be a non-deployable candidate.",
    )
    require(a.get("source_split") == "VAL_ONLY", "Hybrid artifact is not VAL-only.")
    require(a.get("test_data_used") is False, "Hybrid artifact used TEST.")
    require(a.get("runtime_unit") == "5s_window", "Expected 5-s hybrid runtime unit.")

    source_dir = Path(str(a.get("source_run_dir", ""))).resolve()
    require(
        source_dir == run_dir.resolve(),
        f"Hybrid/source run mismatch: {source_dir} != {run_dir.resolve()}",
    )

    aggregation = a.get("aggregation_candidate", {})
    require(
        str(aggregation.get("name", "")).upper() == "RMS",
        "Threshold calibration expects RMS source aggregation candidate.",
    )
    require(
        aggregation.get("production_frozen") is False,
        "Unexpected already-frozen production aggregation.",
    )

    selection = a.get("selection", {})
    require(
        selection.get("hybrid_score_production_frozen") is False,
        "Hybrid score already frozen unexpectedly.",
    )
    require(
        selection.get("quality_threshold_selected") is False,
        "Hybrid artifact already selected a quality threshold.",
    )
    require(
        selection.get("router_selected") is False,
        "Hybrid artifact already selected a router.",
    )

    existing = a.get("existing_deployment", {})
    require(existing.get("tau_0_80_changed") is False, "Source changed tau=0.80.")
    require(
        existing.get("quality_calibrators_changed") is False,
        "Source changed existing quality calibrators.",
    )
    require(existing.get("fusion_weights_changed") is False, "Source changed fusion.")
    require(existing.get("emotion_models_changed") is False, "Source changed emotion models.")

    validation = a.get("validation_protocol", {})
    require(
        validation.get("subject_loso_reference_fit") is True,
        "Hybrid source did not use subject-wise LOSO reference fitting.",
    )

    return a


def normalize_strings(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()

    for col in cols:
        require(col in frame.columns, f"Missing grouping column: {col}")
        frame[col] = frame[col].fillna("").astype(str)

    return frame


def prepare_features(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, low_memory=False)

    required = {
        "source_window_id",
        "subject",
        "pair_key",
        "condition",
        "family",
        "severity",
        "modality",
        "available",
        "feature_path",
        "value",
    }

    require(
        required.issubset(f.columns),
        f"quality_numeric_features.csv missing: {sorted(required-set(f.columns))}",
    )

    f = normalize_strings(
        f,
        (
            "source_window_id",
            "subject",
            "pair_key",
            "condition",
            "family",
            "severity",
            "modality",
            "feature_path",
        ),
    )

    f["available_bool"] = [
        explicit_bool(v, "feature available")
        for v in f["available"]
    ]
    f["value_num"] = pd.to_numeric(f["value"], errors="coerce")
    f = f[np.isfinite(f["value_num"].to_numpy(dtype=float))].copy()

    return f


def prepare_samples(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, low_memory=False)

    required = {
        "source_window_id",
        "subject",
        "pair_key",
        "condition",
        "modality",
        "available",
        "q",
    }

    require(
        required.issubset(s.columns),
        f"quality_distribution_samples.csv missing: {sorted(required-set(s.columns))}",
    )

    s = normalize_strings(
        s,
        (
            "source_window_id",
            "subject",
            "pair_key",
            "condition",
            "modality",
        ),
    )

    s["available_bool"] = [
        explicit_bool(v, "sample available")
        for v in s["available"]
    ]
    s["q_num"] = pd.to_numeric(s["q"], errors="coerce")

    return s


def feature_spec_from_hybrid_artifact(
    artifact: Mapping[str, Any],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Returns:
      modality -> feature -> {
          feature_type: continuous|guard,
          direction: ...,
          hybrid_transform: gaussian|empirical
      }
    """
    full_ref = artifact.get("full_6_subject_VAL_reference_candidate")
    require(isinstance(full_ref, dict), "Hybrid artifact missing full reference candidate.")

    policy = artifact.get("hybrid_policy", {})
    continuous_policy = policy.get("continuous", {})
    guard_policy = policy.get("guards", {})

    require(isinstance(continuous_policy, dict), "Hybrid continuous policy invalid.")
    require(isinstance(guard_policy, dict), "Hybrid guard policy invalid.")

    result: Dict[str, Dict[str, Dict[str, str]]] = {}

    for modality, feature_blocks in full_ref.items():
        require(isinstance(feature_blocks, dict), f"{modality}: invalid full reference block.")
        result[str(modality)] = {}

        for feature, block in feature_blocks.items():
            require(isinstance(block, dict), f"{modality}/{feature}: invalid reference block.")

            feature_type = str(block.get("feature_type", ""))
            direction = str(block.get("direction", ""))

            require(
                feature_type in {"continuous", "guard"},
                f"{modality}/{feature}: invalid feature_type {feature_type!r}",
            )
            require(
                direction in {"lower_is_worse", "higher_is_worse", "two_sided"},
                f"{modality}/{feature}: invalid direction {direction!r}",
            )

            if feature_type == "continuous":
                modality_policy = continuous_policy.get(modality, {})
                require(
                    isinstance(modality_policy, dict) and feature in modality_policy,
                    f"{modality}/{feature}: absent from hybrid continuous policy.",
                )
                hybrid_transform = str(modality_policy[feature])
            else:
                modality_policy = guard_policy.get(modality, {})
                require(
                    isinstance(modality_policy, dict) and feature in modality_policy,
                    f"{modality}/{feature}: absent from hybrid guard policy.",
                )
                hybrid_transform = str(modality_policy[feature])

            require(
                hybrid_transform in {"gaussian", "empirical"},
                f"{modality}/{feature}: invalid hybrid transform {hybrid_transform!r}",
            )

            # Guards must remain empirical in this calibration stage.
            if feature_type == "guard":
                require(
                    hybrid_transform == "empirical",
                    f"{modality}/{feature}: guard unexpectedly non-empirical.",
                )

            result[str(modality)][str(feature)] = {
                "feature_type": feature_type,
                "direction": direction,
                "hybrid_transform": hybrid_transform,
            }

    require(
        set(result) == {"audio", "eeg", "video"},
        f"Unexpected modality set: {sorted(result)}",
    )

    return result


def make_feature_lookup(
    features: pd.DataFrame,
) -> Dict[Tuple[str, str, str, str], float]:
    grouped = (
        features[features["available_bool"]]
        .groupby(
            ["source_window_id", "condition", "modality", "feature_path"],
            as_index=False,
            dropna=False,
        )
        .agg(value=("value_num", "mean"))
    )

    duplicated = grouped.duplicated(
        subset=[
            "source_window_id",
            "condition",
            "modality",
            "feature_path",
        ],
        keep=False,
    )

    require(
        not bool(duplicated.any()),
        "Duplicate fully-qualified feature lookup keys remain.",
    )

    return {
        (
            str(r.source_window_id),
            str(r.condition),
            str(r.modality),
            str(r.feature_path),
        ): float(r.value)
        for r in grouped.itertuples()
    }


def build_base_samples(samples: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_window_id",
        "subject",
        "pair_key",
        "condition",
        "modality",
    ]

    base = (
        samples.groupby(keys, as_index=False, dropna=False)
        .agg(
            available=("available_bool", "min"),
            existing_q=("q_num", "mean"),
        )
    )

    return base


def clean_window_values(
    features: pd.DataFrame,
    modality: str,
    feature: str,
    subjects: Sequence[str],
) -> pd.DataFrame:
    subject_set = set(str(s) for s in subjects)

    g = features[
        (features["condition"] == "reference")
        & (features["modality"] == modality)
        & (features["feature_path"] == feature)
        & (features["available_bool"])
        & (features["subject"].isin(subject_set))
    ].copy()

    w = (
        g.groupby(
            ["source_window_id", "subject", "pair_key"],
            as_index=False,
            dropna=False,
        )
        .agg(value=("value_num", "mean"))
    )

    return w


def fit_reference_block(
    values: np.ndarray,
    transform: str,
    direction: str,
) -> Dict[str, Any]:
    x = finite(values)
    require(len(x) >= 8, "Too few clean values for reference fit.")

    block: Dict[str, Any] = {
        "transform": transform,
        "direction": direction,
        "n": int(len(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
    }

    if transform == "gaussian":
        sigma = float(np.std(x, ddof=1))
        require(
            sigma > 0 and math.isfinite(sigma),
            "Gaussian reference sigma invalid.",
        )
        block["mu"] = float(np.mean(x))
        block["sigma_sample"] = sigma

    elif transform == "empirical":
        block["clean_sorted_values"] = np.sort(x).tolist()
        block["unique_values"] = int(len(np.unique(x)))

    else:
        raise ThresholdCalibrationError(f"Unknown transform {transform!r}")

    return block


def transform_for_representation(
    feature_meta: Mapping[str, str],
    representation: str,
) -> str:
    require(
        representation in REPRESENTATIONS,
        f"Unknown representation: {representation}",
    )

    feature_type = str(feature_meta["feature_type"])

    if feature_type == "guard":
        return "empirical"

    if representation == "hybrid":
        return str(feature_meta["hybrid_transform"])

    # All-Gaussian CONTINUOUS baseline.
    return "gaussian"


def fit_reference(
    features: pd.DataFrame,
    train_subjects: Sequence[str],
    feature_spec: Mapping[str, Mapping[str, Mapping[str, str]]],
    representation: str,
    expected_windows_per_feature: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    reference: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for modality, features_for_modality in feature_spec.items():
        reference[modality] = {}

        for feature, meta in features_for_modality.items():
            direction = str(meta["direction"])
            transform = transform_for_representation(meta, representation)

            w = clean_window_values(
                features,
                modality=modality,
                feature=feature,
                subjects=train_subjects,
            )

            require(
                len(w) == expected_windows_per_feature,
                f"{representation}/{modality}/{feature}: expected "
                f"{expected_windows_per_feature} clean reference windows, got {len(w)}",
            )
            require(
                w["subject"].nunique() == len(train_subjects),
                f"{representation}/{modality}/{feature}: wrong reference subject count.",
            )

            block = fit_reference_block(
                w["value"].to_numpy(dtype=float),
                transform=transform,
                direction=direction,
            )
            block["feature_type"] = str(meta["feature_type"])
            reference[modality][feature] = block

    return reference


def gaussian_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> float:
    mu = float(block["mu"])
    sigma = float(block["sigma_sample"])
    direction = str(block["direction"])

    z = (float(value) - mu) / sigma

    if direction == "lower_is_worse":
        e = max(0.0, -z)
    elif direction == "higher_is_worse":
        e = max(0.0, z)
    elif direction == "two_sided":
        e = abs(z)
    else:
        raise ThresholdCalibrationError(f"Unknown Gaussian direction {direction!r}")

    return float(min(EVIDENCE_CAP, e))


def empirical_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> float:
    stats = import_stats()

    clean = np.asarray(
        block["clean_sorted_values"],
        dtype=np.float64,
    )
    require(len(clean) >= 8, "Empirical reference too small.")

    x = float(value)
    n = len(clean)
    direction = str(block["direction"])

    if direction == "higher_is_worse":
        tail_count = int(np.sum(clean >= x))
        p = (1.0 + tail_count) / (n + 1.0)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        e = max(0.0, float(stats.norm.isf(p)))

    elif direction == "lower_is_worse":
        tail_count = int(np.sum(clean <= x))
        p = (1.0 + tail_count) / (n + 1.0)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        e = max(0.0, float(stats.norm.isf(p)))

    elif direction == "two_sided":
        p_lo = (1.0 + int(np.sum(clean <= x))) / (n + 1.0)
        p_hi = (1.0 + int(np.sum(clean >= x))) / (n + 1.0)
        p_two = min(1.0, 2.0 * min(p_lo, p_hi))
        p_two = float(np.clip(p_two, 1e-12, 1.0))
        e = max(
            0.0,
            float(stats.norm.isf(max(p_two / 2.0, 1e-12))),
        )

    else:
        raise ThresholdCalibrationError(f"Unknown empirical direction {direction!r}")

    return float(min(EVIDENCE_CAP, e))


def feature_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> float:
    transform = str(block["transform"])

    if transform == "gaussian":
        return gaussian_anomaly(value, block)

    if transform == "empirical":
        return empirical_anomaly(value, block)

    raise ThresholdCalibrationError(f"Unknown transform {transform!r}")


def rms_aggregate(values: Sequence[float]) -> float:
    x = finite(values)
    require(len(x) > 0, "Cannot RMS-aggregate empty evidence.")
    x = np.clip(x, 0.0, EVIDENCE_CAP)
    return float(np.sqrt(np.mean(x ** 2)))


def score_subject_rows(
    subject: str,
    base_samples: pd.DataFrame,
    feature_lookup: Mapping[Tuple[str, str, str, str], float],
    reference: Mapping[str, Mapping[str, Mapping[str, Any]]],
    representation: str,
    conditions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows: List[dict] = []

    g = base_samples[base_samples["subject"] == subject].copy()

    if conditions is not None:
        condition_set = set(str(c) for c in conditions)
        g = g[g["condition"].isin(condition_set)].copy()

    for row in g.itertuples():
        wid = str(row.source_window_id)
        condition = str(row.condition)
        modality = str(row.modality)
        available = bool(row.available)

        feature_blocks = reference.get(modality, {})
        require(
            feature_blocks,
            f"{representation}/{subject}/{modality}: missing reference feature block.",
        )

        expected_features = sorted(feature_blocks)

        if not available:
            rows.append({
                "source_window_id": wid,
                "subject": subject,
                "pair_key": str(row.pair_key),
                "condition": condition,
                "modality": modality,
                "representation": representation,
                "available": False,
                "status": "UNAVAILABLE",
                "expected_evidence_count": len(expected_features),
                "observed_evidence_count": 0,
                "anomaly_score": None,
                "existing_q": (
                    float(row.existing_q)
                    if row.existing_q is not None
                    and math.isfinite(float(row.existing_q))
                    else None
                ),
            })
            continue

        evidence: List[float] = []
        missing: List[str] = []

        for feature in expected_features:
            key = (wid, condition, modality, feature)

            if key not in feature_lookup:
                missing.append(feature)
                continue

            value = float(feature_lookup[key])
            e = feature_anomaly(
                value,
                feature_blocks[feature],
            )
            evidence.append(e)

        if missing:
            status = "PARTIAL_EVIDENCE"
            anomaly = rms_aggregate(evidence) if evidence else None
        elif not evidence:
            status = "NO_QUALITY_EVIDENCE"
            anomaly = None
        else:
            status = "OK"
            anomaly = rms_aggregate(evidence)

        rows.append({
            "source_window_id": wid,
            "subject": subject,
            "pair_key": str(row.pair_key),
            "condition": condition,
            "modality": modality,
            "representation": representation,
            "available": True,
            "status": status,
            "expected_evidence_count": len(expected_features),
            "observed_evidence_count": len(evidence),
            "anomaly_score": anomaly,
            "existing_q": (
                float(row.existing_q)
                if row.existing_q is not None
                and math.isfinite(float(row.existing_q))
                else None
            ),
        })

    return pd.DataFrame(rows)


def finite_sample_upper_threshold(
    clean_scores: Sequence[float],
    alpha: float,
) -> Dict[str, Any]:
    """
    Conservative empirical upper order-statistic threshold.

    Detection rule:
        anomaly > threshold

    k = ceil((n + 1) * (1 - alpha)), 1-based.
    If k > n, threshold = +inf.

    This is intentionally described as an engineering finite-sample rule,
    not as an exact conformal guarantee because calibration windows are
    clustered within trials/subjects.
    """
    x = np.sort(finite(clean_scores))
    require(len(x) > 0, "Cannot calibrate threshold from empty clean scores.")
    require(0.0 < alpha < 1.0, "alpha must lie in (0,1).")

    n = len(x)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))

    if k > n:
        threshold = math.inf
    else:
        threshold = float(x[k - 1])

    calibration_fdr = float(np.mean(x > threshold))

    return {
        "n_calibration": n,
        "target_alpha": float(alpha),
        "order_statistic_k_1based": k,
        "threshold_anomaly": threshold,
        "calibration_exceedance_rate": calibration_fdr,
        "calibration_score_min": float(np.min(x)),
        "calibration_score_median": float(np.median(x)),
        "calibration_score_max": float(np.max(x)),
        "calibration_score_q90": float(np.quantile(x, 0.90)),
        "calibration_score_q95": float(np.quantile(x, 0.95)),
        "calibration_score_q99": float(np.quantile(x, 0.99)),
        "coverage_guarantee_claimed": False,
    }


def anomaly_to_q_threshold(
    anomaly_threshold: float,
    mapping: str,
) -> float:
    stats = import_stats()

    t = float(anomaly_threshold)

    if math.isinf(t):
        return 0.0

    t = max(0.0, t)

    if mapping == "gaussian_kernel":
        q = math.exp(-0.5 * t * t)

    elif mapping == "halfnormal_survival":
        q = 2.0 * float(stats.norm.sf(t))

    else:
        raise ThresholdCalibrationError(f"Unknown mapping {mapping!r}")

    return float(np.clip(q, 0.0, 1.0))


def build_inner_oof_clean_scores(
    outer_train_subjects: Sequence[str],
    features: pd.DataFrame,
    base_samples: pd.DataFrame,
    feature_lookup: Mapping[Tuple[str, str, str, str], float],
    feature_spec: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for inner_held in outer_train_subjects:
        inner_train = [
            s for s in outer_train_subjects
            if s != inner_held
        ]

        require(
            len(inner_train) == EXPECTED_SUBJECTS - 2,
            "Inner reference must use four subjects.",
        )

        for representation in REPRESENTATIONS:
            reference = fit_reference(
                features,
                train_subjects=inner_train,
                feature_spec=feature_spec,
                representation=representation,
                expected_windows_per_feature=(
                    EXPECTED_INNER_REFERENCE_WINDOWS_PER_FEATURE
                ),
            )

            scored = score_subject_rows(
                subject=inner_held,
                base_samples=base_samples,
                feature_lookup=feature_lookup,
                reference=reference,
                representation=representation,
                conditions=["reference"],
            )

            require(
                len(scored) > 0,
                f"No inner clean scores for {inner_held}/{representation}",
            )

            require(
                bool((scored["status"] == "OK").all()),
                f"Inner clean scoring incomplete for {inner_held}/{representation}",
            )

            scored["inner_held_subject"] = inner_held
            scored["inner_reference_subjects"] = ";".join(sorted(inner_train))
            frames.append(scored)

    out = pd.concat(frames, ignore_index=True)

    for modality in ("audio", "eeg", "video"):
        for representation in REPRESENTATIONS:
            g = out[
                (out["modality"] == modality)
                & (out["representation"] == representation)
            ]

            require(
                len(g) == EXPECTED_INNER_CALIBRATION_WINDOWS,
                f"Inner calibration {modality}/{representation}: "
                f"expected {EXPECTED_INNER_CALIBRATION_WINDOWS} rows, got {len(g)}",
            )

            require(
                g["inner_held_subject"].nunique() == EXPECTED_SUBJECTS - 1,
                f"Inner calibration {modality}/{representation}: wrong held-subject count.",
            )

    return out


def calibrate_outer_fold_thresholds(
    outer_held: str,
    inner_oof: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for modality in ("audio", "eeg", "video"):
        for representation in REPRESENTATIONS:
            g = inner_oof[
                (inner_oof["modality"] == modality)
                & (inner_oof["representation"] == representation)
                & (inner_oof["status"] == "OK")
            ]

            scores = finite(g["anomaly_score"])

            require(
                len(scores) == EXPECTED_INNER_CALIBRATION_WINDOWS,
                f"{outer_held}/{modality}/{representation}: "
                f"expected 100 inner OOF clean scores, got {len(scores)}",
            )

            for alpha in TARGET_ALPHAS:
                result = finite_sample_upper_threshold(
                    scores,
                    alpha=alpha,
                )

                row = {
                    "outer_held_subject": outer_held,
                    "modality": modality,
                    "representation": representation,
                    **result,
                    "outer_subject_used_in_threshold_calibration": False,
                    "threshold_selected_for_deployment": False,
                }

                row["q_threshold_gaussian_kernel"] = anomaly_to_q_threshold(
                    result["threshold_anomaly"],
                    "gaussian_kernel",
                )
                row["q_threshold_halfnormal_survival"] = anomaly_to_q_threshold(
                    result["threshold_anomaly"],
                    "halfnormal_survival",
                )

                rows.append(row)

    return pd.DataFrame(rows)


def apply_outer_thresholds(
    outer_scores: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    threshold_lookup: Dict[Tuple[str, str, float], float] = {}

    for row in thresholds.itertuples():
        threshold_lookup[
            (
                str(row.modality),
                str(row.representation),
                float(row.target_alpha),
            )
        ] = float(row.threshold_anomaly)

    rows: List[dict] = []

    for score in outer_scores.itertuples():
        modality = str(score.modality)
        representation = str(score.representation)
        available = bool(score.available)
        status = str(score.status)

        for alpha in TARGET_ALPHAS:
            threshold = threshold_lookup[
                (
                    modality,
                    representation,
                    float(alpha),
                )
            ]

            if not available:
                degraded = True
                reason = "UNAVAILABLE"
            elif status != "OK":
                degraded = None
                reason = status
            else:
                anomaly = float(score.anomaly_score)
                degraded = bool(anomaly > threshold)
                reason = (
                    "ANOMALY_THRESHOLD_EXCEEDED"
                    if degraded
                    else "WITHIN_CALIBRATED_NOMINAL_REGION"
                )

            rows.append({
                "source_window_id": str(score.source_window_id),
                "subject": str(score.subject),
                "pair_key": str(score.pair_key),
                "condition": str(score.condition),
                "modality": modality,
                "representation": representation,
                "available": available,
                "status": status,
                "anomaly_score": score.anomaly_score,
                "existing_q": score.existing_q,
                "target_alpha": float(alpha),
                "threshold_anomaly": threshold,
                "degraded": degraded,
                "decision_reason": reason,
                "old_tau_0_80_reused": False,
            })

    return pd.DataFrame(rows)


def clean_false_degradation_by_fold(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    clean = decisions[
        (decisions["condition"] == "reference")
        & (decisions["status"] == "OK")
    ].copy()

    rows: List[dict] = []

    group_cols = [
        "subject",
        "modality",
        "representation",
        "target_alpha",
        "threshold_anomaly",
    ]

    for key, g in clean.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))

        d = g["degraded"].astype(bool).to_numpy()

        require(
            len(d) == EXPECTED_CLEAN_WINDOWS_PER_SUBJECT,
            f"Outer clean fold {meta}: expected 20 windows, got {len(d)}",
        )

        rows.append({
            "outer_held_subject": meta["subject"],
            "modality": meta["modality"],
            "representation": meta["representation"],
            "target_alpha": meta["target_alpha"],
            "threshold_anomaly": meta["threshold_anomaly"],
            "n_clean_windows": len(d),
            "actual_clean_false_degradation_rate": float(np.mean(d)),
            "clean_healthy_rate": float(1.0 - np.mean(d)),
            "outer_subject_used_in_calibration": False,
        })

    return pd.DataFrame(rows)


def condition_metadata() -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])
        reference = str(spec["reference_condition"])

        meta[f"{family}::reference"] = {
            "family": family,
            "modality": modality,
            "severity": "reference",
            "condition": reference,
        }

        for severity, condition in spec["conditions"].items():
            meta[f"{family}::{severity}"] = {
                "family": family,
                "modality": modality,
                "severity": severity,
                "condition": str(condition),
            }

    return meta


def degradation_detection_by_family_fold(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])

        condition_map = {
            "reference": str(spec["reference_condition"]),
            **{
                severity: str(condition)
                for severity, condition in spec["conditions"].items()
            },
        }

        for subject in sorted(decisions["subject"].unique()):
            for representation in REPRESENTATIONS:
                for alpha in TARGET_ALPHAS:
                    for severity, condition in condition_map.items():
                        g = decisions[
                            (decisions["subject"] == subject)
                            & (decisions["modality"] == modality)
                            & (decisions["representation"] == representation)
                            & (decisions["target_alpha"] == float(alpha))
                            & (decisions["condition"] == condition)
                        ].copy()

                        require(
                            len(g) == EXPECTED_CLEAN_WINDOWS_PER_SUBJECT,
                            f"{subject}/{family}/{representation}/{alpha}/{severity}: "
                            f"expected 20 windows, got {len(g)}",
                        )

                        require(
                            not g["degraded"].isna().any(),
                            f"{subject}/{family}/{representation}/{alpha}/{severity}: "
                            "undefined decisions present.",
                        )

                        degraded = g["degraded"].astype(bool).to_numpy()

                        rows.append({
                            "outer_held_subject": subject,
                            "family": family,
                            "modality": modality,
                            "representation": representation,
                            "target_alpha": float(alpha),
                            "severity": severity,
                            "condition": condition,
                            "n_windows": len(g),
                            "degraded_detection_rate": float(np.mean(degraded)),
                            "available_rate": float(g["available"].astype(bool).mean()),
                            "unavailable_rate": float(
                                1.0 - g["available"].astype(bool).mean()
                            ),
                        })

    return pd.DataFrame(rows)


def pooled_family_detection(
    family_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    group_cols = [
        "family",
        "modality",
        "representation",
        "target_alpha",
        "severity",
        "condition",
    ]

    for key, g in family_fold.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        weights = g["n_windows"].to_numpy(dtype=float)
        rates = g["degraded_detection_rate"].to_numpy(dtype=float)

        rows.append({
            **meta,
            "outer_subjects": int(g["outer_held_subject"].nunique()),
            "n_windows": int(np.sum(weights)),
            "pooled_degraded_detection_rate": float(
                np.average(rates, weights=weights)
            ),
            "subject_rate_mean": float(np.mean(rates)),
            "subject_rate_sd": float(np.std(rates, ddof=1)),
            "subject_rate_min": float(np.min(rates)),
            "subject_rate_max": float(np.max(rates)),
        })

    return pd.DataFrame(rows)


def pooled_severity_detection(
    family_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    group_cols = [
        "modality",
        "representation",
        "target_alpha",
        "severity",
    ]

    for key, g in family_fold.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))

        weights = g["n_windows"].to_numpy(dtype=float)
        rates = g["degraded_detection_rate"].to_numpy(dtype=float)

        rows.append({
            **meta,
            "families": int(g["family"].nunique()),
            "outer_subjects": int(g["outer_held_subject"].nunique()),
            "n_windows": int(np.sum(weights)),
            "pooled_degraded_detection_rate": float(
                np.average(rates, weights=weights)
            ),
            "fold_family_rate_mean": float(np.mean(rates)),
            "fold_family_rate_sd": float(np.std(rates, ddof=1)),
            "fold_family_rate_min": float(np.min(rates)),
            "fold_family_rate_max": float(np.max(rates)),
        })

    return pd.DataFrame(rows)


def threshold_stability(
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    group_cols = [
        "modality",
        "representation",
        "target_alpha",
    ]

    for key, g in thresholds.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        x = finite(g["threshold_anomaly"])

        require(
            len(x) == EXPECTED_SUBJECTS,
            f"Threshold stability {meta}: expected 6 outer thresholds, got {len(x)}",
        )

        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))

        rows.append({
            **meta,
            "outer_folds": len(x),
            "threshold_mean": mean,
            "threshold_median": float(np.median(x)),
            "threshold_sd": sd,
            "threshold_min": float(np.min(x)),
            "threshold_max": float(np.max(x)),
            "threshold_range": float(np.max(x) - np.min(x)),
            "threshold_cv_abs": (
                float(sd / abs(mean))
                if abs(mean) > 1e-12
                else None
            ),
        })

    return pd.DataFrame(rows)


def pooled_clean_fdr(
    clean_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    group_cols = [
        "modality",
        "representation",
        "target_alpha",
    ]

    for key, g in clean_fold.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))

        weights = g["n_clean_windows"].to_numpy(dtype=float)
        rates = g["actual_clean_false_degradation_rate"].to_numpy(dtype=float)

        rows.append({
            **meta,
            "outer_subjects": int(g["outer_held_subject"].nunique()),
            "n_clean_windows": int(np.sum(weights)),
            "actual_clean_false_degradation_rate": float(
                np.average(rates, weights=weights)
            ),
            "subject_clean_fdr_mean": float(np.mean(rates)),
            "subject_clean_fdr_sd": float(np.std(rates, ddof=1)),
            "subject_clean_fdr_min": float(np.min(rates)),
            "subject_clean_fdr_max": float(np.max(rates)),
        })

    return pd.DataFrame(rows)


def build_operating_points(
    threshold_summary: pd.DataFrame,
    clean_summary: pd.DataFrame,
    severity_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for row in threshold_summary.itertuples():
        modality = str(row.modality)
        representation = str(row.representation)
        alpha = float(row.target_alpha)

        clean = clean_summary[
            (clean_summary["modality"] == modality)
            & (clean_summary["representation"] == representation)
            & (clean_summary["target_alpha"] == alpha)
        ]
        require(len(clean) == 1, "Expected one pooled clean FDR row.")

        sev = severity_summary[
            (severity_summary["modality"] == modality)
            & (severity_summary["representation"] == representation)
            & (severity_summary["target_alpha"] == alpha)
        ]

        by_severity = {
            str(r.severity): float(r.pooled_degraded_detection_rate)
            for r in sev.itertuples()
        }

        rows.append({
            "modality": modality,
            "representation": representation,
            "target_alpha": alpha,
            "threshold_mean": row.threshold_mean,
            "threshold_sd": row.threshold_sd,
            "threshold_min": row.threshold_min,
            "threshold_max": row.threshold_max,
            "actual_clean_false_degradation_rate": float(
                clean["actual_clean_false_degradation_rate"].iloc[0]
            ),
            "reference_detection_rate_matched_family": by_severity.get("reference"),
            "mild_detection_rate": by_severity.get("mild"),
            "medium_detection_rate": by_severity.get("medium"),
            "severe_detection_rate": by_severity.get("severe"),
            "alpha_selected_for_deployment": False,
            "representation_selected_for_deployment": False,
        })

    return pd.DataFrame(rows)


def policy_comparison(
    operating_points: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    rows: List[dict] = []

    for alpha in TARGET_ALPHAS:
        h = operating_points[
            (operating_points["modality"] == modality)
            & (operating_points["representation"] == "hybrid")
            & (operating_points["target_alpha"] == float(alpha))
        ]
        b = operating_points[
            (operating_points["modality"] == modality)
            & (
                operating_points["representation"]
                == "all_gaussian_baseline"
            )
            & (operating_points["target_alpha"] == float(alpha))
        ]

        require(len(h) == 1 and len(b) == 1, f"{modality}/{alpha}: comparison rows missing.")

        h = h.iloc[0]
        b = b.iloc[0]

        rows.append({
            "modality": modality,
            "target_alpha": float(alpha),
            "hybrid_actual_clean_fdr": h["actual_clean_false_degradation_rate"],
            "all_gaussian_actual_clean_fdr": b["actual_clean_false_degradation_rate"],
            "delta_clean_fdr_hybrid_minus_gaussian": (
                h["actual_clean_false_degradation_rate"]
                - b["actual_clean_false_degradation_rate"]
            ),
            "hybrid_mild_detection": h["mild_detection_rate"],
            "all_gaussian_mild_detection": b["mild_detection_rate"],
            "delta_mild_detection": (
                h["mild_detection_rate"] - b["mild_detection_rate"]
            ),
            "hybrid_medium_detection": h["medium_detection_rate"],
            "all_gaussian_medium_detection": b["medium_detection_rate"],
            "delta_medium_detection": (
                h["medium_detection_rate"] - b["medium_detection_rate"]
            ),
            "hybrid_severe_detection": h["severe_detection_rate"],
            "all_gaussian_severe_detection": b["severe_detection_rate"],
            "delta_severe_detection": (
                h["severe_detection_rate"] - b["severe_detection_rate"]
            ),
            "significance_test_performed": False,
            "policy_selected": False,
        })

    return pd.DataFrame(rows)


def equivalent_q_thresholds(
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for row in thresholds.itertuples():
        threshold = float(row.threshold_anomaly)

        for mapping in MAPPINGS:
            rows.append({
                "outer_held_subject": str(row.outer_held_subject),
                "modality": str(row.modality),
                "representation": str(row.representation),
                "target_alpha": float(row.target_alpha),
                "anomaly_threshold": threshold,
                "mapping": mapping,
                "equivalent_q_threshold": anomaly_to_q_threshold(
                    threshold,
                    mapping,
                ),
                "equivalent_decision": "q < q_threshold",
                "q_mapping_selected": False,
                "anomaly_router_preferred_for_interpretability": True,
            })

    return pd.DataFrame(rows)


def full_val_crossfit_threshold_candidates(
    outer_scores_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pool each subject's CLEAN score obtained from an outer reference fit that
    excluded that subject.  This gives 120 subject-cross-fitted clean scores per
    modality/representation.

    These thresholds are candidate parameters for a later runtime-freeze stage.
    They are NOT used to evaluate the nested outer folds in this script.
    """
    clean = outer_scores_all[
        (outer_scores_all["condition"] == "reference")
        & (outer_scores_all["status"] == "OK")
    ].copy()

    rows: List[dict] = []

    for modality in ("audio", "eeg", "video"):
        for representation in REPRESENTATIONS:
            g = clean[
                (clean["modality"] == modality)
                & (clean["representation"] == representation)
            ]

            scores = finite(g["anomaly_score"])

            require(
                len(scores) == EXPECTED_FULL_CROSSFIT_CLEAN_WINDOWS,
                f"Full crossfit {modality}/{representation}: "
                f"expected 120 clean scores, got {len(scores)}",
            )

            require(
                g["subject"].nunique() == EXPECTED_SUBJECTS,
                "Full crossfit should contain all six subjects.",
            )

            for alpha in TARGET_ALPHAS:
                result = finite_sample_upper_threshold(
                    scores,
                    alpha,
                )

                rows.append({
                    "modality": modality,
                    "representation": representation,
                    **result,
                    "q_threshold_gaussian_kernel": anomaly_to_q_threshold(
                        result["threshold_anomaly"],
                        "gaussian_kernel",
                    ),
                    "q_threshold_halfnormal_survival": anomaly_to_q_threshold(
                        result["threshold_anomaly"],
                        "halfnormal_survival",
                    ),
                    "used_for_nested_outer_evaluation": False,
                    "deployment_frozen": False,
                    "purpose": "candidate_for_next_runtime_freeze_stage",
                })

    return pd.DataFrame(rows)


def create_plots(
    out_dir: Path,
    operating_points: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ThresholdCalibrationError("matplotlib is required for plots.") from exc

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=False)

    # Detection at fixed clean false-degradation budget.
    for modality in ("audio", "eeg", "video"):
        g = operating_points[
            operating_points["modality"] == modality
        ].copy()

        if g.empty:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        for representation in REPRESENTATIONS:
            r = g[g["representation"] == representation].sort_values("target_alpha")
            if r.empty:
                continue

            ax.plot(
                r["actual_clean_false_degradation_rate"].to_numpy(dtype=float),
                r["severe_detection_rate"].to_numpy(dtype=float),
                marker="o",
                label=representation,
            )

        ax.set_xlabel("actual outer-subject clean false-degradation rate")
        ax.set_ylabel("pooled severe degradation detection rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"{modality.upper()}: severe detection at calibrated clean false-degradation"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plots / f"{modality}_clean_fdr_vs_severe_detection.png",
            dpi=170,
        )
        plt.close(fig)

    # Threshold stability by target alpha.
    for modality in ("audio", "eeg", "video"):
        g = threshold_summary[
            threshold_summary["modality"] == modality
        ].copy()

        if g.empty:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        for representation in REPRESENTATIONS:
            r = g[g["representation"] == representation].sort_values("target_alpha")
            if r.empty:
                continue

            ax.errorbar(
                r["target_alpha"].to_numpy(dtype=float),
                r["threshold_mean"].to_numpy(dtype=float),
                yerr=r["threshold_sd"].to_numpy(dtype=float),
                marker="o",
                capsize=4,
                label=representation,
            )

        ax.set_xlabel("target clean false-degradation budget alpha")
        ax.set_ylabel("anomaly threshold mean ± SD across outer folds")
        ax.set_title(f"{modality.upper()}: nested-LOSO threshold stability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plots / f"{modality}_threshold_stability.png",
            dpi=170,
        )
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Nested subject-LOSO calibration of distribution-aware quality "
            "anomaly thresholds on formal balanced VAL only."
        )
    )

    p.add_argument("--deployment-root")
    p.add_argument("--run-dir")
    p.add_argument("--hybrid-artifact")
    p.add_argument("--output-dir")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true")

    return p.parse_args(argv)


def self_test() -> int:
    # Order-statistic threshold test.
    scores = np.arange(1, 101, dtype=float)

    t05 = finite_sample_upper_threshold(scores, 0.05)
    t10 = finite_sample_upper_threshold(scores, 0.10)

    # 5%: k=ceil(101*0.95)=96 -> threshold=96 -> 4/100 exceed.
    # 10%: k=ceil(101*0.90)=91 -> threshold=91 -> 9/100 exceed.
    order_rule_ok = (
        t05["order_statistic_k_1based"] == 96
        and math.isclose(t05["threshold_anomaly"], 96.0)
        and math.isclose(t05["calibration_exceedance_rate"], 0.04)
        and t10["order_statistic_k_1based"] == 91
        and math.isclose(t10["threshold_anomaly"], 91.0)
        and math.isclose(t10["calibration_exceedance_rate"], 0.09)
    )

    # q mappings must decrease with a larger anomaly threshold.
    qk1 = anomaly_to_q_threshold(1.0, "gaussian_kernel")
    qk2 = anomaly_to_q_threshold(2.0, "gaussian_kernel")
    qh1 = anomaly_to_q_threshold(1.0, "halfnormal_survival")
    qh2 = anomaly_to_q_threshold(2.0, "halfnormal_survival")

    # Gaussian vs empirical anomaly sanity.
    gblock = {
        "transform": "gaussian",
        "direction": "lower_is_worse",
        "mu": 0.0,
        "sigma_sample": 1.0,
    }
    g_nom = feature_anomaly(0.0, gblock)
    g_bad = feature_anomaly(-3.0, gblock)

    eblock = {
        "transform": "empirical",
        "direction": "higher_is_worse",
        "clean_sorted_values": np.sort(
            np.asarray([0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1])
        ).tolist(),
    }
    e_nom = feature_anomaly(0.0, eblock)
    e_bad = feature_anomaly(0.9, eblock)

    checks = {
        "finite_sample_order_rule": order_rule_ok,
        "gaussian_kernel_threshold_monotone": qk1 > qk2,
        "halfnormal_threshold_monotone": qh1 > qh2,
        "gaussian_anomaly_direction": g_bad > g_nom,
        "empirical_anomaly_direction": e_bad > e_nom,
        "rms_positive": rms_aggregate([0.0, 1.0, 2.0]) > 0.0,
        "target_alphas": TARGET_ALPHAS == (0.01, 0.05, 0.10),
        "representations": REPRESENTATIONS
        == ("hybrid", "all_gaussian_baseline"),
        "family_count": len(FAMILY_SPECS) == 6,
        "strict_json_nonfinite_sanitized": (
            '"x": null' in json_text({"x": float("nan")})
            and '"y": null' in json_text({"y": np.float64(np.inf)})
        ),
    }

    require(all(checks.values()), f"Self-test failed: {checks}")

    print(
        json_text({
            "status": "PASS",
            "version": VERSION,
            "checks": checks,
            "real_EAV_used": False,
            "test_data_used": False,
            "models_loaded": False,
            "router_changed": False,
            "threshold_selected_for_deployment": False,
        })
    )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    t0 = time.perf_counter()

    if args.deployment_root:
        deployment_root = Path(args.deployment_root).expanduser().resolve()
    else:
        try:
            deployment_root = discover_deployment_root(Path.cwd())
        except ThresholdCalibrationError:
            deployment_root = discover_deployment_root(
                Path(__file__).resolve().parent
            )

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else discover_latest_formal_run(deployment_root)
    )

    require(run_dir.is_dir(), f"Run directory missing: {run_dir}")
    run_summary = validate_run(run_dir)

    hybrid_artifact_path = (
        Path(args.hybrid_artifact).expanduser().resolve()
        if args.hybrid_artifact
        else discover_latest_hybrid_artifact(run_dir)
    )

    hybrid_artifact = validate_hybrid_artifact(
        hybrid_artifact_path,
        run_dir,
    )

    feature_path = run_dir / "quality_numeric_features.csv"
    sample_path = run_dir / "quality_distribution_samples.csv"

    require(feature_path.is_file(), f"Missing {feature_path}")
    require(sample_path.is_file(), f"Missing {sample_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / f"threshold_calibration_{timestamp}"
    )

    require(
        not out_dir.exists(),
        f"Output directory exists; refusing overwrite: {out_dir}",
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    features = prepare_features(feature_path)
    samples = prepare_samples(sample_path)

    feature_spec = feature_spec_from_hybrid_artifact(
        hybrid_artifact
    )

    feature_lookup = make_feature_lookup(features)
    base_samples = build_base_samples(samples)

    subjects = sorted(
        samples.loc[
            samples["condition"] == "reference",
            "subject",
        ].unique().tolist()
    )

    require(
        len(subjects) == EXPECTED_SUBJECTS,
        f"Expected six VAL subjects, got {subjects}",
    )

    # Formal clean geometry.
    for modality in ("audio", "eeg", "video"):
        g = base_samples[
            (base_samples["condition"] == "reference")
            & (base_samples["modality"] == modality)
        ]

        require(
            len(g) == EXPECTED_FULL_CROSSFIT_CLEAN_WINDOWS,
            f"{modality}: expected 120 clean sample rows, got {len(g)}",
        )
        require(
            g["subject"].nunique() == EXPECTED_SUBJECTS,
            f"{modality}: expected six clean subjects.",
        )
        require(
            bool(g["available"].all()),
            f"{modality}: clean reference contains unavailable windows.",
        )

    all_threshold_frames: List[pd.DataFrame] = []
    all_inner_frames: List[pd.DataFrame] = []
    all_outer_score_frames: List[pd.DataFrame] = []
    all_outer_decision_frames: List[pd.DataFrame] = []

    for outer_held in subjects:
        outer_train = [
            s for s in subjects
            if s != outer_held
        ]

        require(
            len(outer_train) == EXPECTED_SUBJECTS - 1,
            "Outer training fold must contain five subjects.",
        )

        # --------------------------------------------------------------
        # INNER: cross-fit clean scores among the five outer-training
        # subjects.  Threshold calibration never sees outer_held.
        # --------------------------------------------------------------
        inner_oof = build_inner_oof_clean_scores(
            outer_train_subjects=outer_train,
            features=features,
            base_samples=base_samples,
            feature_lookup=feature_lookup,
            feature_spec=feature_spec,
        )
        inner_oof["outer_held_subject"] = outer_held

        thresholds = calibrate_outer_fold_thresholds(
            outer_held=outer_held,
            inner_oof=inner_oof,
        )

        # --------------------------------------------------------------
        # OUTER: fit feature references on all five outer-training
        # subjects, then score the held subject.
        # --------------------------------------------------------------
        outer_rep_frames: List[pd.DataFrame] = []

        for representation in REPRESENTATIONS:
            outer_reference = fit_reference(
                features,
                train_subjects=outer_train,
                feature_spec=feature_spec,
                representation=representation,
                expected_windows_per_feature=(
                    EXPECTED_OUTER_REFERENCE_WINDOWS_PER_FEATURE
                ),
            )

            scored = score_subject_rows(
                subject=outer_held,
                base_samples=base_samples,
                feature_lookup=feature_lookup,
                reference=outer_reference,
                representation=representation,
                conditions=None,
            )

            scored["outer_reference_subjects"] = ";".join(
                sorted(outer_train)
            )
            scored["outer_held_subject"] = outer_held
            outer_rep_frames.append(scored)

        outer_scores = pd.concat(
            outer_rep_frames,
            ignore_index=True,
        )

        # Controlled family conditions must have complete evidence if
        # available. Availability itself remains a valid hard trigger.
        family_conditions = set()
        for spec in FAMILY_SPECS.values():
            family_conditions.add(str(spec["reference_condition"]))
            family_conditions.update(
                str(c)
                for c in spec["conditions"].values()
            )

        incomplete = outer_scores[
            (outer_scores["condition"].isin(family_conditions))
            & (outer_scores["available"])
            & (outer_scores["status"] != "OK")
        ]

        require(
            len(incomplete) == 0,
            f"{outer_held}: controlled family scoring contains incomplete evidence.",
        )

        decisions = apply_outer_thresholds(
            outer_scores,
            thresholds,
        )

        all_inner_frames.append(inner_oof)
        all_threshold_frames.append(thresholds)
        all_outer_score_frames.append(outer_scores)
        all_outer_decision_frames.append(decisions)

    inner_all = pd.concat(
        all_inner_frames,
        ignore_index=True,
    )
    thresholds_all = pd.concat(
        all_threshold_frames,
        ignore_index=True,
    )
    outer_scores_all = pd.concat(
        all_outer_score_frames,
        ignore_index=True,
    )
    decisions_all = pd.concat(
        all_outer_decision_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Primary operating-point evaluation.
    # --------------------------------------------------------------
    clean_fold = clean_false_degradation_by_fold(
        decisions_all
    )

    family_fold = degradation_detection_by_family_fold(
        decisions_all
    )

    family_pooled = pooled_family_detection(
        family_fold
    )

    severity_pooled = pooled_severity_detection(
        family_fold
    )

    threshold_summary = threshold_stability(
        thresholds_all
    )

    clean_summary = pooled_clean_fdr(
        clean_fold
    )

    operating_points = build_operating_points(
        threshold_summary=threshold_summary,
        clean_summary=clean_summary,
        severity_summary=severity_pooled,
    )

    audio_compare = policy_comparison(
        operating_points,
        modality="audio",
    )

    eeg_compare = policy_comparison(
        operating_points,
        modality="eeg",
    )

    video_compare = policy_comparison(
        operating_points,
        modality="video",
    )

    q_thresholds = equivalent_q_thresholds(
        thresholds_all
    )

    full_crossfit_candidates = (
        full_val_crossfit_threshold_candidates(
            outer_scores_all
        )
    )

    # --------------------------------------------------------------
    # Write outputs.
    # --------------------------------------------------------------
    atomic_csv(
        out_dir / "inner_oof_clean_anomaly_scores.csv",
        inner_all,
    )
    atomic_csv(
        out_dir / "nested_loso_thresholds.csv",
        thresholds_all,
    )
    atomic_csv(
        out_dir / "outer_window_anomaly_scores.csv",
        outer_scores_all,
    )
    atomic_csv(
        out_dir / "outer_window_threshold_decisions.csv",
        decisions_all,
    )
    atomic_csv(
        out_dir / "clean_false_degradation_by_fold.csv",
        clean_fold,
    )
    atomic_csv(
        out_dir / "clean_false_degradation_pooled.csv",
        clean_summary,
    )
    atomic_csv(
        out_dir / "degradation_detection_by_family_fold.csv",
        family_fold,
    )
    atomic_csv(
        out_dir / "degradation_detection_by_family.csv",
        family_pooled,
    )
    atomic_csv(
        out_dir / "degradation_detection_by_severity.csv",
        severity_pooled,
    )
    atomic_csv(
        out_dir / "threshold_stability.csv",
        threshold_summary,
    )
    atomic_csv(
        out_dir / "modality_operating_points.csv",
        operating_points,
    )
    atomic_csv(
        out_dir / "audio_policy_comparison.csv",
        audio_compare,
    )
    atomic_csv(
        out_dir / "eeg_policy_comparison.csv",
        eeg_compare,
    )
    atomic_csv(
        out_dir / "video_policy_comparison.csv",
        video_compare,
    )
    atomic_csv(
        out_dir / "equivalent_q_thresholds.csv",
        q_thresholds,
    )
    atomic_csv(
        out_dir / "full_val_crossfit_threshold_candidates.csv",
        full_crossfit_candidates,
    )

    if not args.no_plots:
        create_plots(
            out_dir,
            operating_points=operating_points,
            threshold_summary=threshold_summary,
        )

    # Compact operating point object for artifact / summary.
    compact_operating_points: Dict[str, Any] = {}

    for modality in ("audio", "eeg", "video"):
        compact_operating_points[modality] = {}

        for representation in REPRESENTATIONS:
            compact_operating_points[modality][representation] = {}

            g = operating_points[
                (operating_points["modality"] == modality)
                & (operating_points["representation"] == representation)
            ].sort_values("target_alpha")

            for row in g.itertuples():
                key = f"alpha_{float(row.target_alpha):.2f}"

                compact_operating_points[modality][representation][key] = {
                    "target_alpha": float(row.target_alpha),
                    "threshold_mean": float(row.threshold_mean),
                    "threshold_sd": float(row.threshold_sd),
                    "actual_clean_false_degradation_rate": float(
                        row.actual_clean_false_degradation_rate
                    ),
                    "mild_detection_rate": (
                        float(row.mild_detection_rate)
                        if row.mild_detection_rate is not None
                        and math.isfinite(float(row.mild_detection_rate))
                        else None
                    ),
                    "medium_detection_rate": (
                        float(row.medium_detection_rate)
                        if row.medium_detection_rate is not None
                        and math.isfinite(float(row.medium_detection_rate))
                        else None
                    ),
                    "severe_detection_rate": (
                        float(row.severe_detection_rate)
                        if row.severe_detection_rate is not None
                        and math.isfinite(float(row.severe_detection_rate))
                        else None
                    ),
                }

    candidate_artifact = {
        "schema": "eav.distribution_quality_threshold_candidate.v1",
        "version": VERSION,
        "artifact_state": "CANDIDATE_NOT_DEPLOYABLE",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_hybrid_artifact": str(hybrid_artifact_path),
        "source_split": "VAL_ONLY",
        "test_data_used": False,
        "runtime_unit": "5s_window",
        "aggregation": "RMS",
        "target_clean_false_degradation_budgets": list(
            TARGET_ALPHAS
        ),
        "threshold_rule": {
            "score": "modality anomaly B >= 0",
            "decision": "Degraded if unavailable OR B > T",
            "order_statistic": (
                "k=ceil((n+1)*(1-alpha)); T=k-th smallest "
                "inner-OOF clean anomaly"
            ),
            "strict_exceedance": True,
            "exact_coverage_guarantee_claimed": False,
            "reason_no_exact_guarantee": (
                "5-s windows are clustered within trials and subjects"
            ),
        },
        "nested_subject_protocol": {
            "outer_subjects": subjects,
            "outer_train_subjects_per_fold": 5,
            "outer_test_subjects_per_fold": 1,
            "inner_reference_subjects_per_fold": 4,
            "inner_held_subjects_per_subfold": 1,
            "inner_oof_clean_windows_per_modality_representation": (
                EXPECTED_INNER_CALIBRATION_WINDOWS
            ),
            "outer_subject_used_in_threshold_calibration": False,
            "corruption_family_used_as_score_input": False,
        },
        "representations": {
            "hybrid": (
                "feature-specific Gaussian/empirical transforms from source hybrid artifact"
            ),
            "all_gaussian_baseline": (
                "all continuous features Gaussian; guards empirical; RMS aggregation"
            ),
        },
        "operating_points": compact_operating_points,
        "full_val_crossfit_threshold_candidates": (
            full_crossfit_candidates.to_dict(orient="records")
        ),
        "equivalent_q_mappings": {
            "gaussian_kernel": "q=exp(-0.5*B^2)",
            "halfnormal_survival": "q=2*Phi(-B)",
            "mapping_required_by_router": False,
            "probability_of_emotion_correctness": False,
        },
        "availability": {
            "separate_hard_state": True,
            "unavailable_always_degraded": True,
            "anomaly_threshold_never_overrides_unavailable": True,
        },
        "selection": {
            "target_alpha_selected": False,
            "audio_policy_selected": False,
            "eeg_policy_selected": False,
            "video_policy_selected": False,
            "thresholds_frozen_for_production": False,
            "router_selected": False,
        },
        "existing_deployment": {
            "tau_0_80_changed": False,
            "quality_calibrators_changed": False,
            "fusion_weights_changed": False,
            "emotion_models_changed": False,
        },
        "guardrails": [
            "VAL only; no TEST subjects are used.",
            "Outer subject is excluded from threshold calibration.",
            "Inner clean scores are cross-fitted: each inner held subject is scored against the other four outer-training subjects.",
            "Known corruption family is used only for retrospective evaluation, never for score calculation.",
            "Availability remains a separate hard state.",
            "Threshold calibration is performed directly on anomaly B, not by reusing old q=0.80.",
            "Equivalent q thresholds are monotone re-expressions of B thresholds only.",
            "No policy, alpha, threshold, router, or runtime implementation is selected/deployed here.",
        ],
        "source_hashes": {
            "run_summary": sha256_file(
                run_dir / "run_summary.json"
            ),
            "quality_numeric_features": sha256_file(
                feature_path
            ),
            "quality_distribution_samples": sha256_file(
                sample_path
            ),
            "hybrid_artifact": sha256_file(
                hybrid_artifact_path
            ),
        },
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
    }

    atomic_json(
        out_dir
        / "distribution_threshold_candidate_artifact.json",
        candidate_artifact,
    )

    summary = {
        "version": VERSION,
        "status": "PASS",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_hybrid_artifact": str(hybrid_artifact_path),
        "formal_balanced_val": True,
        "test_data_used": False,
        "subjects": subjects,
        "nested_subject_loso": True,
        "target_alphas": list(TARGET_ALPHAS),
        "operating_points": compact_operating_points,
        "deployment_selection_made": False,
        "router_threshold_retuned_in_existing_system": False,
        "models_or_existing_calibrators_modified": False,
        "candidate_artifact": str(
            out_dir
            / "distribution_threshold_candidate_artifact.json"
        ),
        "outputs": sorted(
            p.name
            for p in out_dir.iterdir()
        ),
        "packages": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }

    atomic_json(
        out_dir / "threshold_calibration_summary.json",
        summary,
    )

    # Human-readable README.
    lines = [
        "# Distribution-Aware Quality Threshold Calibration",
        "",
        f"- Version: `{VERSION}`",
        f"- Source run: `{run_dir}`",
        f"- Source hybrid artifact: `{hybrid_artifact_path}`",
        "- Scope: **formal balanced VAL only**",
        "- Nested subject-wise LOSO: **Yes**",
        "- Existing tau=0.80 changed: **No**",
        "- Deployment threshold selected: **No**",
        "",
        "## Threshold rule",
        "",
        "`Degraded if unavailable OR anomaly B > T`",
        "",
        "Targets evaluated: 1%, 5%, 10% clean false-degradation budgets.",
        "",
        "Thresholds are calibrated from INNER out-of-fold clean anomaly scores, "
        "and evaluated only on the OUTER held subject.",
        "",
        "## Operating points",
        "",
    ]

    for modality in ("audio", "eeg", "video"):
        lines.append(f"### {modality.upper()}")

        for representation in REPRESENTATIONS:
            g = operating_points[
                (operating_points["modality"] == modality)
                & (operating_points["representation"] == representation)
            ].sort_values("target_alpha")

            lines.append(f"- `{representation}`")

            for row in g.itertuples():
                lines.append(
                    f"  - alpha={row.target_alpha:.2f}: "
                    f"T={row.threshold_mean:.3f}±{row.threshold_sd:.3f}, "
                    f"actual clean FDR="
                    f"{row.actual_clean_false_degradation_rate:.3f}, "
                    f"mild={row.mild_detection_rate:.3f}, "
                    f"medium={row.medium_detection_rate:.3f}, "
                    f"severe={row.severe_detection_rate:.3f}"
                )

        lines.append("")

    lines += [
        "## Decision constraints",
        "",
        "- These are candidate operating points, not production thresholds.",
        "- No TEST subjects are used.",
        "- No statistical significance claim is made.",
        "- 5-s windows are clustered within trials and subjects.",
        "- Equivalent q thresholds are only alternate monotone representations.",
        "- The next step is policy/alpha review, then a separate runtime implementation and system-level A/B test.",
        "",
    ]

    atomic_text(
        out_dir / "README_threshold_calibration.md",
        "\n".join(lines),
    )

    print("=" * 124)
    print("EAV DISTRIBUTION-AWARE QUALITY THRESHOLD CALIBRATION")
    print("=" * 124)
    print(f"Status                       : PASS")
    print(f"Version                      : {VERSION}")
    print(f"Source formal run            : {run_dir}")
    print(f"Source hybrid artifact       : {hybrid_artifact_path}")
    print(f"Output                       : {out_dir}")
    print(f"Nested subject LOSO          : True")
    print(f"Outer subjects               : {len(subjects)}")
    print(f"Inner clean scores / policy  : {EXPECTED_INNER_CALIBRATION_WINDOWS}")
    print(f"Target alphas                : {TARGET_ALPHAS}")
    print(f"TEST used                    : False")
    print(f"Existing tau changed         : False")
    print(f"Deployment selection made    : False")
    print()

    for modality in ("audio", "eeg", "video"):
        print(f"{modality.upper()} operating points:")

        g = operating_points[
            operating_points["modality"] == modality
        ].sort_values(
            ["representation", "target_alpha"]
        )

        for row in g.itertuples():
            print(
                f"  {row.representation:22s} "
                f"alpha={row.target_alpha:.2f} | "
                f"T={row.threshold_mean:.3f}±{row.threshold_sd:.3f} | "
                f"cleanFDR={row.actual_clean_false_degradation_rate:.3f} | "
                f"mild={row.mild_detection_rate:.3f} | "
                f"medium={row.medium_detection_rate:.3f} | "
                f"severe={row.severe_detection_rate:.3f}"
            )

        print()

    print("Core outputs:")
    print(
        f"  {out_dir / 'nested_loso_thresholds.csv'}"
    )
    print(
        f"  {out_dir / 'modality_operating_points.csv'}"
    )
    print(
        f"  {out_dir / 'audio_policy_comparison.csv'}"
    )
    print(
        f"  {out_dir / 'eeg_policy_comparison.csv'}"
    )
    print(
        f"  {out_dir / 'threshold_stability.csv'}"
    )
    print(
        f"  {out_dir / 'full_val_crossfit_threshold_candidates.csv'}"
    )
    print(
        f"  {out_dir / 'threshold_calibration_summary.json'}"
    )
    print(
        f"  {out_dir / 'distribution_threshold_candidate_artifact.json'}"
    )
    print("=" * 124)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print("\nINTERRUPTED.", file=sys.stderr)
        raise SystemExit(130)

    except Exception as exc:
        print(
            f"\nDISTRIBUTION QUALITY THRESHOLD CALIBRATION ERROR: {exc}",
            file=sys.stderr,
        )
        raise
