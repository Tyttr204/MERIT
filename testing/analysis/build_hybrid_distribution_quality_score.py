#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAV Hybrid Distribution-Aware Quality Score Builder v1.0
========================================================

Purpose
-------
Build and validate the final *analysis candidate* for distribution-aware signal
quality scoring before any deployment/router modification.

This stage follows the evidence established by the preceding audits:

5-s runtime features that remain reasonable Gaussian candidates:
  - Audio: signal_metrics.rms_dbfs
  - Video: physical_quality.technical_raw

5-s runtime features that should be treated non-parametrically:
  - Audio: dnsmos.OVRL_raw
  - EEG: features.line_fraction_mean
  - EEG: features.hf55_90_fraction_mean
  - EEG: features.slow02_1_fraction_mean
  - EEG: features.rms_uv_median
  - EEG discrete/bounded guards:
        features.pyprep_bad_fraction
        features.raw_flat_channel_fraction
        features.raw_hold_fraction_mean
  - Video bounded guard:
        face_observability.raw_detection_rate

Core candidate
--------------
1) Fit reference distributions on CLEAN VAL only.

2) Use subject-wise leave-one-subject-out (LOSO) scoring:
       hold out one VAL subject;
       fit all reference distributions on the other five subjects;
       score the held subject;
       repeat for all six subjects.

   This avoids scoring a subject against a reference distribution that already
   contains that subject's clean windows.

3) Convert each feature into a common non-negative anomaly scale.

   Gaussian feature:
       z = (x - mu) / sigma

       lower_is_worse  -> e = max(0, -z)
       higher_is_worse -> e = max(0, +z)
       two_sided       -> e = abs(z)

   Empirical feature / guard:
       estimate a one- or two-sided clean-tail probability p from the sorted
       training-subject clean reference values, with finite-sample correction;
       convert p into a Gaussian-equivalent anomaly magnitude.

       one-sided:  e = max(0, Phi^-1(1 - p))
       two-sided:  e = max(0, Phi^-1(1 - p_two/2))

   The empirical transform does NOT claim the raw feature is Gaussian.  The
   normal quantile is used only to place heterogeneous tail probabilities onto
   a common anomaly scale.

4) Aggregate modality evidence with RMS:
       B_m = sqrt(mean(e_k^2))

   RMS is carried forward as the analysis candidate from the previous
   aggregation comparison.  This script does NOT deploy or freeze RMS in the
   production router.

5) Study two monotone candidate reliability mappings:
       gaussian_kernel:
           q = exp(-0.5 * B^2)

       halfnormal_survival:
           q = 2 * Phi(-B)

   These q values are engineering candidate scores, not probabilities that the
   emotion classification is correct.

6) Compare the hybrid score against an internal "all-Gaussian continuous"
   baseline using exactly the same LOSO folds and the same RMS aggregation.
   Discrete guards remain empirical in both representations.

7) Report clean false-degradation rates at descriptive candidate thresholds
   0.20 / 0.40 / 0.60 / 0.80, but DO NOT select a threshold.

This script DOES NOT
--------------------
- use TEST subjects;
- train/retrain any emotion classifier;
- train/retrain any fusion network;
- modify existing EEG/Audio/Video quality calibrators;
- modify tau=0.80;
- modify the deployed router;
- choose a final reliability mapping;
- choose a final threshold;
- treat classifier confidence as signal quality;
- use the known corruption family as a score input.

Important statistical interpretation
------------------------------------
The 120 5-s windows are clustered within 30 trials and 6 subjects.  Window rows
are therefore NOT treated as 120 independent subjects.

Runtime reference parameters are necessarily estimated at the 5-s window level,
because deployment consumes 5-s windows.  Robustness ranking is evaluated on
20-s trial trajectories (four 5-s windows per condition), and LOSO keeps the
held subject out of reference fitting.

Recommended location
--------------------
最终部署/testing/analysis/build_hybrid_distribution_quality_score.py

Recommended command from 最终部署
---------------------------------
python -X utf8 .\\testing\\analysis\\build_hybrid_distribution_quality_score.py

The script auto-selects:
- latest completed formal balanced VAL robustness run;
- latest valid distribution_score_build_* candidate artifact in that run.

Explicit paths can be supplied with --run-dir and --score-artifact.
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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "EAV-HYBRID-DIST-QUALITY.1.0"

EXPECTED_SUBJECTS = 6
EXPECTED_CLEAN_TRIALS = 30
EXPECTED_CLEAN_WINDOWS_PER_MODALITY = 120
EXPECTED_TRAIN_WINDOWS_PER_FOLD_FEATURE = 100
EXPECTED_TEST_WINDOWS_PER_FOLD_FEATURE = 20

EVIDENCE_CAP = 12.0
CANDIDATE_THRESHOLDS = (0.20, 0.40, 0.60, 0.80)
MAPPINGS = ("gaussian_kernel", "halfnormal_survival")
REPRESENTATIONS = ("hybrid", "all_gaussian_baseline")

# Carried forward from the previous aggregation analysis as the next candidate.
# It is NOT a deployed/frozen production choice in this script.
AGGREGATION_CANDIDATE = "rms"

# Explicit feature-specific hybrid policy established from the previous
# 5-s runtime normality audit.  Direction is read from the source artifact.
HYBRID_CONTINUOUS_POLICY: Dict[str, Dict[str, str]] = {
    "audio": {
        "dnsmos.OVRL_raw": "empirical",
        "signal_metrics.rms_dbfs": "gaussian",
    },
    "eeg": {
        "features.hf55_90_fraction_mean": "empirical",
        "features.line_fraction_mean": "empirical",
        "features.rms_uv_median": "empirical",
        "features.slow02_1_fraction_mean": "empirical",
    },
    "video": {
        "physical_quality.technical_raw": "gaussian",
    },
}

FAMILY_SPECS: Dict[str, Dict[str, Any]] = {
    "audio_attenuation": {
        "modality": "audio",
        "reference_condition": "audio_matched_reference",
        "conditions": [
            "audio_attenuation_mild",
            "audio_attenuation_medium",
            "audio_attenuation_severe",
        ],
    },
    "audio_white_noise": {
        "modality": "audio",
        "reference_condition": "audio_matched_reference",
        "conditions": [
            "audio_white_noise_mild",
            "audio_white_noise_medium",
            "audio_white_noise_severe",
        ],
    },
    "video_blur": {
        "modality": "video",
        "reference_condition": "video_matched_reference",
        "conditions": [
            "video_blur_mild",
            "video_blur_medium",
            "video_blur_severe",
        ],
    },
    "video_brightness": {
        "modality": "video",
        "reference_condition": "video_matched_reference",
        "conditions": [
            "video_brightness_mild",
            "video_brightness_medium",
            "video_brightness_severe",
        ],
    },
    "eeg_global_line": {
        "modality": "eeg",
        "reference_condition": "reference",
        "conditions": [
            "eeg_global_line_mild",
            "eeg_global_line_medium",
            "eeg_global_line_severe",
        ],
    },
    "eeg_channel_flatline": {
        "modality": "eeg",
        "reference_condition": "reference",
        "conditions": [
            "eeg_channel_flatline_mild",
            "eeg_channel_flatline_medium",
            "eeg_channel_flatline_severe",
        ],
    },
}


class HybridQualityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise HybridQualityError(message)


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
    raise HybridQualityError(f"{name}: invalid boolean {value!r}")


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
        raise HybridQualityError("scipy is required.") from exc
    return stats


def import_auc():
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise HybridQualityError("scikit-learn is required.") from exc
    return roc_auc_score


def discover_deployment_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "main.py").is_file() and (p / "system_checks").is_dir():
            return p
    raise HybridQualityError(
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


def discover_latest_score_artifact(run_dir: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []

    for d in run_dir.glob("distribution_score_build_*"):
        if not d.is_dir():
            continue
        p = d / "distribution_quality_score_candidate_artifact.json"
        if not p.is_file():
            continue
        try:
            a = read_json(p)
        except Exception:
            continue
        if (
            a.get("artifact_state") == "CANDIDATE_NOT_DEPLOYABLE"
            and a.get("test_data_used") is False
            and a.get("source_split") == "VAL_ONLY"
            and a.get("schema") == "eav.distribution_quality_score_candidate.v1"
        ):
            candidates.append((p.stat().st_mtime, p.resolve()))

    require(candidates, "No valid distribution_score_build candidate artifact found.")
    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[-1][1]


def validate_run(run_dir: Path) -> dict:
    s = read_json(run_dir / "run_summary.json")
    require(s.get("status") == "PASS_OFFLINE_PROCESSING", "Source run is not PASS.")
    require(s.get("formal_balanced_val") is True, "Source is not formal balanced VAL.")
    require(s.get("suite") == "robustness", "Source is not robustness suite.")
    require(s.get("test_data_used") is False, "TEST data used; refusing analysis.")
    require(s.get("threshold_retuned") is False, "Source run retuned threshold.")
    require(
        s.get("frozen_model_training_performed") is False,
        "Source run reports training.",
    )
    return s


def validate_score_artifact(path: Path, run_dir: Path) -> dict:
    a = read_json(path)

    require(
        a.get("artifact_state") == "CANDIDATE_NOT_DEPLOYABLE",
        "Score artifact must remain non-deployable.",
    )
    require(a.get("source_split") == "VAL_ONLY", "Score artifact is not VAL-only.")
    require(a.get("test_data_used") is False, "Score artifact used TEST.")
    require(
        a.get("runtime_unit") == "5s_window",
        "Hybrid stage requires 5-s source score artifact.",
    )

    source_dir = Path(str(a.get("source_run_dir", ""))).resolve()
    require(
        source_dir == run_dir.resolve(),
        f"Score artifact/source run mismatch: {source_dir} != {run_dir}",
    )

    existing = a.get("existing_deployment", {})
    require(existing.get("tau_0_80_changed") is False, "Source artifact changed tau.")
    require(existing.get("quality_calibrators_changed") is False, "Source changed calibrators.")
    require(existing.get("fusion_weights_changed") is False, "Source changed fusion.")
    require(existing.get("emotion_models_changed") is False, "Source changed emotion models.")

    selection = a.get("selection", {})
    require(selection.get("router_selected") is False, "Source selected a router.")
    require(selection.get("quality_threshold_selected") is False, "Source selected threshold.")

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

    f["available_bool"] = [explicit_bool(v, "feature available") for v in f["available"]]
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
        ("source_window_id", "subject", "pair_key", "condition", "modality"),
    )

    s["available_bool"] = [explicit_bool(v, "sample available") for v in s["available"]]
    s["q_num"] = pd.to_numeric(s["q"], errors="coerce")

    return s


def continuous_spec_from_artifact(
    artifact: Mapping[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    models = artifact.get("runtime_gaussian_models")
    require(isinstance(models, dict), "Score artifact missing runtime_gaussian_models.")

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for modality, features in models.items():
        require(isinstance(features, dict), f"{modality}: invalid runtime model block.")
        result[str(modality)] = {}

        for feature, block in features.items():
            require(isinstance(block, dict), f"{modality}/{feature}: invalid model block.")

            direction = str(block.get("direction", ""))
            require(
                direction in {"lower_is_worse", "higher_is_worse", "two_sided"},
                f"{modality}/{feature}: invalid direction {direction!r}",
            )

            result[str(modality)][str(feature)] = {
                "direction": direction,
                "shapiro_p": block.get("shapiro_p"),
                "anderson_reject_at_5pct_diagnostic": block.get(
                    "anderson_reject_at_5pct_diagnostic"
                ),
            }

    return result


def guard_spec_from_artifact(
    artifact: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    guards = artifact.get("empirical_discrete_guards", {})
    require(isinstance(guards, dict), "Score artifact discrete guard block invalid.")

    result: Dict[str, Dict[str, str]] = {}

    for modality, features in guards.items():
        require(isinstance(features, dict), f"{modality}: invalid guard block.")
        result[str(modality)] = {}

        for feature, block in features.items():
            require(isinstance(block, dict), f"{modality}/{feature}: invalid guard block.")

            direction = str(block.get("direction", ""))
            require(
                direction in {"lower_is_worse", "higher_is_worse", "two_sided"},
                f"{modality}/{feature}: invalid guard direction {direction!r}",
            )

            result[str(modality)][str(feature)] = direction

    return result


def validate_hybrid_policy(
    continuous_spec: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    rows: List[dict] = []

    expected_modalities = set(HYBRID_CONTINUOUS_POLICY)
    actual_modalities = set(continuous_spec)

    require(
        expected_modalities == actual_modalities,
        f"Continuous modality drift: expected {expected_modalities}, got {actual_modalities}",
    )

    for modality, policy_features in HYBRID_CONTINUOUS_POLICY.items():
        actual_features = set(continuous_spec[modality])
        expected_features = set(policy_features)

        require(
            actual_features == expected_features,
            f"{modality}: feature drift. expected={sorted(expected_features)} "
            f"actual={sorted(actual_features)}",
        )

        for feature, transform in policy_features.items():
            diagnostic = continuous_spec[modality][feature]
            shapiro_p = diagnostic.get("shapiro_p")
            ad_reject = diagnostic.get("anderson_reject_at_5pct_diagnostic")

            shapiro_value = (
                float(shapiro_p)
                if shapiro_p is not None and math.isfinite(float(shapiro_p))
                else None
            )
            ad_value = bool(ad_reject) if ad_reject is not None else None

            # This is a consistency check with the previous 5-s audit, not a new
            # universal normality decision rule.
            if transform == "gaussian":
                support = (
                    shapiro_value is not None
                    and shapiro_value >= 0.05
                    and ad_value is False
                )
            elif transform == "empirical":
                support = (
                    (shapiro_value is not None and shapiro_value < 0.05)
                    or ad_value is True
                )
            else:
                raise HybridQualityError(
                    f"{modality}/{feature}: unknown transform {transform}"
                )

            require(
                support,
                f"{modality}/{feature}: hybrid transform {transform} is not "
                f"consistent with source 5-s diagnostic "
                f"(Shapiro p={shapiro_value}, AD reject={ad_value}).",
            )

            rows.append({
                "modality": modality,
                "feature_path": feature,
                "direction": continuous_spec[modality][feature]["direction"],
                "hybrid_transform": transform,
                "source_shapiro_p": shapiro_value,
                "source_anderson_reject_at_5pct": ad_value,
                "policy_supported_by_source_diagnostic": True,
                "diagnostic_not_population_proof": True,
            })

    return pd.DataFrame(rows)


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
        subset=["source_window_id", "condition", "modality", "feature_path"],
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
    require(len(x) >= 8, "Too few clean reference values.")

    block: Dict[str, Any] = {
        "transform": transform,
        "direction": direction,
        "n": int(len(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
    }

    if transform == "gaussian":
        sigma = float(np.std(x, ddof=1))
        require(
            sigma > 0 and math.isfinite(sigma),
            "Gaussian reference sigma is invalid.",
        )
        block["mu"] = float(np.mean(x))
        block["sigma_sample"] = sigma

    elif transform == "empirical":
        block["clean_sorted_values"] = np.sort(x).tolist()
        block["unique_values"] = int(len(np.unique(x)))
        block["finite_sample_tail_floor"] = float(1.0 / (len(x) + 1.0))

    else:
        raise HybridQualityError(f"Unknown transform: {transform}")

    return block


def fit_fold_reference(
    features: pd.DataFrame,
    train_subjects: Sequence[str],
    continuous_spec: Mapping[str, Mapping[str, Mapping[str, Any]]],
    guard_spec: Mapping[str, Mapping[str, str]],
    representation: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    require(
        representation in REPRESENTATIONS,
        f"Unknown representation: {representation}",
    )

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for modality, feature_blocks in continuous_spec.items():
        result.setdefault(modality, {})

        for feature, meta in feature_blocks.items():
            direction = str(meta["direction"])
            if representation == "hybrid":
                transform = HYBRID_CONTINUOUS_POLICY[modality][feature]
            else:
                transform = "gaussian"

            w = clean_window_values(
                features,
                modality=modality,
                feature=feature,
                subjects=train_subjects,
            )

            require(
                len(w) == EXPECTED_TRAIN_WINDOWS_PER_FOLD_FEATURE,
                f"{representation}/{modality}/{feature}: expected "
                f"{EXPECTED_TRAIN_WINDOWS_PER_FOLD_FEATURE} training windows, got {len(w)}",
            )

            require(
                w["subject"].nunique() == EXPECTED_SUBJECTS - 1,
                f"{representation}/{modality}/{feature}: wrong training subject count.",
            )

            block = fit_reference_block(
                w["value"].to_numpy(dtype=float),
                transform=transform,
                direction=direction,
            )
            block["feature_type"] = "continuous"
            result[modality][feature] = block

    # Guards are empirical in BOTH hybrid and all-Gaussian continuous baseline.
    for modality, features_for_modality in guard_spec.items():
        result.setdefault(modality, {})

        for feature, direction in features_for_modality.items():
            w = clean_window_values(
                features,
                modality=modality,
                feature=feature,
                subjects=train_subjects,
            )

            require(
                len(w) == EXPECTED_TRAIN_WINDOWS_PER_FOLD_FEATURE,
                f"{representation}/{modality}/{feature}: expected "
                f"{EXPECTED_TRAIN_WINDOWS_PER_FOLD_FEATURE} guard training windows, got {len(w)}",
            )

            block = fit_reference_block(
                w["value"].to_numpy(dtype=float),
                transform="empirical",
                direction=direction,
            )
            block["feature_type"] = "guard"
            result[modality][feature] = block

    return result


def fit_full_reference(
    features: pd.DataFrame,
    all_subjects: Sequence[str],
    continuous_spec: Mapping[str, Mapping[str, Mapping[str, Any]]],
    guard_spec: Mapping[str, Mapping[str, str]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for modality, feature_blocks in continuous_spec.items():
        result.setdefault(modality, {})

        for feature, meta in feature_blocks.items():
            direction = str(meta["direction"])
            transform = HYBRID_CONTINUOUS_POLICY[modality][feature]

            w = clean_window_values(
                features,
                modality=modality,
                feature=feature,
                subjects=all_subjects,
            )

            require(
                len(w) == EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
                f"full/{modality}/{feature}: expected 120 clean windows, got {len(w)}",
            )

            block = fit_reference_block(
                w["value"].to_numpy(dtype=float),
                transform=transform,
                direction=direction,
            )
            block["feature_type"] = "continuous"
            block["fit_scope"] = "all_6_VAL_subjects_clean_reference"
            result[modality][feature] = block

    for modality, features_for_modality in guard_spec.items():
        result.setdefault(modality, {})

        for feature, direction in features_for_modality.items():
            w = clean_window_values(
                features,
                modality=modality,
                feature=feature,
                subjects=all_subjects,
            )

            require(
                len(w) == EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
                f"full/{modality}/{feature}: expected 120 clean guard windows.",
            )

            block = fit_reference_block(
                w["value"].to_numpy(dtype=float),
                transform="empirical",
                direction=direction,
            )
            block["feature_type"] = "guard"
            block["fit_scope"] = "all_6_VAL_subjects_clean_reference"
            result[modality][feature] = block

    return result


def gaussian_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
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
        raise HybridQualityError(f"Unknown Gaussian direction: {direction}")

    e = float(min(EVIDENCE_CAP, e))

    return e, {
        "z_raw": float(z),
        "tail_probability": None,
        "transform_detail": "directional_gaussian_z",
    }


def empirical_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    stats = import_stats()

    clean = np.asarray(block["clean_sorted_values"], dtype=np.float64)
    require(len(clean) >= 8, "Empirical reference too small.")

    x = float(value)
    direction = str(block["direction"])
    n = len(clean)

    if direction == "higher_is_worse":
        tail_count = int(np.sum(clean >= x))
        p = (1.0 + tail_count) / (n + 1.0)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        e_raw = float(stats.norm.isf(p))
        e = max(0.0, e_raw)
        detail = "one_sided_upper_empirical_tail_to_z_equivalent"

    elif direction == "lower_is_worse":
        tail_count = int(np.sum(clean <= x))
        p = (1.0 + tail_count) / (n + 1.0)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        e_raw = float(stats.norm.isf(p))
        e = max(0.0, e_raw)
        detail = "one_sided_lower_empirical_tail_to_z_equivalent"

    elif direction == "two_sided":
        p_lo = (1.0 + int(np.sum(clean <= x))) / (n + 1.0)
        p_hi = (1.0 + int(np.sum(clean >= x))) / (n + 1.0)
        p_two = min(1.0, 2.0 * min(p_lo, p_hi))
        p_two = float(np.clip(p_two, 1e-12, 1.0))
        e_raw = float(stats.norm.isf(max(p_two / 2.0, 1e-12)))
        e = max(0.0, e_raw)
        p = p_two
        detail = "two_sided_empirical_tail_to_abs_z_equivalent"

    else:
        raise HybridQualityError(f"Unknown empirical direction: {direction}")

    e = float(min(EVIDENCE_CAP, e))

    return e, {
        "z_raw": None,
        "tail_probability": float(p),
        "transform_detail": detail,
    }


def feature_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    transform = str(block["transform"])

    if transform == "gaussian":
        return gaussian_anomaly(value, block)

    if transform == "empirical":
        return empirical_anomaly(value, block)

    raise HybridQualityError(f"Unknown reference transform: {transform}")


def rms_aggregate(evidence: Sequence[float]) -> float:
    e = finite(evidence)
    require(len(e) > 0, "Cannot RMS-aggregate empty evidence.")
    e = np.clip(e, 0.0, EVIDENCE_CAP)
    return float(np.sqrt(np.mean(e ** 2)))


def reliability_mapping(anomaly: float, mapping: str) -> float:
    stats = import_stats()
    a = max(0.0, float(anomaly))

    if mapping == "gaussian_kernel":
        q = math.exp(-0.5 * a * a)

    elif mapping == "halfnormal_survival":
        q = 2.0 * float(stats.norm.sf(a))

    else:
        raise HybridQualityError(f"Unknown mapping: {mapping}")

    return float(np.clip(q, 0.0, 1.0))


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


def score_fold_subject(
    held_subject: str,
    base_samples: pd.DataFrame,
    feature_lookup: Mapping[Tuple[str, str, str, str], float],
    hybrid_ref: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_ref: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    evidence_rows: List[dict] = []
    score_rows: List[dict] = []

    held = base_samples[base_samples["subject"] == held_subject].copy()

    for row in held.itertuples():
        wid = str(row.source_window_id)
        condition = str(row.condition)
        modality = str(row.modality)
        available = bool(row.available)

        for representation, reference in (
            ("hybrid", hybrid_ref),
            ("all_gaussian_baseline", baseline_ref),
        ):
            expected_blocks = reference.get(modality, {})
            expected_features = sorted(expected_blocks)

            observed_evidence: List[float] = []
            missing_features: List[str] = []

            if available:
                for feature in expected_features:
                    key = (wid, condition, modality, feature)

                    if key not in feature_lookup:
                        missing_features.append(feature)
                        continue

                    value = float(feature_lookup[key])
                    block = expected_blocks[feature]
                    evidence, detail = feature_anomaly(value, block)
                    observed_evidence.append(evidence)

                    evidence_rows.append({
                        "source_window_id": wid,
                        "subject": str(row.subject),
                        "held_subject": held_subject,
                        "pair_key": str(row.pair_key),
                        "condition": condition,
                        "modality": modality,
                        "representation": representation,
                        "feature_path": feature,
                        "feature_type": block["feature_type"],
                        "transform": block["transform"],
                        "direction": block["direction"],
                        "raw_value": value,
                        "anomaly_evidence": evidence,
                        "tail_probability": detail["tail_probability"],
                        "z_raw": detail["z_raw"],
                        "transform_detail": detail["transform_detail"],
                        "corruption_family_used_as_input": False,
                    })

            if not available:
                status = "UNAVAILABLE"
                anomaly = None
                q_kernel = 0.0
                q_survival = 0.0

            elif missing_features:
                status = "PARTIAL_EVIDENCE"
                anomaly = (
                    rms_aggregate(observed_evidence)
                    if observed_evidence
                    else None
                )
                q_kernel = (
                    reliability_mapping(anomaly, "gaussian_kernel")
                    if anomaly is not None
                    else None
                )
                q_survival = (
                    reliability_mapping(anomaly, "halfnormal_survival")
                    if anomaly is not None
                    else None
                )

            elif not observed_evidence:
                status = "NO_QUALITY_EVIDENCE"
                anomaly = None
                q_kernel = None
                q_survival = None

            else:
                status = "OK"
                anomaly = rms_aggregate(observed_evidence)
                q_kernel = reliability_mapping(anomaly, "gaussian_kernel")
                q_survival = reliability_mapping(anomaly, "halfnormal_survival")

            existing_q = (
                float(row.existing_q)
                if row.existing_q is not None
                and math.isfinite(float(row.existing_q))
                else None
            )

            score_rows.append({
                "source_window_id": wid,
                "subject": str(row.subject),
                "held_subject": held_subject,
                "pair_key": str(row.pair_key),
                "condition": condition,
                "modality": modality,
                "representation": representation,
                "available": available,
                "status": status,
                "expected_evidence_count": len(expected_features),
                "observed_evidence_count": len(observed_evidence),
                "missing_feature_count": len(missing_features),
                "missing_features": ";".join(missing_features),
                "anomaly_score_rms": anomaly,
                "q_gaussian_kernel": q_kernel,
                "q_halfnormal_survival": q_survival,
                "existing_q": existing_q,
                "aggregation": AGGREGATION_CANDIDATE,
                "corruption_family_used_as_input": False,
            })

    return pd.DataFrame(evidence_rows), pd.DataFrame(score_rows)


def build_trial_scores(window_scores: pd.DataFrame) -> pd.DataFrame:
    # Primary analysis only accepts complete evidence or explicitly unavailable.
    usable = window_scores[
        window_scores["status"].isin(["OK", "UNAVAILABLE"])
    ].copy()

    t = (
        usable.groupby(
            [
                "subject",
                "held_subject",
                "pair_key",
                "condition",
                "modality",
                "representation",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            windows=("source_window_id", "nunique"),
            available_fraction=("available", "mean"),
            mean_anomaly_score=("anomaly_score_rms", "mean"),
            mean_q_gaussian_kernel=("q_gaussian_kernel", "mean"),
            mean_q_halfnormal_survival=("q_halfnormal_survival", "mean"),
            mean_existing_q=("existing_q", "mean"),
        )
    )

    return t


def evaluate_family_trajectories(
    trial_scores: pd.DataFrame,
) -> pd.DataFrame:
    stats = import_stats()
    roc_auc_score = import_auc()

    rows: List[dict] = []

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])
        reference = str(spec["reference_condition"])
        conditions = [str(x) for x in spec["conditions"]]
        ordered = [reference] + conditions

        for representation in REPRESENTATIONS:
            g = trial_scores[
                (trial_scores["modality"] == modality)
                & (trial_scores["representation"] == representation)
                & (trial_scores["condition"].isin(ordered))
                & (trial_scores["windows"] == 4)
            ].copy()

            pivot = g.pivot_table(
                index=["subject", "pair_key"],
                columns="condition",
                values="mean_anomaly_score",
                aggfunc="mean",
            )

            complete = pivot.dropna(subset=ordered).copy()

            require(
                len(complete) == EXPECTED_CLEAN_TRIALS,
                f"{family}/{representation}: expected 30 complete trial trajectories, "
                f"got {len(complete)}",
            )

            arr = complete[ordered].to_numpy(dtype=float)
            diffs = np.diff(arr, axis=1)

            spearmans: List[float] = []
            for trajectory in arr:
                rho = stats.spearmanr(
                    np.arange(4, dtype=float),
                    trajectory,
                ).statistic
                if math.isfinite(float(rho)):
                    spearmans.append(float(rho))

            require(spearmans, f"{family}/{representation}: all Spearman undefined.")

            severe_diff = arr[:, -1] - arr[:, 0]

            y = np.concatenate([
                np.zeros(len(arr), dtype=int),
                np.ones(len(arr), dtype=int),
            ])
            score = np.concatenate([arr[:, 0], arr[:, -1]])
            auc = float(roc_auc_score(y, score))

            rows.append({
                "family": family,
                "modality": modality,
                "representation": representation,
                "n_complete_trials": len(complete),
                "mean_spearman_severity_vs_anomaly": float(np.mean(spearmans)),
                "median_spearman_severity_vs_anomaly": float(np.median(spearmans)),
                "positive_spearman_rate": float(
                    np.mean(np.asarray(spearmans) > 0)
                ),
                "monotonic_nondecreasing_rate": float(
                    np.mean(np.all(diffs >= -1e-12, axis=1))
                ),
                "strict_change_somewhere_rate": float(
                    np.mean(np.any(diffs > 1e-12, axis=1))
                ),
                "severe_worse_than_reference_rate": float(
                    np.mean(arr[:, -1] > arr[:, 0])
                ),
                "mean_severe_minus_reference_anomaly": float(
                    np.mean(severe_diff)
                ),
                "median_severe_minus_reference_anomaly": float(
                    np.median(severe_diff)
                ),
                "reference_vs_severe_auc": auc,
                "deployment_selected": False,
            })

    return pd.DataFrame(rows)


def summarize_modalities(
    family_eval: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for modality in sorted(family_eval["modality"].unique()):
        for representation in REPRESENTATIONS:
            g = family_eval[
                (family_eval["modality"] == modality)
                & (family_eval["representation"] == representation)
            ].copy()

            require(len(g) > 0, f"No family evaluation for {modality}/{representation}")

            rows.append({
                "modality": modality,
                "representation": representation,
                "families": int(len(g)),
                "mean_family_spearman": float(
                    g["mean_spearman_severity_vs_anomaly"].mean()
                ),
                "worst_family_spearman": float(
                    g["mean_spearman_severity_vs_anomaly"].min()
                ),
                "mean_monotonic_rate": float(
                    g["monotonic_nondecreasing_rate"].mean()
                ),
                "mean_severe_worse_rate": float(
                    g["severe_worse_than_reference_rate"].mean()
                ),
                "mean_reference_vs_severe_auc": float(
                    g["reference_vs_severe_auc"].mean()
                ),
                "aggregation": AGGREGATION_CANDIDATE,
                "deployment_selected": False,
            })

    return pd.DataFrame(rows)


def compare_hybrid_to_baseline(
    modality_summary: pd.DataFrame,
    family_eval: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    modality_rows: List[dict] = []

    for modality in sorted(modality_summary["modality"].unique()):
        h = modality_summary[
            (modality_summary["modality"] == modality)
            & (modality_summary["representation"] == "hybrid")
        ].iloc[0]

        b = modality_summary[
            (modality_summary["modality"] == modality)
            & (modality_summary["representation"] == "all_gaussian_baseline")
        ].iloc[0]

        modality_rows.append({
            "modality": modality,
            "hybrid_mean_family_spearman": h["mean_family_spearman"],
            "baseline_mean_family_spearman": b["mean_family_spearman"],
            "delta_mean_family_spearman": (
                h["mean_family_spearman"] - b["mean_family_spearman"]
            ),
            "hybrid_worst_family_spearman": h["worst_family_spearman"],
            "baseline_worst_family_spearman": b["worst_family_spearman"],
            "delta_worst_family_spearman": (
                h["worst_family_spearman"] - b["worst_family_spearman"]
            ),
            "hybrid_mean_monotonic_rate": h["mean_monotonic_rate"],
            "baseline_mean_monotonic_rate": b["mean_monotonic_rate"],
            "delta_mean_monotonic_rate": (
                h["mean_monotonic_rate"] - b["mean_monotonic_rate"]
            ),
            "hybrid_mean_auc": h["mean_reference_vs_severe_auc"],
            "baseline_mean_auc": b["mean_reference_vs_severe_auc"],
            "delta_mean_auc": (
                h["mean_reference_vs_severe_auc"]
                - b["mean_reference_vs_severe_auc"]
            ),
            "significance_test_performed": False,
        })

    family_rows: List[dict] = []
    for family in sorted(family_eval["family"].unique()):
        h = family_eval[
            (family_eval["family"] == family)
            & (family_eval["representation"] == "hybrid")
        ].iloc[0]

        b = family_eval[
            (family_eval["family"] == family)
            & (family_eval["representation"] == "all_gaussian_baseline")
        ].iloc[0]

        family_rows.append({
            "family": family,
            "modality": h["modality"],
            "hybrid_spearman": h["mean_spearman_severity_vs_anomaly"],
            "baseline_spearman": b["mean_spearman_severity_vs_anomaly"],
            "delta_spearman": (
                h["mean_spearman_severity_vs_anomaly"]
                - b["mean_spearman_severity_vs_anomaly"]
            ),
            "hybrid_monotonic_rate": h["monotonic_nondecreasing_rate"],
            "baseline_monotonic_rate": b["monotonic_nondecreasing_rate"],
            "delta_monotonic_rate": (
                h["monotonic_nondecreasing_rate"]
                - b["monotonic_nondecreasing_rate"]
            ),
            "hybrid_auc": h["reference_vs_severe_auc"],
            "baseline_auc": b["reference_vs_severe_auc"],
            "delta_auc": (
                h["reference_vs_severe_auc"]
                - b["reference_vs_severe_auc"]
            ),
            "significance_test_performed": False,
        })

    return pd.DataFrame(modality_rows), pd.DataFrame(family_rows)


def subject_loso_stability(
    trial_scores: pd.DataFrame,
) -> pd.DataFrame:
    stats = import_stats()
    roc_auc_score = import_auc()

    rows: List[dict] = []

    subjects = sorted(trial_scores["subject"].unique().tolist())

    for subject in subjects:
        for family, spec in FAMILY_SPECS.items():
            modality = str(spec["modality"])
            reference = str(spec["reference_condition"])
            conditions = [str(x) for x in spec["conditions"]]
            ordered = [reference] + conditions

            for representation in REPRESENTATIONS:
                g = trial_scores[
                    (trial_scores["subject"] == subject)
                    & (trial_scores["modality"] == modality)
                    & (trial_scores["representation"] == representation)
                    & (trial_scores["condition"].isin(ordered))
                    & (trial_scores["windows"] == 4)
                ].copy()

                pivot = g.pivot_table(
                    index=["pair_key"],
                    columns="condition",
                    values="mean_anomaly_score",
                    aggfunc="mean",
                )
                complete = pivot.dropna(subset=ordered).copy()

                require(
                    len(complete) == 5,
                    f"{subject}/{family}/{representation}: expected 5 trials, "
                    f"got {len(complete)}",
                )

                arr = complete[ordered].to_numpy(dtype=float)
                diffs = np.diff(arr, axis=1)

                rhos: List[float] = []
                for trajectory in arr:
                    rho = stats.spearmanr(
                        np.arange(4, dtype=float),
                        trajectory,
                    ).statistic
                    if math.isfinite(float(rho)):
                        rhos.append(float(rho))

                y = np.concatenate([
                    np.zeros(len(arr), dtype=int),
                    np.ones(len(arr), dtype=int),
                ])
                score = np.concatenate([arr[:, 0], arr[:, -1]])

                rows.append({
                    "subject": subject,
                    "family": family,
                    "modality": modality,
                    "representation": representation,
                    "n_trials": len(complete),
                    "mean_trial_spearman": (
                        float(np.mean(rhos)) if rhos else None
                    ),
                    "monotonic_trial_rate": float(
                        np.mean(np.all(diffs >= -1e-12, axis=1))
                    ),
                    "severe_worse_than_reference_rate": float(
                        np.mean(arr[:, -1] > arr[:, 0])
                    ),
                    "reference_vs_severe_auc": float(
                        roc_auc_score(y, score)
                    ),
                    "held_subject_not_in_reference_fit": True,
                })

    return pd.DataFrame(rows)


def clean_false_degradation(
    window_scores: pd.DataFrame,
) -> pd.DataFrame:
    clean = window_scores[
        (window_scores["condition"] == "reference")
        & (window_scores["status"] == "OK")
    ].copy()

    rows: List[dict] = []
    mapping_cols = {
        "gaussian_kernel": "q_gaussian_kernel",
        "halfnormal_survival": "q_halfnormal_survival",
    }

    for level in ("overall", "subject"):
        if level == "overall":
            group_cols = ["modality", "representation"]
        else:
            group_cols = ["subject", "modality", "representation"]

        for key, g in clean.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)

            meta = dict(zip(group_cols, key))

            for mapping, col in mapping_cols.items():
                q = finite(g[col])
                if len(q) == 0:
                    continue

                for threshold in CANDIDATE_THRESHOLDS:
                    rows.append({
                        "level": level,
                        **meta,
                        "mapping": mapping,
                        "threshold": threshold,
                        "n_clean_windows": len(q),
                        "mean_clean_q": float(np.mean(q)),
                        "median_clean_q": float(np.median(q)),
                        "q05_clean_q": float(np.quantile(q, 0.05)),
                        "false_degradation_rate_q_below_threshold": float(
                            np.mean(q < threshold)
                        ),
                        "threshold_selected": False,
                    })

    return pd.DataFrame(rows)


def mapping_response(
    window_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []
    mapping_cols = {
        "gaussian_kernel": "q_gaussian_kernel",
        "halfnormal_survival": "q_halfnormal_survival",
    }

    for (
        condition,
        modality,
        representation,
    ), g in window_scores[
        window_scores["status"].isin(["OK", "UNAVAILABLE"])
    ].groupby(
        ["condition", "modality", "representation"],
        dropna=False,
    ):
        for mapping, col in mapping_cols.items():
            q = finite(g[col])
            if len(q) == 0:
                continue

            for threshold in CANDIDATE_THRESHOLDS:
                rows.append({
                    "condition": condition,
                    "modality": modality,
                    "representation": representation,
                    "mapping": mapping,
                    "threshold": threshold,
                    "n_windows": len(q),
                    "mean_q": float(np.mean(q)),
                    "median_q": float(np.median(q)),
                    "fraction_below_threshold": float(
                        np.mean(q < threshold)
                    ),
                    "threshold_selected": False,
                })

    return pd.DataFrame(rows)


def existing_q_comparison(
    window_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    hybrid = window_scores[
        (window_scores["representation"] == "hybrid")
        & (window_scores["status"] == "OK")
    ].copy()

    for (condition, modality), g in hybrid.groupby(
        ["condition", "modality"],
        dropna=False,
    ):
        existing = pd.to_numeric(g["existing_q"], errors="coerce").to_numpy(dtype=float)
        candidate = pd.to_numeric(
            g["q_gaussian_kernel"],
            errors="coerce",
        ).to_numpy(dtype=float)

        mask = np.isfinite(existing) & np.isfinite(candidate)
        if not np.any(mask):
            continue

        x = existing[mask]
        y = candidate[mask]

        corr = (
            float(np.corrcoef(x, y)[0, 1])
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0
            else None
        )

        rows.append({
            "condition": condition,
            "modality": modality,
            "n": int(np.sum(mask)),
            "mean_existing_q": float(np.mean(x)),
            "mean_hybrid_q_gaussian_kernel": float(np.mean(y)),
            "pearson_existing_vs_hybrid": corr,
            "old_tau_0_80_trigger_rate_existing_q": float(np.mean(x < 0.80)),
            "hybrid_q_below_0_80_descriptive_rate": float(np.mean(y < 0.80)),
            "same_0_80_scale_assumed": False,
            "not_used_for_selection": True,
        })

    return pd.DataFrame(rows)


def reference_parameter_summary(
    held_subject: str,
    train_subjects: Sequence[str],
    representation: str,
    reference: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    rows: List[dict] = []

    for modality, features in reference.items():
        for feature, block in features.items():
            rows.append({
                "held_subject": held_subject,
                "train_subjects": ";".join(sorted(train_subjects)),
                "representation": representation,
                "modality": modality,
                "feature_path": feature,
                "feature_type": block["feature_type"],
                "transform": block["transform"],
                "direction": block["direction"],
                "n_reference_windows": block["n"],
                "mean": block["mean"],
                "median": block["median"],
                "min": block["min"],
                "max": block["max"],
                "mu": block.get("mu"),
                "sigma_sample": block.get("sigma_sample"),
                "unique_values": block.get("unique_values"),
                "held_subject_excluded": True,
            })

    return pd.DataFrame(rows)


def create_plots(
    out_dir: Path,
    family_eval: pd.DataFrame,
    modality_summary: pd.DataFrame,
    clean_fd: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise HybridQualityError("matplotlib is required for plots.") from exc

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=False)

    # Hybrid vs all-Gaussian mean Spearman by family.
    pivot = family_eval.pivot(
        index="family",
        columns="representation",
        values="mean_spearman_severity_vs_anomaly",
    )
    if not pivot.empty:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        x = np.arange(len(pivot))
        width = 0.36
        if "hybrid" in pivot.columns:
            ax.bar(
                x - width / 2,
                pivot["hybrid"].to_numpy(dtype=float),
                width=width,
                label="hybrid",
            )
        if "all_gaussian_baseline" in pivot.columns:
            ax.bar(
                x + width / 2,
                pivot["all_gaussian_baseline"].to_numpy(dtype=float),
                width=width,
                label="all_gaussian_baseline",
            )
        ax.set_xticks(x, pivot.index.tolist(), rotation=55, ha="right")
        ax.set_ylabel("mean trial Spearman(severity, anomaly)")
        ax.set_ylim(-1.0, 1.05)
        ax.set_title("Hybrid vs all-Gaussian continuous baseline")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "hybrid_vs_all_gaussian_family_spearman.png", dpi=170)
        plt.close(fig)

    # Modality mean Spearman.
    if not modality_summary.empty:
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)

        modalities = sorted(modality_summary["modality"].unique())
        x = np.arange(len(modalities))
        width = 0.36

        hybrid_vals = []
        baseline_vals = []

        for modality in modalities:
            hybrid_vals.append(
                float(
                    modality_summary[
                        (modality_summary["modality"] == modality)
                        & (modality_summary["representation"] == "hybrid")
                    ]["mean_family_spearman"].iloc[0]
                )
            )
            baseline_vals.append(
                float(
                    modality_summary[
                        (modality_summary["modality"] == modality)
                        & (
                            modality_summary["representation"]
                            == "all_gaussian_baseline"
                        )
                    ]["mean_family_spearman"].iloc[0]
                )
            )

        ax.bar(x - width / 2, hybrid_vals, width=width, label="hybrid")
        ax.bar(x + width / 2, baseline_vals, width=width, label="all_gaussian_baseline")
        ax.set_xticks(x, modalities)
        ax.set_ylim(-1.0, 1.05)
        ax.set_ylabel("mean family Spearman")
        ax.set_title("Modality-level hybrid robustness")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "hybrid_modality_spearman.png", dpi=170)
        plt.close(fig)

    # Clean false degradation at threshold 0.80, descriptive only.
    g = clean_fd[
        (clean_fd["level"] == "overall")
        & (clean_fd["representation"] == "hybrid")
        & (clean_fd["threshold"] == 0.80)
    ].copy()

    if not g.empty:
        labels = [
            f"{r.modality}\n{r.mapping}"
            for r in g.itertuples()
        ]
        values = g["false_degradation_rate_q_below_threshold"].to_numpy(dtype=float)

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        x = np.arange(len(values))
        ax.bar(x, values)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("clean fraction q < 0.80")
        ax.set_title("Descriptive clean false-degradation rate (no threshold selected)")
        fig.tight_layout()
        fig.savefig(plots / "hybrid_clean_false_degradation_q080.png", dpi=170)
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build subject-LOSO hybrid Gaussian/empirical distribution-aware "
            "quality scores on formal VAL only."
        )
    )
    p.add_argument("--deployment-root")
    p.add_argument("--run-dir")
    p.add_argument("--score-artifact")
    p.add_argument("--output-dir")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def self_test() -> int:
    # 1) Gaussian directional evidence.
    gblock = {
        "transform": "gaussian",
        "direction": "lower_is_worse",
        "mu": 10.0,
        "sigma_sample": 2.0,
    }
    g_clean, _ = gaussian_anomaly(10.0, gblock)
    g_bad, _ = gaussian_anomaly(6.0, gblock)

    # 2) Empirical one-sided evidence.
    eblock = {
        "transform": "empirical",
        "direction": "higher_is_worse",
        "clean_sorted_values": np.sort(
            np.asarray([0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1])
        ).tolist(),
    }
    e_clean, _ = empirical_anomaly(0.0, eblock)
    e_bad, _ = empirical_anomaly(0.8, eblock)

    # 3) Empirical two-sided evidence.
    tblock = {
        "transform": "empirical",
        "direction": "two_sided",
        "clean_sorted_values": np.sort(
            np.asarray([-1.0, -0.5, -0.2, 0.0, 0.1, 0.3, 0.5, 1.0])
        ).tolist(),
    }
    e_mid, _ = empirical_anomaly(0.0, tblock)
    e_far, _ = empirical_anomaly(4.0, tblock)

    # 4) Condition-qualified lookup regression.
    mini = pd.DataFrame([
        {
            "source_window_id": "w01",
            "subject": "subject08",
            "pair_key": "subject08_trial",
            "condition": "reference",
            "family": "",
            "severity": "",
            "modality": "audio",
            "available": True,
            "feature_path": "dnsmos.OVRL_raw",
            "value": 3.4,
        },
        {
            "source_window_id": "w01",
            "subject": "subject08",
            "pair_key": "subject08_trial",
            "condition": "audio_white_noise_severe",
            "family": "audio_white_noise",
            "severity": "severe",
            "modality": "audio",
            "available": True,
            "feature_path": "dnsmos.OVRL_raw",
            "value": 1.1,
        },
    ])
    mini = normalize_strings(
        mini,
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
    mini["available_bool"] = True
    mini["value_num"] = pd.to_numeric(mini["value"])
    lookup = make_feature_lookup(mini)

    # 5) RMS + mappings.
    rms_value = rms_aggregate([0.0, 1.0, 2.0])
    q0k = reliability_mapping(0.0, "gaussian_kernel")
    q2k = reliability_mapping(2.0, "gaussian_kernel")
    q0h = reliability_mapping(0.0, "halfnormal_survival")
    q2h = reliability_mapping(2.0, "halfnormal_survival")

    # 6) Reference fit excludes held subject - simple synthetic check.
    synthetic = []
    for sidx in range(6):
        subject = f"subject{sidx:02d}"
        for widx in range(20):
            synthetic.append({
                "source_window_id": f"{subject}_w{widx:02d}",
                "subject": subject,
                "pair_key": f"{subject}_trial{widx//4:02d}",
                "condition": "reference",
                "family": "",
                "severity": "",
                "modality": "audio",
                "available": True,
                "feature_path": "signal_metrics.rms_dbfs",
                "value": -40.0 + sidx + 0.01 * widx,
            })
    syn = pd.DataFrame(synthetic)
    syn = normalize_strings(
        syn,
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
    syn["available_bool"] = True
    syn["value_num"] = pd.to_numeric(syn["value"])

    train_subjects = [f"subject{sidx:02d}" for sidx in range(1, 6)]
    train_values = clean_window_values(
        syn,
        "audio",
        "signal_metrics.rms_dbfs",
        train_subjects,
    )

    checks = {
        "gaussian_bad_exceeds_clean": g_bad > g_clean,
        "empirical_bad_exceeds_clean": e_bad > e_clean,
        "two_sided_far_exceeds_center": e_far > e_mid,
        "condition_specific_lookup": (
            len(lookup) == 2
            and math.isclose(
                lookup[("w01", "reference", "audio", "dnsmos.OVRL_raw")],
                3.4,
            )
            and math.isclose(
                lookup[
                    (
                        "w01",
                        "audio_white_noise_severe",
                        "audio",
                        "dnsmos.OVRL_raw",
                    )
                ],
                1.1,
            )
        ),
        "rms_positive": rms_value > 0,
        "kernel_monotone": q0k > q2k,
        "halfnormal_monotone": q0h > q2h,
        "zero_anomaly_maps_to_one": (
            math.isclose(q0k, 1.0)
            and math.isclose(q0h, 1.0)
        ),
        "loso_reference_has_100_windows": len(train_values) == 100,
        "held_subject_excluded": "subject00" not in set(train_values["subject"]),
        "strict_json_nonfinite_sanitized": (
            '"x": null' in json_text({"x": float("nan")})
            and '"y": null' in json_text({"y": np.float64(np.inf)})
        ),
        "aggregation_candidate_is_rms": AGGREGATION_CANDIDATE == "rms",
    }

    require(all(checks.values()), f"Self-test failed: {checks}")

    print(json_text({
        "status": "PASS",
        "version": VERSION,
        "checks": checks,
        "real_EAV_used": False,
        "models_loaded": False,
        "test_data_used": False,
        "router_changed": False,
        "deployment_selection_made": False,
    }))

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
        except HybridQualityError:
            deployment_root = discover_deployment_root(Path(__file__).resolve().parent)

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else discover_latest_formal_run(deployment_root)
    )
    require(run_dir.is_dir(), f"Run directory missing: {run_dir}")

    run_summary = validate_run(run_dir)

    score_artifact_path = (
        Path(args.score_artifact).expanduser().resolve()
        if args.score_artifact
        else discover_latest_score_artifact(run_dir)
    )
    score_artifact = validate_score_artifact(
        score_artifact_path,
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
        else run_dir / f"hybrid_quality_score_{timestamp}"
    )

    require(
        not out_dir.exists(),
        f"Output directory exists; refusing overwrite: {out_dir}",
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    features = prepare_features(feature_path)
    samples = prepare_samples(sample_path)

    continuous_spec = continuous_spec_from_artifact(score_artifact)
    guard_spec = guard_spec_from_artifact(score_artifact)

    policy_df = validate_hybrid_policy(continuous_spec)
    atomic_csv(out_dir / "hybrid_feature_policy.csv", policy_df)

    subjects = sorted(
        samples.loc[
            samples["condition"] == "reference",
            "subject",
        ].unique().tolist()
    )

    require(
        len(subjects) == EXPECTED_SUBJECTS,
        f"Expected 6 VAL subjects, got {subjects}",
    )

    # Formal geometry checks.
    for modality in sorted(continuous_spec):
        g = samples[
            (samples["condition"] == "reference")
            & (samples["modality"] == modality)
        ]
        require(
            g["source_window_id"].nunique()
            == EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
            f"{modality}: expected 120 clean source windows.",
        )
        require(
            g["pair_key"].nunique() == EXPECTED_CLEAN_TRIALS,
            f"{modality}: expected 30 clean trials.",
        )

    feature_lookup = make_feature_lookup(features)
    base_samples = build_base_samples(samples)

    all_evidence_frames: List[pd.DataFrame] = []
    all_score_frames: List[pd.DataFrame] = []
    all_reference_summary_frames: List[pd.DataFrame] = []

    for held_subject in subjects:
        train_subjects = [s for s in subjects if s != held_subject]

        require(
            len(train_subjects) == EXPECTED_SUBJECTS - 1,
            "LOSO training subject count is not five.",
        )

        hybrid_ref = fit_fold_reference(
            features,
            train_subjects=train_subjects,
            continuous_spec=continuous_spec,
            guard_spec=guard_spec,
            representation="hybrid",
        )

        baseline_ref = fit_fold_reference(
            features,
            train_subjects=train_subjects,
            continuous_spec=continuous_spec,
            guard_spec=guard_spec,
            representation="all_gaussian_baseline",
        )

        all_reference_summary_frames.append(
            reference_parameter_summary(
                held_subject,
                train_subjects,
                "hybrid",
                hybrid_ref,
            )
        )
        all_reference_summary_frames.append(
            reference_parameter_summary(
                held_subject,
                train_subjects,
                "all_gaussian_baseline",
                baseline_ref,
            )
        )

        evidence_df, score_df = score_fold_subject(
            held_subject=held_subject,
            base_samples=base_samples,
            feature_lookup=feature_lookup,
            hybrid_ref=hybrid_ref,
            baseline_ref=baseline_ref,
        )

        all_evidence_frames.append(evidence_df)
        all_score_frames.append(score_df)

    evidence_all = pd.concat(
        all_evidence_frames,
        ignore_index=True,
    )
    window_scores = pd.concat(
        all_score_frames,
        ignore_index=True,
    )
    reference_summary = pd.concat(
        all_reference_summary_frames,
        ignore_index=True,
    )

    # Primary family conditions must have complete quality evidence.
    primary_conditions = set()
    for spec in FAMILY_SPECS.values():
        primary_conditions.add(str(spec["reference_condition"]))
        primary_conditions.update(str(x) for x in spec["conditions"])

    incomplete_primary = window_scores[
        (window_scores["condition"].isin(primary_conditions))
        & (window_scores["available"])
        & (window_scores["status"] != "OK")
    ]

    require(
        len(incomplete_primary) == 0,
        "Primary family evaluation contains available windows with incomplete evidence.",
    )

    trial_scores = build_trial_scores(window_scores)
    family_eval = evaluate_family_trajectories(trial_scores)
    modality_summary = summarize_modalities(family_eval)

    modality_compare, family_compare = compare_hybrid_to_baseline(
        modality_summary,
        family_eval,
    )

    loso_subject = subject_loso_stability(trial_scores)
    clean_fd = clean_false_degradation(window_scores)
    mapping_df = mapping_response(window_scores)
    existing_compare = existing_q_comparison(window_scores)

    # Full six-subject reference artifact for the NEXT stage only.
    # It remains explicitly non-deployable until a later freeze decision.
    full_reference = fit_full_reference(
        features,
        all_subjects=subjects,
        continuous_spec=continuous_spec,
        guard_spec=guard_spec,
    )

    atomic_csv(
        out_dir / "loso_reference_parameter_summary.csv",
        reference_summary,
    )
    atomic_csv(
        out_dir / "hybrid_window_evidence_loso.csv",
        evidence_all,
    )
    atomic_csv(
        out_dir / "hybrid_window_scores_loso.csv",
        window_scores,
    )
    atomic_csv(
        out_dir / "hybrid_trial_scores_loso.csv",
        trial_scores,
    )
    atomic_csv(
        out_dir / "hybrid_family_validation.csv",
        family_eval,
    )
    atomic_csv(
        out_dir / "hybrid_modality_summary.csv",
        modality_summary,
    )
    atomic_csv(
        out_dir / "hybrid_vs_allgaussian_modality.csv",
        modality_compare,
    )
    atomic_csv(
        out_dir / "hybrid_vs_allgaussian_family.csv",
        family_compare,
    )
    atomic_csv(
        out_dir / "hybrid_subject_loso_stability.csv",
        loso_subject,
    )
    atomic_csv(
        out_dir / "hybrid_clean_false_degradation.csv",
        clean_fd,
    )
    atomic_csv(
        out_dir / "hybrid_mapping_threshold_response.csv",
        mapping_df,
    )
    atomic_csv(
        out_dir / "hybrid_vs_existing_q.csv",
        existing_compare,
    )

    if not args.no_plots:
        create_plots(
            out_dir,
            family_eval=family_eval,
            modality_summary=modality_summary,
            clean_fd=clean_fd,
        )

    candidate_artifact = {
        "schema": "eav.hybrid_distribution_quality_score_candidate.v1",
        "version": VERSION,
        "artifact_state": "CANDIDATE_NOT_DEPLOYABLE",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_score_artifact": str(score_artifact_path),
        "source_split": "VAL_ONLY",
        "test_data_used": False,
        "runtime_unit": "5s_window",
        "validation_protocol": {
            "subjects": subjects,
            "subject_loso_reference_fit": True,
            "train_subjects_per_fold": 5,
            "held_subjects_per_fold": 1,
            "clean_windows_per_train_fold_feature": (
                EXPECTED_TRAIN_WINDOWS_PER_FOLD_FEATURE
            ),
            "primary_robustness_unit": "20s_trial_trajectory",
            "windows_per_trial": 4,
            "corruption_family_used_as_score_input": False,
        },
        "hybrid_policy": {
            "continuous": HYBRID_CONTINUOUS_POLICY,
            "guards": {
                modality: {
                    feature: "empirical"
                    for feature in features_for_modality
                }
                for modality, features_for_modality in guard_spec.items()
            },
            "gaussian_meaning": (
                "directional z relative to clean training-subject reference"
            ),
            "empirical_meaning": (
                "clean-tail probability converted to z-equivalent anomaly magnitude; "
                "does not assume raw feature is Gaussian"
            ),
        },
        "aggregation_candidate": {
            "name": "RMS",
            "formula": "sqrt(mean(evidence^2))",
            "carried_forward_from_previous_VAL_aggregation_analysis": True,
            "production_frozen": False,
        },
        "reliability_mappings_evaluated": {
            "gaussian_kernel": "exp(-0.5 * B^2)",
            "halfnormal_survival": "2 * Phi(-B)",
            "probability_of_emotion_correctness": False,
        },
        "candidate_thresholds_evaluated_descriptively": list(
            CANDIDATE_THRESHOLDS
        ),
        "full_6_subject_VAL_reference_candidate": full_reference,
        "comparison_baseline": {
            "name": "all_gaussian_continuous_plus_empirical_guards",
            "aggregation": "RMS",
            "same_LOSO_folds": True,
            "deployment_baseline": False,
        },
        "availability": {
            "separate_hard_state": True,
            "unavailable_candidate_quality": 0.0,
            "distribution_score_never_overrides_unavailable": True,
        },
        "selection": {
            "hybrid_score_production_frozen": False,
            "reliability_mapping_selected": False,
            "quality_threshold_selected": False,
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
            "Held subject is excluded from its own reference distribution fit.",
            "Window rows are clustered within trials/subjects and are not treated as independent subjects.",
            "Known corruption family is never used to calculate the score.",
            "Availability remains a separate hard state.",
            "Classifier confidence is not used as signal quality.",
            "Hybrid q mappings are engineering scores, not probabilities of emotion-classification correctness.",
            "No threshold, mapping, router, or runtime module is selected/deployed here.",
        ],
        "source_hashes": {
            "run_summary": sha256_file(run_dir / "run_summary.json"),
            "quality_numeric_features": sha256_file(feature_path),
            "quality_distribution_samples": sha256_file(sample_path),
            "source_score_artifact": sha256_file(score_artifact_path),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }

    atomic_json(
        out_dir / "hybrid_distribution_quality_score_candidate_artifact.json",
        candidate_artifact,
    )

    # Compact summary for rapid review.
    compact_modalities: Dict[str, Any] = {}

    for modality in sorted(modality_summary["modality"].unique()):
        h = modality_summary[
            (modality_summary["modality"] == modality)
            & (modality_summary["representation"] == "hybrid")
        ].iloc[0]

        b = modality_summary[
            (modality_summary["modality"] == modality)
            & (modality_summary["representation"] == "all_gaussian_baseline")
        ].iloc[0]

        clean_q080 = clean_fd[
            (clean_fd["level"] == "overall")
            & (clean_fd["modality"] == modality)
            & (clean_fd["representation"] == "hybrid")
            & (clean_fd["mapping"] == "gaussian_kernel")
            & (clean_fd["threshold"] == 0.80)
        ]

        compact_modalities[modality] = {
            "hybrid_mean_family_spearman": float(h["mean_family_spearman"]),
            "hybrid_worst_family_spearman": float(h["worst_family_spearman"]),
            "hybrid_mean_monotonic_rate": float(h["mean_monotonic_rate"]),
            "hybrid_mean_reference_vs_severe_auc": float(
                h["mean_reference_vs_severe_auc"]
            ),
            "all_gaussian_baseline_mean_family_spearman": float(
                b["mean_family_spearman"]
            ),
            "delta_mean_family_spearman": float(
                h["mean_family_spearman"] - b["mean_family_spearman"]
            ),
            "clean_false_degradation_q_kernel_at_0_80_descriptive": (
                float(
                    clean_q080[
                        "false_degradation_rate_q_below_threshold"
                    ].iloc[0]
                )
                if len(clean_q080)
                else None
            ),
        }

    summary = {
        "version": VERSION,
        "status": "PASS",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_score_artifact": str(score_artifact_path),
        "formal_balanced_val": True,
        "test_data_used": False,
        "subjects": subjects,
        "subject_loso_reference_fit": True,
        "runtime_unit": "5s_window",
        "primary_robustness_unit": "20s_trial_trajectory",
        "aggregation_candidate": "RMS",
        "modality_results": compact_modalities,
        "deployment_selection_made": False,
        "router_threshold_retuned": False,
        "models_or_existing_calibrators_modified": False,
        "candidate_artifact": str(
            out_dir / "hybrid_distribution_quality_score_candidate_artifact.json"
        ),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
        "packages": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "matplotlib": package_version("matplotlib"),
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }

    atomic_json(
        out_dir / "hybrid_quality_score_summary.json",
        summary,
    )

    # Human-readable README.
    lines = [
        "# Hybrid Distribution-Aware Quality Score",
        "",
        f"- Version: `{VERSION}`",
        f"- Source run: `{run_dir}`",
        f"- Source score artifact: `{score_artifact_path}`",
        "- Scope: **formal balanced VAL only**",
        "- Subject-wise LOSO reference fitting: **Yes**",
        "- Runtime unit: **5-s window**",
        "- Primary robustness unit: **20-s trial trajectory**",
        "- Aggregation candidate: **RMS**",
        "- Production score frozen: **No**",
        "- Router changed: **No**",
        "",
        "## Hybrid feature policy",
        "",
    ]

    for row in policy_df.itertuples():
        lines.append(
            f"- {row.modality} / `{row.feature_path}` -> "
            f"**{row.hybrid_transform}** "
            f"(direction: {row.direction})"
        )

    lines += [
        "",
        "Discrete/bounded guards remain empirical-tail evidence.",
        "",
        "## Hybrid modality results",
        "",
    ]

    for modality, values in compact_modalities.items():
        lines.append(
            f"- **{modality}**: mean rho="
            f"{values['hybrid_mean_family_spearman']:.3f}, "
            f"worst rho={values['hybrid_worst_family_spearman']:.3f}, "
            f"monotonic={values['hybrid_mean_monotonic_rate']:.3f}, "
            f"AUC={values['hybrid_mean_reference_vs_severe_auc']:.3f}, "
            f"delta rho vs all-Gaussian="
            f"{values['delta_mean_family_spearman']:+.3f}"
        )

    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- No TEST data are used.",
        "- No significance claim is made.",
        "- A failure to reject Gaussianity is not proof of a Gaussian population.",
        "- Empirical-tail z-equivalent evidence does not make the raw variable Gaussian.",
        "- Candidate q is not probability of emotion-classification correctness.",
        "- Thresholds 0.20/0.40/0.60/0.80 are descriptive only.",
        "",
        "## Next gate",
        "",
        "Review:",
        "- `hybrid_modality_summary.csv`",
        "- `hybrid_vs_allgaussian_modality.csv`",
        "- `hybrid_family_validation.csv`",
        "- `hybrid_clean_false_degradation.csv`",
        "- `hybrid_subject_loso_stability.csv`",
        "",
        "Only if the hybrid candidate is robust and clean false-degradation is "
        "acceptable should a separate runtime implementation be created under "
        "`最终部署/Quality/`, followed by system-level A/B testing against the "
        "existing tau=0.80 router.",
        "",
    ]

    atomic_text(
        out_dir / "README_hybrid_quality_score.md",
        "\n".join(lines),
    )

    print("=" * 120)
    print("EAV HYBRID DISTRIBUTION-AWARE QUALITY SCORE")
    print("=" * 120)
    print(f"Status                      : PASS")
    print(f"Version                     : {VERSION}")
    print(f"Source formal run           : {run_dir}")
    print(f"Source score artifact       : {score_artifact_path}")
    print(f"Output                      : {out_dir}")
    print(f"VAL subjects                : {len(subjects)}")
    print(f"Subject-wise LOSO           : True")
    print(f"Runtime unit                : 5s window")
    print(f"Primary robustness unit     : 20s trial trajectory")
    print(f"Aggregation candidate       : RMS")
    print(f"TEST used                   : False")
    print(f"Existing tau changed        : False")
    print(f"Deployment selection made   : False")
    print()

    for modality in sorted(compact_modalities):
        v = compact_modalities[modality]
        print(
            f"{modality.upper():5s} hybrid                  : "
            f"mean rho={v['hybrid_mean_family_spearman']:.3f} | "
            f"worst rho={v['hybrid_worst_family_spearman']:.3f} | "
            f"monotonic={v['hybrid_mean_monotonic_rate']:.3f} | "
            f"AUC={v['hybrid_mean_reference_vs_severe_auc']:.3f} | "
            f"delta rho={v['delta_mean_family_spearman']:+.3f}"
        )

    print()
    print("Core outputs:")
    print(f"  {out_dir / 'hybrid_feature_policy.csv'}")
    print(f"  {out_dir / 'hybrid_family_validation.csv'}")
    print(f"  {out_dir / 'hybrid_modality_summary.csv'}")
    print(f"  {out_dir / 'hybrid_vs_allgaussian_modality.csv'}")
    print(f"  {out_dir / 'hybrid_clean_false_degradation.csv'}")
    print(f"  {out_dir / 'hybrid_subject_loso_stability.csv'}")
    print(f"  {out_dir / 'hybrid_quality_score_summary.json'}")
    print(
        f"  {out_dir / 'hybrid_distribution_quality_score_candidate_artifact.json'}"
    )
    print("=" * 120)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nINTERRUPTED.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nHYBRID QUALITY SCORE ERROR: {exc}", file=sys.stderr)
        raise
