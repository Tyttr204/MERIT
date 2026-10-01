#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAV EEG Quality Hybrid-v2 Analysis v1.0
=======================================

Purpose
-------
Resolve the final EEG quality-policy issue found by the preceding nested
threshold-calibration experiment:

- The current empirical EEG hybrid policy detects channel-flatline corruption
  very strongly, because several EEG anomaly features/guards move together.
- The same policy is much weaker on global line interference, where
  `features.line_fraction_mean` is the dominant physical indicator.
- A pure empirical tail transform saturates once `line_fraction_mean` exceeds
  the clean reference support, so mild/medium/severe line interference can
  collapse onto almost the same anomaly magnitude.
- RMS aggregation can then dilute that single saturated line anomaly among the
  other mostly-normal EEG evidences.

This script changes ONLY the transform used for `line_fraction_mean` and keeps
the rest of the EEG Hybrid-v1 policy fixed.

Policies compared
-----------------
1) empirical_line
   Current Hybrid-v1 baseline:
       line_fraction_mean -> empirical clean-tail anomaly

2) gaussian_line
       line_fraction_mean -> directional mean/std z anomaly

3) robust_iqr_line
       line_fraction_mean -> directional robust z:
           center = clean median
           scale  = IQR / 1.349
       with documented fallback only if IQR degenerates

4) robust_mad_line
       line_fraction_mean -> directional robust z:
           center = clean median
           scale  = 1.4826 * MAD
       with documented fallback only if MAD degenerates

All other EEG evidence remains unchanged:
  continuous:
    features.hf55_90_fraction_mean  -> empirical
    features.rms_uv_median          -> empirical, two-sided
    features.slow02_1_fraction_mean -> empirical

  guards:
    features.pyprep_bad_fraction       -> empirical
    features.raw_flat_channel_fraction -> empirical
    features.raw_hold_fraction_mean     -> empirical

Modality aggregation remains:
    B_EEG = sqrt(mean(evidence_k^2))

No corruption family is used as a score input.

Validation protocol
-------------------
Nested subject-wise LOSO on the same six formal VAL subjects.

Outer fold:
    hold one subject out for evaluation.

Inner threshold calibration:
    among the five outer-training subjects, hold each one out in turn,
    fit the EEG reference on the other four,
    score the inner-held clean reference windows,
    pool 100 clean out-of-fold anomaly scores,
    calibrate thresholds for alpha in {0.01, 0.05, 0.10}.

Outer evaluation:
    fit each policy's reference on all five outer-training subjects,
    score the outer-held subject on:
        reference
        eeg_global_line_mild/medium/severe
        eeg_channel_flatline_mild/medium/severe

Primary development operating point:
    alpha = 0.05
but no alpha or policy is frozen by this script.

Threshold rule
--------------
Using the same conservative engineering order-statistic rule as the preceding
threshold-calibration stage:

    k = ceil((n + 1) * (1 - alpha))
    T = k-th smallest inner-OOF clean anomaly

Router candidate rule:
    unavailable OR B_EEG > T -> Degraded

The outer subject is excluded from both:
    - feature-reference fitting
    - threshold calibration

Important interpretation
------------------------
Robust-z does NOT assert that the raw line-fraction population is Gaussian.
It only provides a non-saturating standardized distance from the clean centre.

Empirical-tail -> z-equivalent likewise does NOT make the raw feature Gaussian;
the normal quantile is only a common anomaly scale.

The 5-s windows are clustered within trials and subjects.  Therefore the
order-statistic threshold is an engineering calibration rule, not a claim of
exact independent-sample conformal coverage.

This script DOES NOT
--------------------
- use TEST subjects;
- retrain emotion classifiers;
- retrain fusion;
- modify existing quality calibrators;
- modify the deployed tau=0.80 router;
- change Audio or Video quality policy;
- change RMS aggregation;
- choose/freeze a production EEG policy;
- choose/freeze a production threshold;
- use emotion-classification confidence as signal quality.

Recommended location
--------------------
最终部署/testing/analysis/build_eeg_quality_hybrid_v2.py

Recommended command from 最终部署
---------------------------------
python -X utf8 .\\testing\\analysis\\build_eeg_quality_hybrid_v2.py

Automatic discovery
-------------------
If paths are omitted, the script selects:
  - latest completed formal balanced VAL robustness run;
  - latest valid hybrid_quality_score_* artifact in that run.
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


VERSION = "EAV-EEG-QUALITY-HYBRID-V2.1.0"

EEG_MODALITY = "eeg"
LINE_FEATURE = "features.line_fraction_mean"

EXPECTED_SUBJECTS = 6
EXPECTED_TRIALS_PER_SUBJECT = 5
EXPECTED_WINDOWS_PER_TRIAL = 4
EXPECTED_CLEAN_WINDOWS_PER_SUBJECT = (
    EXPECTED_TRIALS_PER_SUBJECT * EXPECTED_WINDOWS_PER_TRIAL
)  # 20
EXPECTED_INNER_CALIBRATION_WINDOWS = (
    (EXPECTED_SUBJECTS - 1) * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 100
EXPECTED_INNER_REFERENCE_WINDOWS = (
    (EXPECTED_SUBJECTS - 2) * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 80
EXPECTED_OUTER_REFERENCE_WINDOWS = (
    (EXPECTED_SUBJECTS - 1) * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 100
EXPECTED_FULL_CROSSFIT_CLEAN_WINDOWS = (
    EXPECTED_SUBJECTS * EXPECTED_CLEAN_WINDOWS_PER_SUBJECT
)  # 120

TARGET_ALPHAS = (0.01, 0.05, 0.10)
PRIMARY_ALPHA = 0.05
EVIDENCE_CAP = 12.0
AGGREGATION = "RMS"

LINE_POLICIES: Dict[str, str] = {
    "empirical_line": "empirical",
    "gaussian_line": "gaussian",
    "robust_iqr_line": "robust_iqr",
    "robust_mad_line": "robust_mad",
}

EEG_EVAL_CONDITIONS = (
    "reference",
    "eeg_global_line_mild",
    "eeg_global_line_medium",
    "eeg_global_line_severe",
    "eeg_channel_flatline_mild",
    "eeg_channel_flatline_medium",
    "eeg_channel_flatline_severe",
)

FAMILY_SPECS: Dict[str, Dict[str, Any]] = {
    "eeg_global_line": {
        "reference_condition": "reference",
        "conditions": {
            "mild": "eeg_global_line_mild",
            "medium": "eeg_global_line_medium",
            "severe": "eeg_global_line_severe",
        },
    },
    "eeg_channel_flatline": {
        "reference_condition": "reference",
        "conditions": {
            "mild": "eeg_channel_flatline_mild",
            "medium": "eeg_channel_flatline_medium",
            "severe": "eeg_channel_flatline_severe",
        },
    },
}


class EEGHybridV2Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise EEGHybridV2Error(message)


def json_safe(obj: Any) -> Any:
    """Recursively convert values to strict JSON-safe representations."""
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

    raise EEGHybridV2Error(f"{name}: invalid boolean {value!r}")


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
        raise EEGHybridV2Error("scipy is required.") from exc
    return stats


def discover_deployment_root(start: Path) -> Path:
    start = start.resolve()

    for p in [start, *start.parents]:
        if (p / "main.py").is_file() and (p / "system_checks").is_dir():
            return p

    raise EEGHybridV2Error(
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

    require(candidates, "No valid hybrid quality candidate artifact found.")

    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[-1][1]


def validate_run(run_dir: Path) -> dict:
    s = read_json(run_dir / "run_summary.json")

    require(s.get("status") == "PASS_OFFLINE_PROCESSING", "Source run is not PASS.")
    require(s.get("formal_balanced_val") is True, "Source is not formal balanced VAL.")
    require(s.get("suite") == "robustness", "Source is not robustness suite.")
    require(s.get("test_data_used") is False, "TEST data used; refusing EEG-v2 analysis.")
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
        "Hybrid source should still be non-deployable.",
    )
    require(a.get("source_split") == "VAL_ONLY", "Hybrid source is not VAL-only.")
    require(a.get("test_data_used") is False, "Hybrid source used TEST.")
    require(a.get("runtime_unit") == "5s_window", "Expected 5-s source policy.")

    source_dir = Path(str(a.get("source_run_dir", ""))).resolve()
    require(
        source_dir == run_dir.resolve(),
        f"Hybrid/source run mismatch: {source_dir} != {run_dir.resolve()}",
    )

    aggregation = a.get("aggregation_candidate", {})
    require(
        str(aggregation.get("name", "")).upper() == "RMS",
        "EEG Hybrid-v2 expects RMS source aggregation.",
    )

    selection = a.get("selection", {})
    require(
        selection.get("hybrid_score_production_frozen") is False,
        "Source hybrid score already frozen unexpectedly.",
    )
    require(
        selection.get("quality_threshold_selected") is False,
        "Source hybrid artifact already selected threshold.",
    )
    require(
        selection.get("router_selected") is False,
        "Source hybrid artifact already selected router.",
    )

    existing = a.get("existing_deployment", {})
    require(existing.get("tau_0_80_changed") is False, "Source changed old tau.")
    require(
        existing.get("quality_calibrators_changed") is False,
        "Source changed existing calibrators.",
    )
    require(existing.get("fusion_weights_changed") is False, "Source changed fusion.")
    require(existing.get("emotion_models_changed") is False, "Source changed emotion models.")

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


def eeg_feature_spec_from_hybrid_artifact(
    artifact: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    full_ref = artifact.get("full_6_subject_VAL_reference_candidate")
    require(isinstance(full_ref, dict), "Hybrid artifact missing full reference candidate.")

    eeg_ref = full_ref.get("eeg")
    require(isinstance(eeg_ref, dict), "Hybrid artifact missing EEG reference block.")

    policy = artifact.get("hybrid_policy", {})
    continuous_policy = policy.get("continuous", {}).get("eeg", {})
    guard_policy = policy.get("guards", {}).get("eeg", {})

    require(isinstance(continuous_policy, dict), "EEG continuous policy invalid.")
    require(isinstance(guard_policy, dict), "EEG guard policy invalid.")

    spec: Dict[str, Dict[str, str]] = {}

    for feature, block in eeg_ref.items():
        require(isinstance(block, dict), f"EEG/{feature}: invalid source block.")

        feature_type = str(block.get("feature_type", ""))
        direction = str(block.get("direction", ""))

        require(
            feature_type in {"continuous", "guard"},
            f"EEG/{feature}: invalid feature type.",
        )
        require(
            direction in {"lower_is_worse", "higher_is_worse", "two_sided"},
            f"EEG/{feature}: invalid direction.",
        )

        if feature_type == "continuous":
            require(
                feature in continuous_policy,
                f"EEG/{feature}: absent from source continuous policy.",
            )
            base_transform = str(continuous_policy[feature])
        else:
            require(
                feature in guard_policy,
                f"EEG/{feature}: absent from source guard policy.",
            )
            base_transform = str(guard_policy[feature])

        require(
            base_transform in {"gaussian", "empirical"},
            f"EEG/{feature}: unexpected source transform {base_transform!r}",
        )

        spec[str(feature)] = {
            "feature_type": feature_type,
            "direction": direction,
            "source_transform": base_transform,
        }

    expected_features = {
        "features.hf55_90_fraction_mean",
        "features.line_fraction_mean",
        "features.rms_uv_median",
        "features.slow02_1_fraction_mean",
        "features.pyprep_bad_fraction",
        "features.raw_flat_channel_fraction",
        "features.raw_hold_fraction_mean",
    }

    require(
        set(spec) == expected_features,
        f"EEG feature drift: expected={sorted(expected_features)}, "
        f"actual={sorted(spec)}",
    )

    require(
        spec[LINE_FEATURE]["direction"] == "higher_is_worse",
        "line_fraction_mean direction drifted from higher_is_worse.",
    )
    require(
        spec[LINE_FEATURE]["source_transform"] == "empirical",
        "Current EEG Hybrid-v1 line baseline is expected to be empirical.",
    )

    # Everything except line stays fixed as current Hybrid-v1.
    for feature, meta in spec.items():
        if feature == LINE_FEATURE:
            continue
        require(
            meta["source_transform"] == "empirical",
            f"EEG Hybrid-v2 expects non-line feature {feature} to remain empirical.",
        )

    return spec


def make_feature_lookup(
    features: pd.DataFrame,
) -> Dict[Tuple[str, str, str, str], float]:
    eeg = features[
        (features["modality"] == EEG_MODALITY)
        & (features["available_bool"])
    ].copy()

    grouped = (
        eeg.groupby(
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
        "Duplicate fully-qualified EEG feature lookup keys remain.",
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


def build_eeg_base_samples(samples: pd.DataFrame) -> pd.DataFrame:
    eeg = samples[
        (samples["modality"] == EEG_MODALITY)
        & (samples["condition"].isin(EEG_EVAL_CONDITIONS))
    ].copy()

    keys = [
        "source_window_id",
        "subject",
        "pair_key",
        "condition",
        "modality",
    ]

    base = (
        eeg.groupby(keys, as_index=False, dropna=False)
        .agg(
            available=("available_bool", "min"),
            existing_q=("q_num", "mean"),
        )
    )

    return base


def clean_window_values(
    features: pd.DataFrame,
    feature: str,
    subjects: Sequence[str],
) -> pd.DataFrame:
    subject_set = set(str(s) for s in subjects)

    g = features[
        (features["condition"] == "reference")
        & (features["modality"] == EEG_MODALITY)
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


def robust_scale_iqr(x: np.ndarray) -> Tuple[float, str]:
    x = finite(x)
    require(len(x) >= 8, "Too few values for robust IQR scale.")

    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    scale = (q75 - q25) / 1.349

    if math.isfinite(scale) and scale > 1e-12:
        return float(scale), "iqr_over_1.349"

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad

    if math.isfinite(scale) and scale > 1e-12:
        return float(scale), "fallback_mad_times_1.4826"

    scale = float(np.std(x, ddof=1))
    require(
        math.isfinite(scale) and scale > 1e-12,
        "IQR/MAD/std are all degenerate.",
    )

    return scale, "fallback_sample_std"


def robust_scale_mad(x: np.ndarray) -> Tuple[float, str]:
    x = finite(x)
    require(len(x) >= 8, "Too few values for robust MAD scale.")

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad

    if math.isfinite(scale) and scale > 1e-12:
        return float(scale), "mad_times_1.4826"

    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    scale = (q75 - q25) / 1.349

    if math.isfinite(scale) and scale > 1e-12:
        return float(scale), "fallback_iqr_over_1.349"

    scale = float(np.std(x, ddof=1))
    require(
        math.isfinite(scale) and scale > 1e-12,
        "MAD/IQR/std are all degenerate.",
    )

    return scale, "fallback_sample_std"


def fit_reference_block(
    values: np.ndarray,
    transform: str,
    direction: str,
) -> Dict[str, Any]:
    x = finite(values)
    require(len(x) >= 8, "Too few values for reference fit.")

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
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
    }

    if transform == "empirical":
        block["clean_sorted_values"] = np.sort(x).tolist()
        block["unique_values"] = int(len(np.unique(x)))

    elif transform == "gaussian":
        sigma = float(np.std(x, ddof=1))
        require(
            math.isfinite(sigma) and sigma > 1e-12,
            "Gaussian sigma invalid.",
        )
        block["center"] = float(np.mean(x))
        block["scale"] = sigma
        block["scale_definition"] = "sample_std"

    elif transform == "robust_iqr":
        scale, definition = robust_scale_iqr(x)
        block["center"] = float(np.median(x))
        block["scale"] = scale
        block["scale_definition"] = definition

    elif transform == "robust_mad":
        scale, definition = robust_scale_mad(x)
        block["center"] = float(np.median(x))
        block["scale"] = scale
        block["scale_definition"] = definition

    else:
        raise EEGHybridV2Error(f"Unknown transform {transform!r}")

    return block


def transform_for_feature(
    feature: str,
    meta: Mapping[str, str],
    policy: str,
) -> str:
    require(policy in LINE_POLICIES, f"Unknown policy {policy!r}")

    if feature == LINE_FEATURE:
        return LINE_POLICIES[policy]

    # Preserve Hybrid-v1 transform for every other EEG feature.
    return str(meta["source_transform"])


def fit_eeg_reference(
    features: pd.DataFrame,
    train_subjects: Sequence[str],
    feature_spec: Mapping[str, Mapping[str, str]],
    policy: str,
    expected_windows_per_feature: int,
) -> Dict[str, Dict[str, Any]]:
    reference: Dict[str, Dict[str, Any]] = {}

    for feature, meta in feature_spec.items():
        transform = transform_for_feature(
            feature,
            meta,
            policy,
        )
        direction = str(meta["direction"])

        w = clean_window_values(
            features,
            feature=feature,
            subjects=train_subjects,
        )

        require(
            len(w) == expected_windows_per_feature,
            f"{policy}/{feature}: expected {expected_windows_per_feature} "
            f"clean reference windows, got {len(w)}",
        )
        require(
            w["subject"].nunique() == len(train_subjects),
            f"{policy}/{feature}: reference subject count mismatch.",
        )

        block = fit_reference_block(
            w["value"].to_numpy(dtype=float),
            transform=transform,
            direction=direction,
        )
        block["feature_type"] = str(meta["feature_type"])
        block["feature_path"] = feature
        reference[feature] = block

    return reference


def directional_standardized_anomaly(
    value: float,
    center: float,
    scale: float,
    direction: str,
) -> Tuple[float, float]:
    z = (float(value) - float(center)) / float(scale)

    if direction == "higher_is_worse":
        evidence = max(0.0, z)
    elif direction == "lower_is_worse":
        evidence = max(0.0, -z)
    elif direction == "two_sided":
        evidence = abs(z)
    else:
        raise EEGHybridV2Error(f"Unknown standardized direction {direction!r}")

    return float(min(EVIDENCE_CAP, evidence)), float(z)


def empirical_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    stats = import_stats()

    clean = np.asarray(block["clean_sorted_values"], dtype=np.float64)
    require(len(clean) >= 8, "Empirical reference too small.")

    x = float(value)
    n = len(clean)
    direction = str(block["direction"])

    outside_support = False
    empirical_saturated = False

    if direction == "higher_is_worse":
        tail_count = int(np.sum(clean >= x))
        p = (1.0 + tail_count) / (n + 1.0)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        raw = float(stats.norm.isf(p))
        evidence = max(0.0, raw)
        outside_support = bool(x > float(np.max(clean)))
        empirical_saturated = bool(tail_count == 0)

    elif direction == "lower_is_worse":
        tail_count = int(np.sum(clean <= x))
        p = (1.0 + tail_count) / (n + 1.0)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        raw = float(stats.norm.isf(p))
        evidence = max(0.0, raw)
        outside_support = bool(x < float(np.min(clean)))
        empirical_saturated = bool(tail_count == 0)

    elif direction == "two_sided":
        p_lo = (1.0 + int(np.sum(clean <= x))) / (n + 1.0)
        p_hi = (1.0 + int(np.sum(clean >= x))) / (n + 1.0)
        p_two = min(1.0, 2.0 * min(p_lo, p_hi))
        p_two = float(np.clip(p_two, 1e-12, 1.0))
        raw = float(stats.norm.isf(max(p_two / 2.0, 1e-12)))
        evidence = max(0.0, raw)
        p = p_two
        outside_support = bool(
            x < float(np.min(clean))
            or x > float(np.max(clean))
        )
        empirical_saturated = bool(
            x < float(np.min(clean))
            or x > float(np.max(clean))
        )

    else:
        raise EEGHybridV2Error(f"Unknown empirical direction {direction!r}")

    return float(min(EVIDENCE_CAP, evidence)), {
        "standardized_value": None,
        "tail_probability": float(p),
        "outside_reference_support": outside_support,
        "empirical_tail_saturated": empirical_saturated,
    }


def feature_anomaly(
    value: float,
    block: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    transform = str(block["transform"])
    direction = str(block["direction"])

    if transform == "empirical":
        evidence, detail = empirical_anomaly(value, block)
        detail["transform_detail"] = "empirical_tail_to_z_equivalent"
        return evidence, detail

    if transform in {"gaussian", "robust_iqr", "robust_mad"}:
        evidence, z = directional_standardized_anomaly(
            value=float(value),
            center=float(block["center"]),
            scale=float(block["scale"]),
            direction=direction,
        )

        if direction == "higher_is_worse":
            outside = bool(float(value) > float(block["max"]))
        elif direction == "lower_is_worse":
            outside = bool(float(value) < float(block["min"]))
        else:
            outside = bool(
                float(value) < float(block["min"])
                or float(value) > float(block["max"])
            )

        return evidence, {
            "standardized_value": z,
            "tail_probability": None,
            "outside_reference_support": outside,
            "empirical_tail_saturated": False,
            "transform_detail": transform,
        }

    raise EEGHybridV2Error(f"Unknown feature transform {transform!r}")


def rms_aggregate(values: Sequence[float]) -> float:
    x = finite(values)
    require(len(x) > 0, "Cannot RMS-aggregate empty evidence.")

    x = np.clip(x, 0.0, EVIDENCE_CAP)
    return float(np.sqrt(np.mean(x ** 2)))


def finite_sample_upper_threshold(
    clean_scores: Sequence[float],
    alpha: float,
) -> Dict[str, Any]:
    x = np.sort(finite(clean_scores))
    require(len(x) > 0, "Cannot calibrate from empty clean scores.")
    require(0.0 < alpha < 1.0, "alpha must lie in (0,1).")

    n = len(x)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))

    threshold = (
        math.inf
        if k > n
        else float(x[k - 1])
    )

    return {
        "n_calibration": n,
        "target_alpha": float(alpha),
        "order_statistic_k_1based": k,
        "threshold_anomaly": threshold,
        "calibration_exceedance_rate": float(np.mean(x > threshold)),
        "calibration_min": float(np.min(x)),
        "calibration_median": float(np.median(x)),
        "calibration_max": float(np.max(x)),
        "calibration_q90": float(np.quantile(x, 0.90)),
        "calibration_q95": float(np.quantile(x, 0.95)),
        "calibration_q99": float(np.quantile(x, 0.99)),
        "exact_coverage_guarantee_claimed": False,
    }


def score_subject(
    subject: str,
    base_samples: pd.DataFrame,
    feature_lookup: Mapping[Tuple[str, str, str, str], float],
    reference: Mapping[str, Mapping[str, Any]],
    policy: str,
    conditions: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: List[dict] = []
    evidence_rows: List[dict] = []

    g = base_samples[
        (base_samples["subject"] == subject)
        & (base_samples["condition"].isin(set(conditions)))
    ].copy()

    expected_features = sorted(reference)

    for row in g.itertuples():
        wid = str(row.source_window_id)
        condition = str(row.condition)
        available = bool(row.available)

        if not available:
            score_rows.append({
                "source_window_id": wid,
                "subject": subject,
                "pair_key": str(row.pair_key),
                "condition": condition,
                "policy": policy,
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

        evidence_values: List[float] = []
        missing: List[str] = []

        for feature in expected_features:
            key = (
                wid,
                condition,
                EEG_MODALITY,
                feature,
            )

            if key not in feature_lookup:
                missing.append(feature)
                continue

            value = float(feature_lookup[key])
            block = reference[feature]
            evidence, detail = feature_anomaly(
                value,
                block,
            )
            evidence_values.append(evidence)

            evidence_rows.append({
                "source_window_id": wid,
                "subject": subject,
                "pair_key": str(row.pair_key),
                "condition": condition,
                "policy": policy,
                "feature_path": feature,
                "feature_type": block["feature_type"],
                "transform": block["transform"],
                "direction": block["direction"],
                "raw_value": value,
                "reference_min": block["min"],
                "reference_max": block["max"],
                "reference_median": block["median"],
                "reference_center": block.get("center"),
                "reference_scale": block.get("scale"),
                "reference_scale_definition": block.get("scale_definition"),
                "anomaly_evidence": evidence,
                "standardized_value": detail["standardized_value"],
                "tail_probability": detail["tail_probability"],
                "outside_reference_support": detail["outside_reference_support"],
                "empirical_tail_saturated": detail["empirical_tail_saturated"],
                "transform_detail": detail["transform_detail"],
                "corruption_family_used_as_input": False,
            })

        if missing:
            status = "PARTIAL_EVIDENCE"
            anomaly = (
                rms_aggregate(evidence_values)
                if evidence_values
                else None
            )
        elif not evidence_values:
            status = "NO_QUALITY_EVIDENCE"
            anomaly = None
        else:
            status = "OK"
            anomaly = rms_aggregate(evidence_values)

        score_rows.append({
            "source_window_id": wid,
            "subject": subject,
            "pair_key": str(row.pair_key),
            "condition": condition,
            "policy": policy,
            "available": True,
            "status": status,
            "expected_evidence_count": len(expected_features),
            "observed_evidence_count": len(evidence_values),
            "anomaly_score": anomaly,
            "existing_q": (
                float(row.existing_q)
                if row.existing_q is not None
                and math.isfinite(float(row.existing_q))
                else None
            ),
        })

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(evidence_rows),
    )


def build_inner_oof_clean_scores(
    outer_train_subjects: Sequence[str],
    features: pd.DataFrame,
    base_samples: pd.DataFrame,
    feature_lookup: Mapping[Tuple[str, str, str, str], float],
    feature_spec: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for inner_held in outer_train_subjects:
        inner_train = [
            s for s in outer_train_subjects
            if s != inner_held
        ]

        require(
            len(inner_train) == EXPECTED_SUBJECTS - 2,
            "Inner reference must contain four subjects.",
        )

        for policy in LINE_POLICIES:
            reference = fit_eeg_reference(
                features,
                train_subjects=inner_train,
                feature_spec=feature_spec,
                policy=policy,
                expected_windows_per_feature=(
                    EXPECTED_INNER_REFERENCE_WINDOWS
                ),
            )

            scored, _ = score_subject(
                subject=inner_held,
                base_samples=base_samples,
                feature_lookup=feature_lookup,
                reference=reference,
                policy=policy,
                conditions=["reference"],
            )

            require(
                len(scored) == EXPECTED_CLEAN_WINDOWS_PER_SUBJECT,
                f"{inner_held}/{policy}: expected 20 inner clean rows, got {len(scored)}",
            )
            require(
                bool((scored["status"] == "OK").all()),
                f"{inner_held}/{policy}: incomplete inner clean evidence.",
            )

            scored["inner_held_subject"] = inner_held
            scored["inner_reference_subjects"] = ";".join(sorted(inner_train))
            frames.append(scored)

    result = pd.concat(frames, ignore_index=True)

    for policy in LINE_POLICIES:
        g = result[result["policy"] == policy]
        require(
            len(g) == EXPECTED_INNER_CALIBRATION_WINDOWS,
            f"{policy}: expected 100 inner OOF clean rows, got {len(g)}",
        )
        require(
            g["inner_held_subject"].nunique() == EXPECTED_SUBJECTS - 1,
            f"{policy}: inner held-subject count mismatch.",
        )

    return result


def calibrate_outer_thresholds(
    outer_held: str,
    inner_oof: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for policy in LINE_POLICIES:
        g = inner_oof[
            (inner_oof["policy"] == policy)
            & (inner_oof["status"] == "OK")
        ]

        clean_scores = finite(g["anomaly_score"])

        require(
            len(clean_scores) == EXPECTED_INNER_CALIBRATION_WINDOWS,
            f"{outer_held}/{policy}: expected 100 clean calibration scores.",
        )

        for alpha in TARGET_ALPHAS:
            result = finite_sample_upper_threshold(
                clean_scores,
                alpha,
            )

            rows.append({
                "outer_held_subject": outer_held,
                "policy": policy,
                **result,
                "outer_subject_used_in_calibration": False,
                "threshold_selected_for_production": False,
            })

    return pd.DataFrame(rows)


def apply_thresholds(
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    lookup = {
        (
            str(r.policy),
            float(r.target_alpha),
        ): float(r.threshold_anomaly)
        for r in thresholds.itertuples()
    }

    rows: List[dict] = []

    for score in scores.itertuples():
        policy = str(score.policy)
        available = bool(score.available)
        status = str(score.status)

        for alpha in TARGET_ALPHAS:
            threshold = lookup[(policy, float(alpha))]

            if not available:
                degraded = True
                reason = "UNAVAILABLE"
            elif status != "OK":
                degraded = None
                reason = status
            else:
                degraded = bool(
                    float(score.anomaly_score) > threshold
                )
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
                "policy": policy,
                "available": available,
                "status": status,
                "anomaly_score": score.anomaly_score,
                "existing_q": score.existing_q,
                "target_alpha": float(alpha),
                "threshold_anomaly": threshold,
                "degraded": degraded,
                "decision_reason": reason,
            })

    return pd.DataFrame(rows)


def clean_fdr_by_fold(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    clean = decisions[
        (decisions["condition"] == "reference")
        & (decisions["status"] == "OK")
    ].copy()

    rows: List[dict] = []

    for (
        subject,
        policy,
        alpha,
        threshold,
    ), g in clean.groupby(
        [
            "subject",
            "policy",
            "target_alpha",
            "threshold_anomaly",
        ],
        dropna=False,
    ):
        require(
            len(g) == EXPECTED_CLEAN_WINDOWS_PER_SUBJECT,
            f"{subject}/{policy}/{alpha}: expected 20 clean windows.",
        )

        d = g["degraded"].astype(bool).to_numpy()

        rows.append({
            "outer_held_subject": subject,
            "policy": policy,
            "target_alpha": float(alpha),
            "threshold_anomaly": float(threshold),
            "n_clean_windows": len(g),
            "actual_clean_false_degradation_rate": float(np.mean(d)),
            "clean_healthy_rate": float(1.0 - np.mean(d)),
        })

    return pd.DataFrame(rows)


def family_detection_by_fold(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    subjects = sorted(decisions["subject"].unique())

    for subject in subjects:
        for policy in LINE_POLICIES:
            for alpha in TARGET_ALPHAS:
                for family, spec in FAMILY_SPECS.items():
                    condition_map = {
                        "reference": str(spec["reference_condition"]),
                        **{
                            severity: str(condition)
                            for severity, condition in spec["conditions"].items()
                        },
                    }

                    for severity, condition in condition_map.items():
                        g = decisions[
                            (decisions["subject"] == subject)
                            & (decisions["policy"] == policy)
                            & (decisions["target_alpha"] == float(alpha))
                            & (decisions["condition"] == condition)
                        ].copy()

                        require(
                            len(g) == EXPECTED_CLEAN_WINDOWS_PER_SUBJECT,
                            f"{subject}/{policy}/{alpha}/{family}/{severity}: "
                            f"expected 20 windows, got {len(g)}",
                        )
                        require(
                            not g["degraded"].isna().any(),
                            f"{subject}/{policy}/{alpha}/{family}/{severity}: "
                            "undefined decisions present.",
                        )

                        d = g["degraded"].astype(bool).to_numpy()

                        rows.append({
                            "outer_held_subject": subject,
                            "policy": policy,
                            "target_alpha": float(alpha),
                            "family": family,
                            "severity": severity,
                            "condition": condition,
                            "n_windows": len(g),
                            "degraded_detection_rate": float(np.mean(d)),
                        })

    return pd.DataFrame(rows)


def pool_clean_fdr(
    clean_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for (policy, alpha), g in clean_fold.groupby(
        ["policy", "target_alpha"],
        dropna=False,
    ):
        weights = g["n_clean_windows"].to_numpy(dtype=float)
        rates = g["actual_clean_false_degradation_rate"].to_numpy(dtype=float)

        rows.append({
            "policy": policy,
            "target_alpha": float(alpha),
            "outer_subjects": int(g["outer_held_subject"].nunique()),
            "n_clean_windows": int(np.sum(weights)),
            "actual_clean_false_degradation_rate": float(
                np.average(rates, weights=weights)
            ),
            "subject_rate_mean": float(np.mean(rates)),
            "subject_rate_sd": float(np.std(rates, ddof=1)),
            "subject_rate_min": float(np.min(rates)),
            "subject_rate_max": float(np.max(rates)),
        })

    return pd.DataFrame(rows)


def pool_family_detection(
    family_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    group_cols = [
        "policy",
        "target_alpha",
        "family",
        "severity",
        "condition",
    ]

    for key, g in family_fold.groupby(
        group_cols,
        dropna=False,
    ):
        meta = dict(zip(group_cols, key))
        weights = g["n_windows"].to_numpy(dtype=float)
        rates = g["degraded_detection_rate"].to_numpy(dtype=float)

        rows.append({
            **meta,
            "outer_subjects": int(g["outer_held_subject"].nunique()),
            "n_windows": int(np.sum(weights)),
            "pooled_detection_rate": float(
                np.average(rates, weights=weights)
            ),
            "subject_rate_mean": float(np.mean(rates)),
            "subject_rate_sd": float(np.std(rates, ddof=1)),
            "subject_rate_min": float(np.min(rates)),
            "subject_rate_max": float(np.max(rates)),
        })

    return pd.DataFrame(rows)


def threshold_stability(
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for (policy, alpha), g in thresholds.groupby(
        ["policy", "target_alpha"],
        dropna=False,
    ):
        x = finite(g["threshold_anomaly"])

        require(
            len(x) == EXPECTED_SUBJECTS,
            f"{policy}/{alpha}: expected six outer thresholds.",
        )

        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))

        rows.append({
            "policy": policy,
            "target_alpha": float(alpha),
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


def build_operating_points(
    clean_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for policy in LINE_POLICIES:
        for alpha in TARGET_ALPHAS:
            clean = clean_summary[
                (clean_summary["policy"] == policy)
                & (clean_summary["target_alpha"] == float(alpha))
            ]
            threshold = threshold_summary[
                (threshold_summary["policy"] == policy)
                & (threshold_summary["target_alpha"] == float(alpha))
            ]

            require(len(clean) == 1, "Expected one clean summary row.")
            require(len(threshold) == 1, "Expected one threshold summary row.")

            family_rates: Dict[Tuple[str, str], float] = {}

            g = family_summary[
                (family_summary["policy"] == policy)
                & (family_summary["target_alpha"] == float(alpha))
            ]

            for r in g.itertuples():
                family_rates[
                    (str(r.family), str(r.severity))
                ] = float(r.pooled_detection_rate)

            line_rates = [
                family_rates.get(("eeg_global_line", sev))
                for sev in ("mild", "medium", "severe")
            ]
            flat_rates = [
                family_rates.get(("eeg_channel_flatline", sev))
                for sev in ("mild", "medium", "severe")
            ]

            severe_values = [
                x
                for x in (
                    family_rates.get(("eeg_global_line", "severe")),
                    family_rates.get(("eeg_channel_flatline", "severe")),
                )
                if x is not None
            ]

            rows.append({
                "policy": policy,
                "target_alpha": float(alpha),
                "actual_clean_false_degradation_rate": float(
                    clean["actual_clean_false_degradation_rate"].iloc[0]
                ),
                "threshold_mean": float(
                    threshold["threshold_mean"].iloc[0]
                ),
                "threshold_sd": float(
                    threshold["threshold_sd"].iloc[0]
                ),
                "threshold_cv_abs": (
                    float(threshold["threshold_cv_abs"].iloc[0])
                    if pd.notna(threshold["threshold_cv_abs"].iloc[0])
                    else None
                ),
                "global_line_mild_detection": line_rates[0],
                "global_line_medium_detection": line_rates[1],
                "global_line_severe_detection": line_rates[2],
                "flatline_mild_detection": flat_rates[0],
                "flatline_medium_detection": flat_rates[1],
                "flatline_severe_detection": flat_rates[2],
                "mean_severe_detection_across_families": (
                    float(np.mean(severe_values))
                    if severe_values
                    else None
                ),
                "worst_severe_detection_across_families": (
                    float(np.min(severe_values))
                    if severe_values
                    else None
                ),
                "policy_selected_for_production": False,
            })

    return pd.DataFrame(rows)


def primary_policy_scorecard(
    operating_points: pd.DataFrame,
) -> pd.DataFrame:
    g = operating_points[
        operating_points["target_alpha"] == PRIMARY_ALPHA
    ].copy()

    g["clean_fdr_absolute_error_from_target"] = np.abs(
        g["actual_clean_false_degradation_rate"]
        - PRIMARY_ALPHA
    )

    columns = [
        "policy",
        "target_alpha",
        "actual_clean_false_degradation_rate",
        "clean_fdr_absolute_error_from_target",
        "threshold_mean",
        "threshold_sd",
        "threshold_cv_abs",
        "global_line_mild_detection",
        "global_line_medium_detection",
        "global_line_severe_detection",
        "flatline_mild_detection",
        "flatline_medium_detection",
        "flatline_severe_detection",
        "mean_severe_detection_across_families",
        "worst_severe_detection_across_families",
        "policy_selected_for_production",
    ]

    return g[columns].sort_values("policy").reset_index(drop=True)


def summarize_line_transform_response(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    line = evidence[
        evidence["feature_path"] == LINE_FEATURE
    ].copy()

    rows: List[dict] = []

    condition_order = {
        "reference": 0,
        "eeg_global_line_mild": 1,
        "eeg_global_line_medium": 2,
        "eeg_global_line_severe": 3,
        "eeg_channel_flatline_mild": 1,
        "eeg_channel_flatline_medium": 2,
        "eeg_channel_flatline_severe": 3,
    }

    for (policy, condition), g in line.groupby(
        ["policy", "condition"],
        dropna=False,
    ):
        e = finite(g["anomaly_evidence"])
        raw = finite(g["raw_value"])

        require(len(e) == len(g), "Line evidence contains non-finite anomaly values.")

        rows.append({
            "policy": policy,
            "condition": condition,
            "severity_rank": condition_order.get(condition),
            "n_windows": len(g),
            "mean_raw_line_fraction": float(np.mean(raw)),
            "median_raw_line_fraction": float(np.median(raw)),
            "mean_line_anomaly_evidence": float(np.mean(e)),
            "median_line_anomaly_evidence": float(np.median(e)),
            "max_line_anomaly_evidence": float(np.max(e)),
            "outside_reference_support_rate": float(
                g["outside_reference_support"].astype(bool).mean()
            ),
            "empirical_tail_saturation_rate": float(
                g["empirical_tail_saturated"].astype(bool).mean()
            ),
            "unique_anomaly_values": int(
                len(np.unique(np.round(e, 12)))
            ),
        })

    return pd.DataFrame(rows)


def line_trial_severity_monotonicity(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    stats = import_stats()

    line = evidence[
        evidence["feature_path"] == LINE_FEATURE
    ].copy()

    ordered = [
        "reference",
        "eeg_global_line_mild",
        "eeg_global_line_medium",
        "eeg_global_line_severe",
    ]

    # Average four 5-s evidence values per 20-s trial condition.
    trial = (
        line[line["condition"].isin(ordered)]
        .groupby(
            ["subject", "pair_key", "policy", "condition"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_line_evidence=("anomaly_evidence", "mean"),
            windows=("source_window_id", "nunique"),
        )
    )

    rows: List[dict] = []

    for policy in LINE_POLICIES:
        g = trial[trial["policy"] == policy]

        pivot = g.pivot_table(
            index=["subject", "pair_key"],
            columns="condition",
            values="mean_line_evidence",
            aggfunc="mean",
        )

        complete = pivot.dropna(subset=ordered).copy()

        require(
            len(complete) == EXPECTED_SUBJECTS * EXPECTED_TRIALS_PER_SUBJECT,
            f"{policy}: expected 30 complete global-line trial trajectories, "
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

        rows.append({
            "policy": policy,
            "n_complete_trials": len(complete),
            "mean_spearman_severity_vs_line_evidence": (
                float(np.mean(rhos))
                if rhos
                else None
            ),
            "median_spearman": (
                float(np.median(rhos))
                if rhos
                else None
            ),
            "monotonic_nondecreasing_rate": float(
                np.mean(np.all(diffs >= -1e-12, axis=1))
            ),
            "strict_increase_somewhere_rate": float(
                np.mean(np.any(diffs > 1e-12, axis=1))
            ),
            "severe_greater_than_reference_rate": float(
                np.mean(arr[:, -1] > arr[:, 0])
            ),
        })

    return pd.DataFrame(rows)


def full_val_crossfit_threshold_candidates(
    outer_scores: pd.DataFrame,
) -> pd.DataFrame:
    clean = outer_scores[
        (outer_scores["condition"] == "reference")
        & (outer_scores["status"] == "OK")
    ].copy()

    rows: List[dict] = []

    for policy in LINE_POLICIES:
        g = clean[clean["policy"] == policy]
        scores = finite(g["anomaly_score"])

        require(
            len(scores) == EXPECTED_FULL_CROSSFIT_CLEAN_WINDOWS,
            f"{policy}: expected 120 full crossfit clean scores.",
        )
        require(
            g["subject"].nunique() == EXPECTED_SUBJECTS,
            f"{policy}: full crossfit missing subjects.",
        )

        for alpha in TARGET_ALPHAS:
            result = finite_sample_upper_threshold(
                scores,
                alpha,
            )

            rows.append({
                "policy": policy,
                **result,
                "used_for_nested_outer_evaluation": False,
                "production_frozen": False,
                "purpose": "candidate_for_later_runtime_freeze",
            })

    return pd.DataFrame(rows)


def create_plots(
    out_dir: Path,
    operating_points: pd.DataFrame,
    line_response: pd.DataFrame,
    line_monotonicity: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EEGHybridV2Error("matplotlib is required for plots.") from exc

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=False)

    # Global-line detection at primary alpha.
    g = operating_points[
        operating_points["target_alpha"] == PRIMARY_ALPHA
    ].copy()

    if not g.empty:
        labels = g["policy"].tolist()
        mild = g["global_line_mild_detection"].to_numpy(dtype=float)
        medium = g["global_line_medium_detection"].to_numpy(dtype=float)
        severe = g["global_line_severe_detection"].to_numpy(dtype=float)

        x = np.arange(len(labels))
        width = 0.24

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(x - width, mild, width=width, label="mild")
        ax.bar(x, medium, width=width, label="medium")
        ax.bar(x + width, severe, width=width, label="severe")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("degraded detection rate")
        ax.set_title("EEG global-line detection at alpha=0.05")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plots / "global_line_detection_alpha005.png",
            dpi=170,
        )
        plt.close(fig)

    # Flatline severe + global-line severe vs clean FDR.
    if not g.empty:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        for row in g.itertuples():
            ax.scatter(
                [row.actual_clean_false_degradation_rate],
                [row.global_line_severe_detection],
                label=f"{row.policy} / line severe",
            )
            ax.scatter(
                [row.actual_clean_false_degradation_rate],
                [row.flatline_severe_detection],
                marker="x",
                label=f"{row.policy} / flat severe",
            )

        ax.set_xlabel("actual clean false-degradation rate")
        ax.set_ylabel("severe degradation detection")
        ax.set_xlim(0.0, max(0.20, float(g["actual_clean_false_degradation_rate"].max()) + 0.03))
        ax.set_ylim(0.0, 1.05)
        ax.set_title("EEG Hybrid-v2 operating trade-off at alpha=0.05")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            plots / "primary_operating_tradeoff.png",
            dpi=170,
        )
        plt.close(fig)

    # Line evidence by global-line severity.
    ordered = [
        "reference",
        "eeg_global_line_mild",
        "eeg_global_line_medium",
        "eeg_global_line_severe",
    ]

    r = line_response[
        line_response["condition"].isin(ordered)
    ].copy()

    if not r.empty:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)

        for policy in LINE_POLICIES:
            p = r[r["policy"] == policy].copy()
            means = []
            for condition in ordered:
                row = p[p["condition"] == condition]
                means.append(
                    float(row["mean_line_anomaly_evidence"].iloc[0])
                    if len(row)
                    else np.nan
                )

            ax.plot(
                [0, 1, 2, 3],
                means,
                marker="o",
                label=policy,
            )

        ax.set_xticks(
            [0, 1, 2, 3],
            ["reference", "mild", "medium", "severe"],
        )
        ax.set_ylabel("mean line-fraction anomaly evidence")
        ax.set_title("Line-transform response to global line interference")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            plots / "line_transform_response.png",
            dpi=170,
        )
        plt.close(fig)

    # Monotonicity summary.
    if not line_monotonicity.empty:
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)

        labels = line_monotonicity["policy"].tolist()
        values = line_monotonicity[
            "mean_spearman_severity_vs_line_evidence"
        ].to_numpy(dtype=float)

        x = np.arange(len(labels))
        ax.bar(x, values)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(-1.0, 1.05)
        ax.set_ylabel("mean Spearman")
        ax.set_title("Global-line severity ordering by line transform")
        fig.tight_layout()
        fig.savefig(
            plots / "line_transform_monotonicity.png",
            dpi=170,
        )
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Focused EEG Hybrid-v2 comparison of line_fraction anomaly transforms "
            "under nested subject-LOSO threshold calibration."
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
    # Robust scales should remain positive on non-degenerate data.
    clean = np.asarray(
        [0.001, 0.0015, 0.002, 0.0022, 0.0025, 0.003, 0.0035, 0.004],
        dtype=float,
    )

    iqr_scale, _ = robust_scale_iqr(clean)
    mad_scale, _ = robust_scale_mad(clean)

    # Empirical tail should saturate above clean max; robust z should continue.
    empirical_block = fit_reference_block(
        clean,
        transform="empirical",
        direction="higher_is_worse",
    )
    robust_block = fit_reference_block(
        clean,
        transform="robust_iqr",
        direction="higher_is_worse",
    )

    e_emp_1, _ = feature_anomaly(0.010, empirical_block)
    e_emp_2, _ = feature_anomaly(0.100, empirical_block)

    e_rob_1, _ = feature_anomaly(0.010, robust_block)
    e_rob_2, _ = feature_anomaly(0.100, robust_block)

    # Gaussian higher-is-worse sanity.
    gaussian_block = fit_reference_block(
        clean,
        transform="gaussian",
        direction="higher_is_worse",
    )
    e_g_nom, _ = feature_anomaly(float(np.mean(clean)), gaussian_block)
    e_g_bad, _ = feature_anomaly(0.02, gaussian_block)

    # Finite-sample threshold rule.
    scores = np.arange(1, 101, dtype=float)
    t05 = finite_sample_upper_threshold(
        scores,
        0.05,
    )

    # Condition-qualified lookup identity.
    mini = pd.DataFrame([
        {
            "source_window_id": "w01",
            "subject": "subject08",
            "pair_key": "subject08_trial",
            "condition": "reference",
            "family": "",
            "severity": "",
            "modality": "eeg",
            "available": True,
            "feature_path": LINE_FEATURE,
            "value": 0.003,
        },
        {
            "source_window_id": "w01",
            "subject": "subject08",
            "pair_key": "subject08_trial",
            "condition": "eeg_global_line_severe",
            "family": "eeg_global_line",
            "severity": "severe",
            "modality": "eeg",
            "available": True,
            "feature_path": LINE_FEATURE,
            "value": 0.50,
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

    # Synthetic held-subject exclusion geometry.
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
                "modality": "eeg",
                "available": True,
                "feature_path": LINE_FEATURE,
                "value": 0.002 + 0.0001 * sidx + 0.000001 * widx,
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

    four_subjects = [
        "subject01",
        "subject02",
        "subject03",
        "subject04",
    ]

    clean_four = clean_window_values(
        syn,
        feature=LINE_FEATURE,
        subjects=four_subjects,
    )

    checks = {
        "policy_count": len(LINE_POLICIES) == 4,
        "iqr_scale_positive": iqr_scale > 0,
        "mad_scale_positive": mad_scale > 0,
        "empirical_outside_support_saturates": math.isclose(
            e_emp_1,
            e_emp_2,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "robust_z_continues_beyond_support": e_rob_2 > e_rob_1,
        "gaussian_directional_anomaly": e_g_bad > e_g_nom,
        "finite_sample_threshold_rule": (
            t05["order_statistic_k_1based"] == 96
            and math.isclose(
                t05["threshold_anomaly"],
                96.0,
            )
        ),
        "condition_specific_lookup": (
            len(lookup) == 2
            and math.isclose(
                lookup[
                    (
                        "w01",
                        "reference",
                        "eeg",
                        LINE_FEATURE,
                    )
                ],
                0.003,
            )
            and math.isclose(
                lookup[
                    (
                        "w01",
                        "eeg_global_line_severe",
                        "eeg",
                        LINE_FEATURE,
                    )
                ],
                0.50,
            )
        ),
        "inner_reference_geometry_80_windows": len(clean_four) == 80,
        "strict_json_nonfinite_sanitized": (
            '"x": null' in json_text({"x": float("nan")})
            and '"y": null' in json_text({"y": np.float64(np.inf)})
        ),
        "primary_alpha_is_005": math.isclose(PRIMARY_ALPHA, 0.05),
        "aggregation_is_rms": AGGREGATION == "RMS",
    }

    require(
        all(checks.values()),
        f"Self-test failed: {checks}",
    )

    print(
        json_text({
            "status": "PASS",
            "version": VERSION,
            "checks": checks,
            "real_EAV_used": False,
            "test_data_used": False,
            "models_loaded": False,
            "router_changed": False,
            "policy_selected_for_production": False,
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
        except EEGHybridV2Error:
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
        else run_dir / f"eeg_quality_hybrid_v2_{timestamp}"
    )

    require(
        not out_dir.exists(),
        f"Output directory exists; refusing overwrite: {out_dir}",
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    features = prepare_features(feature_path)
    samples = prepare_samples(sample_path)

    feature_spec = eeg_feature_spec_from_hybrid_artifact(
        hybrid_artifact
    )

    feature_lookup = make_feature_lookup(features)
    base_samples = build_eeg_base_samples(samples)

    subjects = sorted(
        samples.loc[
            (samples["condition"] == "reference")
            & (samples["modality"] == EEG_MODALITY),
            "subject",
        ].unique().tolist()
    )

    require(
        len(subjects) == EXPECTED_SUBJECTS,
        f"Expected six VAL subjects, got {subjects}",
    )

    # Formal condition geometry.
    for subject in subjects:
        for condition in EEG_EVAL_CONDITIONS:
            g = base_samples[
                (base_samples["subject"] == subject)
                & (base_samples["condition"] == condition)
            ]

            require(
                len(g) == EXPECTED_CLEAN_WINDOWS_PER_SUBJECT,
                f"{subject}/{condition}: expected 20 EEG windows, got {len(g)}",
            )

            require(
                bool(g["available"].all()),
                f"{subject}/{condition}: controlled EEG family unexpectedly unavailable.",
            )

    all_inner: List[pd.DataFrame] = []
    all_thresholds: List[pd.DataFrame] = []
    all_outer_scores: List[pd.DataFrame] = []
    all_outer_evidence: List[pd.DataFrame] = []
    all_decisions: List[pd.DataFrame] = []
    outer_reference_rows: List[dict] = []

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
        # INNER cross-fit clean anomaly scores for threshold calibration.
        # --------------------------------------------------------------
        inner_oof = build_inner_oof_clean_scores(
            outer_train_subjects=outer_train,
            features=features,
            base_samples=base_samples,
            feature_lookup=feature_lookup,
            feature_spec=feature_spec,
        )
        inner_oof["outer_held_subject"] = outer_held

        thresholds = calibrate_outer_thresholds(
            outer_held,
            inner_oof,
        )

        # --------------------------------------------------------------
        # OUTER subject score under each line-transform policy.
        # --------------------------------------------------------------
        fold_scores: List[pd.DataFrame] = []
        fold_evidence: List[pd.DataFrame] = []

        for policy in LINE_POLICIES:
            reference = fit_eeg_reference(
                features,
                train_subjects=outer_train,
                feature_spec=feature_spec,
                policy=policy,
                expected_windows_per_feature=(
                    EXPECTED_OUTER_REFERENCE_WINDOWS
                ),
            )

            # Record line-reference parameters for stability / interpretation.
            line_ref = reference[LINE_FEATURE]

            outer_reference_rows.append({
                "outer_held_subject": outer_held,
                "policy": policy,
                "outer_reference_subjects": ";".join(sorted(outer_train)),
                "line_transform": line_ref["transform"],
                "line_reference_n": line_ref["n"],
                "line_reference_min": line_ref["min"],
                "line_reference_max": line_ref["max"],
                "line_reference_mean": line_ref["mean"],
                "line_reference_median": line_ref["median"],
                "line_reference_center": line_ref.get("center"),
                "line_reference_scale": line_ref.get("scale"),
                "line_reference_scale_definition": line_ref.get(
                    "scale_definition"
                ),
                "outer_subject_excluded": True,
            })

            scores, evidence = score_subject(
                subject=outer_held,
                base_samples=base_samples,
                feature_lookup=feature_lookup,
                reference=reference,
                policy=policy,
                conditions=EEG_EVAL_CONDITIONS,
            )

            require(
                bool((scores["status"] == "OK").all()),
                f"{outer_held}/{policy}: controlled EEG evaluation has incomplete evidence.",
            )

            scores["outer_held_subject"] = outer_held
            scores["outer_reference_subjects"] = ";".join(sorted(outer_train))
            evidence["outer_held_subject"] = outer_held
            evidence["outer_reference_subjects"] = ";".join(sorted(outer_train))

            fold_scores.append(scores)
            fold_evidence.append(evidence)

        outer_scores = pd.concat(
            fold_scores,
            ignore_index=True,
        )
        outer_evidence = pd.concat(
            fold_evidence,
            ignore_index=True,
        )

        decisions = apply_thresholds(
            outer_scores,
            thresholds,
        )

        all_inner.append(inner_oof)
        all_thresholds.append(thresholds)
        all_outer_scores.append(outer_scores)
        all_outer_evidence.append(outer_evidence)
        all_decisions.append(decisions)

    inner_all = pd.concat(
        all_inner,
        ignore_index=True,
    )
    thresholds_all = pd.concat(
        all_thresholds,
        ignore_index=True,
    )
    outer_scores_all = pd.concat(
        all_outer_scores,
        ignore_index=True,
    )
    outer_evidence_all = pd.concat(
        all_outer_evidence,
        ignore_index=True,
    )
    decisions_all = pd.concat(
        all_decisions,
        ignore_index=True,
    )
    outer_reference_df = pd.DataFrame(
        outer_reference_rows
    )

    # --------------------------------------------------------------
    # Evaluate nested operating points.
    # --------------------------------------------------------------
    clean_fold = clean_fdr_by_fold(
        decisions_all
    )
    family_fold = family_detection_by_fold(
        decisions_all
    )

    clean_summary = pool_clean_fdr(
        clean_fold
    )
    family_summary = pool_family_detection(
        family_fold
    )
    threshold_summary = threshold_stability(
        thresholds_all
    )

    operating_points = build_operating_points(
        clean_summary=clean_summary,
        family_summary=family_summary,
        threshold_summary=threshold_summary,
    )

    scorecard = primary_policy_scorecard(
        operating_points
    )

    line_response = summarize_line_transform_response(
        outer_evidence_all
    )

    line_monotonicity = line_trial_severity_monotonicity(
        outer_evidence_all
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
        out_dir / "eeg_v2_inner_oof_clean_scores.csv",
        inner_all,
    )
    atomic_csv(
        out_dir / "eeg_v2_nested_loso_thresholds.csv",
        thresholds_all,
    )
    atomic_csv(
        out_dir / "eeg_v2_outer_window_scores.csv",
        outer_scores_all,
    )
    atomic_csv(
        out_dir / "eeg_v2_outer_window_evidence.csv",
        outer_evidence_all,
    )
    atomic_csv(
        out_dir / "eeg_v2_outer_window_decisions.csv",
        decisions_all,
    )
    atomic_csv(
        out_dir / "eeg_v2_outer_line_reference_parameters.csv",
        outer_reference_df,
    )
    atomic_csv(
        out_dir / "eeg_v2_clean_fdr_by_fold.csv",
        clean_fold,
    )
    atomic_csv(
        out_dir / "eeg_v2_clean_fdr_pooled.csv",
        clean_summary,
    )
    atomic_csv(
        out_dir / "eeg_v2_family_detection_by_fold.csv",
        family_fold,
    )
    atomic_csv(
        out_dir / "eeg_v2_family_detection_pooled.csv",
        family_summary,
    )
    atomic_csv(
        out_dir / "eeg_v2_threshold_stability.csv",
        threshold_summary,
    )
    atomic_csv(
        out_dir / "eeg_v2_operating_points.csv",
        operating_points,
    )
    atomic_csv(
        out_dir / "eeg_v2_primary_alpha005_scorecard.csv",
        scorecard,
    )
    atomic_csv(
        out_dir / "eeg_v2_line_transform_response.csv",
        line_response,
    )
    atomic_csv(
        out_dir / "eeg_v2_line_severity_monotonicity.csv",
        line_monotonicity,
    )
    atomic_csv(
        out_dir / "eeg_v2_full_val_crossfit_threshold_candidates.csv",
        full_crossfit_candidates,
    )

    if not args.no_plots:
        create_plots(
            out_dir,
            operating_points=operating_points,
            line_response=line_response,
            line_monotonicity=line_monotonicity,
        )

    # --------------------------------------------------------------
    # Compact summary.
    # --------------------------------------------------------------
    compact_policies: Dict[str, Any] = {}

    for row in scorecard.itertuples():
        compact_policies[str(row.policy)] = {
            "target_alpha": float(row.target_alpha),
            "actual_clean_false_degradation_rate": float(
                row.actual_clean_false_degradation_rate
            ),
            "clean_fdr_absolute_error_from_target": float(
                row.clean_fdr_absolute_error_from_target
            ),
            "threshold_mean": float(row.threshold_mean),
            "threshold_sd": float(row.threshold_sd),
            "threshold_cv_abs": (
                float(row.threshold_cv_abs)
                if row.threshold_cv_abs is not None
                and math.isfinite(float(row.threshold_cv_abs))
                else None
            ),
            "global_line": {
                "mild_detection": float(
                    row.global_line_mild_detection
                ),
                "medium_detection": float(
                    row.global_line_medium_detection
                ),
                "severe_detection": float(
                    row.global_line_severe_detection
                ),
            },
            "channel_flatline": {
                "mild_detection": float(
                    row.flatline_mild_detection
                ),
                "medium_detection": float(
                    row.flatline_medium_detection
                ),
                "severe_detection": float(
                    row.flatline_severe_detection
                ),
            },
            "mean_severe_detection_across_families": float(
                row.mean_severe_detection_across_families
            ),
            "worst_severe_detection_across_families": float(
                row.worst_severe_detection_across_families
            ),
            "production_selected": False,
        }

    compact_line_monotonicity = {
        str(r.policy): {
            "mean_spearman": (
                float(r.mean_spearman_severity_vs_line_evidence)
                if r.mean_spearman_severity_vs_line_evidence is not None
                and math.isfinite(
                    float(r.mean_spearman_severity_vs_line_evidence)
                )
                else None
            ),
            "monotonic_nondecreasing_rate": float(
                r.monotonic_nondecreasing_rate
            ),
            "severe_greater_than_reference_rate": float(
                r.severe_greater_than_reference_rate
            ),
        }
        for r in line_monotonicity.itertuples()
    }

    artifact = {
        "schema": "eav.eeg_quality_hybrid_v2_candidate.v1",
        "version": VERSION,
        "artifact_state": "CANDIDATE_NOT_DEPLOYABLE",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_hybrid_artifact": str(hybrid_artifact_path),
        "source_split": "VAL_ONLY",
        "test_data_used": False,
        "scope": "EEG_QUALITY_POLICY_ONLY",
        "runtime_unit": "5s_window",
        "aggregation": {
            "name": "RMS",
            "changed_from_hybrid_v1": False,
            "production_frozen": False,
        },
        "line_transform_policies": {
            "empirical_line": {
                "transform": "empirical_clean_tail_to_z_equivalent",
                "role": "current_hybrid_v1_baseline",
            },
            "gaussian_line": {
                "transform": "directional_mean_std_z",
                "gaussian_population_claimed": False,
            },
            "robust_iqr_line": {
                "transform": "directional_robust_z",
                "center": "median",
                "scale": "IQR/1.349",
                "non_saturating_beyond_clean_support": True,
                "gaussian_population_claimed": False,
            },
            "robust_mad_line": {
                "transform": "directional_robust_z",
                "center": "median",
                "scale": "1.4826*MAD",
                "non_saturating_beyond_clean_support": True,
                "gaussian_population_claimed": False,
            },
        },
        "fixed_non_line_eeg_policy": {
            feature: {
                "transform": meta["source_transform"],
                "direction": meta["direction"],
                "feature_type": meta["feature_type"],
            }
            for feature, meta in feature_spec.items()
            if feature != LINE_FEATURE
        },
        "nested_subject_protocol": {
            "subjects": subjects,
            "outer_train_subjects_per_fold": 5,
            "outer_test_subjects_per_fold": 1,
            "inner_reference_subjects_per_fold": 4,
            "inner_held_subjects_per_subfold": 1,
            "inner_oof_clean_windows_per_policy": (
                EXPECTED_INNER_CALIBRATION_WINDOWS
            ),
            "outer_subject_used_in_threshold_calibration": False,
            "corruption_family_used_as_score_input": False,
        },
        "target_alphas": list(TARGET_ALPHAS),
        "primary_development_alpha": PRIMARY_ALPHA,
        "primary_alpha005_scorecard": compact_policies,
        "line_severity_monotonicity": compact_line_monotonicity,
        "full_val_crossfit_threshold_candidates": (
            full_crossfit_candidates.to_dict(orient="records")
        ),
        "selection": {
            "line_transform_selected": False,
            "target_alpha_selected": False,
            "eeg_policy_frozen_for_production": False,
            "router_selected": False,
        },
        "existing_deployment": {
            "audio_policy_changed": False,
            "video_policy_changed": False,
            "tau_0_80_changed": False,
            "quality_calibrators_changed": False,
            "fusion_weights_changed": False,
            "emotion_models_changed": False,
        },
        "availability": {
            "separate_hard_state": True,
            "unavailable_always_degraded": True,
        },
        "guardrails": [
            "VAL only; no TEST subjects are used.",
            "Only EEG line_fraction transform changes across compared policies.",
            "All non-line EEG evidence and RMS aggregation are held fixed.",
            "Outer subject is excluded from feature-reference fitting and threshold calibration.",
            "Known corruption family is used only for retrospective evaluation.",
            "Robust-z does not assert a Gaussian raw-feature population.",
            "Empirical-tail saturation is measured explicitly.",
            "No EEG policy, alpha, threshold, router, or runtime implementation is selected/deployed here.",
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
            "source_hybrid_artifact": sha256_file(
                hybrid_artifact_path
            ),
        },
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
    }

    atomic_json(
        out_dir / "eeg_quality_hybrid_v2_candidate_artifact.json",
        artifact,
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
        "primary_alpha": PRIMARY_ALPHA,
        "policies": compact_policies,
        "line_severity_monotonicity": compact_line_monotonicity,
        "deployment_selection_made": False,
        "router_threshold_retuned_in_existing_system": False,
        "models_or_existing_calibrators_modified": False,
        "candidate_artifact": str(
            out_dir / "eeg_quality_hybrid_v2_candidate_artifact.json"
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
        out_dir / "eeg_quality_hybrid_v2_summary.json",
        summary,
    )

    lines = [
        "# EEG Quality Hybrid-v2",
        "",
        f"- Version: `{VERSION}`",
        f"- Source run: `{run_dir}`",
        f"- Source Hybrid-v1 artifact: `{hybrid_artifact_path}`",
        "- Scope: **EEG quality policy only**",
        "- Formal balanced VAL only: **Yes**",
        "- Nested subject-wise LOSO: **Yes**",
        "- Primary development alpha: **0.05**",
        "- RMS aggregation changed: **No**",
        "- Production policy selected: **No**",
        "",
        "## What changes",
        "",
        "Only `features.line_fraction_mean` changes across policies.",
        "Every other EEG continuous feature and guard remains the current "
        "Hybrid-v1 empirical transform.",
        "",
        "## Alpha=0.05 operating-point scorecard",
        "",
    ]

    for row in scorecard.itertuples():
        lines.append(
            f"- **{row.policy}**: "
            f"cleanFDR={row.actual_clean_false_degradation_rate:.3f}, "
            f"line(m/m/s)="
            f"{row.global_line_mild_detection:.3f}/"
            f"{row.global_line_medium_detection:.3f}/"
            f"{row.global_line_severe_detection:.3f}, "
            f"flat(m/m/s)="
            f"{row.flatline_mild_detection:.3f}/"
            f"{row.flatline_medium_detection:.3f}/"
            f"{row.flatline_severe_detection:.3f}, "
            f"T={row.threshold_mean:.3f}±{row.threshold_sd:.3f}"
        )

    lines += [
        "",
        "## Line-evidence severity ordering",
        "",
    ]

    for row in line_monotonicity.itertuples():
        rho = (
            f"{row.mean_spearman_severity_vs_line_evidence:.3f}"
            if row.mean_spearman_severity_vs_line_evidence is not None
            and math.isfinite(
                float(row.mean_spearman_severity_vs_line_evidence)
            )
            else "NA"
        )
        lines.append(
            f"- **{row.policy}**: mean rho={rho}, "
            f"monotonic={row.monotonic_nondecreasing_rate:.3f}, "
            f"severe>reference={row.severe_greater_than_reference_rate:.3f}"
        )

    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- Robust-z is a standardized distance, not a Gaussian population claim.",
        "- Empirical-tail saturation is expected beyond finite clean support and is measured explicitly.",
        "- No TEST data are used.",
        "- No policy/threshold is frozen in this stage.",
        "",
        "## Next gate",
        "",
        "Review:",
        "- `eeg_v2_primary_alpha005_scorecard.csv`",
        "- `eeg_v2_line_transform_response.csv`",
        "- `eeg_v2_line_severity_monotonicity.csv`",
        "- `eeg_v2_threshold_stability.csv`",
        "- `eeg_v2_full_val_crossfit_threshold_candidates.csv`",
        "",
        "A later freeze decision should require a candidate that preserves strong "
        "flatline detection, restores global-line sensitivity/order, keeps clean "
        "false-degradation near the target, and remains stable across held subjects.",
        "",
    ]

    atomic_text(
        out_dir / "README_eeg_quality_hybrid_v2.md",
        "\n".join(lines),
    )

    print("=" * 124)
    print("EAV EEG QUALITY HYBRID-v2")
    print("=" * 124)
    print(f"Status                       : PASS")
    print(f"Version                      : {VERSION}")
    print(f"Source formal run            : {run_dir}")
    print(f"Source Hybrid-v1 artifact    : {hybrid_artifact_path}")
    print(f"Output                       : {out_dir}")
    print(f"Nested subject LOSO          : True")
    print(f"VAL subjects                 : {len(subjects)}")
    print(f"Primary alpha                : {PRIMARY_ALPHA:.2f}")
    print(f"Policies                     : {', '.join(LINE_POLICIES)}")
    print(f"TEST used                    : False")
    print(f"Existing tau changed         : False")
    print(f"Production selection made    : False")
    print()

    for row in scorecard.itertuples():
        print(
            f"{row.policy:18s} | "
            f"cleanFDR={row.actual_clean_false_degradation_rate:.3f} | "
            f"line={row.global_line_mild_detection:.3f}/"
            f"{row.global_line_medium_detection:.3f}/"
            f"{row.global_line_severe_detection:.3f} | "
            f"flat={row.flatline_mild_detection:.3f}/"
            f"{row.flatline_medium_detection:.3f}/"
            f"{row.flatline_severe_detection:.3f} | "
            f"T={row.threshold_mean:.3f}±{row.threshold_sd:.3f}"
        )

    print()
    print("Core outputs:")
    print(
        f"  {out_dir / 'eeg_v2_primary_alpha005_scorecard.csv'}"
    )
    print(
        f"  {out_dir / 'eeg_v2_line_transform_response.csv'}"
    )
    print(
        f"  {out_dir / 'eeg_v2_line_severity_monotonicity.csv'}"
    )
    print(
        f"  {out_dir / 'eeg_v2_threshold_stability.csv'}"
    )
    print(
        f"  {out_dir / 'eeg_v2_full_val_crossfit_threshold_candidates.csv'}"
    )
    print(
        f"  {out_dir / 'eeg_quality_hybrid_v2_summary.json'}"
    )
    print(
        f"  {out_dir / 'eeg_quality_hybrid_v2_candidate_artifact.json'}"
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
            f"\nEEG QUALITY HYBRID-v2 ERROR: {exc}",
            file=sys.stderr,
        )
        raise
