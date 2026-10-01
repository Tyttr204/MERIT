#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAV Distribution-Aware Quality Score Builder v1.0
=================================================

Purpose
-------
Build and compare *candidate* modality-level distribution-aware quality scores
from the already-completed formal balanced VAL robustness run.

This script is analysis-only. It does NOT modify the deployed EEG / Audio /
Video emotion heads, quality modules, calibrators, fusion checkpoints, or the
frozen tau=0.80 router.

Why this stage exists
---------------------
The previous audit established that several important raw continuous quality
features are reasonably modeled by Gaussian nominal distributions, while
discrete anomaly indicators and availability should remain separate.

The previous candidate-fitting stage also showed strong corruption-specific
responses:
  - Audio attenuation: RMS dBFS
  - Audio white noise: DNSMOS OVRL_raw
  - Video blur / brightness: DOVER technical_raw
  - EEG line noise: line_fraction_mean
  - EEG flatline: RMS + explicit flat-channel guards

A real robot, however, does not know the corruption family in advance. Therefore
runtime quality must calculate all relevant evidence simultaneously and combine
it without using the known corruption label.

Scientific design
-----------------
A) Continuous raw features
   Fit 5-s runtime candidate Gaussian parameters on CLEAN VAL windows:
       z = (x - mu_clean) / sigma_clean

   Convert to non-negative directional anomaly evidence:
       lower_is_worse : e = max(0, -z)
       higher_is_worse: e = max(0, +z)
       two_sided      : e = abs(z)

B) Discrete / bounded anomaly guards
   Do NOT force them into Gaussian models. Convert them to an equivalent
   one-sided empirical tail z-score using the CLEAN VAL reference distribution.

C) Availability
   Remains a separate hard state:
       unavailable -> candidate reliability = 0
   Gaussian / empirical scores never override missing/unavailable sensing.

D) Fault-agnostic aggregation
   Compare four strategies, all using the same evidence set:
       max
       top2_mean
       rms
       logmeanexp_beta2

   Known corruption family is used ONLY for retrospective evaluation, NEVER for
   score calculation.

E) Candidate reliability mappings
   Two monotone mappings are exported for study, not selected:
       gaussian_kernel:
           q = exp(-0.5 * anomaly^2)
       halfnormal_survival:
           q = 2 * Phi(-anomaly)

   No mapping is declared final in this script.

Important 5-s vs 20-s distinction
---------------------------------
The deployed system operates on 5-s windows. Therefore this script fits a
separate set of *runtime-candidate* 5-s Gaussian parameters from the 120 clean
VAL windows per modality.

This does NOT invalidate the earlier 20-s trial-mean audit. The 20-s trial means
were appropriate for distribution-shape diagnostics with reduced within-trial
dependence. The 5-s parameters here are necessary because a runtime score must
use the same temporal unit as deployment.

To avoid pseudo-replication in evaluation, strategy ranking is based primarily
on complete 20-s trial trajectories (four 5-s windows averaged per condition),
while subject-wise LOSO stability is also reported.

This script does NOT
--------------------
- use TEST data;
- train an emotion classifier or fusion network;
- change the existing quality calibrators;
- change tau=0.80;
- alter the existing final deployment path;
- choose a final aggregation strategy;
- choose a final reliability mapping;
- choose a final quality threshold;
- use corruption family as a score input.

Expected source files
---------------------
Formal robustness run:
  run_summary.json
  quality_numeric_features.csv
  quality_distribution_samples.csv

Previous candidate fit:
  distribution_quality_candidate_artifact.json

Recommended location
--------------------
最终部署/testing/analysis/build_distribution_aware_quality_score.py

Recommended command from 最终部署
---------------------------------
python -X utf8 .\\testing\\analysis\\build_distribution_aware_quality_score.py

If paths are omitted:
- latest completed formal balanced VAL robustness run is selected;
- latest distribution_candidate_fit_* artifact inside that run is selected.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "EAV-DIST-QUALITY-SCORE.1.0.2"
EXPECTED_SUBJECTS = 6
EXPECTED_CLEAN_WINDOWS_PER_MODALITY = 120
EXPECTED_CLEAN_TRIALS = 30
EVIDENCE_CAP = 12.0
LOSO_SEED = 20260924

STRATEGIES = (
    "max",
    "top2_mean",
    "rms",
    "logmeanexp_beta2",
)

MAPPINGS = (
    "gaussian_kernel",
    "halfnormal_survival",
)

CANDIDATE_THRESHOLDS = (0.20, 0.40, 0.60, 0.80)

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


class ScoreBuildError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ScoreBuildError(message)


def json_safe(obj: Any) -> Any:
    """Recursively convert objects to strict JSON-safe values.

    Important: json.dumps(default=...) is NOT called for an ordinary Python
    float, so float("nan") can bypass json_default and fail when
    allow_nan=False.  Sanitize recursively before serialization.
    """
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

    # Pandas may surface nullable scalar values such as pd.NA.
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # Last-resort support for objects exposing a scalar .item().
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
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON must be an object: {path}")
    return value


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
    raise ScoreBuildError(f"{name}: cannot parse boolean {value!r}")


def finite(values: Iterable[Any]) -> np.ndarray:
    result: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            result.append(x)
    return np.asarray(result, dtype=np.float64)


def import_stats():
    try:
        from scipy import stats
    except Exception as exc:
        raise ScoreBuildError("scipy is required.") from exc
    return stats


def import_auc():
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise ScoreBuildError("scikit-learn is required.") from exc
    return roc_auc_score


def discover_deployment_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "main.py").is_file() and (p / "system_checks").is_dir():
            return p
    raise ScoreBuildError(
        "Cannot locate 最终部署 root. Run from 最终部署 or pass --deployment-root."
    )


def discover_latest_formal_run(root: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    system_checks = root / "system_checks"
    require(system_checks.is_dir(), f"Missing system_checks: {system_checks}")

    for d in system_checks.iterdir():
        if not d.is_dir():
            continue
        summary_path = d / "run_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        if (
            summary.get("formal_balanced_val") is True
            and summary.get("status") == "PASS_OFFLINE_PROCESSING"
            and summary.get("suite") == "robustness"
            and summary.get("test_data_used") is False
        ):
            candidates.append((d.stat().st_mtime, d.resolve()))

    require(candidates, "No completed formal balanced VAL robustness run found.")
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def discover_latest_candidate_artifact(run_dir: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    for d in run_dir.glob("distribution_candidate_fit_*"):
        if not d.is_dir():
            continue
        artifact = d / "distribution_quality_candidate_artifact.json"
        if artifact.is_file():
            candidates.append((artifact.stat().st_mtime, artifact.resolve()))
    require(candidates, "No distribution candidate artifact found under source run.")
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def validate_run(run_dir: Path) -> dict:
    summary = read_json(run_dir / "run_summary.json")
    require(summary.get("status") == "PASS_OFFLINE_PROCESSING", "Source run is not PASS.")
    require(summary.get("formal_balanced_val") is True, "Source is not formal balanced VAL.")
    require(summary.get("suite") == "robustness", "Source is not robustness suite.")
    require(summary.get("test_data_used") is False, "TEST data used; refusing analysis.")
    require(summary.get("threshold_retuned") is False, "Source run retuned threshold.")
    require(
        summary.get("frozen_model_training_performed") is False,
        "Source run performed training.",
    )
    return summary


def validate_candidate_artifact(path: Path, run_dir: Path) -> dict:
    artifact = read_json(path)
    require(
        artifact.get("artifact_state") == "CANDIDATE_NOT_DEPLOYABLE",
        "Expected a non-deployable candidate artifact.",
    )
    require(artifact.get("test_data_used") is False, "Candidate artifact used TEST.")
    require(artifact.get("source_split") == "VAL_ONLY", "Candidate artifact is not VAL-only.")
    source_dir = Path(str(artifact.get("source_run_dir", ""))).resolve()
    require(
        source_dir == run_dir.resolve(),
        f"Candidate/source run mismatch: {source_dir} != {run_dir}",
    )
    router = artifact.get("router", {})
    require(router.get("tau_0_80_changed") is False, "Candidate changed tau=0.80.")
    require(router.get("fusion_weights_changed") is False, "Candidate changed fusion.")
    require(router.get("runtime_router_selected") is False, "Candidate selected runtime router.")
    return artifact


def normalize_grouping_strings(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
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
    f = normalize_grouping_strings(
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
    s = normalize_grouping_strings(
        s,
        ("source_window_id", "subject", "pair_key", "condition", "modality"),
    )
    s["available_bool"] = [explicit_bool(v, "sample available") for v in s["available"]]
    s["q_num"] = pd.to_numeric(s["q"], errors="coerce")
    return s


def continuous_spec_from_artifact(artifact: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    models = artifact.get("models")
    require(isinstance(models, dict), "Candidate artifact missing models.")
    spec: Dict[str, Dict[str, str]] = {}
    for modality, features in models.items():
        require(isinstance(features, dict), f"{modality}: models must be object.")
        spec[str(modality)] = {}
        for feature, block in features.items():
            require(isinstance(block, dict), f"{modality}/{feature}: model block invalid.")
            direction = str(block.get("direction", ""))
            require(
                direction in {"lower_is_worse", "higher_is_worse", "two_sided"},
                f"{modality}/{feature}: invalid direction {direction!r}",
            )
            spec[str(modality)][str(feature)] = direction
    return spec


def guard_spec_from_artifact(artifact: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    guards = artifact.get("discrete_guards", {})
    require(isinstance(guards, dict), "Candidate artifact discrete_guards invalid.")
    result: Dict[str, Dict[str, str]] = {}
    for modality, features in guards.items():
        require(isinstance(features, dict), "Guard modality block invalid.")
        result[str(modality)] = {}
        for feature, direction in features.items():
            direction = str(direction)
            require(
                direction in {"lower_is_worse", "higher_is_worse", "two_sided"},
                f"Invalid guard direction {modality}/{feature}: {direction}",
            )
            result[str(modality)][str(feature)] = direction
    return result


def normality_diagnostic(x: np.ndarray) -> Dict[str, Any]:
    stats = import_stats()
    x = finite(x)
    require(len(x) >= 8, "Too few values for normality diagnostic.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = stats.shapiro(x)
        ad = stats.anderson(x, dist="norm")
        levels = np.asarray(ad.significance_level, dtype=float)
        idx = int(np.argmin(np.abs(levels - 5.0)))
    return {
        "n": int(len(x)),
        "shapiro_W": float(sh.statistic),
        "shapiro_p": float(sh.pvalue),
        "anderson_statistic": float(ad.statistic),
        "anderson_5pct_critical": float(ad.critical_values[idx]),
        "anderson_reject_at_5pct_diagnostic": bool(ad.statistic > ad.critical_values[idx]),
        "diagnostic_only": True,
    }


def fit_runtime_gaussians(
    features: pd.DataFrame,
    continuous_spec: Mapping[str, Mapping[str, str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[dict] = []
    artifact: Dict[str, Any] = {}

    for modality, feature_spec in continuous_spec.items():
        artifact[modality] = {}
        for feature, direction in feature_spec.items():
            g = features[
                (features["condition"] == "reference")
                & (features["modality"] == modality)
                & (features["feature_path"] == feature)
                & (features["available_bool"])
            ].copy()

            # One numeric scalar per 5-s source window for this feature.
            w = (
                g.groupby(
                    ["source_window_id", "subject", "pair_key"],
                    as_index=False,
                    dropna=False,
                )
                .agg(value=("value_num", "mean"))
            )

            x = finite(w["value"])
            require(
                len(x) == EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
                f"{modality}/{feature}: expected 120 clean 5-s windows, got {len(x)}",
            )
            require(
                w["subject"].nunique() == EXPECTED_SUBJECTS,
                f"{modality}/{feature}: expected 6 subjects.",
            )

            mu = float(np.mean(x))
            sigma = float(np.std(x, ddof=1))
            require(sigma > 0 and math.isfinite(sigma), f"{modality}/{feature}: sigma invalid.")

            diag = normality_diagnostic(x)
            row = {
                "modality": modality,
                "feature_path": feature,
                "direction": direction,
                "fit_unit": "clean_available_5s_window",
                "n_windows": len(x),
                "n_subjects": int(w["subject"].nunique()),
                "mu": mu,
                "sigma_sample": sigma,
                "median": float(np.median(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                **diag,
                "runtime_selected": False,
            }
            rows.append(row)
            artifact[modality][feature] = dict(row)

    return pd.DataFrame(rows), artifact


def gaussian_loso_subject_stability(
    features: pd.DataFrame,
    runtime_params: pd.DataFrame,
) -> pd.DataFrame:
    stats = import_stats()
    rows: List[dict] = []

    for _, param in runtime_params.iterrows():
        modality = str(param["modality"])
        feature = str(param["feature_path"])
        g = features[
            (features["condition"] == "reference")
            & (features["modality"] == modality)
            & (features["feature_path"] == feature)
            & (features["available_bool"])
        ].copy()
        w = (
            g.groupby(
                ["source_window_id", "subject", "pair_key"],
                as_index=False,
                dropna=False,
            )
            .agg(value=("value_num", "mean"))
        )
        subjects = sorted(w["subject"].unique().tolist())

        fold_mus = []
        fold_sigmas = []
        all_nll: List[float] = []

        for held in subjects:
            train = finite(w.loc[w["subject"] != held, "value"])
            test = finite(w.loc[w["subject"] == held, "value"])
            mu = float(np.mean(train))
            sigma = float(np.std(train, ddof=1))
            require(sigma > 0, f"{modality}/{feature}/{held}: LOSO sigma invalid.")
            nll = -stats.norm.logpdf(test, loc=mu, scale=sigma)
            require(np.all(np.isfinite(nll)), "Non-finite LOSO NLL.")
            fold_mus.append(mu)
            fold_sigmas.append(sigma)
            all_nll.extend(nll.tolist())
            rows.append({
                "level": "fold",
                "modality": modality,
                "feature_path": feature,
                "held_subject": held,
                "n_train_windows": len(train),
                "n_test_windows": len(test),
                "mu_train": mu,
                "sigma_train": sigma,
                "mean_test_nll": float(np.mean(nll)),
                "runtime_selected": False,
            })

        rows.append({
            "level": "aggregate",
            "modality": modality,
            "feature_path": feature,
            "held_subject": "",
            "n_train_windows": None,
            "n_test_windows": len(all_nll),
            "mu_train": float(np.mean(fold_mus)),
            "sigma_train": float(np.mean(fold_sigmas)),
            "mean_test_nll": float(np.mean(all_nll)),
            "mu_fold_sd": float(np.std(fold_mus, ddof=1)),
            "sigma_fold_sd": float(np.std(fold_sigmas, ddof=1)),
            "runtime_selected": False,
        })

    return pd.DataFrame(rows)


def fit_guard_empirical_reference(
    features: pd.DataFrame,
    guard_spec: Mapping[str, Mapping[str, str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[dict] = []
    artifact: Dict[str, Any] = {}

    for modality, feature_spec in guard_spec.items():
        artifact[modality] = {}
        for feature, direction in feature_spec.items():
            g = features[
                (features["condition"] == "reference")
                & (features["modality"] == modality)
                & (features["feature_path"] == feature)
                & (features["available_bool"])
            ].copy()
            w = (
                g.groupby(
                    ["source_window_id", "subject", "pair_key"],
                    as_index=False,
                    dropna=False,
                )
                .agg(value=("value_num", "mean"))
            )
            x = finite(w["value"])
            if len(x) == 0:
                rows.append({
                    "modality": modality,
                    "feature_path": feature,
                    "direction": direction,
                    "status": "MISSING",
                    "runtime_selected": False,
                })
                continue

            require(
                len(x) == EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
                f"{modality}/{feature}: expected 120 clean guard windows, got {len(x)}",
            )
            sorted_values = np.sort(x)
            row = {
                "modality": modality,
                "feature_path": feature,
                "direction": direction,
                "status": "OK",
                "n_windows": len(x),
                "n_subjects": int(w["subject"].nunique()),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "q95": float(np.quantile(x, 0.95)),
                "q99": float(np.quantile(x, 0.99)),
                "unique_values": int(len(np.unique(x))),
                "gaussian_assumption_used": False,
                "runtime_selected": False,
            }
            rows.append(row)
            artifact[modality][feature] = {
                **row,
                "clean_sorted_values": sorted_values.tolist(),
                "empirical_tail_to_z_equivalent": True,
            }

    return pd.DataFrame(rows), artifact


def directional_continuous_evidence(
    value: float,
    mu: float,
    sigma: float,
    direction: str,
) -> float:
    z = (float(value) - float(mu)) / float(sigma)
    if direction == "lower_is_worse":
        evidence = max(0.0, -z)
    elif direction == "higher_is_worse":
        evidence = max(0.0, z)
    elif direction == "two_sided":
        evidence = abs(z)
    else:
        raise ScoreBuildError(f"Unknown direction: {direction}")
    return float(min(EVIDENCE_CAP, evidence))


def empirical_guard_evidence(
    value: float,
    clean_sorted: np.ndarray,
    direction: str,
) -> float:
    """
    Convert a discrete/bounded guard to a one-sided z-equivalent anomaly using
    an empirical clean tail probability with a +1 finite-sample correction.
    """
    stats = import_stats()
    clean = np.asarray(clean_sorted, dtype=np.float64)
    require(len(clean) >= 8, "Too few clean guard values.")

    x = float(value)
    n = len(clean)

    if direction == "higher_is_worse":
        tail_count = int(np.sum(clean >= x))
        p = (1.0 + tail_count) / (n + 1.0)
    elif direction == "lower_is_worse":
        tail_count = int(np.sum(clean <= x))
        p = (1.0 + tail_count) / (n + 1.0)
    elif direction == "two_sided":
        lo = (1.0 + int(np.sum(clean <= x))) / (n + 1.0)
        hi = (1.0 + int(np.sum(clean >= x))) / (n + 1.0)
        p = min(1.0, 2.0 * min(lo, hi))
    else:
        raise ScoreBuildError(f"Unknown guard direction: {direction}")

    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    z_equiv = float(stats.norm.isf(p))
    return float(min(EVIDENCE_CAP, max(0.0, z_equiv)))


def aggregate_evidence(values: np.ndarray, strategy: str) -> float:
    e = np.asarray(values, dtype=np.float64)
    e = e[np.isfinite(e)]
    require(len(e) > 0, "Cannot aggregate empty evidence.")
    e = np.clip(e, 0.0, EVIDENCE_CAP)

    if strategy == "max":
        return float(np.max(e))

    if strategy == "top2_mean":
        k = min(2, len(e))
        return float(np.mean(np.sort(e)[-k:]))

    if strategy == "rms":
        return float(np.sqrt(np.mean(e ** 2)))

    if strategy == "logmeanexp_beta2":
        beta = 2.0
        m = float(np.max(e))
        value = m + math.log(float(np.mean(np.exp(beta * (e - m))))) / beta
        return float(max(0.0, value))

    raise ScoreBuildError(f"Unknown aggregation strategy: {strategy}")


def reliability_mapping(anomaly: float, mapping: str) -> float:
    stats = import_stats()
    a = max(0.0, float(anomaly))
    if mapping == "gaussian_kernel":
        q = math.exp(-0.5 * a * a)
    elif mapping == "halfnormal_survival":
        q = 2.0 * float(stats.norm.sf(a))
    else:
        raise ScoreBuildError(f"Unknown mapping: {mapping}")
    return float(np.clip(q, 0.0, 1.0))


def make_feature_lookup(
    features: pd.DataFrame,
) -> Dict[Tuple[str, str, str, str], float]:
    """
    key = (source_window_id, condition, modality, feature_path)

    CONDITION MUST BE PART OF THE KEY.

    The formal robustness suite reuses the same source_window_id across
    reference and all synthetic corruption conditions.  Omitting condition
    would average the raw feature over reference/mild/medium/severe/missing
    variants and then feed that same averaged value back to every condition.
    That destroys the degradation signal and produces the pathological
    pattern: AUC≈0.5, undefined Spearman, yet "monotonic"=1 because all
    condition scores are tied.
    """
    grouped = (
        features[features["available_bool"]]
        .groupby(
            ["source_window_id", "condition", "modality", "feature_path"],
            as_index=False,
            dropna=False,
        )
        .agg(value=("value_num", "mean"))
    )

    # A fully qualified window/condition/modality/feature tuple must be unique.
    duplicated = grouped.duplicated(
        subset=["source_window_id", "condition", "modality", "feature_path"],
        keep=False,
    )
    require(
        not bool(duplicated.any()),
        "Feature lookup still contains duplicate fully-qualified keys.",
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


def build_window_scores(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    runtime_params: pd.DataFrame,
    guard_artifact: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    param_lookup = {
        (str(r.modality), str(r.feature_path)): {
            "mu": float(r.mu),
            "sigma": float(r.sigma_sample),
            "direction": str(r.direction),
        }
        for r in runtime_params.itertuples()
    }

    feature_lookup = make_feature_lookup(features)

    # Use one sample row per window/modality/condition.
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

    evidence_rows: List[dict] = []
    score_rows: List[dict] = []

    for row in base.itertuples():
        wid = str(row.source_window_id)
        modality = str(row.modality)
        available = bool(row.available)

        continuous_e: List[float] = []
        guard_e: List[float] = []

        # Continuous Gaussian evidence.
        for (m, feature), p in param_lookup.items():
            if m != modality:
                continue
            key = (wid, str(row.condition), modality, feature)
            if key not in feature_lookup:
                continue
            value = feature_lookup[key]
            e = directional_continuous_evidence(
                value,
                p["mu"],
                p["sigma"],
                p["direction"],
            )
            continuous_e.append(e)
            evidence_rows.append({
                "source_window_id": wid,
                "subject": str(row.subject),
                "pair_key": str(row.pair_key),
                "condition": str(row.condition),
                "modality": modality,
                "available": available,
                "evidence_type": "continuous_gaussian",
                "feature_path": feature,
                "direction": p["direction"],
                "raw_value": value,
                "anomaly_evidence_z_equivalent": e,
            })

        # Discrete empirical-tail guards.
        modality_guards = guard_artifact.get(modality, {})
        if isinstance(modality_guards, dict):
            for feature, g in modality_guards.items():
                if not isinstance(g, dict) or g.get("status") != "OK":
                    continue
                key = (wid, str(row.condition), modality, feature)
                if key not in feature_lookup:
                    continue
                value = feature_lookup[key]
                clean_sorted = np.asarray(g["clean_sorted_values"], dtype=np.float64)
                direction = str(g["direction"])
                e = empirical_guard_evidence(value, clean_sorted, direction)
                guard_e.append(e)
                evidence_rows.append({
                    "source_window_id": wid,
                    "subject": str(row.subject),
                    "pair_key": str(row.pair_key),
                    "condition": str(row.condition),
                    "modality": modality,
                    "available": available,
                    "evidence_type": "discrete_empirical_guard",
                    "feature_path": feature,
                    "direction": direction,
                    "raw_value": value,
                    "anomaly_evidence_z_equivalent": e,
                })

        all_e = np.asarray(continuous_e + guard_e, dtype=np.float64)

        for strategy in STRATEGIES:
            if not available:
                anomaly = None
                q_kernel = 0.0
                q_survival = 0.0
                status = "UNAVAILABLE"
            elif len(all_e) == 0:
                anomaly = None
                q_kernel = None
                q_survival = None
                status = "NO_QUALITY_EVIDENCE"
            else:
                anomaly = aggregate_evidence(all_e, strategy)
                q_kernel = reliability_mapping(anomaly, "gaussian_kernel")
                q_survival = reliability_mapping(anomaly, "halfnormal_survival")
                status = "OK"

            score_rows.append({
                "source_window_id": wid,
                "subject": str(row.subject),
                "pair_key": str(row.pair_key),
                "condition": str(row.condition),
                "modality": modality,
                "available": available,
                "existing_q": (
                    float(row.existing_q)
                    if row.existing_q is not None and math.isfinite(float(row.existing_q))
                    else None
                ),
                "strategy": strategy,
                "n_continuous_evidence": len(continuous_e),
                "n_guard_evidence": len(guard_e),
                "n_total_evidence": len(all_e),
                "anomaly_score": anomaly,
                "q_gaussian_kernel": q_kernel,
                "q_halfnormal_survival": q_survival,
                "status": status,
                "family_used_as_score_input": False,
            })

    return pd.DataFrame(evidence_rows), pd.DataFrame(score_rows)


def build_trial_scores(window_scores: pd.DataFrame) -> pd.DataFrame:
    ok = window_scores[window_scores["status"].isin(["OK", "UNAVAILABLE"])].copy()

    grouped = (
        ok.groupby(
            ["subject", "pair_key", "condition", "modality", "strategy"],
            as_index=False,
            dropna=False,
        )
        .agg(
            windows=("source_window_id", "nunique"),
            available_fraction=("available", "mean"),
            mean_anomaly_score=("anomaly_score", "mean"),
            mean_q_gaussian_kernel=("q_gaussian_kernel", "mean"),
            mean_q_halfnormal_survival=("q_halfnormal_survival", "mean"),
            mean_existing_q=("existing_q", "mean"),
        )
    )
    return grouped


def evaluate_aggregation_strategies(
    trial_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stats = import_stats()
    roc_auc_score = import_auc()

    family_rows: List[dict] = []

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])
        reference = str(spec["reference_condition"])
        conditions = [str(x) for x in spec["conditions"]]
        ordered = [reference] + conditions

        for strategy in STRATEGIES:
            g = trial_scores[
                (trial_scores["modality"] == modality)
                & (trial_scores["strategy"] == strategy)
                & (trial_scores["condition"].isin(ordered))
            ].copy()

            pivot = g.pivot_table(
                index=["subject", "pair_key"],
                columns="condition",
                values="mean_anomaly_score",
                aggfunc="mean",
            )
            complete = pivot.dropna(subset=ordered).copy()
            if complete.empty:
                family_rows.append({
                    "family": family,
                    "modality": modality,
                    "strategy": strategy,
                    "status": "NO_COMPLETE_TRAJECTORIES",
                    "deployment_selected": False,
                })
                continue

            arr = complete[ordered].to_numpy(dtype=float)
            diffs = np.diff(arr, axis=1)
            spearmans: List[float] = []
            for row in arr:
                rho = stats.spearmanr(np.arange(4, dtype=float), row).statistic
                if math.isfinite(float(rho)):
                    spearmans.append(float(rho))

            severe_diff = arr[:, -1] - arr[:, 0]
            y = np.concatenate([
                np.zeros(len(arr), dtype=int),
                np.ones(len(arr), dtype=int),
            ])
            score = np.concatenate([arr[:, 0], arr[:, -1]])
            auc = float(roc_auc_score(y, score))

            family_rows.append({
                "family": family,
                "modality": modality,
                "strategy": strategy,
                "status": "OK",
                "n_complete_trials": int(len(complete)),
                "mean_spearman_severity_vs_anomaly": float(np.mean(spearmans)),
                "positive_spearman_rate": float(np.mean(np.asarray(spearmans) > 0)),
                "monotonic_nondecreasing_rate": float(
                    np.mean(np.all(diffs >= -1e-12, axis=1))
                ),
                "severe_worse_than_reference_rate": float(
                    np.mean(arr[:, -1] > arr[:, 0])
                ),
                "mean_severe_minus_reference_anomaly": float(np.mean(severe_diff)),
                "median_severe_minus_reference_anomaly": float(np.median(severe_diff)),
                "reference_vs_severe_auc": auc,
                "deployment_selected": False,
            })

    family_df = pd.DataFrame(family_rows)

    modality_rows: List[dict] = []
    ok = family_df[family_df["status"] == "OK"].copy()

    for modality in sorted(ok["modality"].unique().tolist()):
        for strategy in STRATEGIES:
            g = ok[
                (ok["modality"] == modality)
                & (ok["strategy"] == strategy)
            ].copy()
            if g.empty:
                continue
            modality_rows.append({
                "modality": modality,
                "strategy": strategy,
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
                "deployment_selected": False,
            })

    modality_df = pd.DataFrame(modality_rows)
    if not modality_df.empty:
        modality_df["diagnostic_rank"] = np.nan
        for modality, idx in modality_df.groupby("modality").groups.items():
            local = modality_df.loc[idx].sort_values(
                [
                    "mean_family_spearman",
                    "worst_family_spearman",
                    "mean_monotonic_rate",
                    "mean_reference_vs_severe_auc",
                    "strategy",
                ],
                ascending=[False, False, False, False, True],
            )
            for rank, ridx in enumerate(local.index, start=1):
                modality_df.loc[ridx, "diagnostic_rank"] = rank

    return family_df, modality_df


def mapping_threshold_response(window_scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    mapping_cols = {
        "gaussian_kernel": "q_gaussian_kernel",
        "halfnormal_survival": "q_halfnormal_survival",
    }

    for (condition, modality, strategy), g in window_scores.groupby(
        ["condition", "modality", "strategy"], dropna=False
    ):
        for mapping, col in mapping_cols.items():
            q = pd.to_numeric(g[col], errors="coerce")
            valid = q.notna()
            qv = q[valid].to_numpy(dtype=float)
            if len(qv) == 0:
                continue
            for threshold in CANDIDATE_THRESHOLDS:
                rows.append({
                    "condition": condition,
                    "modality": modality,
                    "strategy": strategy,
                    "mapping": mapping,
                    "threshold": threshold,
                    "n_windows": int(len(qv)),
                    "mean_candidate_q": float(np.mean(qv)),
                    "median_candidate_q": float(np.median(qv)),
                    "fraction_below_threshold": float(np.mean(qv < threshold)),
                    "threshold_selected": False,
                })

    return pd.DataFrame(rows)


def existing_q_comparison(
    window_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for (condition, modality, strategy), g in window_scores.groupby(
        ["condition", "modality", "strategy"], dropna=False
    ):
        existing = pd.to_numeric(g["existing_q"], errors="coerce").to_numpy(dtype=float)
        newq = pd.to_numeric(g["q_gaussian_kernel"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(existing) & np.isfinite(newq)
        if not np.any(mask):
            continue
        x = existing[mask]
        y = newq[mask]
        corr = (
            float(np.corrcoef(x, y)[0, 1])
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0
            else None
        )
        rows.append({
            "condition": condition,
            "modality": modality,
            "strategy": strategy,
            "n": int(np.sum(mask)),
            "mean_existing_q": float(np.mean(x)),
            "mean_candidate_q_gaussian_kernel": float(np.mean(y)),
            "pearson_existing_vs_candidate": corr,
            "not_used_for_selection": True,
        })
    return pd.DataFrame(rows)


def create_plots(
    out_dir: Path,
    trial_scores: pd.DataFrame,
    family_eval: pd.DataFrame,
    modality_ranking: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ScoreBuildError("matplotlib is required for plots.") from exc

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=False)

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])
        ordered = [str(spec["reference_condition"])] + [
            str(x) for x in spec["conditions"]
        ]
        g = trial_scores[
            (trial_scores["modality"] == modality)
            & (trial_scores["condition"].isin(ordered))
        ].copy()
        if g.empty:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for strategy in STRATEGIES:
            s = g[g["strategy"] == strategy]
            means = []
            for condition in ordered:
                c = s[s["condition"] == condition]
                means.append(
                    float(c["mean_anomaly_score"].mean())
                    if len(c) else np.nan
                )
            ax.plot([0, 1, 2, 3], means, marker="o", label=strategy)

        ax.set_xticks([0, 1, 2, 3], ["reference", "mild", "medium", "severe"])
        ax.set_ylabel("mean trial anomaly score")
        ax.set_xlabel("controlled degradation severity")
        ax.set_title(f"{family}: fault-agnostic aggregation candidates")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / f"{family}_aggregation_by_severity.png", dpi=170)
        plt.close(fig)

    if not modality_ranking.empty:
        for modality in sorted(modality_ranking["modality"].unique()):
            g = modality_ranking[
                modality_ranking["modality"] == modality
            ].sort_values("diagnostic_rank")
            fig = plt.figure(figsize=(7, 4.5))
            ax = fig.add_subplot(111)
            x = np.arange(len(g))
            ax.bar(x, g["mean_family_spearman"].to_numpy(dtype=float))
            ax.set_xticks(x, g["strategy"].tolist(), rotation=25, ha="right")
            ax.set_ylim(-1.0, 1.05)
            ax.set_ylabel("mean family Spearman")
            ax.set_title(f"{modality.upper()}: aggregation diagnostic ranking")
            fig.tight_layout()
            fig.savefig(plots / f"{modality}_aggregation_ranking.png", dpi=170)
            plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build and compare fault-agnostic distribution-aware modality quality "
            "scores on formal VAL only; does not modify deployment."
        )
    )
    p.add_argument("--deployment-root")
    p.add_argument("--run-dir")
    p.add_argument("--candidate-artifact")
    p.add_argument("--output-dir")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def self_test() -> int:
    stats = import_stats()

    clean = np.asarray([0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1], dtype=float)
    guard_normal = empirical_guard_evidence(0.0, np.sort(clean), "higher_is_worse")
    guard_bad = empirical_guard_evidence(0.8, np.sort(clean), "higher_is_worse")

    e = np.asarray([0.0, 1.0, 2.0], dtype=float)
    aggregates = {name: aggregate_evidence(e, name) for name in STRATEGIES}

    q0k = reliability_mapping(0.0, "gaussian_kernel")
    q2k = reliability_mapping(2.0, "gaussian_kernel")
    q0s = reliability_mapping(0.0, "halfnormal_survival")
    q2s = reliability_mapping(2.0, "halfnormal_survival")

    # Regression test: the formal robustness suite deliberately reuses the
    # same source_window_id for reference and corrupted copies.  Condition must
    # therefore remain in the feature-lookup identity.
    mini_features = pd.DataFrame([
        {
            "source_window_id": "subject08_instance001_w01",
            "subject": "subject08",
            "pair_key": "subject08_instance001",
            "condition": "reference",
            "family": "",
            "severity": "",
            "modality": "audio",
            "available": True,
            "feature_path": "dnsmos.OVRL_raw",
            "value": 3.5,
        },
        {
            "source_window_id": "subject08_instance001_w01",
            "subject": "subject08",
            "pair_key": "subject08_instance001",
            "condition": "audio_white_noise_severe",
            "family": "audio_white_noise",
            "severity": "severe",
            "modality": "audio",
            "available": True,
            "feature_path": "dnsmos.OVRL_raw",
            "value": 1.2,
        },
    ])
    mini_prepared = mini_features.copy()
    mini_prepared = normalize_grouping_strings(
        mini_prepared,
        (
            "source_window_id", "subject", "pair_key", "condition",
            "family", "severity", "modality", "feature_path",
        ),
    )
    mini_prepared["available_bool"] = True
    mini_prepared["value_num"] = pd.to_numeric(mini_prepared["value"])
    mini_lookup = make_feature_lookup(mini_prepared)
    condition_key_separation = (
        len(mini_lookup) == 2
        and math.isclose(
            mini_lookup[
                (
                    "subject08_instance001_w01",
                    "reference",
                    "audio",
                    "dnsmos.OVRL_raw",
                )
            ],
            3.5,
        )
        and math.isclose(
            mini_lookup[
                (
                    "subject08_instance001_w01",
                    "audio_white_noise_severe",
                    "audio",
                    "dnsmos.OVRL_raw",
                )
            ],
            1.2,
        )
    )

    checks = {
        "continuous_lower_direction": (
            directional_continuous_evidence(8.0, 10.0, 2.0, "lower_is_worse")
            >
            directional_continuous_evidence(10.0, 10.0, 2.0, "lower_is_worse")
        ),
        "guard_bad_exceeds_clean": guard_bad > guard_normal,
        "all_aggregations_nonnegative": all(v >= 0 for v in aggregates.values()),
        "max_is_two": math.isclose(aggregates["max"], 2.0),
        "kernel_decreases": q0k > q2k,
        "survival_decreases": q0s > q2s,
        "zero_anomaly_maps_to_one": (
            math.isclose(q0k, 1.0) and math.isclose(q0s, 1.0)
        ),
        "family_count": len(FAMILY_SPECS) == 6,
        "strategy_count": len(STRATEGIES) == 4,
        "condition_specific_feature_lookup": condition_key_separation,
        "strict_json_nan_sanitized": (
            '"x": null' in json_text({"x": float("nan")})
            and '"y": null' in json_text({"y": np.float64(np.inf)})
        ),
    }
    require(all(checks.values()), f"Self-test failed: {checks}")

    print(json_text({
        "status": "PASS",
        "version": VERSION,
        "checks": checks,
        "real_EAV_used": False,
        "models_loaded": False,
        "router_changed": False,
        "runtime_score_selected": False,
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
        except ScoreBuildError:
            deployment_root = discover_deployment_root(Path(__file__).resolve().parent)

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else discover_latest_formal_run(deployment_root)
    )
    require(run_dir.is_dir(), f"Run directory missing: {run_dir}")
    run_summary = validate_run(run_dir)

    candidate_path = (
        Path(args.candidate_artifact).expanduser().resolve()
        if args.candidate_artifact
        else discover_latest_candidate_artifact(run_dir)
    )
    candidate = validate_candidate_artifact(candidate_path, run_dir)

    feature_path = run_dir / "quality_numeric_features.csv"
    sample_path = run_dir / "quality_distribution_samples.csv"
    require(feature_path.is_file(), f"Missing {feature_path}")
    require(sample_path.is_file(), f"Missing {sample_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / f"distribution_score_build_{timestamp}"
    )
    require(not out.exists(), f"Output exists; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)

    features = prepare_features(feature_path)
    samples = prepare_samples(sample_path)
    continuous_spec = continuous_spec_from_artifact(candidate)
    guard_spec = guard_spec_from_artifact(candidate)

    # Formal geometry checks on source samples.
    clean_samples = samples[samples["condition"] == "reference"].copy()
    for modality in sorted(continuous_spec):
        g = clean_samples[clean_samples["modality"] == modality]
        require(
            g["source_window_id"].nunique() == EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
            f"{modality}: expected 120 clean source windows.",
        )
        require(
            g["subject"].nunique() == EXPECTED_SUBJECTS,
            f"{modality}: expected 6 clean subjects.",
        )
        require(
            g["pair_key"].nunique() == EXPECTED_CLEAN_TRIALS,
            f"{modality}: expected 30 clean trials.",
        )

    runtime_params_df, runtime_param_artifact = fit_runtime_gaussians(
        features,
        continuous_spec,
    )
    loso_df = gaussian_loso_subject_stability(features, runtime_params_df)
    guard_df, guard_artifact = fit_guard_empirical_reference(features, guard_spec)

    evidence_df, window_scores_df = build_window_scores(
        samples,
        features,
        runtime_params_df,
        guard_artifact,
    )
    trial_scores_df = build_trial_scores(window_scores_df)

    family_eval_df, modality_ranking_df = evaluate_aggregation_strategies(
        trial_scores_df
    )
    threshold_df = mapping_threshold_response(window_scores_df)
    existing_compare_df = existing_q_comparison(window_scores_df)

    atomic_csv(out / "runtime_5s_gaussian_parameters.csv", runtime_params_df)
    atomic_csv(out / "runtime_5s_gaussian_loso_stability.csv", loso_df)
    atomic_csv(out / "guard_clean_empirical_reference.csv", guard_df)
    atomic_csv(out / "window_evidence_long.csv", evidence_df)
    atomic_csv(out / "window_distribution_scores.csv", window_scores_df)
    atomic_csv(out / "trial_distribution_scores.csv", trial_scores_df)
    atomic_csv(out / "aggregation_family_validation.csv", family_eval_df)
    atomic_csv(out / "aggregation_modality_ranking.csv", modality_ranking_df)
    atomic_csv(out / "mapping_threshold_response.csv", threshold_df)
    atomic_csv(out / "existing_q_comparison.csv", existing_compare_df)

    if not args.no_plots:
        create_plots(out, trial_scores_df, family_eval_df, modality_ranking_df)

    candidate_artifact = {
        "schema": "eav.distribution_quality_score_candidate.v1",
        "version": VERSION,
        "artifact_state": "CANDIDATE_NOT_DEPLOYABLE",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_candidate_artifact": str(candidate_path),
        "source_split": "VAL_ONLY",
        "test_data_used": False,
        "runtime_unit": "5s_window",
        "runtime_gaussian_models": runtime_param_artifact,
        "empirical_discrete_guards": guard_artifact,
        "availability": {
            "separate_hard_state": True,
            "unavailable_candidate_quality": 0.0,
            "distribution_score_never_overrides_unavailable": True,
        },
        "evidence": {
            "continuous": {
                "formula": "directional positive Gaussian z anomaly",
                "cap": EVIDENCE_CAP,
            },
            "discrete_guard": {
                "formula": "one-sided empirical clean-tail z-equivalent anomaly",
                "gaussian_assumption": False,
                "cap": EVIDENCE_CAP,
            },
            "corruption_family_is_runtime_input": False,
        },
        "aggregation_strategies_evaluated": list(STRATEGIES),
        "reliability_mappings_evaluated": {
            "gaussian_kernel": "exp(-0.5 * anomaly^2)",
            "halfnormal_survival": "2 * Phi(-anomaly)",
        },
        "candidate_thresholds_evaluated_descriptively": list(CANDIDATE_THRESHOLDS),
        "selection": {
            "aggregation_strategy_selected": False,
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
            "5-s runtime parameters are fit on clean VAL windows because runtime operates on 5-s windows.",
            "Trial-level paired trajectories are primary for corruption-severity strategy evaluation.",
            "Window rows are correlated within trials/subjects; no naive independence claim is made.",
            "Discrete guards are not Gaussianized; empirical tails are used.",
            "Availability is a separate hard state.",
            "Known corruption family is never used in score calculation.",
            "No TEST data, threshold retuning, emotion-model training, or fusion-model changes occur.",
        ],
        "source_hashes": {
            "run_summary": sha256_file(run_dir / "run_summary.json"),
            "quality_numeric_features": sha256_file(feature_path),
            "quality_distribution_samples": sha256_file(sample_path),
            "candidate_artifact": sha256_file(candidate_path),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(
        out / "distribution_quality_score_candidate_artifact.json",
        candidate_artifact,
    )

    # Compact diagnostic tops, explicitly not deployment selections.
    tops = {}
    for modality in sorted(modality_ranking_df["modality"].unique()):
        g = modality_ranking_df[
            modality_ranking_df["modality"] == modality
        ].sort_values("diagnostic_rank")
        tops[modality] = (
            g[
                [
                    "diagnostic_rank",
                    "strategy",
                    "mean_family_spearman",
                    "worst_family_spearman",
                    "mean_monotonic_rate",
                    "mean_reference_vs_severe_auc",
                ]
            ].to_dict(orient="records")
        )

    summary = {
        "version": VERSION,
        "status": "PASS",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_candidate_artifact": str(candidate_path),
        "formal_balanced_val": True,
        "test_data_used": False,
        "runtime_unit": "5s_window",
        "clean_subjects": EXPECTED_SUBJECTS,
        "clean_trials": EXPECTED_CLEAN_TRIALS,
        "clean_windows_per_modality": EXPECTED_CLEAN_WINDOWS_PER_MODALITY,
        "continuous_runtime_features_fitted": int(len(runtime_params_df)),
        "empirical_guard_features_fitted": int(
            (guard_df["status"] == "OK").sum()
        ) if len(guard_df) else 0,
        "window_score_rows": int(len(window_scores_df)),
        "trial_score_rows": int(len(trial_scores_df)),
        "diagnostic_aggregation_order": tops,
        "deployment_selection_made": False,
        "router_threshold_retuned": False,
        "models_or_existing_calibrators_modified": False,
        "candidate_artifact": str(
            out / "distribution_quality_score_candidate_artifact.json"
        ),
        "packages": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "matplotlib": package_version("matplotlib"),
        },
        "outputs": sorted(p.name for p in out.iterdir()),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    atomic_json(out / "quality_score_build_summary.json", summary)

    lines = [
        "# EAV Distribution-Aware Quality Score Build",
        "",
        f"- Version: `{VERSION}`",
        f"- Source run: `{run_dir}`",
        f"- Source candidate: `{candidate_path}`",
        "- Scope: formal balanced **VAL only**",
        "- Runtime unit: **5-s window**",
        "- Existing tau=0.80 changed: **No**",
        "- Existing quality calibrators changed: **No**",
        "- Emotion/Fusion models changed: **No**",
        "- Deployment score selected: **No**",
        "",
        "## Candidate score architecture",
        "",
        "Continuous raw feature -> 5-s clean Gaussian z anomaly",
        "",
        "Discrete guard -> clean empirical-tail z-equivalent anomaly",
        "",
        "Availability -> independent hard state",
        "",
        "All evidence -> fault-agnostic aggregation -> candidate reliability mapping",
        "",
        "## Diagnostic aggregation ranking",
        "",
    ]
    for modality in sorted(tops):
        lines.append(f"### {modality}")
        for r in tops[modality]:
            lines.append(
                f"- rank {int(r['diagnostic_rank'])}: `{r['strategy']}` — "
                f"mean rho={r['mean_family_spearman']:.3f}, "
                f"worst rho={r['worst_family_spearman']:.3f}, "
                f"monotonic={r['mean_monotonic_rate']:.3f}, "
                f"AUC={r['mean_reference_vs_severe_auc']:.3f}"
            )
        lines.append("")

    lines += [
        "## Next design gate",
        "",
        "Review `aggregation_modality_ranking.csv`, "
        "`aggregation_family_validation.csv`, and "
        "`mapping_threshold_response.csv`.",
        "",
        "Only after one aggregation strategy and one mapping are frozen on VAL "
        "should a separate runtime module be implemented under "
        "`最终部署/Quality/` and compared against the existing tau=0.80 router.",
        "",
    ]
    atomic_text(out / "README_quality_score_build.md", "\n".join(lines))

    print("=" * 116)
    print("EAV DISTRIBUTION-AWARE QUALITY SCORE BUILD")
    print("=" * 116)
    print(f"Status                     : PASS")
    print(f"Version                    : {VERSION}")
    print(f"Source formal run          : {run_dir}")
    print(f"Source candidate artifact  : {candidate_path}")
    print(f"Output                     : {out}")
    print(f"Runtime unit               : 5s window")
    print(f"Clean subjects / trials    : {EXPECTED_SUBJECTS} / {EXPECTED_CLEAN_TRIALS}")
    print(f"Clean windows/modality     : {EXPECTED_CLEAN_WINDOWS_PER_MODALITY}")
    print(f"Continuous features fitted : {len(runtime_params_df)}")
    print(
        "Empirical guards fitted    : "
        f"{int((guard_df['status'] == 'OK').sum()) if len(guard_df) else 0}"
    )
    print("TEST used                  : False")
    print("Existing tau changed       : False")
    print("Runtime score selected     : False")
    print()
    for modality in sorted(tops):
        first = tops[modality][0] if tops[modality] else None
        if first:
            print(
                f"{modality.upper():5s} diagnostic top         : "
                f"{first['strategy']} | "
                f"mean rho={first['mean_family_spearman']:.3f} | "
                f"worst rho={first['worst_family_spearman']:.3f} | "
                f"AUC={first['mean_reference_vs_severe_auc']:.3f}"
            )
    print()
    print("Core outputs:")
    print(f"  {out / 'runtime_5s_gaussian_parameters.csv'}")
    print(f"  {out / 'window_distribution_scores.csv'}")
    print(f"  {out / 'aggregation_family_validation.csv'}")
    print(f"  {out / 'aggregation_modality_ranking.csv'}")
    print(f"  {out / 'mapping_threshold_response.csv'}")
    print(f"  {out / 'quality_score_build_summary.json'}")
    print("=" * 116)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nINTERRUPTED.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nDISTRIBUTION QUALITY SCORE BUILD ERROR: {exc}", file=sys.stderr)
        raise
