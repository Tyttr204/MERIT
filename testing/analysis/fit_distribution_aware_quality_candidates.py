#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAV Distribution-Aware Quality Candidate Fitting v1.0
=====================================================

Purpose
-------
Fit and validate *candidate* distribution-aware quality models from the already
completed formal balanced VAL robustness run, without changing the deployed
emotion classifiers, quality modules, calibrators, fusion checkpoints, or the
frozen tau=0.80 router.

This is an ANALYSIS script. It is deliberately not a runtime module.

Scientific design
-----------------
1) Fit nominal Gaussian parameters ONLY on CLEAN VAL trial means.
2) Use raw continuous quality features, not the already-calibrated q values.
3) Keep availability and discrete anomaly evidence separate.
4) Standardize each continuous feature relative to its clean distribution:
       z = (x - mu_clean) / sigma_clean
5) Convert z only into a *directional degradation score* for validation:
       lower_is_worse : bad_z = -z
       higher_is_worse: bad_z = +z
       two_sided      : bad_z = abs(z)
   No [0,1] deployment q is defined here.
6) Validate whether bad_z rises from reference -> mild -> medium -> severe.
7) Use subject-disjoint LOSO diagnostics for clean Gaussian stability.
8) Export a candidate JSON artifact, but mark it NOT DEPLOYABLE.

The script does NOT:
- use TEST data;
- retrain any emotion/fusion network;
- alter existing q_EEG/q_Audio/q_Video;
- alter tau=0.80;
- select a final router;
- claim all raw features are Gaussian;
- force discrete bad-channel / flatline / face-observability variables into
  a Gaussian model.

Default continuous candidates
-----------------------------
Audio:
  dnsmos.OVRL_raw                 lower_is_worse
  signal_metrics.rms_dbfs         lower_is_worse

Video:
  physical_quality.technical_raw  lower_is_worse

EEG:
  features.line_fraction_mean     higher_is_worse
  features.hf55_90_fraction_mean  higher_is_worse
  features.slow02_1_fraction_mean higher_is_worse
  features.rms_uv_median          two_sided

Default explicit non-Gaussian guards
------------------------------------
EEG:
  features.raw_flat_channel_fraction
  features.raw_hold_fraction_mean
  features.pyprep_bad_fraction

Video:
  face_observability.raw_detection_rate

Availability remains outside all distribution models.

Expected input files in the formal run directory
------------------------------------------------
run_summary.json
quality_numeric_features.csv
quality_distribution_samples.csv

Recommended location
--------------------
最终部署/testing/analysis/fit_distribution_aware_quality_candidates.py

Recommended command from 最终部署
---------------------------------
python -X utf8 .\\testing\\analysis\\fit_distribution_aware_quality_candidates.py

If --run-dir is omitted, the latest completed formal balanced VAL robustness
run below ./system_checks is selected.
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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "EAV-DIST-QUALITY-CANDIDATES.1.0.1"
SEED = 20260924
EXPECTED_SUBJECTS = 6
EXPECTED_TRIALS = 30
EXPECTED_WINDOWS = 120

SEVERITY_RANK = {
    "": 0,
    "reference": 0,
    "clean": 0,
    "mild": 1,
    "medium": 2,
    "severe": 3,
}

CONTINUOUS_FEATURES: Dict[str, Dict[str, str]] = {
    "audio": {
        "dnsmos.OVRL_raw": "lower_is_worse",
        "signal_metrics.rms_dbfs": "lower_is_worse",
    },
    "video": {
        "physical_quality.technical_raw": "lower_is_worse",
    },
    "eeg": {
        "features.line_fraction_mean": "higher_is_worse",
        "features.hf55_90_fraction_mean": "higher_is_worse",
        "features.slow02_1_fraction_mean": "higher_is_worse",
        "features.rms_uv_median": "two_sided",
    },
}

DISCRETE_GUARDS: Dict[str, Dict[str, str]] = {
    "eeg": {
        "features.raw_flat_channel_fraction": "higher_is_worse",
        "features.raw_hold_fraction_mean": "higher_is_worse",
        "features.pyprep_bad_fraction": "higher_is_worse",
    },
    "video": {
        "face_observability.raw_detection_rate": "lower_is_worse",
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
        "continuous_candidates": [
            "signal_metrics.rms_dbfs",
            "dnsmos.OVRL_raw",
        ],
        "guard_candidates": [],
    },
    "audio_white_noise": {
        "modality": "audio",
        "reference_condition": "audio_matched_reference",
        "conditions": [
            "audio_white_noise_mild",
            "audio_white_noise_medium",
            "audio_white_noise_severe",
        ],
        "continuous_candidates": [
            "dnsmos.OVRL_raw",
            "signal_metrics.rms_dbfs",
        ],
        "guard_candidates": [],
    },
    "video_blur": {
        "modality": "video",
        "reference_condition": "video_matched_reference",
        "conditions": [
            "video_blur_mild",
            "video_blur_medium",
            "video_blur_severe",
        ],
        "continuous_candidates": [
            "physical_quality.technical_raw",
        ],
        "guard_candidates": [
            "face_observability.raw_detection_rate",
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
        "continuous_candidates": [
            "physical_quality.technical_raw",
        ],
        "guard_candidates": [
            "face_observability.raw_detection_rate",
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
        "continuous_candidates": [
            "features.line_fraction_mean",
            "features.rms_uv_median",
            "features.hf55_90_fraction_mean",
        ],
        "guard_candidates": [
            "features.pyprep_bad_fraction",
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
        "continuous_candidates": [
            "features.rms_uv_median",
            "features.line_fraction_mean",
        ],
        "guard_candidates": [
            "features.raw_flat_channel_fraction",
            "features.pyprep_bad_fraction",
        ],
    },
}


class CandidateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CandidateError(message)


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    raise TypeError(type(obj).__name__)


def json_text(obj: Any) -> str:
    return json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        default=json_default,
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
    raise CandidateError(f"{name}: invalid boolean {value!r}")


def finite(values: Iterable[Any]) -> np.ndarray:
    result: List[float] = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            result.append(x)
    return np.asarray(result, dtype=np.float64)


def discover_deployment_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "main.py").is_file() and (p / "system_checks").is_dir():
            return p
    raise CandidateError(
        "Cannot locate 最终部署 root. Run from 最终部署 or pass --deployment-root."
    )


def discover_latest_formal_run(root: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    for d in (root / "system_checks").iterdir():
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
            s.get("formal_balanced_val") is True
            and s.get("status") == "PASS_OFFLINE_PROCESSING"
            and s.get("suite") == "robustness"
            and s.get("test_data_used") is False
        ):
            candidates.append((d.stat().st_mtime, d.resolve()))
    require(candidates, "No completed formal balanced VAL robustness run found.")
    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[-1][1]


def validate_run(run_dir: Path) -> dict:
    s = read_json(run_dir / "run_summary.json")
    require(s.get("status") == "PASS_OFFLINE_PROCESSING", "Run status is not PASS.")
    require(s.get("formal_balanced_val") is True, "Not a formal balanced VAL run.")
    require(s.get("suite") == "robustness", "Not the robustness suite.")
    require(s.get("test_data_used") is False, "TEST data used; refusing analysis.")
    require(s.get("threshold_retuned") is False, "Threshold was retuned in source run.")
    require(
        s.get("frozen_model_training_performed") is False,
        "Source run reports training.",
    )
    return s


def severity_rank_from_condition(condition: str, severity: str) -> int:
    s = str(severity or "").strip().lower()
    if s in SEVERITY_RANK and s:
        return SEVERITY_RANK[s]
    c = str(condition)
    if c.endswith("_mild"):
        return 1
    if c.endswith("_medium"):
        return 2
    if c.endswith("_severe"):
        return 3
    return 0


def import_scipy_stats():
    try:
        from scipy import stats
    except Exception as exc:
        raise CandidateError("scipy is required.") from exc
    return stats


def import_auc():
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise CandidateError("scikit-learn is required.") from exc
    return roc_auc_score


def gaussian_fit(x: np.ndarray) -> Dict[str, Any]:
    x = finite(x)
    require(len(x) >= 8, "Too few clean values for Gaussian fit.")
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    require(sigma > 0 and math.isfinite(sigma), "Gaussian sigma is non-positive.")
    stats = import_scipy_stats()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = stats.shapiro(x)
        ad = stats.anderson(x, dist="norm")
        levels = np.asarray(ad.significance_level, dtype=float)
        idx = int(np.argmin(np.abs(levels - 5.0)))
        try:
            nt = stats.normaltest(x)
            dag_k2, dag_p = float(nt.statistic), float(nt.pvalue)
        except Exception:
            dag_k2, dag_p = None, None
        jb = stats.jarque_bera(x)

    centered = (x - mu) / np.std(x, ddof=0)
    skew = float(np.mean(centered**3))
    excess = float(np.mean(centered**4) - 3.0)

    return {
        "n": int(len(x)),
        "mu": mu,
        "sigma_sample": sigma,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "skewness_moment": skew,
        "excess_kurtosis_moment": excess,
        "shapiro_W": float(sh.statistic),
        "shapiro_p": float(sh.pvalue),
        "dagostino_K2": dag_k2,
        "dagostino_p": dag_p,
        "jarque_bera": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "anderson_statistic": float(ad.statistic),
        "anderson_5pct_critical": float(ad.critical_values[idx]),
        "anderson_reject_at_5pct_diagnostic": bool(
            ad.statistic > ad.critical_values[idx]
        ),
        "normality_is_diagnostic_only": True,
    }


def bad_z(values: np.ndarray, mu: float, sigma: float, direction: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    z = (x - float(mu)) / float(sigma)
    if direction == "lower_is_worse":
        return -z
    if direction == "higher_is_worse":
        return z
    if direction == "two_sided":
        return np.abs(z)
    raise CandidateError(f"Unknown direction: {direction}")


def build_trial_feature_table(features: pd.DataFrame) -> pd.DataFrame:
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
        required.issubset(features.columns),
        f"quality_numeric_features.csv missing columns: {sorted(required-set(features.columns))}",
    )
    f = features.copy()
    f["available_bool"] = [explicit_bool(v, "available") for v in f["available"]]
    f["value_num"] = pd.to_numeric(f["value"], errors="coerce")
    f = f[np.isfinite(f["value_num"].to_numpy(dtype=float))].copy()

    # IMPORTANT:
    # In quality_numeric_features.csv, reference / matched-reference rows have
    # intentionally blank family and severity fields. pandas.read_csv reads
    # those blanks as NaN, and pandas.groupby(dropna=True) would silently remove
    # every such row. Normalize identity/grouping columns first and also keep
    # dropna=False as a second guard. This is essential because CLEAN reference
    # rows are exactly the rows used to fit nominal Gaussian parameters.
    for col in (
        "subject",
        "pair_key",
        "condition",
        "family",
        "severity",
        "modality",
        "feature_path",
    ):
        f[col] = f[col].fillna("").astype(str)

    # Distribution fitting/validation uses current, usable raw-quality evidence.
    # Unavailable remains a separate hard state.
    f = f[f["available_bool"]].copy()

    group_cols = [
        "subject",
        "pair_key",
        "condition",
        "family",
        "severity",
        "modality",
        "feature_path",
    ]
    t = (
        f.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            mean_value=("value_num", "mean"),
            windows=("value_num", "size"),
        )
    )
    t["severity_rank"] = [
        severity_rank_from_condition(c, s)
        for c, s in zip(t["condition"], t["severity"])
    ]
    return t


def clean_gaussian_parameters(
    trial_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[dict] = []
    artifact: Dict[str, Any] = {}

    for modality, feature_map in CONTINUOUS_FEATURES.items():
        artifact[modality] = {}
        for feature, direction in feature_map.items():
            g = trial_features[
                (trial_features["modality"].astype(str) == modality)
                & (trial_features["feature_path"].astype(str) == feature)
                & (trial_features["condition"].astype(str) == "reference")
            ].copy()

            x = finite(g["mean_value"])
            if len(x) == 0:
                rows.append({
                    "modality": modality,
                    "feature_path": feature,
                    "direction": direction,
                    "status": "MISSING",
                    "deployment_selected": False,
                })
                continue

            params = gaussian_fit(x)
            subjects = sorted(g["subject"].astype(str).unique().tolist())
            pairs = sorted(g["pair_key"].astype(str).unique().tolist())
            require(
                len(subjects) == EXPECTED_SUBJECTS,
                f"{modality}/{feature}: clean fit has {len(subjects)} subjects, expected 6.",
            )
            require(
                len(pairs) == EXPECTED_TRIALS,
                f"{modality}/{feature}: clean fit has {len(pairs)} trials, expected 30.",
            )

            row = {
                "modality": modality,
                "feature_path": feature,
                "direction": direction,
                "status": "OK",
                **params,
                "deployment_selected": False,
            }
            rows.append(row)
            artifact[modality][feature] = {
                "distribution": "gaussian_candidate",
                "direction": direction,
                "fit_unit": "clean_20s_trial_mean",
                **params,
                "deployment_selected": False,
            }

    return pd.DataFrame(rows), artifact


def loso_clean_stability(
    trial_features: pd.DataFrame,
    parameter_table: pd.DataFrame,
) -> pd.DataFrame:
    stats = import_scipy_stats()
    rows: List[dict] = []

    for _, p in parameter_table[parameter_table["status"] == "OK"].iterrows():
        modality = str(p["modality"])
        feature = str(p["feature_path"])
        g = trial_features[
            (trial_features["modality"].astype(str) == modality)
            & (trial_features["feature_path"].astype(str) == feature)
            & (trial_features["condition"].astype(str) == "reference")
        ].copy()

        subjects = sorted(g["subject"].astype(str).unique().tolist())
        fold_mus = []
        fold_sigmas = []
        nll_all: List[float] = []

        for held in subjects:
            train = finite(g.loc[g["subject"].astype(str) != held, "mean_value"])
            test = finite(g.loc[g["subject"].astype(str) == held, "mean_value"])
            mu = float(np.mean(train))
            sigma = float(np.std(train, ddof=1))
            require(sigma > 0, f"{modality}/{feature}/{held}: zero LOSO sigma.")
            lp = stats.norm.logpdf(test, loc=mu, scale=sigma)
            require(np.all(np.isfinite(lp)), "Non-finite LOSO Gaussian logpdf.")
            fold_mus.append(mu)
            fold_sigmas.append(sigma)
            nll_all.extend((-lp).tolist())
            rows.append({
                "level": "fold",
                "modality": modality,
                "feature_path": feature,
                "held_subject": held,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "mu_train": mu,
                "sigma_train": sigma,
                "mean_test_nll": float(np.mean(-lp)),
                "deployment_selected": False,
            })

        rows.append({
            "level": "aggregate",
            "modality": modality,
            "feature_path": feature,
            "held_subject": "",
            "train_n": None,
            "test_n": int(len(nll_all)),
            "mu_train": float(np.mean(fold_mus)),
            "sigma_train": float(np.mean(fold_sigmas)),
            "mean_test_nll": float(np.mean(nll_all)),
            "mu_fold_sd": float(np.std(fold_mus, ddof=1)),
            "sigma_fold_sd": float(np.std(fold_sigmas, ddof=1)),
            "deployment_selected": False,
        })

    return pd.DataFrame(rows)


def family_feature_validation(
    trial_features: pd.DataFrame,
    gaussian_params: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    roc_auc_score = import_auc()
    stats = import_scipy_stats()
    condition_rows: List[dict] = []
    summary_rows: List[dict] = []

    param_lookup = {}
    for _, r in gaussian_params[gaussian_params["status"] == "OK"].iterrows():
        param_lookup[(str(r["modality"]), str(r["feature_path"]))] = r.to_dict()

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])
        ref_condition = str(spec["reference_condition"])
        corruption_conditions = [str(x) for x in spec["conditions"]]

        for feature in spec["continuous_candidates"]:
            key = (modality, feature)
            if key not in param_lookup:
                summary_rows.append({
                    "family": family,
                    "modality": modality,
                    "feature_path": feature,
                    "candidate_type": "gaussian_continuous",
                    "status": "NO_CLEAN_GAUSSIAN_PARAMS",
                    "deployment_selected": False,
                })
                continue

            p = param_lookup[key]
            direction = str(p["direction"])
            mu = float(p["mu"])
            sigma = float(p["sigma_sample"])

            g = trial_features[
                (trial_features["modality"].astype(str) == modality)
                & (trial_features["feature_path"].astype(str) == feature)
                & (
                    trial_features["condition"].astype(str).isin(
                        [ref_condition] + corruption_conditions
                    )
                )
            ].copy()

            if g.empty:
                summary_rows.append({
                    "family": family,
                    "modality": modality,
                    "feature_path": feature,
                    "candidate_type": "gaussian_continuous",
                    "status": "NO_FAMILY_DATA",
                    "deployment_selected": False,
                })
                continue

            g["bad_z"] = bad_z(
                g["mean_value"].to_numpy(dtype=float),
                mu,
                sigma,
                direction,
            )

            # Condition summaries.
            for condition in [ref_condition] + corruption_conditions:
                c = g[g["condition"].astype(str) == condition]
                if c.empty:
                    continue
                rank = 0 if condition == ref_condition else severity_rank_from_condition(
                    condition, c["severity"].iloc[0]
                )
                condition_rows.append({
                    "family": family,
                    "modality": modality,
                    "feature_path": feature,
                    "direction": direction,
                    "condition": condition,
                    "severity_rank": rank,
                    "n_trials": int(len(c)),
                    "mean_raw": float(c["mean_value"].mean()),
                    "median_raw": float(c["mean_value"].median()),
                    "mean_bad_z": float(c["bad_z"].mean()),
                    "median_bad_z": float(c["bad_z"].median()),
                    "std_bad_z": float(c["bad_z"].std(ddof=1)) if len(c) > 1 else 0.0,
                })

            # Complete paired trajectories: one row/trial, four severity columns.
            pivot = g.pivot_table(
                index=["subject", "pair_key"],
                columns="condition",
                values="bad_z",
                aggfunc="mean",
            )
            required_conditions = [ref_condition] + corruption_conditions
            complete = pivot.dropna(subset=required_conditions).copy()

            if complete.empty:
                summary_rows.append({
                    "family": family,
                    "modality": modality,
                    "feature_path": feature,
                    "candidate_type": "gaussian_continuous",
                    "status": "NO_COMPLETE_TRAJECTORIES",
                    "deployment_selected": False,
                })
                continue

            arr = complete[required_conditions].to_numpy(dtype=float)
            diffs = np.diff(arr, axis=1)
            monotonic_rate = float(np.mean(np.all(diffs >= -1e-12, axis=1)))
            strict_any_rate = float(np.mean(np.any(diffs > 1e-12, axis=1)))
            severe_worse_rate = float(np.mean(arr[:, -1] > arr[:, 0]))

            spearmans: List[float] = []
            for row in arr:
                rho = stats.spearmanr(np.arange(4, dtype=float), row).statistic
                if math.isfinite(float(rho)):
                    spearmans.append(float(rho))

            paired_diff = arr[:, -1] - arr[:, 0]
            paired_sd = float(np.std(paired_diff, ddof=1)) if len(paired_diff) > 1 else 0.0
            dz = (
                float(np.mean(paired_diff) / paired_sd)
                if paired_sd > 0
                else None
            )

            y = np.concatenate([
                np.zeros(len(arr), dtype=int),
                np.ones(len(arr), dtype=int),
            ])
            score = np.concatenate([arr[:, 0], arr[:, -1]])
            try:
                auc = float(roc_auc_score(y, score))
            except Exception:
                auc = None

            summary_rows.append({
                "family": family,
                "modality": modality,
                "feature_path": feature,
                "candidate_type": "gaussian_continuous",
                "direction": direction,
                "status": "OK",
                "n_complete_trials": int(len(complete)),
                "mean_spearman_severity_vs_bad_z": (
                    float(np.mean(spearmans)) if spearmans else None
                ),
                "positive_spearman_rate": (
                    float(np.mean(np.asarray(spearmans) > 0)) if spearmans else None
                ),
                "monotonic_nondecreasing_rate": monotonic_rate,
                "strict_change_somewhere_rate": strict_any_rate,
                "severe_worse_than_reference_rate": severe_worse_rate,
                "mean_severe_minus_reference_bad_z": float(np.mean(paired_diff)),
                "median_severe_minus_reference_bad_z": float(np.median(paired_diff)),
                "paired_effect_size_dz": dz,
                "reference_vs_severe_auc": auc,
                "deployment_selected": False,
            })

    summary = pd.DataFrame(summary_rows)

    # Diagnostic family ranks only. No runtime selection.
    if not summary.empty:
        summary["diagnostic_rank"] = np.nan
        for family, idx in summary[summary["status"] == "OK"].groupby("family").groups.items():
            local = summary.loc[idx].copy()
            # Primary: severity correlation; ties: monotonicity, AUC, severe delta.
            local = local.sort_values(
                [
                    "mean_spearman_severity_vs_bad_z",
                    "monotonic_nondecreasing_rate",
                    "reference_vs_severe_auc",
                    "mean_severe_minus_reference_bad_z",
                    "feature_path",
                ],
                ascending=[False, False, False, False, True],
                na_position="last",
            )
            for rank, ridx in enumerate(local.index, start=1):
                summary.loc[ridx, "diagnostic_rank"] = rank

    return pd.DataFrame(condition_rows), summary


def discrete_guard_validation(
    trial_features: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    for family, spec in FAMILY_SPECS.items():
        modality = str(spec["modality"])
        ref = str(spec["reference_condition"])
        conditions = [str(x) for x in spec["conditions"]]

        for feature in spec.get("guard_candidates", []):
            direction = DISCRETE_GUARDS.get(modality, {}).get(feature)
            if direction is None:
                continue

            g = trial_features[
                (trial_features["modality"].astype(str) == modality)
                & (trial_features["feature_path"].astype(str) == feature)
                & (
                    trial_features["condition"].astype(str).isin(
                        [ref] + conditions
                    )
                )
            ].copy()
            if g.empty:
                rows.append({
                    "family": family,
                    "modality": modality,
                    "feature_path": feature,
                    "status": "NO_DATA",
                    "gaussian_model_used": False,
                    "deployment_selected": False,
                })
                continue

            for condition in [ref] + conditions:
                c = g[g["condition"].astype(str) == condition]
                if c.empty:
                    continue
                x = finite(c["mean_value"])
                rows.append({
                    "family": family,
                    "modality": modality,
                    "feature_path": feature,
                    "direction": direction,
                    "condition": condition,
                    "severity_rank": (
                        0 if condition == ref
                        else severity_rank_from_condition(condition, c["severity"].iloc[0])
                    ),
                    "n_trials": int(len(x)),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "nonzero_rate": float(np.mean(np.abs(x) > 1e-12)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                    "status": "OK",
                    "gaussian_model_used": False,
                    "deployment_selected": False,
                })
    return pd.DataFrame(rows)


def availability_validation(samples: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject", "pair_key", "condition", "modality", "available"
    }
    require(required.issubset(samples.columns),
            f"quality_distribution_samples.csv missing {sorted(required-set(samples.columns))}")
    s = samples.copy()
    s["available_bool"] = [explicit_bool(v, "sample available") for v in s["available"]]

    rows = []
    for (condition, modality), g in s.groupby(["condition", "modality"]):
        rows.append({
            "condition": condition,
            "modality": modality,
            "n_windows": int(len(g)),
            "availability_rate": float(g["available_bool"].mean()),
            "modelled_separately_from_gaussian_quality": True,
        })
    return pd.DataFrame(rows)


def create_plots(
    out: Path,
    condition_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise CandidateError("matplotlib is required for plots.") from exc

    plots = out / "plots"
    plots.mkdir(exist_ok=False)

    ok = family_summary[family_summary["status"] == "OK"].copy()

    for family, spec in FAMILY_SPECS.items():
        g = condition_summary[condition_summary["family"] == family].copy()
        if g.empty:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for feature, fg in g.groupby("feature_path"):
            fg = fg.sort_values("severity_rank")
            ax.plot(
                fg["severity_rank"].to_numpy(),
                fg["mean_bad_z"].to_numpy(),
                marker="o",
                label=str(feature),
            )
        ax.set_xticks([0, 1, 2, 3], ["reference", "mild", "medium", "severe"])
        ax.set_xlabel("corruption severity")
        ax.set_ylabel("mean directional bad-z (higher = more degraded)")
        ax.set_title(f"{family}: Gaussian-standardized raw quality response")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / f"{family}_bad_z_by_severity.png", dpi=170)
        plt.close(fig)

    if not ok.empty:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        labels = [
            f"{r.family}\n{r.feature_path}"
            for r in ok.sort_values(["family", "diagnostic_rank"]).itertuples()
        ]
        values = ok.sort_values(["family", "diagnostic_rank"])[
            "mean_spearman_severity_vs_bad_z"
        ].to_numpy()
        ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values)), labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("mean Spearman(severity, bad-z)")
        ax.set_title("Distribution-aware candidate degradation sensitivity")
        fig.tight_layout()
        fig.savefig(plots / "candidate_severity_correlations.png", dpi=170)
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit raw-feature Gaussian quality candidates on clean VAL and validate under controlled degradation."
    )
    p.add_argument("--deployment-root")
    p.add_argument("--run-dir")
    p.add_argument("--output-dir")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def self_test() -> int:
    rng = np.random.default_rng(SEED)
    x = rng.normal(10.0, 2.0, 30)
    fit = gaussian_fit(x)
    degraded = np.array([8.0, 6.0, 4.0])
    scores = bad_z(degraded, fit["mu"], fit["sigma_sample"], "lower_is_worse")
    # Regression test for CSV blank family/severity -> NaN. Reference rows must
    # survive trial aggregation; v1.0 incorrectly let pandas groupby drop them.
    mini = pd.DataFrame([
        {
            "source_window_id": f"w{i}",
            "subject": "subject08",
            "pair_key": "subject08_instance001",
            "condition": "reference",
            "family": np.nan,
            "severity": np.nan,
            "modality": "audio",
            "available": True,
            "feature_path": "dnsmos.OVRL_raw",
            "value": 0.8 + 0.01 * i,
        }
        for i in range(4)
    ])
    mini_trial = build_trial_feature_table(mini)

    checks = {
        "fit_n": fit["n"] == 30,
        "sigma_positive": fit["sigma_sample"] > 0,
        "lower_bad_score_orders": bool(scores[0] < scores[1] < scores[2]),
        "feature_spec_has_all_modalities": set(CONTINUOUS_FEATURES) == {"eeg", "audio", "video"},
        "family_count": len(FAMILY_SPECS) == 6,
        "blank_reference_group_preserved": (
            len(mini_trial) == 1
            and str(mini_trial.iloc[0]["condition"]) == "reference"
            and int(mini_trial.iloc[0]["windows"]) == 4
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
        except CandidateError:
            deployment_root = discover_deployment_root(Path(__file__).resolve().parent)

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else discover_latest_formal_run(deployment_root)
    )
    require(run_dir.is_dir(), f"Missing run directory: {run_dir}")
    run_summary = validate_run(run_dir)

    feature_path = run_dir / "quality_numeric_features.csv"
    sample_path = run_dir / "quality_distribution_samples.csv"
    require(feature_path.is_file(), f"Missing {feature_path}")
    require(sample_path.is_file(), f"Missing {sample_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / f"distribution_candidate_fit_{timestamp}"
    )
    require(not out.exists(), f"Output exists; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)

    features = pd.read_csv(feature_path, low_memory=False)
    samples = pd.read_csv(sample_path, low_memory=False)
    trial_features = build_trial_feature_table(features)

    # Audit source trial geometry.
    clean_pairs = sorted(
        trial_features.loc[
            trial_features["condition"].astype(str) == "reference",
            "pair_key",
        ].astype(str).unique().tolist()
    )
    clean_subjects = sorted(
        trial_features.loc[
            trial_features["condition"].astype(str) == "reference",
            "subject",
        ].astype(str).unique().tolist()
    )
    require(len(clean_pairs) == EXPECTED_TRIALS,
            f"Expected {EXPECTED_TRIALS} clean trials, got {len(clean_pairs)}")
    require(len(clean_subjects) == EXPECTED_SUBJECTS,
            f"Expected {EXPECTED_SUBJECTS} subjects, got {len(clean_subjects)}")

    params_df, artifact_models = clean_gaussian_parameters(trial_features)
    loso_df = loso_clean_stability(trial_features, params_df)
    condition_df, family_df = family_feature_validation(trial_features, params_df)
    guard_df = discrete_guard_validation(trial_features)
    availability_df = availability_validation(samples)

    atomic_csv(out / "clean_gaussian_parameters.csv", params_df)
    atomic_csv(out / "clean_gaussian_loso_stability.csv", loso_df)
    atomic_csv(out / "condition_gaussian_bad_z.csv", condition_df)
    atomic_csv(out / "family_candidate_validation.csv", family_df)
    atomic_csv(out / "discrete_guard_response.csv", guard_df)
    atomic_csv(out / "availability_response.csv", availability_df)

    if not args.no_plots:
        create_plots(out, condition_df, family_df)

    candidate_artifact = {
        "schema": "eav.distribution_quality_candidates.v1",
        "version": VERSION,
        "artifact_state": "CANDIDATE_NOT_DEPLOYABLE",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_run_version": run_summary.get("version"),
        "source_split": "VAL_ONLY",
        "test_data_used": False,
        "fit_unit": "clean 20s trial mean of raw continuous quality feature",
        "models": artifact_models,
        "discrete_guards": DISCRETE_GUARDS,
        "availability": {
            "separate_hard_state": True,
            "not_overridden_by_gaussian_score": True,
        },
        "score_definition_for_validation_only": {
            "z": "(x - clean_mu) / clean_sigma",
            "lower_is_worse": "bad_z = -z",
            "higher_is_worse": "bad_z = +z",
            "two_sided": "bad_z = abs(z)",
            "final_0_1_quality_defined": False,
        },
        "router": {
            "tau_0_80_changed": False,
            "fusion_weights_changed": False,
            "runtime_router_selected": False,
        },
        "guardrails": [
            "Continuous Gaussian evidence and discrete anomaly guards remain separate.",
            "Availability remains a separate hard state.",
            "No [0,1] replacement q is defined in this candidate artifact.",
            "No Gaussian assumption is made for discrete bad-channel/flatline/face-detection ratios.",
            "Family rankings are diagnostic VAL evidence only.",
            "No TEST data or new model training are used.",
        ],
        "source_hashes": {
            "run_summary": sha256_file(run_dir / "run_summary.json"),
            "quality_numeric_features": sha256_file(feature_path),
            "quality_distribution_samples": sha256_file(sample_path),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(out / "distribution_quality_candidate_artifact.json", candidate_artifact)

    # Compact family report.
    compact_families: Dict[str, Any] = {}
    for family in FAMILY_SPECS:
        g = family_df[
            (family_df["family"] == family)
            & (family_df["status"] == "OK")
        ].sort_values("diagnostic_rank")
        compact_families[family] = {
            "modality": FAMILY_SPECS[family]["modality"],
            "diagnostic_candidates": (
                g[
                    [
                        "diagnostic_rank",
                        "feature_path",
                        "mean_spearman_severity_vs_bad_z",
                        "monotonic_nondecreasing_rate",
                        "severe_worse_than_reference_rate",
                        "reference_vs_severe_auc",
                    ]
                ].to_dict(orient="records")
                if not g.empty else []
            ),
            "deployment_candidate_selected": False,
        }

    summary = {
        "version": VERSION,
        "status": "PASS",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "formal_balanced_val": True,
        "test_data_used": False,
        "models_or_calibrators_modified": False,
        "router_threshold_retuned": False,
        "distribution_runtime_deployed": False,
        "continuous_features_fitted": int((params_df["status"] == "OK").sum()),
        "continuous_features_requested": int(len(params_df)),
        "clean_subjects": clean_subjects,
        "clean_trials": len(clean_pairs),
        "family_validation": compact_families,
        "candidate_artifact": str(out / "distribution_quality_candidate_artifact.json"),
        "outputs": sorted(p.name for p in out.iterdir()),
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
    atomic_json(out / "candidate_fit_summary.json", summary)

    lines = [
        "# Distribution-Aware Quality Candidate Fit",
        "",
        f"- Version: `{VERSION}`",
        f"- Source: `{run_dir}`",
        "- Scope: **VAL only**",
        "- Existing tau=0.80 changed: **No**",
        "- Emotion/fusion models retrained: **No**",
        "- Runtime distribution model deployed: **No**",
        "",
        "## What was fitted",
        "",
        "Gaussian parameters were fitted only to selected continuous raw-quality "
        "features on clean 20-s trial means.",
        "",
        "Discrete guards and availability were kept separate.",
        "",
        "## Family diagnostic ranking",
        "",
    ]
    for family in FAMILY_SPECS:
        lines.append(f"### {family}")
        g = family_df[
            (family_df["family"] == family)
            & (family_df["status"] == "OK")
        ].sort_values("diagnostic_rank")
        if g.empty:
            lines.append("No complete continuous candidate.")
        else:
            for r in g.itertuples():
                lines.append(
                    f"- rank {int(r.diagnostic_rank)}: `{r.feature_path}` — "
                    f"Spearman={r.mean_spearman_severity_vs_bad_z:.3f}, "
                    f"monotonic={r.monotonic_nondecreasing_rate:.3f}, "
                    f"AUC={r.reference_vs_severe_auc:.3f}"
                )
        lines.append("")
    lines += [
        "## Next gate",
        "",
        "Review `family_candidate_validation.csv`, `discrete_guard_response.csv`, "
        "and the plots. Only after a candidate set is frozen should a separate "
        "`Quality/distribution_calibrator.py` runtime module be implemented.",
        "",
    ]
    atomic_text(out / "README_candidate_fit.md", "\n".join(lines))

    print("=" * 112)
    print("EAV DISTRIBUTION-AWARE QUALITY CANDIDATE FIT")
    print("=" * 112)
    print(f"Status                    : PASS")
    print(f"Version                   : {VERSION}")
    print(f"Source formal run         : {run_dir}")
    print(f"Output                    : {out}")
    print(f"Clean subjects            : {len(clean_subjects)}")
    print(f"Clean trials              : {len(clean_pairs)}")
    print(f"Gaussian features fitted  : {(params_df['status'] == 'OK').sum()} / {len(params_df)}")
    print("TEST used                 : False")
    print("tau=0.80 changed          : False")
    print("Runtime model deployed    : False")
    print()
    for family in FAMILY_SPECS:
        g = family_df[
            (family_df["family"] == family)
            & (family_df["status"] == "OK")
        ].sort_values("diagnostic_rank")
        if g.empty:
            print(f"{family:24s} : no complete candidate")
        else:
            r = g.iloc[0]
            print(
                f"{family:24s} : diagnostic top={r['feature_path']} | "
                f"rho={r['mean_spearman_severity_vs_bad_z']:.3f} | "
                f"monotonic={r['monotonic_nondecreasing_rate']:.3f} | "
                f"AUC={r['reference_vs_severe_auc']:.3f}"
            )
    print()
    print("Core outputs:")
    print(f"  {out / 'clean_gaussian_parameters.csv'}")
    print(f"  {out / 'family_candidate_validation.csv'}")
    print(f"  {out / 'discrete_guard_response.csv'}")
    print(f"  {out / 'distribution_quality_candidate_artifact.json'}")
    print(f"  {out / 'candidate_fit_summary.json'}")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nINTERRUPTED.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nDISTRIBUTION QUALITY CANDIDATE ERROR: {exc}", file=sys.stderr)
        raise
