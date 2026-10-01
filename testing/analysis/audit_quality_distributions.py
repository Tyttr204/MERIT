#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAV Quality Distribution Audit v1.0
===================================

Purpose
-------
Offline, analysis-only audit of the quality distributions exported by the
formal balanced VAL robustness run of system_test_runner.py.

This script:
  * NEVER trains or changes EEG / Audio / Video / Fusion models.
  * NEVER changes the frozen router threshold tau=0.80.
  * NEVER writes into deployment model/calibration directories.
  * NEVER opens TEST data.
  * Uses CLEAN REFERENCE quality observations for nominal-distribution fitting.
  * Keeps availability separate from non-missing quality.
  * Treats 5-s window rows as correlated descriptive observations.
  * Uses 20-s trial means as the PRIMARY distribution-audit unit.
  * Still notes that trials are clustered within subjects (6 VAL subjects).
  * Audits calibrated q and raw numeric quality-detector features separately.
  * Compares Normal, Skew-Normal, Beta, and 1/2/3-component Gaussian mixtures.
  * Uses leave-one-subject-out (LOSO) predictive NLL as a diagnostic comparison.

Important interpretation
------------------------
Normality tests are diagnostics, not proof of population normality/non-normality:
the 30 trial means are clustered within six subjects. Window-level p-values are
therefore not used as the primary inferential result.

The candidate-fit table is NOT a deployment-model selection. It is evidence for
the next design review. A future distribution-aware runtime calibrator should be
implemented only after this audit is reviewed and frozen.

Expected source files in --run-dir
----------------------------------
run_summary.json
quality_distribution_samples.csv
quality_numeric_features.csv
quality_distribution_trial_means.csv

Optional:
quality_distribution_summary.csv
quality_distribution_trial_mean_summary.csv
quality_response_summary.csv

Recommended location
--------------------
最终部署/testing/analysis/audit_quality_distributions.py

Recommended command from 最终部署
---------------------------------
python -X utf8 .\\testing\\analysis\\audit_quality_distributions.py ^
  --run-dir ".\\system_checks\\run_YYYYMMDD_HHMMSS_xxxxxx_robustness"

If --run-dir is omitted, the latest completed formal balanced VAL robustness
run under ./system_checks is selected deterministically by directory mtime.
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "EAV-QDIST-AUDIT.1.0"
MODALITIES = ("eeg", "audio", "video")
EXPECTED_FORMAL_WINDOWS_PER_MODALITY = 120
EXPECTED_FORMAL_TRIALS_PER_MODALITY = 30
EXPECTED_FORMAL_SUBJECTS = 6
EPS = 1e-6
DEFAULT_SEED = 20260924


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def json_text(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
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


def as_bool(value: Any, name: str = "value") -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and float(value) in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
    raise AuditError(f"{name}: cannot parse explicit boolean from {value!r}")


def finite_array(values: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return np.asarray(out, dtype=np.float64)


def clean_numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    require(column in frame.columns, f"Missing column: {column}")
    return finite_array(frame[column].tolist())


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def find_deployment_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for p in candidates:
        if (p / "system_checks").is_dir() and (p / "main.py").is_file():
            return p
    raise AuditError(
        "Cannot locate 最终部署 root. Run from 最终部署 or pass --deployment-root."
    )


def discover_latest_formal_run(deployment_root: Path) -> Path:
    system_checks = deployment_root / "system_checks"
    require(system_checks.is_dir(), f"Missing system_checks: {system_checks}")
    candidates: List[Tuple[float, Path]] = []
    for d in system_checks.iterdir():
        if not d.is_dir():
            continue
        summary = d / "run_summary.json"
        if not summary.is_file():
            continue
        try:
            obj = read_json(summary)
        except Exception:
            continue
        if (
            obj.get("formal_balanced_val") is True
            and obj.get("status") == "PASS_OFFLINE_PROCESSING"
            and obj.get("suite") == "robustness"
            and obj.get("test_data_used") is False
        ):
            candidates.append((d.stat().st_mtime, d.resolve()))
    require(candidates, "No completed formal balanced VAL robustness run found.")
    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[-1][1]


def validate_run(run_dir: Path) -> dict:
    summary = read_json(run_dir / "run_summary.json")

    require(
        summary.get("status") == "PASS_OFFLINE_PROCESSING",
        "Source run did not finish with PASS_OFFLINE_PROCESSING.",
    )
    require(
        summary.get("formal_balanced_val") is True,
        "Source run is not a formal balanced VAL run.",
    )
    require(summary.get("suite") == "robustness", "Source run is not robustness.")
    require(summary.get("test_data_used") is False, "TEST data was used; audit refuses source.")
    require(
        summary.get("threshold_retuned") is False,
        "Source run reports threshold retuning; audit refuses source.",
    )
    require(
        summary.get("frozen_model_training_performed") is False,
        "Source run reports model training; audit refuses source.",
    )

    qlog = summary.get("quality_distribution_logging", {})
    require(
        isinstance(qlog, dict) and qlog.get("status") == "OBSERVATIONAL_EXPORT_COMPLETE",
        "Quality-distribution export was not completed.",
    )
    require(
        qlog.get("distribution_model_fitted") is False,
        "Source run already fitted a distribution model unexpectedly.",
    )
    require(
        qlog.get("router_threshold_retuned") is False,
        "Source quality export reports router threshold retuning.",
    )
    return summary


def validate_source_files(run_dir: Path) -> Dict[str, Path]:
    names = {
        "summary": "run_summary.json",
        "samples": "quality_distribution_samples.csv",
        "features": "quality_numeric_features.csv",
        "trials": "quality_distribution_trial_means.csv",
    }
    paths = {k: (run_dir / v).resolve() for k, v in names.items()}
    for key, path in paths.items():
        require(path.is_file(), f"Required source file missing ({key}): {path}")
    return paths


def describe(x: np.ndarray) -> Dict[str, Any]:
    x = finite_array(x)
    n = len(x)
    if n == 0:
        return {"n": 0}
    q = np.quantile(x, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n >= 3 and std > 0:
        centered = (x - mean) / np.std(x, ddof=0)
        skew = float(np.mean(centered ** 3))
        kurt = float(np.mean(centered ** 4) - 3.0)
    else:
        skew = None
        kurt = None
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std_sample": std,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q10": float(q[2]),
        "q25": float(q[3]),
        "q50": float(q[4]),
        "q75": float(q[5]),
        "q90": float(q[6]),
        "q95": float(q[7]),
        "q99": float(q[8]),
        "mean_minus_median": mean - median,
        "skewness_moment": skew,
        "excess_kurtosis_moment": kurt,
        "unique_values": int(len(np.unique(x))),
    }


def import_stats():
    try:
        from scipy import stats
    except Exception as exc:
        raise AuditError(
            "scipy is required. Install/use the validated .venv-video environment."
        ) from exc
    return stats


def normality_diagnostics(x: np.ndarray) -> Dict[str, Any]:
    """
    Diagnostic only. Parameter-estimation and subject clustering mean these
    p-values must not be interpreted as definitive population proofs.
    """
    stats = import_stats()
    x = finite_array(x)
    n = len(x)
    result: Dict[str, Any] = {
        "n": n,
        "diagnostic_only": True,
        "population_normality_proof_claimed": False,
    }
    if n < 3 or np.std(x) <= 0:
        result["status"] = "INSUFFICIENT_VARIATION"
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sh = stats.shapiro(x)
            result["shapiro_W"] = float(sh.statistic)
            result["shapiro_p"] = float(sh.pvalue)
        except Exception as exc:
            result["shapiro_error"] = repr(exc)

        if n >= 8:
            try:
                nt = stats.normaltest(x)
                result["dagostino_K2"] = float(nt.statistic)
                result["dagostino_p"] = float(nt.pvalue)
            except Exception as exc:
                result["dagostino_error"] = repr(exc)

        try:
            jb = stats.jarque_bera(x)
            result["jarque_bera"] = float(jb.statistic)
            result["jarque_bera_p"] = float(jb.pvalue)
        except Exception as exc:
            result["jarque_bera_error"] = repr(exc)

        try:
            ad = stats.anderson(x, dist="norm")
            result["anderson_statistic"] = float(ad.statistic)
            result["anderson_significance_levels"] = [float(v) for v in ad.significance_level]
            result["anderson_critical_values"] = [float(v) for v in ad.critical_values]
            # Locate the 5% critical value when available.
            levels = np.asarray(ad.significance_level, dtype=float)
            idx = int(np.argmin(np.abs(levels - 5.0)))
            result["anderson_5pct_critical"] = float(ad.critical_values[idx])
            result["anderson_reject_at_5pct_diagnostic"] = bool(
                ad.statistic > ad.critical_values[idx]
            )
        except Exception as exc:
            result["anderson_error"] = repr(exc)

    result["status"] = "OK"
    return result


@dataclass
class FitResult:
    model: str
    n: int
    params_json: str
    log_likelihood: float
    aic: float
    bic: float
    cdf_rmse: float
    ks_like_D: float
    boundary_clip_used: bool
    fit_status: str
    note: str = ""


def empirical_targets(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) / float(n)


def cdf_metrics(x: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> Tuple[float, float]:
    xs = np.sort(np.asarray(x, dtype=np.float64))
    emp = empirical_targets(len(xs))
    pred = np.clip(np.asarray(cdf(xs), dtype=np.float64), 0.0, 1.0)
    require(pred.shape == xs.shape, "CDF shape mismatch.")
    rmse = float(np.sqrt(np.mean((pred - emp) ** 2)))
    d = float(np.max(np.abs(pred - emp)))
    return rmse, d


def fit_normal(x: np.ndarray) -> FitResult:
    stats = import_stats()
    x = finite_array(x)
    mu, sigma = stats.norm.fit(x)
    require(sigma > 0, "Normal sigma <= 0")
    ll = float(np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma)))
    k = 2
    rmse, d = cdf_metrics(x, lambda z: stats.norm.cdf(z, loc=mu, scale=sigma))
    return FitResult(
        "normal", len(x), json.dumps({"mu": mu, "sigma": sigma}),
        ll, 2*k - 2*ll, k*math.log(len(x)) - 2*ll, rmse, d, False, "OK"
    )


def fit_skew_normal(x: np.ndarray) -> FitResult:
    stats = import_stats()
    x = finite_array(x)
    a, loc, scale = stats.skewnorm.fit(x)
    require(scale > 0, "Skew-normal scale <= 0")
    ll = float(np.sum(stats.skewnorm.logpdf(x, a, loc=loc, scale=scale)))
    k = 3
    rmse, d = cdf_metrics(
        x, lambda z: stats.skewnorm.cdf(z, a, loc=loc, scale=scale)
    )
    return FitResult(
        "skew_normal", len(x),
        json.dumps({"shape": a, "loc": loc, "scale": scale}),
        ll, 2*k - 2*ll, k*math.log(len(x)) - 2*ll, rmse, d, False, "OK"
    )


def fit_beta(x: np.ndarray) -> FitResult:
    stats = import_stats()
    x = finite_array(x)
    clipped = np.clip(x, EPS, 1.0 - EPS)
    clip_used = bool(np.any(clipped != x))
    a, b, loc, scale = stats.beta.fit(clipped, floc=0.0, fscale=1.0)
    require(a > 0 and b > 0, "Beta shapes must be > 0")
    ll = float(np.sum(stats.beta.logpdf(clipped, a, b, loc=0.0, scale=1.0)))
    k = 2
    rmse, d = cdf_metrics(
        clipped, lambda z: stats.beta.cdf(z, a, b, loc=0.0, scale=1.0)
    )
    return FitResult(
        "beta_fixed_0_1", len(x),
        json.dumps({"alpha": a, "beta": b, "loc": 0.0, "scale": 1.0}),
        ll, 2*k - 2*ll, k*math.log(len(x)) - 2*ll, rmse, d, clip_used, "OK",
        note=f"Boundary values clipped to [{EPS}, {1-EPS}] for finite likelihood."
    )


def _gmm_import():
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise AuditError(
            "scikit-learn is required. Use the validated .venv-video environment."
        ) from exc
    return GaussianMixture


def fit_gmm(x: np.ndarray, components: int, seed: int) -> FitResult:
    stats = import_stats()
    GaussianMixture = _gmm_import()
    x = finite_array(x)
    require(len(x) >= max(8, components * 3), "Too few values for requested GMM.")
    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        random_state=seed,
        n_init=10,
        reg_covar=1e-6,
    )
    X = x.reshape(-1, 1)
    model.fit(X)
    ll = float(model.score(X) * len(x))
    weights = model.weights_.astype(float)
    means = model.means_.reshape(-1).astype(float)
    variances = model.covariances_.reshape(-1).astype(float)
    sigmas = np.sqrt(np.maximum(variances, 1e-12))

    def cdf(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        out = np.zeros_like(z)
        for w, mu, sigma in zip(weights, means, sigmas):
            out += w * stats.norm.cdf(z, loc=mu, scale=sigma)
        return out

    rmse, d = cdf_metrics(x, cdf)
    params = {
        "components": components,
        "weights": weights.tolist(),
        "means": means.tolist(),
        "sigmas": sigmas.tolist(),
    }
    return FitResult(
        f"gmm_{components}", len(x), json.dumps(params),
        ll, float(model.aic(X)), float(model.bic(X)), rmse, d, False, "OK"
    )


def fit_candidates(x: np.ndarray, seed: int) -> List[FitResult]:
    x = finite_array(x)
    rows: List[FitResult] = []
    funcs = [
        ("normal", lambda: fit_normal(x)),
        ("skew_normal", lambda: fit_skew_normal(x)),
        ("beta_fixed_0_1", lambda: fit_beta(x)),
        ("gmm_1", lambda: fit_gmm(x, 1, seed)),
        ("gmm_2", lambda: fit_gmm(x, 2, seed)),
        ("gmm_3", lambda: fit_gmm(x, 3, seed)),
    ]
    for name, fn in funcs:
        try:
            rows.append(fn())
        except Exception as exc:
            rows.append(
                FitResult(
                    name, len(x), "{}", float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), False, "FAILED", repr(exc)
                )
            )
    return rows


def logpdf_for_model(
    model_name: str,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    stats = import_stats()
    train = finite_array(train)
    test = finite_array(test)
    require(len(train) >= 8 and len(test) >= 1, "Insufficient fold data.")

    if model_name == "normal":
        mu, sigma = stats.norm.fit(train)
        require(sigma > 0, "Normal sigma <= 0")
        return stats.norm.logpdf(test, loc=mu, scale=sigma)

    if model_name == "skew_normal":
        a, loc, scale = stats.skewnorm.fit(train)
        require(scale > 0, "Skew-normal scale <= 0")
        return stats.skewnorm.logpdf(test, a, loc=loc, scale=scale)

    if model_name == "beta_fixed_0_1":
        tr = np.clip(train, EPS, 1-EPS)
        te = np.clip(test, EPS, 1-EPS)
        a, b, _, _ = stats.beta.fit(tr, floc=0.0, fscale=1.0)
        return stats.beta.logpdf(te, a, b, loc=0.0, scale=1.0)

    if model_name.startswith("gmm_"):
        components = int(model_name.split("_")[1])
        GaussianMixture = _gmm_import()
        require(len(train) >= max(8, components * 3), "Too few training points for GMM.")
        g = GaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=seed,
            n_init=10,
            reg_covar=1e-6,
        )
        g.fit(train.reshape(-1, 1))
        return g.score_samples(test.reshape(-1, 1))

    raise AuditError(f"Unknown model: {model_name}")


def loso_predictive_nll(
    frame: pd.DataFrame,
    value_col: str,
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    require("subject" in frame.columns, "LOSO requires subject column.")
    subjects = sorted(frame["subject"].dropna().astype(str).unique().tolist())
    require(len(subjects) >= 3, "LOSO requires >=3 subjects.")
    fold_rows = []
    all_logpdf: List[float] = []

    for fold_index, held in enumerate(subjects):
        train = finite_array(frame.loc[frame["subject"].astype(str) != held, value_col])
        test = finite_array(frame.loc[frame["subject"].astype(str) == held, value_col])
        try:
            lp = np.asarray(
                logpdf_for_model(model_name, train, test, seed + fold_index * 1009),
                dtype=np.float64,
            )
            ok = bool(len(lp) == len(test) and np.all(np.isfinite(lp)))
            if not ok:
                raise AuditError("Non-finite LOSO log likelihood.")
            all_logpdf.extend(lp.tolist())
            fold_rows.append({
                "held_subject": held,
                "n_train": len(train),
                "n_test": len(test),
                "mean_test_nll": float(-np.mean(lp)),
                "status": "OK",
            })
        except Exception as exc:
            fold_rows.append({
                "held_subject": held,
                "n_train": len(train),
                "n_test": len(test),
                "mean_test_nll": None,
                "status": "FAILED",
                "error": repr(exc),
            })

    successful = sum(r["status"] == "OK" for r in fold_rows)
    return {
        "model": model_name,
        "folds_total": len(subjects),
        "folds_successful": successful,
        "n_test_total_successful": len(all_logpdf),
        "mean_test_nll": float(-np.mean(all_logpdf)) if all_logpdf else None,
        "folds": fold_rows,
        "subject_cluster_aware_split": True,
        "not_a_population_independence_proof": True,
    }


def fit_table_for_scope(
    modality: str,
    scope: str,
    values: np.ndarray,
    seed: int,
) -> List[dict]:
    out = []
    for r in fit_candidates(values, seed):
        out.append({
            "modality": modality,
            "scope": scope,
            "model": r.model,
            "n": r.n,
            "fit_status": r.fit_status,
            "log_likelihood": r.log_likelihood,
            "aic": r.aic,
            "bic": r.bic,
            "cdf_rmse": r.cdf_rmse,
            "ks_like_D_descriptive": r.ks_like_D,
            "boundary_clip_used": r.boundary_clip_used,
            "params_json": r.params_json,
            "note": r.note,
            "deployment_family_selected": False,
        })
    return out


def create_plots(
    out_dir: Path,
    modality: str,
    scope: str,
    x: np.ndarray,
    fit_rows: List[dict],
    seed: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise AuditError("matplotlib is required for plots.") from exc

    stats = import_stats()
    x = finite_array(x)
    if len(x) < 3 or np.std(x) <= 0:
        return

    # Histogram + candidate densities.
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(x, bins="auto", density=True, alpha=0.45, label="observed")

    grid_lo = max(0.0, float(np.min(x)) - 0.08)
    grid_hi = min(1.0, float(np.max(x)) + 0.08)
    if grid_hi <= grid_lo:
        grid_lo, grid_hi = float(np.min(x)) - 0.01, float(np.max(x)) + 0.01
    grid = np.linspace(grid_lo, grid_hi, 500)

    for row in fit_rows:
        if row["fit_status"] != "OK":
            continue
        name = row["model"]
        params = json.loads(row["params_json"])
        try:
            if name == "normal":
                y = stats.norm.pdf(grid, loc=params["mu"], scale=params["sigma"])
            elif name == "skew_normal":
                y = stats.skewnorm.pdf(
                    grid, params["shape"], loc=params["loc"], scale=params["scale"]
                )
            elif name == "beta_fixed_0_1":
                y = stats.beta.pdf(
                    np.clip(grid, EPS, 1-EPS),
                    params["alpha"], params["beta"], loc=0.0, scale=1.0
                )
            elif name.startswith("gmm_"):
                y = np.zeros_like(grid)
                for w, mu, sigma in zip(
                    params["weights"], params["means"], params["sigmas"]
                ):
                    y += float(w) * stats.norm.pdf(
                        grid, loc=float(mu), scale=float(sigma)
                    )
            else:
                continue
            if np.all(np.isfinite(y)):
                ax.plot(grid, y, linewidth=1.4, label=name)
        except Exception:
            continue

    ax.set_title(f"{modality.upper()} quality distribution — {scope}")
    ax.set_xlabel("quality")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scope}_{modality}_hist_fits.png", dpi=170)
    plt.close(fig)

    # Normal Q-Q.
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    stats.probplot(x, dist="norm", plot=ax)
    ax.set_title(f"{modality.upper()} Normal Q-Q — {scope}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{scope}_{modality}_qq_normal.png", dpi=170)
    plt.close(fig)

    # ECDF.
    xs = np.sort(x)
    ecdf = np.arange(1, len(xs)+1, dtype=float) / len(xs)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.step(xs, ecdf, where="post")
    ax.set_title(f"{modality.upper()} ECDF — {scope}")
    ax.set_xlabel("quality")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scope}_{modality}_ecdf.png", dpi=170)
    plt.close(fig)


def raw_feature_audit(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "subject", "pair_key", "condition", "modality",
        "available", "feature_path", "value"
    }
    require(required.issubset(features.columns),
            f"quality_numeric_features.csv missing: {sorted(required-set(features.columns))}")

    f = features.copy()
    f["available_bool"] = [as_bool(v, "feature available") for v in f["available"]]
    f = f[(f["condition"].astype(str) == "reference") & f["available_bool"]].copy()
    f["value_num"] = pd.to_numeric(f["value"], errors="coerce")
    f = f[np.isfinite(f["value_num"].to_numpy(dtype=float))].copy()

    window_rows = []
    for (modality, feature), g in f.groupby(["modality", "feature_path"], dropna=False):
        x = finite_array(g["value_num"])
        d = describe(x)
        nrm = normality_diagnostics(x) if len(x) >= 8 and len(np.unique(x)) >= 5 else {
            "status": "NOT_TESTED", "n": len(x), "diagnostic_only": True
        }
        window_rows.append({
            "level": "window",
            "modality": modality,
            "feature_path": feature,
            **d,
            "normality_json": json.dumps(nrm, ensure_ascii=False),
        })

    # Mean each raw feature over the 4 windows of the same clean trial.
    trial = (
        f.groupby(["subject", "pair_key", "modality", "feature_path"], as_index=False)
         .agg(mean_value=("value_num", "mean"), windows=("value_num", "size"))
    )

    trial_rows = []
    for (modality, feature), g in trial.groupby(["modality", "feature_path"], dropna=False):
        x = finite_array(g["mean_value"])
        d = describe(x)
        nrm = normality_diagnostics(x) if len(x) >= 8 and len(np.unique(x)) >= 5 else {
            "status": "NOT_TESTED", "n": len(x), "diagnostic_only": True
        }
        trial_rows.append({
            "level": "trial_mean",
            "modality": modality,
            "feature_path": feature,
            **d,
            "normality_json": json.dumps(nrm, ensure_ascii=False),
            "subject_clustered": True,
        })

    return (
        pd.DataFrame(window_rows),
        trial,
        pd.DataFrame(trial_rows),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit clean VAL multimodal quality distributions without changing deployment."
    )
    p.add_argument(
        "--deployment-root",
        help="最终部署 root. Defaults to auto-discovery from cwd/script parents.",
    )
    p.add_argument(
        "--run-dir",
        help="Formal balanced VAL robustness run directory. If omitted, latest valid run is used.",
    )
    p.add_argument(
        "--output-dir",
        help="New output directory. Default: <run-dir>/distribution_audit_<timestamp>",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG hist/Q-Q/ECDF plots.",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Synthetic math test only; does not open EAV/model/run files.",
    )
    return p.parse_args(argv)


def self_test() -> int:
    rng = np.random.default_rng(DEFAULT_SEED)
    normal = np.clip(rng.normal(0.8, 0.05, 60), 0.01, 0.99)
    skewed = np.clip(1.0 - rng.beta(2.0, 8.0, 60), 0.01, 0.99)
    checks = {
        "normal_describe_n": describe(normal)["n"] == 60,
        "normality_runs": normality_diagnostics(normal).get("status") == "OK",
        "candidate_count": len(fit_candidates(normal, DEFAULT_SEED)) == 6,
        "skew_detected_direction": describe(skewed)["skewness_moment"] < 0,
    }
    require(all(checks.values()), f"Self-test failed: {checks}")
    print(json_text({
        "status": "PASS",
        "version": VERSION,
        "checks": checks,
        "real_EAV_used": False,
        "trained_models_loaded": False,
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
            deployment_root = find_deployment_root(Path.cwd())
        except AuditError:
            deployment_root = find_deployment_root(Path(__file__).resolve().parent)

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else discover_latest_formal_run(deployment_root)
    )
    require(run_dir.is_dir(), f"Run directory not found: {run_dir}")

    run_summary = validate_run(run_dir)
    source_paths = validate_source_files(run_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / f"distribution_audit_{timestamp}"
    )
    require(not out.exists(), f"Output already exists; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)
    plots_dir = out / "plots"
    if not args.no_plots:
        plots_dir.mkdir(exist_ok=False)

    source_hashes = {k: sha256_file(v) for k, v in source_paths.items()}

    samples = pd.read_csv(source_paths["samples"], low_memory=False)
    features = pd.read_csv(source_paths["features"], low_memory=False)
    trials = pd.read_csv(source_paths["trials"], low_memory=False)

    # ------------------------------------------------------------------
    # Source contracts
    # ------------------------------------------------------------------
    sample_required = {
        "source_window_id", "subject", "pair_key", "condition", "kind",
        "modality", "available", "q"
    }
    trial_required = {
        "subject", "pair_key", "modality", "windows", "available_windows",
        "mean_q_available_only"
    }
    require(sample_required.issubset(samples.columns),
            f"Samples missing columns: {sorted(sample_required-set(samples.columns))}")
    require(trial_required.issubset(trials.columns),
            f"Trials missing columns: {sorted(trial_required-set(trials.columns))}")

    samples = samples.copy()
    samples["available_bool"] = [as_bool(v, "available") for v in samples["available"]]
    samples["q_num"] = pd.to_numeric(samples["q"], errors="coerce")
    require(np.all(np.isfinite(samples["q_num"].to_numpy(dtype=float))),
            "Non-finite q found in sample export.")
    require(np.all((samples["q_num"] >= 0.0) & (samples["q_num"] <= 1.0)),
            "q outside [0,1].")

    clean = samples[samples["condition"].astype(str) == "reference"].copy()
    clean_av = clean[clean["available_bool"]].copy()

    # Formal design checks.
    clean_counts = clean.groupby("modality").size().to_dict()
    trial_counts = trials.groupby("modality").size().to_dict()
    subjects = sorted(trials["subject"].dropna().astype(str).unique().tolist())

    for m in MODALITIES:
        require(clean_counts.get(m, 0) == EXPECTED_FORMAL_WINDOWS_PER_MODALITY,
                f"{m}: expected 120 clean window rows, got {clean_counts.get(m,0)}")
        require(trial_counts.get(m, 0) == EXPECTED_FORMAL_TRIALS_PER_MODALITY,
                f"{m}: expected 30 clean trial rows, got {trial_counts.get(m,0)}")
    require(len(subjects) == EXPECTED_FORMAL_SUBJECTS,
            f"Expected 6 VAL subjects in trial export, got {len(subjects)}")

    # ------------------------------------------------------------------
    # Calibrated-q audit: window descriptive + trial-mean primary.
    # ------------------------------------------------------------------
    descriptive_rows = []
    normality_rows = []
    fit_rows: List[dict] = []
    loso_rows: List[dict] = []
    loso_folds: List[dict] = []

    model_names = (
        "normal", "skew_normal", "beta_fixed_0_1",
        "gmm_1", "gmm_2", "gmm_3"
    )

    for mi, modality in enumerate(MODALITIES):
        w = clean_av[clean_av["modality"].astype(str) == modality].copy()
        wx = finite_array(w["q_num"])

        t = trials[trials["modality"].astype(str) == modality].copy()
        t["mean_q_available_only_num"] = pd.to_numeric(
            t["mean_q_available_only"], errors="coerce"
        )
        t = t[np.isfinite(t["mean_q_available_only_num"].to_numpy(dtype=float))].copy()
        tx = finite_array(t["mean_q_available_only_num"])

        for scope, x, primary in (
            ("clean_window_available_q", wx, False),
            ("clean_trial_mean_available_q", tx, True),
        ):
            d = describe(x)
            descriptive_rows.append({
                "modality": modality,
                "scope": scope,
                "primary_distribution_audit_unit": primary,
                **d,
                "window_rows_correlated_within_trial": scope.startswith("clean_window"),
                "trial_rows_clustered_within_subject": scope.startswith("clean_trial"),
            })
            nd = normality_diagnostics(x)
            normality_rows.append({
                "modality": modality,
                "scope": scope,
                "primary_distribution_audit_unit": primary,
                **nd,
                "subject_cluster_caveat": True,
            })

            local_fits = fit_table_for_scope(
                modality, scope, x, args.seed + mi * 10000
            )
            fit_rows.extend(local_fits)
            if not args.no_plots:
                create_plots(
                    plots_dir, modality, scope, x, local_fits,
                    args.seed + mi * 10000
                )

        # Subject-held-out predictive diagnostic on trial means.
        for model_index, model_name in enumerate(model_names):
            result = loso_predictive_nll(
                t,
                "mean_q_available_only_num",
                model_name,
                args.seed + mi * 10000 + model_index * 101,
            )
            loso_rows.append({
                "modality": modality,
                "scope": "clean_trial_mean_available_q",
                "model": model_name,
                "folds_total": result["folds_total"],
                "folds_successful": result["folds_successful"],
                "n_test_total_successful": result["n_test_total_successful"],
                "mean_test_nll": result["mean_test_nll"],
                "subject_cluster_aware_split": True,
                "diagnostic_only": True,
                "deployment_family_selected": False,
            })
            for fold in result["folds"]:
                loso_folds.append({
                    "modality": modality,
                    "model": model_name,
                    **fold,
                })

    desc_df = pd.DataFrame(descriptive_rows)
    norm_df = pd.DataFrame(normality_rows)
    fit_df = pd.DataFrame(fit_rows)
    loso_df = pd.DataFrame(loso_rows)
    folds_df = pd.DataFrame(loso_folds)

    # Diagnostic rankings only, not deployment selection.
    ranking_rows = []
    for modality in MODALITIES:
        g = loso_df[
            (loso_df["modality"] == modality)
            & (loso_df["folds_successful"] == EXPECTED_FORMAL_SUBJECTS)
            & loso_df["mean_test_nll"].notna()
        ].copy()
        g = g.sort_values(["mean_test_nll", "model"], ascending=[True, True])
        for rank, (_, row) in enumerate(g.iterrows(), start=1):
            ranking_rows.append({
                "modality": modality,
                "diagnostic_rank_by_loso_nll": rank,
                "model": row["model"],
                "mean_test_nll": row["mean_test_nll"],
                "deployment_family_selected": False,
            })
    ranking_df = pd.DataFrame(ranking_rows)

    # ------------------------------------------------------------------
    # Raw detector-feature audit.
    # ------------------------------------------------------------------
    raw_window_df, raw_trial_means_df, raw_trial_audit_df = raw_feature_audit(features)

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    atomic_csv(out / "calibrated_q_descriptive.csv", desc_df)
    atomic_csv(out / "calibrated_q_normality_diagnostics.csv", norm_df)
    atomic_csv(out / "calibrated_q_candidate_fits.csv", fit_df)
    atomic_csv(out / "calibrated_q_loso_predictive_fit.csv", loso_df)
    atomic_csv(out / "calibrated_q_loso_folds.csv", folds_df)
    atomic_csv(out / "calibrated_q_diagnostic_ranking.csv", ranking_df)
    atomic_csv(out / "raw_feature_window_audit.csv", raw_window_df)
    atomic_csv(out / "raw_feature_trial_means.csv", raw_trial_means_df)
    atomic_csv(out / "raw_feature_trial_audit.csv", raw_trial_audit_df)

    # Compact per-modality summary.
    compact = {}
    for modality in MODALITIES:
        drow = desc_df[
            (desc_df["modality"] == modality)
            & (desc_df["scope"] == "clean_trial_mean_available_q")
        ].iloc[0].to_dict()

        nrow = norm_df[
            (norm_df["modality"] == modality)
            & (norm_df["scope"] == "clean_trial_mean_available_q")
        ].iloc[0].to_dict()

        ranked = ranking_df[ranking_df["modality"] == modality].sort_values(
            "diagnostic_rank_by_loso_nll"
        )
        compact[modality] = {
            "trial_mean_descriptive": drow,
            "trial_mean_normality_diagnostic": nrow,
            "diagnostic_candidate_order_by_subject_LOSO_NLL":
                ranked[["model", "mean_test_nll"]].to_dict(orient="records"),
            "deployment_family_selected": False,
        }

    summary = {
        "version": VERSION,
        "status": "PASS",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "source_run_version": run_summary.get("version"),
        "source_status": run_summary.get("status"),
        "formal_balanced_val": True,
        "test_data_used": False,
        "router_threshold_retuned": False,
        "models_or_calibrators_modified": False,
        "distribution_model_deployed": False,
        "primary_unit": "clean 20s trial mean of available 5s quality values",
        "secondary_unit": "clean available 5s windows (descriptive; correlated within trial)",
        "subjects": subjects,
        "n_subjects": len(subjects),
        "source_hashes": source_hashes,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "packages": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "matplotlib": package_version("matplotlib"),
        },
        "interpretation_guardrails": [
            "Normality tests are diagnostics, not proof of population distribution.",
            "Thirty trial means are clustered within six VAL subjects.",
            "Window-level rows are correlated within trials and are secondary descriptive evidence.",
            "Calibrated q and raw detector features are audited separately.",
            "No corruption-condition rows are used to fit the nominal clean distribution.",
            "No TEST data are used.",
            "No router threshold, model, checkpoint or calibrator is modified.",
            "LOSO NLL ranking is diagnostic only and does not select a deployment family.",
        ],
        "modalities": compact,
        "raw_feature_groups_window": int(len(raw_window_df)),
        "raw_feature_groups_trial": int(len(raw_trial_audit_df)),
        "outputs": sorted([p.name for p in out.iterdir()]),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    atomic_json(out / "distribution_audit_summary.json", summary)

    # Human-readable report.
    lines = [
        "# EAV Quality Distribution Audit",
        "",
        f"- Version: `{VERSION}`",
        f"- Source: `{run_dir}`",
        "- Scope: formal balanced VAL, clean reference only for nominal fitting",
        "- TEST used: **No**",
        "- Router/model/calibration changed: **No**",
        "- Deployment distribution selected: **No**",
        "",
        "## Primary unit",
        "",
        "Clean 20-s trial means (30 trials per modality).",
        "The 120 five-second windows remain secondary descriptive evidence.",
        "Trials are still clustered within six subjects.",
        "",
        "## Calibrated q summary",
        "",
        "| Modality | Trial mean | Median | SD | Skewness | Excess kurtosis |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for modality in MODALITIES:
        r = desc_df[
            (desc_df["modality"] == modality)
            & (desc_df["scope"] == "clean_trial_mean_available_q")
        ].iloc[0]
        lines.append(
            f"| {modality} | {r['mean']:.4f} | {r['median']:.4f} | "
            f"{r['std_sample']:.4f} | {r['skewness_moment']:.4f} | "
            f"{r['excess_kurtosis_moment']:.4f} |"
        )

    lines += [
        "",
        "## Candidate fit interpretation",
        "",
        "Use `calibrated_q_loso_predictive_fit.csv` and "
        "`calibrated_q_diagnostic_ranking.csv` as diagnostic evidence only.",
        "Do not change the frozen deployment router from this script alone.",
        "",
        "## Raw detector features",
        "",
        "`raw_feature_trial_audit.csv` is the primary table for checking whether "
        "DNSMOS OVRL/RMS, DOVER Technical, EEG physical-quality features, etc. "
        "have different distributional forms before q calibration.",
        "",
        "## Next design gate",
        "",
        "After reviewing these outputs, freeze one candidate family per modality "
        "on VAL only, then implement a separate distribution-aware runtime "
        "calibrator alongside—not over—the existing tau=0.80 router.",
        "",
    ]
    atomic_text(out / "README_audit.md", "\n".join(lines))

    print("=" * 108)
    print("EAV QUALITY DISTRIBUTION AUDIT")
    print("=" * 108)
    print(f"Status                    : PASS")
    print(f"Version                   : {VERSION}")
    print(f"Source formal run         : {run_dir}")
    print(f"Output                    : {out}")
    print(f"Clean windows/modality    : {EXPECTED_FORMAL_WINDOWS_PER_MODALITY}")
    print(f"Clean trials/modality     : {EXPECTED_FORMAL_TRIALS_PER_MODALITY}")
    print(f"VAL subjects              : {len(subjects)}")
    print("TEST used                 : False")
    print("Router/model changed      : False")
    print("Deployment family selected: False")
    print()
    for modality in MODALITIES:
        r = desc_df[
            (desc_df["modality"] == modality)
            & (desc_df["scope"] == "clean_trial_mean_available_q")
        ].iloc[0]
        top = ranking_df[
            ranking_df["modality"] == modality
        ].sort_values("diagnostic_rank_by_loso_nll").head(1)
        top_text = (
            f"{top.iloc[0]['model']} (LOSO NLL={top.iloc[0]['mean_test_nll']:.4f})"
            if len(top) else "no complete candidate"
        )
        print(
            f"{modality.upper():5s} trial q: mean={r['mean']:.4f} "
            f"median={r['median']:.4f} skew={r['skewness_moment']:.4f} "
            f"kurt={r['excess_kurtosis_moment']:.4f} | "
            f"diagnostic best={top_text}"
        )
    print()
    print("Primary outputs:")
    print(f"  {out / 'distribution_audit_summary.json'}")
    print(f"  {out / 'calibrated_q_normality_diagnostics.csv'}")
    print(f"  {out / 'calibrated_q_loso_predictive_fit.csv'}")
    print(f"  {out / 'raw_feature_trial_audit.csv'}")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nINTERRUPTED.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nQUALITY DISTRIBUTION AUDIT ERROR: {exc}", file=sys.stderr)
        raise
