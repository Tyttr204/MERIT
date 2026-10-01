#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""EEG Quality V1 -- source EEG -> EQ1 features -> EQ2 q -> frozen AF4-C.

Single-file runtime, based on the supplied EQ1 and EQ2, not a new detector fit.
Default candidate: pyprep_physical (explicit runtime choice; JSON is unchanged).
Required asset: eeg_quality_calibration.json (full EQ2 export, never a CV fold).
No EQ1/EQ2/training script import, fitting, automatic downloads or installation.
No E4 checkpoint is loaded; CURRENT five-class E4 probabilities are external.

Frozen quality geometry: 30 E4 electrodes, 500 Hz, exactly 2500 samples / 5 s,
BEFORE E4's 500->200 Hz resampling and frozen Train standardization. The first
version deliberately rejects other rates instead of silently moving the model's
input boundary. New acquisition hardware/reference/filtering needs validation.

Unit conversion is explicit (V/mV/uV or known volts/source-unit scalar). We do
not establish the physical unit from amplitude. Calibration's original unit
provenance remains recorded, including its unverified-evidence flag.

On a COPY: exact EQ1 PyPREP methods and Welch/flat/hold computations. No repair,
ICA, interpolation, re-reference, notch, normalization, or E4 input mutation.
Partial finite flat channels / low q are NOT independently hard-masked. Fewer
than two PyPREP-usable channels makes relative detection incomplete: an ERROR,
not a fabricated calibrated score. Default detector error policy is 'raise';
explicit 'unavailable' is a protective software fallback, NOT fault diagnosis.

Capture error, incomplete/nonfinite payload, all-channel flatline, stale or
noncontiguous samples -> unavailable; fixed [.2]*5 clears old EEG probabilities
BEFORE F4/AF4-B. Live freshness is checked before AND after scoring and again
before fusion. Device timestamps must be converted to time.monotonic's domain.
A source waveform hash/window ID can be echoed by the E4 head for alignment.
One EEGQualityV1/EEGWindowBuffer per acquisition stream; full new windows after
buffer gaps. No temporal q smoothing or 20-second quality aggregation is added.

q_eeg is NOT a contribution weight. Optional AF4CEEGBridge calls your trusted
local FinalAF4CSystem, never its main/evaluate function. It reports actual EEG
alpha/class weights ONLY on the active AF4-B adaptive branch, not attribution
through the F4 residual path. F4 / NO_DECISION -> eeg_weight=None.

Commands (existing repaired .venv-video; do not upgrade dependencies):
  python -X utf8 eeg_quality_v1_deployment.py --self-test
  python -X utf8 eeg_quality_v1_deployment.py --check-assets --calibration PATH
  python -X utf8 eeg_quality_v1_deployment.py --preflight --calibration PATH
  python -X utf8 eeg_quality_v1_deployment.py --eeg window.npy --sample-rate 500 \
    --eeg-unit uV --assume-e4-channel-order --window-id w001 --calibration PATH

Hardware acquisition drivers, E4 emotion inference, and other modality heads
remain external. Software preflight is not evidence of real sensor reliability.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

VERSION = 'EEG-QUALITY-V1.0'
SCHEMA = 'eav.eeg_quality.monotonic.v1'
TAU = 0.80
FS, CHANNEL_COUNT, WINDOW_SAMPLES = 500, 30, 2500
WINDOW_SECONDS = 5.0
PYPREP_VALIDATION_VERSION = '0.6.0'
CALIBRATION_NAME = 'eeg_quality_calibration.json'
KNOWN_EQ2_RUN = 'eq2_eeg_quality_20260919_185311_974821'
CANDIDATES = ('pyprep_only', 'pyprep_physical')
SEVERITIES = ('clean', 'mild', 'medium', 'severe')
FEATURES = ('pyprep_bad_fraction', 'pyprep_hf_bad_fraction',
            'pyprep_correlation_bad_fraction', 'raw_flat_channel_fraction',
            'raw_hold_fraction_mean', 'line_fraction_mean',
            'hf55_90_fraction_mean', 'slow02_1_fraction_mean')
CHANNELS = ['FP1','FP2','F7','F3','FZ','F4','F8','FC5','FC1','FC2','FC6','T7','C3','CZ','C4','T8',
            'CP5','CP1','CP2','CP6','P7','P3','PZ','P4','P8','PO9','O1','OZ','O2','PO10']
EMOTIONS = ['Neutral','Sadness','Anger','Happiness','Calmness']
MODALITIES = ['eeg','audio','video']
BAD_TYPES = ('bad_by_nan','bad_by_flat','bad_by_deviation','bad_by_hf_noise',
             'bad_by_correlation','bad_by_dropout','bad_by_SNR')
# PyPREP's output-capture/logging contexts modify process-wide state. Serialize
# our detector calls; keep this work OUT of a hardware acquisition callback.
_DETECTOR_LOCK = threading.RLock()

class InputContractError(ValueError):
    """Incompatible metadata/model/feature contract; never guess or relabel."""

ContractError = InputContractError

class QualityDetectionError(RuntimeError):
    """Detector/calibrator failed; this is NOT evidence of a hardware failure."""

class PayloadUnavailable(ValueError):
    """A current payload cannot be used; the reason code is preserved."""

SOURCE_SCRIPT_SHA256 = {'eav_eq1_eeg_quality_validation(1).py': '17a2f1c5b1d4347f133f1573ea13630bdc0b87d2518ed4e05f334750c2438b41', 'eav_eq2_calibrate_eeg_quality(1).py': '99e118de2725de6220833ca91a215eecc82f8303801fabf82cf4a90199ac633f'}

def require(ok: bool, message: str) -> None:
    if not ok:
        raise InputContractError(message)

def strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ('true','false','0','1'):
        return value.strip().lower() in ('true','1')
    raise InputContractError(f'{name}: explicit boolean/0/1 required, got {value!r}')

def finite(value: Any, name: str) -> float:
    require(not isinstance(value, (bool, np.bool_)), f'{name} is not boolean')
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise InputContractError(f'{name}: numeric scalar required') from exc
    require(math.isfinite(x), f'{name}: NaN/Inf or missing value')
    return x

def exact_int(value: Any, name: str) -> int:
    x = finite(value, name)
    require(x == math.floor(x), f'{name}: integer required')
    return int(x)

def json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj

def canonical_json(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False)

def object_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode('utf-8')).hexdigest()

def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def valid_sha(value: Any) -> bool:
    return bool(re.fullmatch('[a-fA-F0-9]{64}', str(value)))

def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(obj, dict), f'Expected JSON object: {path}')
    return obj

def validate_targets(targets: Mapping[str, float]) -> dict[str, float]:
    require(set(targets) == set(SEVERITIES), 'Need clean/mild/medium/severe targets')
    t = {s: finite(targets[s], s) for s in SEVERITIES}
    require(1 >= t['clean'] >= TAU > t['mild'] > t['medium'] > t['severe'] > 0,
            'Targets must satisfy 1>=clean>=.8>mild>medium>severe>0')
    return t

def validate_curve(curve: Mapping[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    require(curve.get('kind') == 'decreasing_piecewise_linear' and curve.get('feature') == FEATURES[0]
            and curve.get('out_of_bounds') == 'clip', 'Unsupported isotonic curve schema')
    x,y = np.asarray(curve.get('x'),float), np.asarray(curve.get('y'),float)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 0, 'Bad curve dimensions')
    require(np.isfinite(x).all() and np.isfinite(y).all() and (np.diff(x)>0).all()
            and (np.diff(y)<=1e-12).all(), 'Non-monotone or nonfinite curve')
    require(np.all((x>=0)&(x<=1)) and np.all((y>=0)&(y<=1)), 'Curve outside fractional domain')
    return x,y

def validate_trees(spec: Mapping[str,Any]) -> None:
    require(spec.get('kind') == 'numeric_additive_trees_identity' and spec.get('features') == list(FEATURES), 'Wrong tree feature order/kind')
    require(spec.get('leaf_values_include_learning_rate') is True and spec.get('missing_feature_policy') == 'raise',
            'Wrong tree scaling/missing policy')
    finite(spec.get('initial_prediction'), 'initial_prediction')
    trees = spec.get('trees')
    require(isinstance(trees,list) and 1 <= len(trees) <= 2000, 'Invalid tree count')
    for tree in trees:
        require(isinstance(tree,list) and 1 <= len(tree) <= 511, 'Invalid node count')
        parents = np.zeros(len(tree),int)
        for i,n in enumerate(tree):
            require(isinstance(n,dict) and type(n.get('leaf')) is bool, 'Invalid node')
            if n['leaf']:
                finite(n.get('value'), 'leaf value')
            else:
                k = exact_int(n.get('feature'), 'split feature')
                require(0 <= k < len(FEATURES), 'Invalid split feature')
                t = finite(n.get('threshold'), 'split threshold')
                require(0 <= t <= 1, 'Split outside feature fraction domain')
                l,r = exact_int(n.get('left'),'left'), exact_int(n.get('right'),'right')
                require(i < l < len(tree) and i < r < len(tree) and l != r, 'Cyclic/invalid tree edge')
                parents[l] += 1; parents[r] += 1
        require(parents[0] == 0 and (parents[1:] == 1).all(), 'Disconnected/shared tree nodes')

def tree_predict(spec: Mapping[str,Any], x: np.ndarray) -> np.ndarray:
    """Finite float64 [N,8]; caller validates schema/input once."""
    result = np.full(len(x), float(spec['initial_prediction']), dtype=np.float64)
    for tree in spec['trees']:
        todo = [(0,np.arange(len(x)))]
        while todo:
            i, idx = todo.pop()
            if not len(idx):
                continue
            n = tree[i]
            if n['leaf']:
                result[idx] += n['value']
            else:
                left = x[idx,n['feature']] <= n['threshold']
                todo.append((n['right'],idx[~left])); todo.append((n['left'],idx[left]))
    return result

class EEGQualityCalibrator:
    """Portable JSON candidate. No sklearn/PyPREP/Torch needed to apply a map."""
    def __init__(self, artifact: str | Path | Mapping[str,Any], *, candidate: str,
                 expected_feature_contract_id: str | None = None, allow_preflight: bool = False):
        a = read_json(Path(artifact)) if isinstance(artifact,(str,Path)) else copy.deepcopy(dict(artifact))
        require(a.get('schema') == SCHEMA and a.get('stage') == 'EQ2', 'Unsupported artifact')
        require(candidate in CANDIDATES, f'Explicit candidate must be one of {CANDIDATES}')
        require(a.get('router_tau') == TAU and a.get('formal_test_used') is False and a.get('class_order') == EMOTIONS,
                'Frozen fusion protocol mismatch')
        require(a.get('artifact_state') in ('PREFLIGHT_ONLY','CANDIDATES_NOT_SELECTED'), 'Artifact invalid/incomplete')
        require(allow_preflight or a['artifact_state'] != 'PREFLIGHT_ONLY', 'PREFLIGHT_ONLY artifact cannot be deployed')
        t = validate_targets(a['proxy_targets'])
        require(a.get('quality_clip') == [t['severe'],t['clean']], 'Invalid clipping contract')
        ic = a.get('input_contract',{})
        require(ic.get('sample_rate_hz') == 500 and ic.get('window_samples') == 2500 and
                ic.get('channel_order') == CHANNELS and ic.get('runtime_features') == list(FEATURES), 'EEG feature geometry/order mismatch')
        require(object_hash(ic) == a.get('feature_contract_id'), 'Feature contract identity mismatch')
        if expected_feature_contract_id is not None:
            require(a['feature_contract_id'] == expected_feature_contract_id, 'Caller extractor identity mismatch')
        spec = a['candidates'][candidate]
        features = [FEATURES[0]] if candidate == 'pyprep_only' else list(FEATURES)
        require(spec.get('features') == features, 'Candidate feature order changed')
        if candidate == 'pyprep_only':
            self.knots = validate_curve(spec['model'])
        else:
            require(spec.get('coordinate_constraints') == [-1]*len(FEATURES), 'Wrong monotonic direction')
            validate_trees(spec['model'])
            self.knots = None
        for f in FEATURES:
            rr = a.get('observed_feature_ranges',{}).get(f)
            require(isinstance(rr,list) and len(rr) == 2, 'Missing training feature range')
            lo,hi = finite(rr[0],f),finite(rr[1],f)
            require(0 <= lo <= hi <= 1, 'Invalid observed feature range')
        self.artifact = a; self.candidate = candidate; self.features = features
        self.model_spec = spec['model']; self.clip = a['quality_clip']
        self.feature_contract_id = a['feature_contract_id']

    def predict_values(self, values: Any) -> np.ndarray:
        x = np.asarray(values,dtype=np.float64)
        require(x.ndim == 2 and x.shape[1] == len(self.features), 'Expected [N,number_of_candidate_features]')
        require(np.isfinite(x).all() and np.all((x>=0)&(x<=1)), 'Current features must be finite fractions [0,1]')
        if self.candidate == 'pyprep_only':
            p = np.interp(x[:,0], *self.knots)
        else:
            p = tree_predict(self.model_spec,x)
        require(np.isfinite(p).all(), 'Calibrator produced nonfinite q')
        return np.clip(p,*self.clip)

    def assess_features(self, features: Mapping[str,Any] | None, *, payload_available: bool,
                        detector_status: str | None = None, pyprep_method_status: str | None = None,
                        feature_contract_id: str | None = None) -> dict[str,Any]:
        available = strict_bool(payload_available,'payload_available')
        if not available:
            return dict(candidate=self.candidate,q_eeg=0.,eeg_available=False,quality_state='UNAVAILABLE',
                        reason='NO_CURRENT_EEG_EVIDENCE',quality_model_used=False,eeg_side_healthy=False,
                        router_tau=TAU,feature_contract_verified=False)
        require(isinstance(features,Mapping), 'Available EEG needs current feature dictionary')
        # Status may be passed explicitly or supplied in an unmodified EQ1-like row.
        status = detector_status if detector_status is not None else features.get('detector_status')
        method = pyprep_method_status if pyprep_method_status is not None else features.get('pyprep_method_status')
        require(status == 'SCORED' and method == 'FULL_SUBSET', 'Detector failed/limited or status not supplied')
        if feature_contract_id is not None:
            require(feature_contract_id == self.feature_contract_id, 'Feature extractor identity differs from calibration')
        vals = [finite(features.get(f),f) for f in self.features]
        if 'raw_flat_channel_fraction' in features:
            flat = finite(features['raw_flat_channel_fraction'],'raw_flat_channel_fraction')
            require(0 <= flat < 1, 'All-flat or invalid payload cannot be declared available')
        q = float(self.predict_values([vals])[0])
        oor = [f for f,v in zip(self.features,vals) if not self.artifact['observed_feature_ranges'][f][0] <= v <= self.artifact['observed_feature_ranges'][f][1]]
        return dict(candidate=self.candidate,q_eeg=q,eeg_available=True,
                    quality_state='HEALTHY' if q>=TAU else 'DEGRADED',
                    reason='CALIBRATED_CURRENT_FEATURES',quality_model_used=True,
                    eeg_side_healthy=q>=TAU,router_tau=TAU,outside_fit_range_features=oor,
                    feature_contract_verified=feature_contract_id is not None,
                    q_is_actual_fusion_weight=False)

    def make_fusion_input(self, eeg_probs: Sequence[float] | None, *, features: Mapping[str,Any] | None = None,
                          payload_available: bool, classifier_ok: bool = True,
                          detector_status: str | None = None, pyprep_method_status: str | None = None,
                          feature_contract_id: str | None = None) -> dict[str,Any]:
        available = strict_bool(payload_available,'payload_available')
        classifier = strict_bool(classifier_ok,'classifier_ok')
        if not available or not classifier:
            return dict(eeg_probs=[.2]*5,q_eeg=0.,eeg_available=False)
        assessment = self.assess_features(features,payload_available=True,detector_status=detector_status,
                                         pyprep_method_status=pyprep_method_status,feature_contract_id=feature_contract_id)
        p = np.asarray(eeg_probs,dtype=float)
        require(p.shape == (5,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and
                abs(float(p.sum())-1) <= 1e-3, 'Need current five-class probabilities, not logits/stale values')
        return dict(eeg_probs=(p/p.sum()).tolist(),q_eeg=assessment['q_eeg'],eeg_available=True)


# Exact EQ1 arithmetic and detector core retained below.

def json_text(obj: Any, pretty: bool = False) -> str:
    return json.dumps(json_safe(obj), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2 if pretty else None)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def array_sha(x: np.ndarray) -> str:
    a = np.ascontiguousarray(x, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()

def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def unit_scale(unit: str | None, explicit_scale: float | None) -> tuple[float, str]:
    require(not (unit and explicit_scale is not None), "Use --eeg-unit OR --scale-to-volts")
    require(unit is not None or explicit_scale is not None,
            "EEG amplitude units are not established by the supplied E4 script. "
            "Scoring requires --eeg-unit V|mV|uV or --scale-to-volts after verifying the source unit. "
            "Do NOT guess uV from amplitude. --dry-run and --self-test remain available.")
    value = {"V": 1., "mV": 1e-3, "uV": 1e-6}[unit] if unit else float(explicit_scale)
    require(math.isfinite(value) and value > 0, "Scale to volts must be finite and positive")
    return value, unit or "custom_scale"

def centered_rms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    a = x - np.mean(x, axis=axis, keepdims=True)
    return np.sqrt(np.mean(a * a, axis=axis))

def flat_run_metrics(x: np.ndarray, minimum_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Exact repeated ADC samples. Long holds are clues, not proof of device failure."""
    longest, fraction = [], []
    for a in x:
        cuts = np.r_[0, np.flatnonzero(np.diff(a) != 0) + 1, len(a)]
        lengths = np.diff(cuts)
        longest.append(float(lengths.max() / FS))
        fraction.append(float(lengths[lengths >= minimum_samples].sum() / len(a)))
    return np.array(longest), np.array(fraction)

def spectral_and_raw_features(x_volts: np.ndarray, line_hz: float) -> dict[str, Any]:
    from scipy.signal import welch
    require(x_volts.shape == (30, WINDOW_SAMPLES) and np.isfinite(x_volts).all(), "Bad feature input")
    long, hold = flat_run_metrics(x_volts)
    f, psd = welch(x_volts, fs=FS, window="hann", nperseg=WINDOW_SAMPLES,
                   noverlap=0, detrend="constant", scaling="density", axis=-1)
    df = f[1] - f[0]
    def power(lo: float, hi: float) -> np.ndarray:
        return psd[:, (f >= lo) & (f < hi)].sum(axis=1) * df
    def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    # Ratios use raw quality input, not PyPREP's internally high/low-pass-filtered copy.
    base = power(1., 100.)
    line = ratio(power(line_hz-1., line_hz+1.), base)
    hf = ratio(power(55., 90.), power(1., 90.))
    slow = ratio(power(.2, 1.), power(.2, 90.))
    rms = centered_rms(x_volts) * 1e6
    return {
        "raw_flat_channel_fraction": float(np.mean(np.ptp(x_volts, axis=1) == 0)),
        "raw_hold_fraction_mean": float(hold.mean()), "raw_hold_fraction_max": float(hold.max()),
        "raw_hold_longest_seconds_max": float(long.max()),
        "rms_uv_median": float(np.median(rms)), "rms_uv_p90": float(np.percentile(rms, 90)),
        "peak_to_peak_uv_max": float(np.max(np.ptp(x_volts, axis=1))*1e6),
        "line_fraction_mean": float(line.mean()), "line_fraction_p90": float(np.percentile(line, 90)),
        "hf55_90_fraction_mean": float(hf.mean()), "hf55_90_fraction_p90": float(np.percentile(hf, 90)),
        "slow02_1_fraction_mean": float(slow.mean()), "slow02_1_fraction_p90": float(np.percentile(slow, 90)),
        "max_abs_channel_median_uv": float(np.max(np.abs(np.median(x_volts, axis=1)))*1e6),
    }

class _EQ1PyPREPDetector:
    """Explicit public methods; no pipeline cleaning and no implicit new methods."""
    def __init__(self, *, do_detrend: bool = True, allow_other_version: bool = False):
        try:
            import mne
            from pyprep import NoisyChannels
        except ImportError as exc:
            raise RuntimeError("PyPREP is unavailable. Use the verified multimodal environment; do not blindly upgrade its dependencies.") from exc
        pv = package_version("pyprep")
        require(pv == PYPREP_VALIDATION_VERSION or allow_other_version,
                f"EQ1 protocol pins pyprep=={PYPREP_VALIDATION_VERSION}; found {pv}. "
                "Use the pinned version, or explicitly --allow-other-pyprep-version for a separate exploratory run.")
        required = ["find_bad_by_deviation", "find_bad_by_hfnoise", "find_bad_by_correlation", "find_bad_by_SNR", "get_bads"]
        require(all(callable(getattr(NoisyChannels, m, None)) for m in required), "PyPREP method contract changed")
        self.mne, self.cls, self.do_detrend = mne, NoisyChannels, bool(do_detrend)
        module_path = Path(inspect.getfile(NoisyChannels))
        self.identity = {
            "algorithm": "PyPREP NoisyChannels detection-only; explicit method subset",
            "pyprep": pv, "mne": mne.__version__, "numpy": np.__version__, "scipy": package_version("scipy"),
            "source_module_sha256": sha256_file(module_path), "do_detrend": self.do_detrend,
            "ransac_used": False, "pyprep_PSD_extension_used": False, "interpolation_used": False,
            "reference_changed": False, "expected_input_units": "volts", "analysis_seconds": 5.,
            "flat_threshold_volts": 1e-15, "deviation_threshold": 5., "HF_zscore_threshold": 5.,
            "correlation_secs": 1., "correlation_threshold": .4, "frac_bad": .01,
            "five_second_adaptation_validated": False,
        }

    def score(self, x_volts: np.ndarray, seed: int = 0) -> dict[str, Any]:
        require(x_volts.shape == (30, WINDOW_SAMPLES) and np.isfinite(x_volts).all(), "Invalid PyPREP input")
        before = array_sha(x_volts)
        info = self.mne.create_info(CHANNELS, sfreq=FS, ch_types=["eeg"]*30)
        captured_stdout = io.StringIO()
        with warnings.catch_warnings(record=True) as ww, contextlib.redirect_stdout(captured_stdout), self.mne.use_log_level("ERROR"):
            warnings.simplefilter("always")
            raw = self.mne.io.RawArray(np.array(x_volts, dtype=np.float64, copy=True), info, verbose=False)
            nc = self.cls(raw, do_detrend=self.do_detrend, random_state=int(seed), matlab_strict=False)
            require(hasattr(nc, "n_chans_new"), "PyPREP usable-channel result contract changed")
            usable = int(nc.n_chans_new)
            method_status = "FULL_SUBSET"
            if usable >= 2:
                nc.find_bad_by_deviation(deviation_threshold=5.)
                nc.find_bad_by_hfnoise(HF_zscore_threshold=5.)
                nc.find_bad_by_correlation(correlation_secs=1., correlation_threshold=.4, frac_bad=.01)
                nc.find_bad_by_SNR()
            else:
                # No valid inter-channel statistic exists; not silently filled as good.
                method_status = "INSUFFICIENT_USABLE_CHANNELS_FOR_RELATIVE_METHODS"
            bad_by = {}
            for name in BAD_TYPES:
                require(hasattr(nc, name), f"PyPREP result contract changed: missing {name}")
                names = list(getattr(nc, name))
                require(set(names).issubset(CHANNELS), f"Unknown channel from PyPREP: {names}")
                bad_by[name] = [c for c in CHANNELS if c in names]
        require(array_sha(x_volts) == before, "Detector changed the caller's signal")
        bad = [c for c in CHANNELS if any(c in v for v in bad_by.values())]
        out = {
            "pyprep_method_status": method_status, "pyprep_usable_channels": usable,
            "pyprep_bad_count": len(bad), "pyprep_bad_fraction": len(bad)/30.,
            "pyprep_hf_bad_fraction": len(bad_by["bad_by_hf_noise"])/30.,
            "pyprep_correlation_bad_fraction": len(bad_by["bad_by_correlation"])/30.,
            "pyprep_dropout_bad_fraction": len(bad_by["bad_by_dropout"])/30.,
            "bad_channels_json": json_text(bad), "bad_by_type_json": json_text(bad_by),
            "pyprep_warnings_json": json_text(sorted({str(w.message) for w in ww})),
            "pyprep_stdout_excerpt": captured_stdout.getvalue()[:1200],
        }
        # Deterministic public flags are required. Private diagnostics are NOT needed for calibration.
        return out

# =============================================================================
# Runtime assets, immutable feature semantics, and input adapters
# =============================================================================

def atomic_json(path: str | Path, obj: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json_text(obj, True) + '\n', encoding='utf-8')
        os.replace(temp, p)
    finally:
        temp.unlink(missing_ok=True)


def resolve_calibration(explicit: str | Path | None = None,
                        assets_dir: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        require(p.is_file(), f'Calibration not found: {p}')
        return p
    if assets_dir is not None:
        root = Path(assets_dir).expanduser().resolve()
        choices = [root / CALIBRATION_NAME, root / 'models' / CALIBRATION_NAME]
    else:
        here = Path(__file__).resolve().parent
        choices = [here / CALIBRATION_NAME, Path.cwd() / CALIBRATION_NAME]
        for root in dict.fromkeys([Path.cwd(), here, *here.parents]):
            choices.append(root / 'EAV_dataset/models/eeg_quality' / KNOWN_EQ2_RUN / CALIBRATION_NAME)
    existing = list(dict.fromkeys(p.resolve() for p in choices if p.is_file()))
    require(bool(existing), 'No eeg_quality_calibration.json found. Use --calibration PATH or '
            '--assets-dir. No latest-run search, automatic refit, or download is performed.')
    hashes = {digest_file(p) for p in existing}
    require(len(hashes) == 1, 'Conflicting calibration copies found; pin --calibration explicitly: '
            + '; '.join(map(str, existing)))
    return existing[0]


def validate_extractor_contract(a: Mapping[str, Any]) -> None:
    """Validate EVERYTHING computed here, not only number of feature columns."""
    ic = a['input_contract']; d = ic['detector']
    fixed = {'schema': 'eav.eq1.features.v1', 'sample_rate_hz': FS,
             'window_samples': WINDOW_SAMPLES, 'window_seconds': WINDOW_SECONDS,
             'channel_order': CHANNELS, 'runtime_features': list(FEATURES),
             'bad_fraction_denominator': 30, 'expected_method_status': 'FULL_SUBSET',
             'waveform_stage': 'released EAV source window BEFORE E4 500->200 resampling and Train standardization'}
    for k, value in fixed.items():
        require(ic.get(k) == value, f'Unsupported feature contract: {k}={ic.get(k)!r}')
    require(ic.get('source_eq1_version') in ('EQ1-PYPREP.1', 'EQ1-PYPREP.1.1-SPEAKING-DISCOVERY'),
            'Unknown source EQ1 implementation.')
    require(ic.get('line_hz') in (50.0, 60.0), 'Invalid line-frequency feature definition.')
    spectral = dict(method='scipy.signal.welch', fs=500, window='hann', nperseg=2500,
                    noverlap=0, detrend='constant', scaling='density', bands_half_open=True,
                    line_fraction='power(line_hz-1,line_hz+1)/power(1,100), then channel mean',
                    hf55_90_fraction='power(55,90)/power(1,90), then channel mean',
                    slow02_1_fraction='power(.2,1)/power(.2,90), then channel mean',
                    zero_denominator='ratio=0, paired with flat/hold flags; not automatically healthy')
    require(ic.get('spectral') == spectral, 'Welch/band semantics differ from the supplied EQ1.')
    require(ic.get('hold') == dict(rule='exact consecutive repeated values, run length>=10 samples',
            aggregation='mean over 30 channels of qualifying samples/2500'), 'Hold feature rule mismatch.')
    for k, value in {'pyprep': '0.6.0', 'analysis_seconds': 5., 'expected_input_units': 'volts',
                     'flat_threshold_volts': 1e-15, 'deviation_threshold': 5.,
                     'HF_zscore_threshold': 5., 'correlation_secs': 1.,
                     'correlation_threshold': .4, 'frac_bad': .01,
                     'ransac_used': False, 'pyprep_PSD_extension_used': False,
                     'interpolation_used': False, 'reference_changed': False}.items():
        require(d.get(k) == value, f'Unsupported detector parameter {k}: {d.get(k)!r}')
    require(type(d.get('do_detrend')) is bool and valid_sha(d.get('source_module_sha256')),
            'Detector detrend/source fingerprint missing.')
    for k in ('mne', 'numpy', 'scipy'):
        require(isinstance(d.get(k), str) and bool(d[k]), f'Missing pinned {k} version.')
    unit = ic.get('unit_contract', {})
    require(finite(unit.get('volts_per_source_unit'), 'calibration source scale') > 0,
            'Bad source scale provenance.')
    if unit.get('unit') in ('V', 'mV', 'uV'):
        require(unit['volts_per_source_unit'] == {'V': 1., 'mV': 1e-3, 'uV': 1e-6}[unit['unit']],
                'Source named unit disagrees with its scale.')
    require(set(FEATURES).isdisjoint(a.get('forbidden_runtime_predictors', [])),
            'A forbidden metadata field has appeared among runtime features.')


def environment_check(artifact: Mapping[str, Any], *, verify_import: bool = False) -> dict[str, Any]:
    expected = artifact['input_contract']['detector']
    actual = {name: package_version(name) for name in ('numpy', 'scipy', 'mne', 'pyprep')}
    mismatches = {name: {'expected': expected[name], 'actual': v}
                  for name, v in actual.items() if v != expected[name]}
    report = {'status': 'PASS' if not mismatches else 'MISMATCH',
              'expected': {n: expected[n] for n in actual}, 'actual': actual,
              'mismatches': mismatches, 'module_sha256_verified': False,
              'packages_modified': False}
    if verify_import and not mismatches:
        det = _get_verified_detector(artifact)
        report['detector_identity'] = det.identity
        report['module_sha256_verified'] = True
    return report


def _get_verified_detector(artifact: Mapping[str, Any]) -> _EQ1PyPREPDetector:
    expected = artifact['input_contract']['detector']
    env = environment_check(artifact)
    require(env['status'] == 'PASS', 'Detector dependency versions differ from the calibrated run: '
            + canonical_json(env['mismatches']) + '. Use your verified environment; do not upgrade blindly.')
    det = _EQ1PyPREPDetector(do_detrend=expected['do_detrend'])
    for key in ('pyprep', 'mne', 'numpy', 'scipy', 'source_module_sha256', 'do_detrend',
                'expected_input_units', 'analysis_seconds', 'flat_threshold_volts',
                'deviation_threshold', 'HF_zscore_threshold', 'correlation_secs',
                'correlation_threshold', 'frac_bad', 'ransac_used', 'interpolation_used',
                'reference_changed', 'pyprep_PSD_extension_used'):
        require(det.identity.get(key) == expected.get(key), f'Detector identity mismatch: {key}. '
                'No scoring performed with this differing implementation.')
    return det


def canonical_missing_eeg(cached_probs: Any = None) -> dict[str, Any]:
    # Ignore even malformed stale probabilities when no current evidence exists.
    return {'eeg_probs': [.2] * 5, 'q_eeg': 0.0, 'eeg_available': False}


def normalize_probs(values: Any, name: str = 'probabilities') -> list[float]:
    try:
        p = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputContractError(f'{name}: five current probabilities required.') from exc
    require(p.shape == (5,) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            and abs(float(p.sum()) - 1) <= 1e-3, f'{name}: expected 5 normalized probabilities, not logits.')
    return (p / p.sum()).tolist()


def _window_id(value: str | None) -> str:
    if value is None:
        return 'eeg_' + uuid.uuid4().hex
    require(isinstance(value, str) and bool(value.strip()), 'window_id must be a nonempty string.')
    return value.strip()


def _ordered_array(values: Any, channel_names: Sequence[str] | None,
                   channel_axis: int, assume_e4_order: bool) -> tuple[np.ndarray, dict]:
    require(channel_axis in (0, 1) and type(channel_axis) is int, 'channel_axis must be 0 or 1.')
    if values is None:
        raise PayloadUnavailable('NO_EEG_SAMPLES')
    try:
        a = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise PayloadUnavailable('INVALID_EEG_ARRAY') from exc
    if a.size == 0:
        raise PayloadUnavailable('NO_EEG_SAMPLES')
    if a.ndim != 2 or a.shape[channel_axis] != CHANNEL_COUNT:
        raise PayloadUnavailable('E4_INPUT_GEOMETRY_MISMATCH')
    if a.dtype.kind not in 'iuf':
        raise PayloadUnavailable('NON_REAL_NUMERIC_EEG')
    if channel_axis == 1:
        a = a.T
    # No automatic resampling, padding, deletion, or imputation.
    if a.shape[1] != WINDOW_SAMPLES:
        raise PayloadUnavailable('INCOMPLETE_WINDOW' if a.shape[1] < WINDOW_SAMPLES else 'OVERSIZED_WINDOW')
    if channel_names is None:
        require(strict_bool(assume_e4_order, 'assume_e4_order'),
                'Supply the actual channel_names or explicitly declare assume_e4_order=True.')
        order = list(range(30)); declaration = 'CALLER_DECLARED_E4_ORDER'
    else:
        require(not isinstance(channel_names, (str, bytes)), 'channel_names must be a list, not one string.')
        names = [str(v).strip().upper() for v in channel_names]
        require(len(names) == 30 and len(set(names)) == 30 and set(names) == set(CHANNELS),
                'Channel names must contain each of the 30 E4 electrodes exactly once; no aliases/substitution.')
        order = [names.index(ch) for ch in CHANNELS]; declaration = 'EXPLICIT_CHANNEL_LABELS'
    a = np.array(a[order], dtype=np.float64, copy=True, order='C')
    if not np.isfinite(a).all():
        raise PayloadUnavailable('NONFINITE_EEG_SAMPLES')
    if np.all(np.ptp(a, axis=1) == 0):
        raise PayloadUnavailable('ALL_CHANNELS_FLAT')
    return a, {'channel_order': CHANNELS.copy(), 'channel_axis_received': channel_axis,
               'channel_order_declaration': declaration, 'channels_reordered': order != list(range(30)),
               'input_in_e4_order_indices': order, 'source_waveform_sha256': array_sha(a)}


def _freshness_reason(live: bool, newest: float | None, clock: Callable[[], float],
                      max_age: float) -> tuple[str | None, float | None]:
    if not live:
        return None, None
    if newest is None:
        return 'MISSING_SAMPLE_TIMESTAMP', None
    try:
        newest = finite(newest, 'newest_sample_monotonic')
        now = finite(clock(), 'local monotonic clock')
    except InputContractError:
        return 'INVALID_SAMPLE_TIMESTAMP', None
    age = now - newest
    if age < -1e-6:
        return 'FUTURE_OR_WRONG_CLOCK_TIMESTAMP', age
    age = max(age, 0.0)
    return ('STALE_EEG_INPUT' if age > max_age else None), age


def validate_sample_timestamps(values: Any, n: int, tolerance: float) -> np.ndarray:
    try:
        t = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PayloadUnavailable('INVALID_SAMPLE_TIMESTAMPS') from exc
    if t.shape != (n,) or not np.isfinite(t).all():
        raise PayloadUnavailable('INVALID_SAMPLE_TIMESTAMPS')
    if n > 1:
        delta = np.diff(t)
        if np.any(delta <= 0) or np.any(np.abs(delta - 1 / FS) > tolerance):
            raise PayloadUnavailable('NONCONTIGUOUS_SAMPLE_TIMESTAMPS')
    return t.copy()


class EEGQualityV1:
    """Initialize once, reuse for one EEG stream. Does not contain E4 inference.

    'raw' means the calibrated pre-E4 boundary, not arbitrary sensor ADC bytes.
    Hardware input needs an explicit equivalent montage/reference/filter policy.
    Default on_quality_error='raise' preserves EQ2's error contract. Choosing
    'unavailable' is a labelled protective fallback, not a learned rejection.
    """
    def __init__(self, calibration: str | Path | None = None, *,
                 assets_dir: str | Path | None = None, candidate: str = 'pyprep_physical',
                 eeg_unit: str | None = None, scale_to_volts: float | None = None,
                 unit_evidence: str = 'USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED',
                 stale_after_sec: float = 1.0, timestamp_tolerance_sec: float = .0005,
                 on_quality_error: str = 'raise', clock: Callable[[], float] = time.monotonic):
        self.calibration_path = resolve_calibration(calibration, assets_dir)
        self.calibration_sha256 = digest_file(self.calibration_path)
        self.calibrator = EEGQualityCalibrator(self.calibration_path, candidate=candidate)
        validate_extractor_contract(self.calibrator.artifact)
        require(on_quality_error in ('raise', 'unavailable'), 'on_quality_error must be raise or unavailable.')
        self.on_quality_error = on_quality_error
        self.scale = None
        self.eeg_unit = eeg_unit
        if eeg_unit is not None or scale_to_volts is not None:
            self.scale, self.eeg_unit = unit_scale(eeg_unit, scale_to_volts)
        require(isinstance(unit_evidence, str) and bool(unit_evidence.strip()), 'Record the unit evidence string.')
        self.unit_evidence = unit_evidence
        self.stale_after_sec = finite(stale_after_sec, 'stale_after_sec')
        self.timestamp_tolerance_sec = finite(timestamp_tolerance_sec, 'timestamp_tolerance_sec')
        require(self.stale_after_sec > 0, 'stale_after_sec must be positive.')
        require(0 < self.timestamp_tolerance_sec < 1/FS, 'Timestamp tolerance must be positive and < one sample.')
        self.clock = clock
        self._detector = None
        self._lock = threading.RLock()
        self._instance_id = uuid.uuid4().hex
        self._generation = 0
        self._previous_state: str | None = None
        self.identity = dict(module_version=VERSION, candidate=candidate,
            selection='explicit runtime configuration; original EQ2 artifact unmodified',
            calibration_path=str(self.calibration_path), calibration_sha256=self.calibration_sha256,
            source_artifact_state=self.calibrator.artifact['artifact_state'],
            feature_contract_id=self.calibrator.feature_contract_id,
            source_scripts=SOURCE_SCRIPT_SHA256, router_tau=TAU,
            calibration_unit_provenance=copy.deepcopy(self.calibrator.artifact['input_contract']['unit_contract']),
            eeg_emotion_model_included=False, q_is_actual_fusion_weight=False,
            runtime_policy=dict(stale_after_sec=self.stale_after_sec,
                timestamp_tolerance_sec=self.timestamp_tolerance_sec, on_quality_error=on_quality_error,
                quality_smoothing='none', trial_quality_aggregation='not_implemented',
                recovery='next complete fresh valid window', partial_channels='no additional hard cutoff',
                hardware_policy_validated=False, automatic_package_install=False))

    def check_assets(self, *, verify_detector_import: bool = True) -> dict[str, Any]:
        env = environment_check(self.calibrator.artifact, verify_import=verify_detector_import)
        return {'status': env['status'], 'identity': copy.deepcopy(self.identity), 'environment': env,
                'calibrator_loaded': True, 'raw_eeg_scored': False,
                'input_voltage_scale': self.scale, 'input_unit_evidence': self.unit_evidence}

    def load_detector(self) -> _EQ1PyPREPDetector:
        with self._lock:
            if self._detector is None:
                self._detector = _get_verified_detector(self.calibrator.artifact)
            return self._detector

    def _finish(self, report: dict, started: float) -> dict:
        report.update(module_version=VERSION, calibration_sha256=self.calibration_sha256,
                      feature_contract_id=self.calibrator.feature_contract_id,
                      candidate=self.calibrator.candidate, router_tau=TAU,
                      runtime_instance_id=self._instance_id, generation=self._generation,
                      previous_state=self._previous_state,
                      transition=(f'{self._previous_state or "START"}->{report["quality_state"]}'),
                      elapsed_seconds=time.perf_counter() - started,
                      q_is_actual_fusion_weight=False, eeg_weight=None)
        self._previous_state = report['quality_state']
        return json_safe(report)

    def assess_array(self, eeg: Any, *, sample_rate: float = 500,
                     channel_names: Sequence[str] | None = None, channel_axis: int = 0,
                     assume_e4_order: bool = False, window_id: str | None = None,
                     eeg_unit: str | None = None, scale_to_volts: float | None = None,
                     live: bool = False, newest_sample_monotonic: float | None = None,
                     sample_timestamps_monotonic: Any = None, contiguous: bool = True,
                     backend_error: bool = False) -> dict[str, Any]:
        """Score ONE 5s window; statuses are explicit, raw input is never mutated."""
        started = time.perf_counter()
        with self._lock:
            # Revoke an older live assessment even if this call raises later.
            self._generation += 1
            live = strict_bool(live, 'live')
            contiguous = strict_bool(contiguous, 'contiguous')
            backend_error = strict_bool(backend_error, 'backend_error')
            wid = _window_id(window_id)
            sr = finite(sample_rate, 'sample_rate')
            require(sr == FS, 'This calibration needs 500 Hz PRE-E4 input. Do not pass 200 Hz '
                    'standardized E4 tensors or silently resample a new sensor here.')
            report = dict(window_id=wid, window_seconds=WINDOW_SECONDS, sample_rate_hz=FS,
                status='OK', quality_state='UNAVAILABLE', reason=None, q_eeg=0.0,
                eeg_available=False, payload_available=False, quality_model_used=False,
                detector_status='SKIPPED_NO_VALID_PAYLOAD', pyprep_method_status=None,
                live=live, newest_sample_monotonic=newest_sample_monotonic,
                newest_sample_age_seconds=None, input_contiguous=contiguous,
                features=None, input_metadata=None, calibration_result=None, error=None,
                hardware_fault_confirmed=False)
            reason = 'CAPTURE_BACKEND_ERROR' if backend_error else ('NONCONTIGUOUS_EEG_WINDOW' if not contiguous else None)
            if reason:
                report['reason'] = reason
                return self._finish(report, started)
            try:
                if sample_timestamps_monotonic is not None:
                    timestamps = validate_sample_timestamps(sample_timestamps_monotonic, WINDOW_SAMPLES,
                                                            self.timestamp_tolerance_sec)
                    last = float(timestamps[-1])
                    if newest_sample_monotonic is not None and abs(finite(newest_sample_monotonic, 'newest sample') - last) > 1e-6:
                        raise PayloadUnavailable('NEWEST_TIMESTAMP_MISMATCH')
                    newest_sample_monotonic = last
                    report['newest_sample_monotonic'] = last
                reason, age = _freshness_reason(live, newest_sample_monotonic, self.clock, self.stale_after_sec)
                report['newest_sample_age_seconds'] = age
                if reason:
                    raise PayloadUnavailable(reason)
                ordered, meta = _ordered_array(eeg, channel_names, channel_axis, assume_e4_order)
            except PayloadUnavailable as exc:
                report['reason'] = str(exc)
                return self._finish(report, started)
            if eeg_unit is not None or scale_to_volts is not None:
                scale, unit = unit_scale(eeg_unit, scale_to_volts)
            else:
                require(self.scale is not None, 'Supply confirmed eeg_unit or scale_to_volts; no amplitude guessing.')
                scale, unit = self.scale, self.eeg_unit
            with np.errstate(over='ignore', invalid='ignore'):
                volts = ordered * scale
            require(np.isfinite(volts).all(), 'Conversion to volts overflowed; verify input units/ADC scaling.')
            meta.update(source_shape=[30,2500], volts_per_source_unit=scale, source_unit=unit,
                        unit_evidence=self.unit_evidence, waveform_volts_sha256=array_sha(volts),
                        waveform_stage=self.calibrator.artifact['input_contract']['waveform_stage'],
                        resampling_applied=False, reference_changed=False, normalization_applied=False)
            report.update(payload_available=True, input_metadata=meta)
            # Missing packages, incompatible versions and artifact errors are setup
            # errors, not runtime signal faults; NEVER hide them with fail-closed.
            detector = self.load_detector()
            try:
                with _DETECTOR_LOCK:
                    features = spectral_and_raw_features(volts, self.calibrator.artifact['input_contract']['line_hz'])
                    features.update(detector.score(volts, seed=0))
                require(array_sha(ordered) == meta['source_waveform_sha256'] and
                        array_sha(volts) == meta['waveform_volts_sha256'], 'Detector mutated signal copy.')
                features['detector_status'] = 'SCORED'
                require(features.get('pyprep_method_status') == 'FULL_SUBSET',
                        'INSUFFICIENT_USABLE_CHANNELS_FOR_RELATIVE_METHODS: no calibrated q is produced.')
                scored = self.calibrator.assess_features(features, payload_available=True,
                    detector_status='SCORED', pyprep_method_status=features['pyprep_method_status'],
                    feature_contract_id=self.calibrator.feature_contract_id)
            except Exception as exc:
                if self.on_quality_error == 'raise':
                    self._previous_state = 'ERROR'
                    raise QualityDetectionError(f'{type(exc).__name__}: {exc}') from exc
                report.update(status='ERROR', quality_state='ERROR', reason='QUALITY_DETECTOR_ERROR_GUARD',
                              detector_status='ERROR', error=f'{type(exc).__name__}: {exc}',
                              guard_policy='explicit on_quality_error=unavailable; NOT detected sensor failure')
                return self._finish(report, started)
            report.update(features=features, calibration_result=scored,
                          pyprep_method_status=features['pyprep_method_status'], detector_status='SCORED')
            reason, age = _freshness_reason(live, newest_sample_monotonic, self.clock, self.stale_after_sec)
            report['newest_sample_age_seconds'] = age
            if reason:
                report.update(reason=reason + '_AFTER_SCORING', detector_status='SCORED_BUT_STALE',
                              quality_model_used=True, computed_q_not_used=scored['q_eeg'])
                return self._finish(report, started)
            report.update(q_eeg=scored['q_eeg'], eeg_available=True, quality_model_used=True,
                          quality_state=scored['quality_state'], reason=scored['reason'],
                          outside_fit_range_features=scored.get('outside_fit_range_features', []))
            return self._finish(report, started)

    def make_fusion_input(self, assessment: Mapping[str, Any], eeg_probs: Any, *,
                          prediction_window_id: str | None = None,
                          prediction_window_seconds: float = 5.0, classifier_ok: bool = True,
                          prediction_source_sha256: str | None = None) -> dict[str, Any]:
        """q and probs must refer to the same window. Recheck live freshness."""
        require(isinstance(assessment, Mapping), 'Expected an EEGQualityV1 assessment.')
        require(assessment.get('runtime_instance_id') == self._instance_id and
                assessment.get('calibration_sha256') == self.calibration_sha256 and
                assessment.get('feature_contract_id') == self.calibrator.feature_contract_id,
                'Assessment belongs to a different runtime/calibrator.')
        classifier_ok = strict_bool(classifier_ok, 'classifier_ok')
        if not strict_bool(assessment.get('eeg_available'), 'eeg_available') or not classifier_ok:
            return canonical_missing_eeg()
        if assessment['live']:
            if assessment.get('generation') != self._generation:
                return canonical_missing_eeg()
            reason, _ = _freshness_reason(True, assessment.get('newest_sample_monotonic'),
                                         self.clock, self.stale_after_sec)
            if reason:
                return canonical_missing_eeg()
        require(prediction_window_id == assessment['window_id'], 'E4 probability window_id differs from quality window.')
        require(finite(prediction_window_seconds, 'prediction_window_seconds') == WINDOW_SECONDS,
                'A 20s trial prediction cannot be paired with this 5s quality implicitly.')
        if prediction_source_sha256 is not None:
            require(prediction_source_sha256 == assessment['input_metadata']['source_waveform_sha256'],
                    'E4 source-waveform fingerprint differs from quality input.')
        # Recompute from its persisted features, rather than trusting a mutable q field.
        return self.calibrator.make_fusion_input(eeg_probs, features=assessment['features'],
                   payload_available=True, classifier_ok=True, detector_status=assessment['detector_status'],
                   pyprep_method_status=assessment['pyprep_method_status'],
                   feature_contract_id=self.calibrator.feature_contract_id)

    def assess_file(self, path: str | Path, *, window_id: str | None = None,
                    start_sample: int | None = None, mat_variable: str = 'seg',
                    trial_index: int | None = None, window_index: int | None = None,
                    sample_rate: float = 500, channel_names: Sequence[str] | None = None,
                    assume_e4_order: bool = False, channel_axis: int = 0, **kwargs: Any) -> dict:
        x, info = load_eeg_file(path, start_sample=start_sample, mat_variable=mat_variable,
                               trial_index=trial_index, window_index=window_index, channel_axis=channel_axis)
        if info.get('sample_rate') is not None:
            require(float(info['sample_rate']) == float(sample_rate), 'File samplerate conflicts with supplied sample_rate.')
        if info.get('channel_names') is not None:
            if channel_names is not None:
                require(list(channel_names) == info['channel_names'], 'File/supplied channel labels disagree.')
            channel_names = info['channel_names']
        r = self.assess_array(x, sample_rate=sample_rate, channel_names=channel_names,
                             assume_e4_order=assume_e4_order, window_id=window_id, channel_axis=0, **kwargs)
        r['file_source'] = info
        return r

    def process_array(self, eeg: Any, *, context: Mapping[str, Any],
                      bridge: 'AF4CEEGBridge | None' = None, **kwargs: Any) -> dict:
        require(isinstance(context, Mapping), 'A same-window fusion context is required.')
        require('window_id' not in kwargs or kwargs['window_id'] == context.get('window_id'), 'Conflicting window IDs.')
        kwargs['window_id'] = context.get('window_id')
        assessment = self.assess_array(eeg, **kwargs)
        return self._complete_process(assessment, context, bridge)

    def _complete_process(self, assessment: dict, context: Mapping[str, Any], bridge: Any) -> dict:
        sample, call = build_fusion_input(self, assessment, context)
        result = {'quality': assessment, 'fusion_eeg_input': {k: call[k] for k in ('eeg_probs','q_eeg','eeg_available')},
                  'fusion_input': sample, 'fusion': None, 'eeg_weight': None}
        if bridge is not None:
            # Recheck again inside bridge immediately before the actual forward.
            fused = bridge.predict(self, assessment, context)
            result.update(fusion=fused, eeg_weight=fused['eeg_weight'], fusion_input=fused['fusion_input'])
            result['fusion_eeg_input'] = dict(eeg_probs=fused['fusion_input']['eeg_probs'],
                                            q_eeg=fused['q_eeg'], eeg_available=fused['eeg_available'])
        return result

    def process_file(self, path: str | Path, *, context: Mapping[str, Any],
                     bridge: 'AF4CEEGBridge | None' = None, **kwargs: Any) -> dict:
        require(isinstance(context, Mapping), 'Same-window context required.')
        require('window_id' not in kwargs or kwargs['window_id'] == context.get('window_id'), 'Conflicting window IDs.')
        kwargs['window_id'] = context.get('window_id')
        assessment = self.assess_file(path, **kwargs)
        return self._complete_process(assessment, context, bridge)

    def preflight(self) -> dict[str, Any]:
        """Real detector + exported map on synthetic EEG; no accuracy claim."""
        checks = {}
        x = synthetic_eeg_volts()
        initial = x.copy()
        r = self.assess_array(x, eeg_unit='V', channel_names=CHANNELS, window_id='synthetic_preflight')
        require(r['status'] == 'OK' and r['eeg_available'], 'Synthetic scoring failed; see assessment/error.')
        p = [.1, .2, .3, .25, .15]
        a = self.make_fusion_input(r, p, prediction_window_id=r['window_id'])
        checks['synthetic_real_detector_scored'] = r['detector_status'] == 'SCORED'
        checks['source_unchanged'] = np.array_equal(x, initial)
        checks['finite_calibrated_q'] = self.calibrator.clip[0] <= a['q_eeg'] <= self.calibrator.clip[1]
        for name, value, extra in [
            ('none', None, {}), ('all_zero', np.zeros_like(x), {}),
            ('constant', np.ones_like(x), {}), ('capture_error', x, {'backend_error': True}),
            ('gap', x, {'contiguous': False}), ('stale', x, {'live': True, 'newest_sample_monotonic': self.clock()-2*self.stale_after_sec})]:
            m = self.assess_array(value, eeg_unit='V', channel_names=CHANNELS, window_id=name, **extra)
            u = self.make_fusion_input(m, [float('nan')]*5)
            checks[name + '_canonical_missing'] = u == canonical_missing_eeg()
        restored = self.assess_array(x, eeg_unit='V', channel_names=CHANNELS, window_id='restored')
        checks['identical_input_recovery'] = restored['eeg_available'] and abs(restored['q_eeg']-r['q_eeg']) < 1e-12
        require(all(checks.values()), 'Preflight failed: '+canonical_json(checks))
        return {'status': 'PASS', 'checks': checks, 'n_checks': len(checks), 'reference_assessment': r,
                'real_pyprep_inference': True, 'real_EAV_waveform_used': False,
                'synthetic_waveform_only': True, 'real_hardware_tested': False,
                'e4_inference_performed': False, 'fusion_inference_performed': False}
# =============================================================================
# File adapters and a hardware-agnostic online buffer
# =============================================================================

def load_eeg_file(path: str | Path, *, start_sample: int | None = None,
                  mat_variable: str = 'seg', trial_index: int | None = None,
                  window_index: int | None = None, channel_axis: int = 0) -> tuple[np.ndarray, dict]:
    """Read only a requested file/window. No manifest discovery or batch evaluation.

    .npy: numeric [30,T] (or explicit channel_axis=1).
    .npz: key 'eeg', optional scalar 'sample_rate' and Unicode 'channel_names'.
    .mat: explicit variable; 3-D EAV needs BOTH trial_index and window_index.
    Long 2-D input requires explicit start_sample; short input is never padded.
    """
    p = Path(path).expanduser().resolve()
    require(p.is_file(), f'EEG input file not found: {p}')
    info: dict[str, Any] = {'path': str(p)}
    require(channel_axis in (0,1) and type(channel_axis) is int, 'channel_axis must be 0/1.')
    suffix = p.suffix.lower()
    if suffix == '.npy':
        a = np.load(p, allow_pickle=False, mmap_mode='r')
    elif suffix == '.npz':
        with np.load(p, allow_pickle=False) as data:
            require('eeg' in data, 'NPZ requires the numeric key "eeg".')
            a = data['eeg'].copy()
            if 'sample_rate' in data:
                sr = np.asarray(data['sample_rate'])
                require(sr.size == 1, 'NPZ sample_rate must be scalar.')
                info['sample_rate'] = finite(sr.item(), 'NPZ sample_rate')
            if 'channel_names' in data:
                names = np.asarray(data['channel_names'])
                require(names.ndim == 1 and names.dtype.kind in 'US', 'NPZ channel_names must be Unicode/ASCII array, not pickled objects.')
                info['channel_names'] = names.astype(str).tolist()
    elif suffix == '.mat':
        from scipy.io import loadmat
        try:
            obj = loadmat(str(p), variable_names=[mat_variable])
        except NotImplementedError as exc:
            raise InputContractError('MAT v7.3/HDF5 is not supported by this reader; export an explicit numeric window.') from exc
        actual = mat_variable
        if actual not in obj and actual in ('seg', 'seg1'):
            actual = 'seg1' if actual == 'seg' else 'seg'
            obj = loadmat(str(p), variable_names=[actual])
        require(actual in obj, f'MAT variable {mat_variable!r} (or seg/seg1 fallback) missing.')
        info['mat_variable_actual'] = actual
        a = np.asarray(obj[actual])
    else:
        raise InputContractError('Supported EEG file types are .npy, .npz and classic .mat.')
    if a.ndim == 3:
        require(suffix == '.mat' and set(a.shape) == {10000,30,200}, 'Only explicit original EAV 3-D MAT geometry is supported.')
        require(trial_index is not None and window_index is not None and start_sample is None,
                'EAV 3-D input requires zero-based --trial-index and --window-index, not --start-sample.')
        trial = exact_int(trial_index, 'trial_index'); win = exact_int(window_index, 'window_index')
        require(0 <= trial < 200 and 0 <= win < 4, 'EAV trial/window indices out of range.')
        arr = a.transpose(a.shape.index(10000), a.shape.index(30), a.shape.index(200))
        x = np.array(arr[win*2500:(win+1)*2500,:,trial].T, copy=True)
        info.update(trial_index_0based=trial, window_index_0based=win,
                    start_sample=win*2500, end_sample_exclusive=(win+1)*2500,
                    channel_names=CHANNELS.copy(), sample_rate=500,
                    montage_source='caller-selected original EAV MAT/E4 protocol; not inferred from values')
    else:
        require(trial_index is None and window_index is None, 'trial/window selection applies only to EAV 3-D MAT.')
        require(a.ndim == 2 and a.shape[channel_axis] == 30, 'EEG file must have exactly 30 channels on the declared axis.')
        if channel_axis == 1:
            a = a.T
        if start_sample is None:
            require(a.shape[1] == 2500, 'File is not a 5s window; use --start-sample for an explicit complete segment.')
            start = 0
        else:
            start = exact_int(start_sample, 'start_sample')
            require(start >= 0 and start+2500 <= a.shape[1], 'Requested 5s segment is outside file.')
        x = np.array(a[:,start:start+2500], copy=True)
        info.update(start_sample=start, end_sample_exclusive=start+2500)
    require(x.dtype.kind in 'iuf', 'EEG must be real numeric; object arrays are not allowed.')
    info['selected_source_array_sha256'] = array_sha(x)
    return x, info


def load_eav_val_window(manifest: str | Path, window_key: str) -> tuple[np.ndarray, dict]:
    """Convenience reader for ONE named frozen VAL window, never Train/Test.

    No discovery and no relabelling. Legacy task-less manifests are recorded as
    caller-declared original Speaking sources (the same explicit pin as EQ1).
    """
    p = Path(manifest).expanduser().resolve()
    require(p.name.lower() == 'val_window_manifest.csv', 'Use the explicit Stage0C val_window_manifest.csv.')
    require(p.is_file(), f'VAL manifest missing: {p}')
    with p.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    require(rows and all(str(r.get('split','')).strip().lower() == 'val' for r in rows), 'Non-VAL manifest refused.')
    val_subjects = {'subject08','subject09','subject10','subject13','subject14','subject33'}
    require(all(str(r.get('subject','')).strip() in val_subjects for r in rows), 'Unexpected frozen VAL subject.')
    for r in rows:
        if 'task_condition' in r:
            require(str(r['task_condition']).strip().lower() == 'speaking', 'Only original Speaking VAL is supported here.')
    sp = p.parent / 'stage0c_summary.json'
    if sp.is_file():
        s = read_json(sp)
        require(str(s.get('status','')).strip().upper() == 'PASS', 'Stage0C must be PASS.')
        if s.get('task_condition'):
            require(str(s['task_condition']).strip().lower() == 'speaking', 'Listening summary refused.')
    chosen = [r for r in rows if r.get('window_key') == window_key]
    require(len(chosen) == 1, f'Expected one VAL row for {window_key!r}, found {len(chosen)}.')
    r = chosen[0]
    require(exact_int(r['eeg_fs_hz'], 'eeg_fs_hz') == FS, 'VAL source rate mismatch.')
    w = exact_int(r['window_idx_0based'], 'window index')
    require(exact_int(r['eeg_start_sample'], 'start') == w*2500 and
            exact_int(r['eeg_end_sample_exclusive'], 'end') == (w+1)*2500, 'VAL window timing mismatch.')
    ep = Path(r['eeg_path']).expanduser()
    if not ep.is_absolute():
        root = p.parents[3]  # project/EAV_dataset/manifests/stage0c_*/val...
        paths = list(dict.fromkeys([(root/ep).resolve(), (p.parent/ep).resolve()]))
        found = [q for q in paths if q.is_file()]
        require(len(found) == 1, 'Relative MAT path ambiguous/missing. Use explicit --eeg MAT and indices.')
        ep = found[0]
    require(not any(v == 'test' or v.startswith('test_') for v in re.split(r'[\\/]', str(ep).lower())), 'TEST-named EEG path refused.')
    subject_ids = {f'subject{int(v):02d}' for v in re.findall(r'subject0*(\d+)(?!\d)',str(ep),re.I)}
    require(not subject_ids or subject_ids == {r['subject']}, 'Selected MAT subject conflicts with VAL row.')
    x, info = load_eeg_file(ep, mat_variable=r['eeg_variable'],
                           trial_index=exact_int(r['eeg_trial_idx_0based'],'trial'), window_index=w)
    info.update(val_manifest=str(p), window_key=window_key, selected_subject=r['subject'],
                task_evidence='val_row' if 'task_condition' in r else 'caller_pinned_legacy_speaking',
                test_manifest_read=False)
    return x, info


class EEGWindowBuffer:
    """Device-independent 500Hz buffer; acquisition SDK remains external.

    push() accepts numeric [30,N] and actual sample timestamps converted to the
    local monotonic clock. It does not run PyPREP. A gap/error invalidates buffered
    data, then at least one full fresh 5s span is required again. poll assess on a
    worker/main thread, not inside a hardware callback. One buffer per stream.
    """
    def __init__(self, *, channel_names: Sequence[str], stale_after_sec: float = 1.0,
                 timestamp_tolerance_sec: float = .0005, stream_id: str = 'eeg',
                 clock: Callable[[],float] = time.monotonic):
        names = [str(c).strip().upper() for c in channel_names]
        require(len(names)==30 and len(set(names))==30 and set(names)==set(CHANNELS), 'Buffer channel labels must match all E4 electrodes.')
        self.channel_names = names
        self.stale_after_sec = finite(stale_after_sec,'stale_after_sec')
        self.tolerance = finite(timestamp_tolerance_sec,'timestamp_tolerance_sec')
        require(self.stale_after_sec>0 and 0<self.tolerance<1/FS,'Invalid timing policy.')
        self.stream_id = _window_id(stream_id); self.clock = clock
        self._lock = threading.RLock(); self.epoch=0; self.sequence=0
        self.data=np.empty((30,0),float); self.times=np.empty(0,float)
        self.last_reason='WARMING_UP'

    def invalidate(self, reason: str = 'CAPTURE_BACKEND_ERROR') -> None:
        with self._lock:
            self.data=np.empty((30,0),float); self.times=np.empty(0,float)
            self.epoch += 1; self.last_reason=str(reason)

    def push(self, samples: Any, sample_timestamps_monotonic: Any, *, backend_error: bool = False) -> dict:
        with self._lock:
            if strict_bool(backend_error, 'backend_error'):
                self.invalidate('CAPTURE_BACKEND_ERROR')
                return self.status()
            try:
                a=np.asarray(samples)
                if a.ndim!=2 or a.shape[0]!=30 or a.shape[1]<1 or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
                    raise PayloadUnavailable('INVALID_STREAM_CHUNK')
                t=validate_sample_timestamps(sample_timestamps_monotonic,a.shape[1],self.tolerance)
                reason,_=_freshness_reason(True,float(t[-1]),self.clock,self.stale_after_sec)
                if reason:
                    raise PayloadUnavailable(reason)
            except (PayloadUnavailable,ValueError,TypeError) as exc:
                self.invalidate(str(exc)); return self.status()
            if len(self.times) and abs(float(t[0]-self.times[-1])-1/FS)>self.tolerance:
                self.invalidate('STREAM_GAP_OR_OVERLAP')
            self.data=np.concatenate([self.data,np.asarray(a,dtype=float)],axis=1)[:,-2500:].copy()
            self.times=np.concatenate([self.times,t])[-2500:].copy()
            self.sequence+=1
            self.last_reason='READY' if len(self.times)==2500 else 'RECOVERING_FULL_WINDOW'
            return self.status()

    def status(self) -> dict:
        return {'buffered_samples':int(len(self.times)), 'required_samples':2500,
                'ready':len(self.times)==2500,'reason':self.last_reason,'stream_epoch':self.epoch}

    def snapshot(self) -> dict:
        with self._lock:
            if len(self.times):
                reason,_=_freshness_reason(True,float(self.times[-1]),self.clock,self.stale_after_sec)
                if reason:
                    self.invalidate(reason)
            ready=len(self.times)==2500
            return {'samples':self.data.copy() if ready else None,
                    'sample_timestamps_monotonic':self.times.copy() if ready else None,
                    'newest_sample_monotonic':float(self.times[-1]) if ready else None,
                    'channel_names':self.channel_names.copy(), 'sample_rate':500,
                    'window_id':f'{self.stream_id}:{self.epoch}:{self.sequence}',
                    'live':True, 'contiguous':ready, 'buffer_status':self.status()}

    def assess(self, runtime: EEGQualityV1) -> dict:
        packet=self.snapshot()
        s=packet.pop('samples'); state=packet.pop('buffer_status')
        result=runtime.assess_array(s,**packet)
        result['buffer_status']=state
        return result


# =============================================================================
# Safe bridge to the supplied frozen F4 / AF4-B / AF4-C implementation
# =============================================================================

def build_fusion_input(runtime: EEGQualityV1, assessment: Mapping[str,Any],
                       context: Mapping[str,Any]) -> tuple[dict,dict]:
    require(isinstance(context,Mapping), 'Same-window fusion context must be a mapping.')
    require(context.get('window_id') == assessment['window_id'],'Fusion/EEG quality window_id mismatch.')
    require(context.get('window_seconds') == 5.0, 'Fusion context must explicitly describe this 5-second window.')
    require(context.get('class_order') == EMOTIONS, 'Explicit frozen class_order required.')
    av=context.get('available'); quality=context.get('quality')
    require(isinstance(av,Mapping) and set(MODALITIES)<=set(av), 'All three availability flags must be explicit.')
    require(isinstance(quality,Mapping) and {'audio','video'}<=set(quality), 'Audio/Video quality must be explicit, never default healthy.')
    e=runtime.make_fusion_input(assessment,context.get('eeg_probs'),
        prediction_window_id=context['window_id'],prediction_window_seconds=context['window_seconds'],
        classifier_ok=strict_bool(av['eeg'],'available.eeg'),
        prediction_source_sha256=context.get('eeg_source_waveform_sha256'))
    ps={'eeg':e['eeg_probs']}; qs={'eeg':e['q_eeg']}; masks={'eeg':e['eeg_available']}
    for m in ('audio','video'):
        masks[m]=strict_bool(av[m],f'available.{m}')
        if not masks[m]:
            ps[m]=[.2]*5; qs[m]=0.0
        else:
            ps[m]=normalize_probs(context.get(m+'_probs'),m+'_probs')
            v=finite(quality[m],m+' quality')
            require(0<=v<=1,m+' quality must be [0,1].')
            qs[m]=v
    sample={m+'_probs':ps[m] for m in MODALITIES}
    sample.update(quality=qs,available={m:int(masks[m]) for m in MODALITIES})
    call={m+'_probs':ps[m] for m in MODALITIES}
    call.update({f'q_{m}':qs[m] for m in MODALITIES})
    call.update({f'{m}_available':masks[m] for m in MODALITIES})
    return sample,call


def import_fusion_script(path: str | Path) -> Any:
    """Import only a caller-specified trusted local source; never invoke main()."""
    p=Path(path).expanduser().resolve()
    require(p.is_file(),f'Fusion script not found: {p}')
    identity=hashlib.sha256((str(p)+digest_file(p)).encode()).hexdigest()[:20]
    name='_eeg_quality_frozen_af4c_'+identity
    if name in sys.modules:
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,str(p))
    require(spec is not None and spec.loader is not None,'Cannot load fusion module.')
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        require(callable(getattr(module,'FinalAF4CSystem',None)) and
                float(getattr(module,'ROUTER_TAU',float('nan')))==TAU and
                list(getattr(module,'EMOTIONS',[]))==EMOTIONS and
                list(getattr(module,'MODALITIES',[]))==MODALITIES, 'Frozen fusion interface/threshold/order mismatch.')
    except BaseException:
        sys.modules.pop(name,None); raise
    return module


class AF4CEEGBridge:
    """Actual frozen network inference; no invented total EEG contribution."""
    def __init__(self, system: Any, identity: Mapping[str,Any] | None = None):
        require(callable(getattr(system,'predict_batch',None)), 'Expected loaded FinalAF4CSystem.predict_batch.')
        require(float(getattr(system,'tau',float('nan')))==TAU,'Frozen fusion tau must remain .80.')
        self.system=system; self.identity=dict(identity or {'source':'caller-provided loaded system'})
        self._lock=threading.RLock()

    @classmethod
    def from_paths(cls, fusion_script: str | Path, f4_checkpoint: str | Path,
                   af4b_checkpoint: str | Path, device: str = 'cuda') -> 'AF4CEEGBridge':
        f4=Path(f4_checkpoint).expanduser().resolve(); b=Path(af4b_checkpoint).expanduser().resolve()
        require(f4.is_file() and b.is_file(), 'Both actual trained frozen checkpoints are required.')
        require(device in ('cpu','cuda'), 'Fusion device must be cpu/cuda.')
        module=import_fusion_script(fusion_script)
        system=module.FinalAF4CSystem(f4_checkpoint=f4,af4b_checkpoint=b,
                                     device=module.choose_device(device),batch_size=1,tau=TAU)
        return cls(system,{'fusion_script':str(Path(fusion_script).resolve()),
            'fusion_script_sha256':digest_file(Path(fusion_script)), 'f4_checkpoint':str(f4),
            'f4_sha256':digest_file(f4),'af4b_checkpoint':str(b),'af4b_sha256':digest_file(b),
            'benchmark_data_read':False,'mode':'predict_only'})

    def predict(self, runtime: EEGQualityV1, assessment: Mapping[str,Any], context: Mapping[str,Any]) -> dict:
        with self._lock:
            sample,_=build_fusion_input(runtime,assessment,context)
            return self.predict_sample(sample,window_id=assessment['window_id'])

    def predict_sample(self, sample: Mapping[str,Any], *, window_id: str) -> dict:
        # All unavailable modalities are canonicalized BEFORE x15 goes to F4.
        require(isinstance(sample,Mapping) and isinstance(sample.get('available'),Mapping) and
                isinstance(sample.get('quality'),Mapping),'Explicit probabilities, quality and masks required.')
        ps=[]; qs=[]; masks=[]
        for m in MODALITIES:
            available=strict_bool(sample['available'].get(m),m+' availability')
            if available:
                p=normalize_probs(sample.get(m+'_probs'),m+'_probs')
                q=finite(sample['quality'].get(m),m+' quality')
                require(0<=q<=1,'Quality must be [0,1].')
            else:
                p=[.2]*5; q=0.0
            ps.extend(p); qs.append(q); masks.append(float(available))
        x=np.asarray([ps],dtype=np.float32); q=np.asarray([qs],dtype=np.float32)
        mask=np.asarray([masks],dtype=np.float32)
        with self._lock:
            r=self.system.predict_batch(x,q,mask)
        route=str(np.asarray(r['route']).reshape(-1)[0]); state=str(np.asarray(r['system_state']).reshape(-1)[0])
        no_decision=bool(np.asarray(r['router_output']['no_decision']).reshape(-1)[0]>.5)
        expected_missing=not any(masks)
        expected_route='AF4-B' if not all(masks) or bool(np.min(q)<np.float32(TAU)) else 'F4'
        require(route==expected_route and no_decision==expected_missing,'Frozen routing result violates .80/mask contract.')
        final_array=np.asarray(r['router_output']['final_probs'],dtype=float)
        require(final_array.shape==(1,5),'Unexpected final probability shape.')
        final=normalize_probs(final_array[0],'final probabilities')
        require(not no_decision or np.allclose(final,[.2]*5,atol=1e-7),'NO_DECISION must return uniform probabilities.')
        alpha=np.asarray(r['af4b_output']['alpha'],dtype=float)[0]
        eff=np.asarray(r['af4b_output']['effective_weights'],dtype=float)[0]
        gamma=float(np.asarray(r['af4b_output']['gamma']).reshape(-1)[0])
        require(alpha.shape==(3,) and eff.shape==(3,5) and np.isfinite(alpha).all()
                and np.isfinite(eff).all() and math.isfinite(gamma),'Invalid adaptive branch output.')
        require(np.all(alpha>=0) and np.all(eff>=0) and 0<=gamma<=1,'Invalid adaptive weight domain.')
        if not no_decision:
            for j, available in enumerate(masks):
                if not available:
                    require(abs(float(alpha[j]))<=1e-7 and np.max(np.abs(eff[j]))<=1e-7,
                            f'Unavailable {MODALITIES[j]} still has adaptive weight.')
        active=route=='AF4-B' and not no_decision
        kind=('NO_DECISION_NO_ACTIVE_EVIDENCE' if no_decision else
              'F4_HAS_NO_SINGLE_MODALITY_MIXING_WEIGHT' if route=='F4' else 'AF4B_ADAPTIVE_BRANCH_ALPHA')
        return {'window_id':window_id,'window_seconds':5.,'route':route,'system_state':state,
                'no_decision':no_decision,'router_tau':TAU,
                'final':{'emotion':'NO_DECISION' if no_decision else EMOTIONS[int(np.argmax(final))],
                         'probabilities':final,'confidence':None if no_decision else max(final)},
                'eeg_available':bool(masks[0]),'q_eeg':qs[0],
                'eeg_weight':float(alpha[0]) if active else None,'eeg_weight_kind':kind,
                'eeg_weight_is_total_attribution':False,'adaptive_branch_active':active,
                'af4b_alpha_diagnostic':dict(zip(MODALITIES,alpha.tolist())),
                'af4b_eeg_effective_weights_by_class':dict(zip(EMOTIONS,eff[0].tolist())) if active else None,
                'af4b_gamma_diagnostic':gamma,'fusion_identity':self.identity,
                'fusion_input':{**{m+'_probs':ps[5*i:5*i+5] for i,m in enumerate(MODALITIES)},
                                'quality':dict(zip(MODALITIES,qs)),
                                'available':dict(zip(MODALITIES,[int(v) for v in masks]))}}

    def contract_preflight(self) -> dict:
        """Run loaded checkpoints on synthetic probabilities, not on EAV Test."""
        p=[.05,.15,.2,.5,.1]
        def sample(e: bool, ep: Any, qe: float, all_missing: bool=False) -> dict:
            return dict(eeg_probs=ep,audio_probs=p,video_probs=p,
                        quality={'eeg':qe,'audio':1.,'video':1.},
                        available={'eeg':e,'audio':not all_missing,'video':not all_missing})
        h=self.predict_sample(sample(True,p,1.),window_id='fusion_preflight')
        d=self.predict_sample(sample(True,p,.4),window_id='fusion_preflight')
        m1=self.predict_sample(sample(False,[1,0,0,0,0],0.),window_id='fusion_preflight')
        m2=self.predict_sample(sample(False,[0,0,0,0,1],1.),window_id='fusion_preflight')
        lost=self.predict_sample(sample(False,[float('nan')]*5,0.,True),window_id='fusion_preflight')
        back=self.predict_sample(sample(True,p,1.),window_id='fusion_preflight')
        checks={'healthy_F4':h['route']=='F4','degraded_AF4B':d['route']=='AF4-B',
                'missing_eeg_alpha_zero':m1['eeg_weight']==0.,
                'missing_eeg_class_weights_zero':all(v==0 for v in m1['af4b_eeg_effective_weights_by_class'].values()),
                'stale_eeg_contents_invariant':np.allclose(m1['final']['probabilities'],m2['final']['probabilities'],atol=1e-7,rtol=0),
                'all_missing_NO_DECISION':lost['no_decision'] and lost['eeg_weight'] is None,
                'F4_no_invented_weight':h['eeg_weight'] is None,
                'same_payload_recovers':np.allclose(h['final']['probabilities'],back['final']['probabilities'],atol=1e-7,rtol=0)}
        require(all(checks.values()),'Fusion contract checks failed: '+canonical_json(checks))
        return {'status':'PASS','checks':checks,'n_checks':len(checks),'actual_loaded_fusion_forward':True,
                'synthetic_probabilities_only':True,'eeg_emotion_inference':False,'hardware_tested':False,
                'fusion_identity':self.identity}


# Alias for callers that used the shorter bridge name in the Audio module.
AF4CBridge = AF4CEEGBridge
# =============================================================================
# CLI and local checks
# =============================================================================

def synthetic_eeg_volts() -> np.ndarray:
    """Synthetic physiology-like signal for interface checks, NOT clean EEG truth."""
    rng=np.random.default_rng(20260919); t=np.arange(2500)/500
    shared=np.sin(2*np.pi*10*t)+.25*np.sin(2*np.pi*5*t)+.1*np.sin(2*np.pi*19*t)
    return (np.linspace(12,30,30)[:,None]*shared[None,:]+rng.normal(0,2,(30,2500)))*1e-6


def run_self_test() -> dict:
    checks={}
    def rejected(fn: Callable) -> bool:
        try:
            fn()
        except (InputContractError,PayloadUnavailable):
            return True
        return False
    x=synthetic_eeg_volts(); original=x.copy()
    a,info=_ordered_array(x,CHANNELS,0,False)
    checks['source_not_mutated']=np.array_equal(x,original) and np.array_equal(a,x) and not np.shares_memory(a,x)
    order=np.arange(29,-1,-1)
    reordered,_=_ordered_array(x[order],[CHANNELS[i] for i in order],0,False)
    checks['named_channel_reordering']=np.array_equal(reordered,x)
    transposed,_=_ordered_array(x.T,CHANNELS,1,False)
    checks['explicit_channel_axis']=np.array_equal(transposed,x)
    checks['missing_rejected']=rejected(lambda:_ordered_array(None,CHANNELS,0,False))
    checks['short_not_padded']=rejected(lambda:_ordered_array(x[:,:100],CHANNELS,0,False))
    checks['all_flat_rejected']=rejected(lambda:_ordered_array(np.zeros_like(x),CHANNELS,0,False))
    bad=x.copy(); bad[0,0]=np.nan
    checks['nan_rejected']=rejected(lambda:_ordered_array(bad,CHANNELS,0,False))
    checks['unknown_channel_order_not_guessed']=rejected(lambda:_ordered_array(x,None,0,False))
    checks['unit_not_guessed']=rejected(lambda:unit_scale(None,None))
    checks['unit_conversion']=unit_scale('uV',None)[0]==1e-6 and unit_scale('mV',None)[0]==1e-3
    checks['missing_erases_any_probabilities']=canonical_missing_eeg([1,0,0,0,0])==canonical_missing_eeg([float('nan')]*5)
    checks['bad_probs_rejected']=rejected(lambda:normalize_probs([1,2,3,4,5]))
    checks['stale_identified']=_freshness_reason(True,7.,lambda:10.,1.)[0]=='STALE_EEG_INPUT'
    checks['timestamp_required']=_freshness_reason(True,None,lambda:10.,1.)[0]=='MISSING_SAMPLE_TIMESTAMP'
    timestamps=10.+np.arange(2500)/500
    checks['contiguous_timestamps']=np.array_equal(validate_sample_timestamps(timestamps,2500,.0005),timestamps)
    broken=timestamps.copy(); broken[1000:]+=.01
    checks['gap_identified']=rejected(lambda:validate_sample_timestamps(broken,2500,.0005))
    clock=[15.]
    buf=EEGWindowBuffer(channel_names=CHANNELS,clock=lambda:clock[0])
    buf.push(x,timestamps)
    checks['complete_buffer_ready']=buf.snapshot()['samples'].shape==(30,2500)
    buf.invalidate('known_error')
    checks['buffer_error_flushes']=buf.snapshot()['samples'] is None
    buf.push(x,timestamps)
    checks['full_window_recovers']=buf.snapshot()['samples'] is not None
    clock[0]=20.
    checks['stale_buffer_flushes']=buf.snapshot()['samples'] is None
    spec={'kind':'numeric_additive_trees_identity','features':list(FEATURES),
          'leaf_values_include_learning_rate':True,'missing_feature_policy':'raise','initial_prediction':.5,
          'trees':[[{'leaf':False,'feature':0,'threshold':.3,'left':1,'right':2},
                    {'leaf':True,'value':.3},{'leaf':True,'value':-.1}]]}
    validate_trees(spec)
    xx=np.zeros((2,8)); xx[1,0]=.5
    checks['portable_numeric_tree']=np.allclose(tree_predict(spec,xx),[.8,.4])
    require(all(checks.values()),'Self-test failed: '+canonical_json(checks))
    return {'status':'PASS','n_checks':len(checks),'checks':checks,
            'real_pyprep_used':False,'EAV_used':False,'calibration_asset_used':False,'fusion_used':False}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='EEG Quality V1: calibrated source EEG -> safe frozen AF4-C inputs.')
    p.add_argument('--version',action='version',version=VERSION)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--check-assets',action='store_true')
    mode.add_argument('--preflight',action='store_true',help='Real PyPREP + real JSON on a synthetic waveform.')
    mode.add_argument('--fusion-preflight',action='store_true',help='Actual loaded fusion checkpoints on synthetic probabilities.')
    mode.add_argument('--eeg',type=str,help='One explicit NPY / NPZ / classic MAT input.')
    mode.add_argument('--val-manifest',type=str,help='One explicit original Stage0C Speaking VAL manifest.')
    mode.add_argument('--no-eeg',action='store_true',help='Explicit no-payload request; no quality features fabricated.')
    p.add_argument('--calibration',type=str)
    p.add_argument('--assets-dir',type=str)
    p.add_argument('--candidate',choices=CANDIDATES,default='pyprep_physical')
    unit=p.add_mutually_exclusive_group()
    unit.add_argument('--eeg-unit',choices=['V','mV','uV'])
    unit.add_argument('--scale-to-volts',type=float)
    p.add_argument('--unit-evidence',default='USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED')
    p.add_argument('--sample-rate',type=float,default=500)
    p.add_argument('--channel-axis',type=int,choices=[0,1],default=0)
    p.add_argument('--channel-names',help='Comma-separated 30 actual electrode labels.')
    p.add_argument('--assume-e4-channel-order',action='store_true')
    p.add_argument('--start-sample',type=int)
    p.add_argument('--mat-variable',default='seg')
    p.add_argument('--trial-index',type=int,help='Explicit EAV MAT trial, zero-based.')
    p.add_argument('--window-index',type=int,help='Explicit EAV MAT window 0..3.')
    p.add_argument('--window-key',help='One unique key in --val-manifest.')
    p.add_argument('--window-id',help='Same ID as the E4 and other-modality predictions.')
    p.add_argument('--eeg-probs',nargs=5,type=float,help='Current same-window E4 probabilities, fixed class order.')
    p.add_argument('--classifier-failed',action='store_true')
    p.add_argument('--on-quality-error',choices=['raise','unavailable'],default='raise')
    p.add_argument('--stale-after-sec',type=float,default=1.)
    p.add_argument('--context-json',help='Same 5s three-modality probabilities/quality/masks; q_eeg is recomputed.')
    p.add_argument('--fusion-script',help='Trusted original fusion .py; main/evaluate is NOT called.')
    p.add_argument('--f4-checkpoint')
    p.add_argument('--af4b-checkpoint')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda',help='Fusion only; quality detector uses CPU.')
    p.add_argument('--save-fusion-input',help='Optional JSON accepted by original --mode predict.')
    p.add_argument('--output',help='Result JSON. Does not overwrite a model or input file.')
    p.add_argument('--show-features',action='store_true')
    return p.parse_args()


def _check_output_destinations(paths: Sequence[str | Path | None], protected: Sequence[str | Path | None]) -> None:
    outputs=[Path(p).expanduser().resolve() for p in paths if p]
    sources={Path(p).expanduser().resolve() for p in protected if p}
    require(len(outputs)==len(set(outputs)), 'Result and fusion-input paths must be distinct.')
    require(not set(outputs)&sources,'Refusing to overwrite source/calibration/script/checkpoint.')


def main() -> int:
    for out in (sys.stdout,sys.stderr):
        if hasattr(out,'reconfigure'):
            try:
                out.reconfigure(encoding='utf-8',errors='backslashreplace')
            except (OSError,ValueError):
                pass
    args=parse_args()
    print('Script version            :',VERSION)
    if args.self_test:
        r=run_self_test(); print(json_text(r,True))
        if args.output:
            _check_output_destinations([args.output],[__file__]); atomic_json(args.output,r)
        return 0
    runtime=EEGQualityV1(calibration=args.calibration,assets_dir=args.assets_dir,candidate=args.candidate,
                        eeg_unit=args.eeg_unit,scale_to_volts=args.scale_to_volts,unit_evidence=args.unit_evidence,
                        stale_after_sec=args.stale_after_sec,on_quality_error=args.on_quality_error)
    _check_output_destinations([args.output,args.save_fusion_input],
        [runtime.calibration_path,__file__,args.eeg,args.context_json,args.val_manifest,
         args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint])
    print('Calibration               :',runtime.calibration_path)
    print('Runtime candidate         :',args.candidate,'(source EQ2 JSON unchanged)')
    print('Detector input            : 30 x 2500 @ 500 Hz; BEFORE E4 normalization/resampling')
    print('Quality is fusion weight  : NO')
    print('Packages automatically modified: NO')
    if runtime.identity['calibration_unit_provenance'].get('evidence')=='USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED':
        print('[UNIT NOTE] Calibration inherited caller-declared units; this runtime does not independently verify physical scaling.')
    bridge=None
    paths=[args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint]
    if any(paths) or args.fusion_preflight:
        require(all(paths),'Specify --fusion-script, --f4-checkpoint and --af4b-checkpoint together.')
        bridge=AF4CEEGBridge.from_paths(*paths,device=args.device)
    if args.check_assets:
        result=runtime.check_assets()
        print(json_text(result,True))
        if args.output: atomic_json(args.output,result)
        return 0 if result['status']=='PASS' else 2
    if args.preflight or args.fusion_preflight:
        result=runtime.preflight() if args.preflight else {'status':'PASS'}
        if bridge is not None:
            result['fusion_contract_preflight']=bridge.contract_preflight()
        require(not args.fusion_preflight or bridge is not None,'Fusion checkpoints required.')
        print(json_text(result,True))
        print('EEG QUALITY PREFLIGHT STATUS : PASS (see explicit test scope)')
        if args.output: atomic_json(args.output,result)
        return 0
    context=read_json(Path(args.context_json)) if args.context_json else None
    if context is not None:
        require(args.eeg_probs is None,'Supply eeg_probs in context OR --eeg-probs, not both.')
        require(not args.window_id or args.window_id==context.get('window_id'),'CLI/context window_id differ.')
    require(bridge is None or context is not None,'Actual fusion needs current EEG/Audio/Video context JSON.')
    require(not args.save_fusion_input or context is not None,'--save-fusion-input needs complete --context-json.')
    wid=args.window_id or (context.get('window_id') if context else None) or args.window_key
    names=args.channel_names.split(',') if args.channel_names else None
    if args.val_manifest:
        require(bool(args.window_key),'--val-manifest needs --window-key; no first/latest sample is guessed.')
        x,meta=load_eav_val_window(args.val_manifest,args.window_key)
        _check_output_destinations([args.output,args.save_fusion_input],[meta['path']])
        assessment=runtime.assess_array(x,channel_names=CHANNELS,window_id=wid,sample_rate=args.sample_rate)
        assessment['file_source']=meta
    elif args.eeg:
        assessment=runtime.assess_file(args.eeg,window_id=wid,start_sample=args.start_sample,
            mat_variable=args.mat_variable,trial_index=args.trial_index,window_index=args.window_index,
            sample_rate=args.sample_rate,channel_names=names,channel_axis=args.channel_axis,
            assume_e4_order=args.assume_e4_channel_order)
    else:
        assessment=runtime.assess_array(None,window_id=wid)
    if context is not None:
        if args.classifier_failed:
            context=copy.deepcopy(context); context['available']['eeg']=False
        result=runtime._complete_process(assessment,context,bridge)
    elif args.eeg_probs is not None or args.classifier_failed or not assessment['eeg_available']:
        payload=runtime.make_fusion_input(assessment,args.eeg_probs,prediction_window_id=assessment['window_id'],
                                          classifier_ok=not args.classifier_failed)
        result={'quality':assessment,'fusion_eeg_input':payload,'fusion':None,'eeg_weight':None}
    else:
        result={'quality':assessment,'fusion_eeg_input':None,'fusion':None,'eeg_weight':None}
    if args.save_fusion_input:
        atomic_json(args.save_fusion_input,result['fusion_input'])
    print('Window ID                 :',assessment['window_id'])
    print('Quality state             :',assessment['quality_state'])
    print('Reason                    :',assessment['reason'])
    print('q_eeg                     :',f"{assessment['q_eeg']:.6f}")
    print('Quality-side available    :',assessment['eeg_available'])
    if result.get('fusion_eeg_input') is not None:
        print('Effective fusion EEG      :',json_text(result['fusion_eeg_input']))
    print('Actual EEG adaptive weight:',result.get('eeg_weight'),'[None = no active reported AF4-B branch]')
    if result.get('fusion'):
        print('Fusion route / state      :',result['fusion']['route'],result['fusion']['system_state'])
        print('Final                     :',json_text(result['fusion']['final']))
    if args.show_features:
        print('Features                  :',json_text(assessment.get('features'),True))
    if args.output:
        atomic_json(args.output,{'result':result,'runtime_identity':runtime.identity})
        print('Output                    :',Path(args.output).resolve())
    return 2 if assessment['status']=='ERROR' else 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('EEG Quality interrupted; no source data or model changed.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'EEG QUALITY ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2)
