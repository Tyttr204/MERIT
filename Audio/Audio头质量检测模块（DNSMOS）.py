#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio Quality V1: raw audio -> DNSMOS P.835 -> AQ3 -> frozen AF4-C.

Single runtime file. It does not import the AQ2/AQ3 research scripts, train a
model, denoise/normalize loudness, infer emotions from MOS, or open EAV manifests.
Required assets: sig_bak_ovr.onnx and audio_quality_calibration.json.

The deployment candidate defaults explicitly to ovrl_level_cap:
    q_audio = min(g(polynomial-mapped OVRL), h(whole-window RMS dBFS)).
The original AQ3 JSON is not modified or retrospectively labelled selected.

Input contract: 5-second windows, converted to mono / 16 kHz without gain
normalization. Integer PCM is scaled using its numeric full scale, NOT peak/RMS.
Other sample rates use scipy.signal.resample_poly. For a stream, accumulate
native-rate samples FIRST, then resample each complete window. Short windows
are not padded into invented evidence. The DNSMOS-only repetition 5s -> 10s
-> one 9.01s model segment is retained exactly from supplied AQ2.

Output q_audio is a quality input, NOT the final Audio contribution weight.
With the supplied frozen fusion implementation and both trained checkpoints,
AF4CBridge reports its real alpha and class-dependent weights. Those describe
the AF4-B adaptive branch, NOT total attribution of the F4 residual mixture.
F4-active and all-missing states have no meaningful single active Audio weight.

No data/flatline/capture failure/stale data -> unavailable and canonical uniform
probabilities. A quality-model failure is status ERROR, not a microphone fault;
default fail-closed policy excludes that input with a labelled guard q=0.
Set --quality-error-policy raise to propagate it instead. No VAD-only or
low-RMS-only hard exclusion is added. Staleness/recovery are engineering runtime
policies, not claimed hardware-validated rules.

Core usage:
    q = AudioQualityV1(assets_dir='audio', candidate='ovrl_level_cap')
    r = q.assess_file('window.wav', window_id='interaction_001')
    a = q.make_fusion_input(r, current_audio_probs,
                           prediction_window_id='interaction_001')
    fusion.predict_one(..., **a)

For one-call quality + fusion use process_file/process_array with an explicit
same-window context and AF4CBridge. The Audio emotion head remains external;
this module cannot derive emotion probabilities or three-modal weights from
an audio waveform alone.

CLI: --self-test | --check-assets | --preflight | --audio WAV | --mic
No automatic downloads or package installation. Microphone access is opt-in.
The script never calls the fusion script's main/evaluate routines.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

VERSION = 'AUDIO-QUALITY-V1.0'
SCHEMA = 'eav.audio_quality.isotonic.v1'
TAU = 0.80
SR, SAMPLES = 16000, 80000
WINDOW_SECONDS = 5.0
DNS_SECONDS = 9.01
DNS_SAMPLES = int(DNS_SECONDS * SR)
MOS_COLUMNS = ('SIG_raw','BAK_raw','OVRL_raw','SIG','BAK','OVRL')
CANDIDATES = ('ovrl_only','ovrl_level_cap')
EMOTIONS = ['Neutral','Sadness','Anger','Happiness','Calmness']
MODALITIES = ['eeg','audio','video']
STANDARD_DNSMOS_SHA = '269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd'
KNOWN_AQ3_RUN = 'aq3_dnsmos_qaudio_20260918_211337_671658'
MODEL_NAME = 'sig_bak_ovr.onnx'
CALIBRATION_NAME = 'audio_quality_calibration.json'

class InputContractError(ValueError):
    """Malformed or incompatible input; do not guess or silently normalize it."""

class QualityModelError(RuntimeError):
    """Quality estimation failed, not evidence that capture hardware failed."""

class IncompleteWindow(InputContractError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise InputContractError(message)


def json_safe(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): json_safe(v) for k,v in x.items()}
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (list,tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x,(bool,np.bool_)):
        return bool(x)
    if isinstance(x,np.integer):
        return int(x)
    if isinstance(x,(float,np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x,Path):
        return str(x)
    return x


def read_json(path: Path) -> dict[str,Any]:
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(obj,dict), f'Expected JSON object: {path}')
    return obj


def atomic_json(path: Path, obj: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(json_safe(obj),ensure_ascii=False,indent=2,
                                  allow_nan=False)+'\n',encoding='utf-8')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def digest_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


SOURCE_SCRIPT_SHA256 = {'Audio头质量检测模块（DNSMOS）.py': '17c66166045a20c930ead92aa76c8c339a45c860de9805a7d66908283f534ffc', 'eav_aq3_calibrate_dnsmos_to_qaudio(1).py': '456ec97af391092dfcf8218bb7dae3c5cf946f97000276078bbf0afbcff7d4ee'}


def strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if value == 0:
            return False
        if value == 1:
            return True
    if isinstance(value, str) and value.strip().lower() in ('0', '1', 'false', 'true'):
        return value.strip().lower() in ('1', 'true')
    raise InputContractError(f'{name} must be explicit bool or 0/1; got {value!r}')


def finite_scalar(value: Any, name: str) -> float:
    require(not isinstance(value, (bool, np.bool_)), f'{name} is not a boolean.')
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise InputContractError(f'{name} must be a finite number.') from exc
    require(math.isfinite(x), f'{name} must be finite; got {value!r}')
    return x


def validate_curve(curve: Mapping[str, Any], feature: str) -> tuple[np.ndarray, np.ndarray]:
    require(curve.get('feature') == feature and curve.get('out_of_bounds') == 'clip', 'Wrong curve input/extrapolation contract.')
    require(curve.get('kind') == 'increasing_piecewise_linear', 'Unsupported calibration curve.')
    x, y = np.asarray(curve.get('x'), dtype=float), np.asarray(curve.get('y'), dtype=float)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) >= 1, 'Invalid knot shapes.')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite curve knots.')
    require(np.all(np.diff(x) > 0) and np.all(np.diff(y) >= -1e-12), 'Curve must be monotonic with distinct x knots.')
    require((y >= 0).all() and (y <= 1).all(), 'Curve outputs must be in [0,1].')
    return x, y


# DNSMOS inference retained from the supplied AQ2 runtime.
def official_segments(x: np.ndarray) -> list[np.ndarray]:
    """Reproduce the waveform segmentation in Microsoft's regular P.835 code."""
    audio = np.asarray(x, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise InputContractError('DNSMOS input must be nonempty finite mono audio.')
    # Prevent an infinite doubling loop on empty audio (checked above).
    while len(audio) < DNS_SAMPLES:
        audio = np.concatenate([audio, audio])
    num_hops = int(np.floor(len(audio) / SR) - DNS_SECONDS) + 1
    segments = []
    for idx in range(num_hops):
        segment = audio[int(idx * SR):int((idx + DNS_SECONDS) * SR)]
        if len(segment) >= DNS_SAMPLES:
            segments.append(np.ascontiguousarray(segment[:DNS_SAMPLES], dtype=np.float32))
    if not segments:
        raise InputContractError('Official DNSMOS segmentation produced no complete segment.')
    return segments


def standard_p835(raw: np.ndarray) -> np.ndarray:
    """Official non-personalized polynomial fits, in SIG / BAK / OVRL order.

    These polynomials belong to DNSMOS's perceptual scale, not the future EAV
    fusion q_audio calibrator. We keep outputs un-clipped, as official code does.
    """
    r = np.asarray(raw, dtype=np.float64)
    if r.shape[-1] != 3:
        raise InputContractError('Expected [...,3] SIG/BAK/OVRL outputs.')
    return np.stack([
        np.polyval([-0.08397278, 1.22083953, 0.0052439], r[...,0]),
        np.polyval([-0.13166888, 1.60915514, -0.39604546], r[...,1]),
        np.polyval([-0.06766283, 1.11546468, 0.04602535], r[...,2]),
    ], axis=-1)


class DNSMOSP835:
    """Local CPU-only ONNX runner. Initialize once and reuse; no Torch import."""
    def __init__(self, model_path: Path, threads: int = 2):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('Install onnxruntime in .venv-video: python -m pip install onnxruntime') from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Passing bytes avoids dependence on Windows Unicode path handling.
        self.session = ort.InferenceSession(model_path.read_bytes(), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != 'input_1' or inputs[0].type != 'tensor(float)':
            raise InputContractError('Wrong ONNX input contract; use DNSMOS/DNSMOS/sig_bak_ovr.onnx, not P808 or VAD.')
        shape = inputs[0].shape
        if len(shape) != 2 or (isinstance(shape[0], int) and shape[0] != 1) or (
            isinstance(shape[1], int) and shape[1] != DNS_SAMPLES
        ):
            raise InputContractError(f'DNSMOS waveform input shape mismatch: {shape}')
        self.info = {'onnxruntime':ort.__version__, 'providers':self.session.get_providers(),
                     'input_name':inputs[0].name, 'input_shape':shape,
                     'input_samples':DNS_SAMPLES, 'model_sha256':digest_file(model_path)}
        # Interface/finite output test ONLY, not an acoustic-quality validation.
        probe = self._run(np.zeros(DNS_SAMPLES, dtype=np.float32))
        if probe.shape != (3,) or not np.isfinite(probe).all():
            raise InputContractError('DNSMOS forward preflight failed.')

    def _run(self, segment: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.session.run(None, {'input_1':segment[None,:]})[0])
        if raw.shape != (1,3):
            raise InputContractError(f'DNSMOS output shape {raw.shape}; expected [1,3].')
        if not np.isfinite(raw).all():
            raise RuntimeError('DNSMOS produced NaN/Inf.')
        return raw[0].astype(np.float64)

    def score(self, audio: np.ndarray) -> dict[str, Any]:
        raw = np.stack([self._run(s) for s in official_segments(audio)])
        calibrated = standard_p835(raw)
        values = np.r_[raw.mean(axis=0), calibrated.mean(axis=0)]
        return dict(zip(MOS_COLUMNS, values.tolist()), dnsmos_segments=len(raw))


# Portable interpolation retained from the supplied AQ3 runtime.
class AudioQualityCalibrator:
    """Portable JSON adapter. No sklearn/ONNX/Torch imports on this runtime path.

    candidate is REQUIRED. An available signal with low q remains available.
    Invalid/missing quality features on a valid payload raise explicitly.
    """
    def __init__(self, artifact: str | Path | Mapping[str, Any], *, candidate: str,
                 expected_model_sha256: str | None = None, allow_preflight: bool = False):
        obj = read_json(Path(artifact)) if isinstance(artifact, (str, Path)) else dict(artifact)
        require(obj.get('schema') == SCHEMA, 'Unsupported calibration artifact.')
        require(candidate in CANDIDATES, f'Explicit candidate must be one of {CANDIDATES}.')
        require(obj.get('router_tau') == TAU, 'Frozen fusion threshold mismatch.')
        require(obj.get('formal_test_used') is False, 'Artifact must be VAL-only.')
        require(obj.get('artifact_state') in ('CANDIDATES_NOT_SELECTED', 'PREFLIGHT_ONLY'), 'Unrecognized artifact status.')
        require(allow_preflight or obj.get('artifact_state') != 'PREFLIGHT_ONLY', 'Do not deploy a preflight artifact.')
        ic = obj.get('input_contract', {})
        require(ic.get('mos_column') == 'OVRL' and ic.get('level_column') == 'rms_dbfs', 'Feature semantics mismatch.')
        require(ic.get('sample_rate') == 16000 and ic.get('window_samples') == 80000, 'Input geometry mismatch.')
        if expected_model_sha256 is not None:
            require(ic.get('dns_model_sha256', '').lower() == expected_model_sha256.lower(), 'Runtime DNSMOS identity mismatch.')
        self.curves = {'ovrl': validate_curve(obj['curves']['ovrl'], 'OVRL'),
                       'level': validate_curve(obj['curves']['level'], 'rms_dbfs')}
        self.candidate = candidate
        self.artifact = obj

    def assess_scores(self, *, payload_available: bool, ovrl: float | None = None,
                      rms_dbfs: float | None = None) -> dict[str, Any]:
        available = strict_bool(payload_available, 'payload_available')
        if not available:
            return {'candidate': self.candidate, 'audio_available': False, 'q_audio': 0.0,
                    'q_mos': None, 'q_level_cap': None, 'quality_state': 'UNAVAILABLE',
                    'reason': 'NO_CURRENT_AUDIO_EVIDENCE', 'quality_model_used': False,
                    'audio_side_healthy': False, 'router_tau': TAU}
        v = finite_scalar(ovrl, 'OVRL')
        xg, yg = self.curves['ovrl']
        qg = float(np.interp(v, xg, yg))
        qh, low_out = None, None
        q = qg
        if self.candidate == 'ovrl_level_cap':
            level = finite_scalar(rms_dbfs, 'rms_dbfs')
            xh, yh = self.curves['level']
            qh = float(np.interp(level, xh, yh))
            low_out = bool(level < xh[0] or level > xh[-1])
            q = min(qg, qh)
        return {'candidate': self.candidate, 'audio_available': True, 'q_audio': float(q),
                'q_mos': qg, 'q_level_cap': qh, 'quality_state': 'HEALTHY' if q >= TAU else 'DEGRADED',
                'reason': 'CALIBRATED_CURRENT_INPUT', 'quality_model_used': True,
                'audio_side_healthy': bool(q >= TAU), 'router_tau': TAU,
                'ovrl_out_of_fit_range': bool(v < xg[0] or v > xg[-1]),
                'rms_out_of_fit_range': low_out,
                'level_cap_active': bool(qh is not None and qh < qg - 1e-12)}

    def make_fusion_input(self, audio_probs: Sequence[float] | None, *, payload_available: bool,
                          ovrl: float | None = None, rms_dbfs: float | None = None,
                          classifier_ok: bool = True) -> dict[str, Any]:
        available = strict_bool(payload_available, 'payload_available')
        classifier = strict_bool(classifier_ok, 'classifier_ok')
        # Clear stale/invalid cached probabilities BEFORE inspecting their values.
        if not available or not classifier:
            return {'audio_probs': [0.2] * 5, 'q_audio': 0.0, 'audio_available': False}
        assessment = self.assess_scores(payload_available=True, ovrl=ovrl, rms_dbfs=rms_dbfs)
        try:
            p = np.asarray(audio_probs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise InputContractError('Expected current five-class probabilities.') from exc
        require(p.shape == (5,) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
                'Expected a current, finite, nonnegative probability vector of length five (not logits).')
        require(abs(float(p.sum()) - 1) <= 1e-3, 'Probability sum is not one.')
        return {'audio_probs': (p / p.sum()).tolist(), 'q_audio': assessment['q_audio'], 'audio_available': True}

# =============================================================================
# Portable assets and raw waveform conversion
# =============================================================================

def asset_path(name: str, explicit: str | Path | None, assets_dir: str | Path | None) -> Path:
    if explicit is not None:
        p=Path(explicit).expanduser().resolve()
        require(p.is_file(),f'Asset not found: {p}')
        return p
    if assets_dir is not None:
        root=Path(assets_dir).expanduser().resolve()
        choices=[root/name,root/'models'/name]
    else:
        here=Path(__file__).resolve().parent
        choices=[here/name,here/'models'/name,Path.cwd()/name,Path.cwd()/'models'/name]
        # Pinned known project locations, not a recursive/latest-run search.
        for root in dict.fromkeys([Path.cwd(),here,*here.parents]):
            aq=root/'EAV_dataset/models/audio_quality'
            choices.append(aq/'DNSMOS_official'/name if name==MODEL_NAME else aq/KNOWN_AQ3_RUN/name)
    for p in dict.fromkeys(choices):
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f'{name} not found. Place it beside this script or use --assets-dir / '
                            '--model-path / --calibration. No download or new calibration is performed.')


def verified_assets(model_path: str | Path | None = None,
                    calibration: str | Path | None = None,
                    assets_dir: str | Path | None = None,
                    candidate: str = 'ovrl_level_cap') -> tuple[Path,Path,AudioQualityCalibrator,dict]:
    cp=asset_path(CALIBRATION_NAME,calibration,assets_dir)
    mp=asset_path(MODEL_NAME,model_path,assets_dir)
    obj=read_json(cp)
    ic=obj.get('input_contract',{})
    require(ic.get('dns_model_sha256')==STANDARD_DNSMOS_SHA,'Calibrator is not for the validated DNSMOS ONNX.')
    require(ic.get('level_gain_normalization') is False and ic.get('level_dc_removal') is False,
            'Only the supplied no-gain-normalization/no-DC-removal contract is supported.')
    require(ic.get('window_seconds')==WINDOW_SECONDS,'Expected calibrated 5-second windows.')
    require(obj.get('input_feature')=='OVRL','Do not use OVRL_raw.')
    require(100_000 <= mp.stat().st_size <= 100_000_000,'Unexpected DNSMOS size; possible LFS pointer/error page.')
    sha=digest_file(mp)
    require(sha==STANDARD_DNSMOS_SHA,'DNSMOS SHA256 mismatch; no silent weight replacement.')
    adapter=AudioQualityCalibrator(obj,candidate=candidate,expected_model_sha256=sha)
    identity={
        'module_version':VERSION,'runtime_candidate':candidate,
        'candidate_selection':'explicit runtime configuration; source AQ3 JSON unchanged',
        'source_artifact_state':obj.get('artifact_state'),
        'model_path':str(mp),'model_sha256':sha,
        'calibration_path':str(cp),'calibration_sha256':digest_file(cp),
        'router_tau':TAU,'source_scripts':SOURCE_SCRIPT_SHA256,
        'quality_is_fusion_weight':False,'emotion_model_included':False,
        'runtime_policy':{'window_seconds':5.0,'signal_floor_hard_rejection':False,
                          'VAD_hard_rejection':False,'exact_flatline_unavailable':True,
                          'quality_smoothing':'none','recovery':'next complete fresh valid window'},
    }
    return mp,cp,adapter,identity


def normalize_probs(values: Any, name: str = 'probabilities') -> list[float]:
    try:
        p=np.asarray(values,dtype=np.float64)
    except (ValueError,TypeError) as exc:
        raise InputContractError(f'{name}: expected current 5-class probabilities.') from exc
    require(p.shape==(5,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),
            f'{name}: expected five finite probabilities, not logits.')
    require(abs(float(p.sum())-1.0)<=1e-3,f'{name}: probabilities must sum to one.')
    return (p/p.sum()).tolist()


def audio_id(window_id: str | None) -> str:
    if window_id is None:
        return 'audio-'+uuid.uuid4().hex
    require(isinstance(window_id,str) and bool(window_id.strip()),'window_id must be a nonempty string.')
    return window_id.strip()


def native_mono(audio: Any, sample_rate: int, channel: str | int = 'mean') -> tuple[np.ndarray,dict]:
    """Convert the numeric PCM representation only; never auto-normalize gain."""
    rate=finite_scalar(sample_rate,'sample_rate')
    require(rate.is_integer() and 8000<=rate<=192000,'sample_rate must be an integer 8000..192000 Hz.')
    a=np.asarray(audio)
    require(a.ndim in (1,2),'Audio shape must be [samples] or [samples,channels], not channels-first.')
    require(a.size>0,'No audio samples.')
    nch=1 if a.ndim==1 else a.shape[1]
    require(1<=nch<=32,'Invalid channel dimension; expected samples-first.')
    dtype=str(a.dtype)
    if np.issubdtype(a.dtype,np.signedinteger) and a.dtype.itemsize in (1,2,4):
        scale=float(2**(a.dtype.itemsize*8-1))
        x=a.astype(np.float64)/scale
        conversion=f'signed PCM {a.dtype.itemsize*8}-bit / {scale:g}'
    elif a.dtype==np.dtype('uint8'):
        x=(a.astype(np.float64)-128.0)/128.0
        conversion='unsigned PCM8 (x-128)/128'
    elif np.issubdtype(a.dtype,np.floating):
        x=a.astype(np.float64)
        require(np.isfinite(x).all(),'NaN/Inf audio samples.')
        require(np.max(np.abs(x))<=1.000001,
                'Float audio must use digital full scale [-1,1]; do not pass integer PCM as float.')
        conversion='float full-scale preserved'
    else:
        raise InputContractError(f'Unsupported PCM dtype {dtype}; use float32/64, int16/int32 or explicit PCM bytes.')
    require(np.isfinite(x).all(),'Nonfinite PCM samples.')
    if channel=='mean':
        mono=x if x.ndim==1 else x.mean(axis=1,dtype=np.float64)
        policy='arithmetic channel mean' if nch>1 else 'mono'
    else:
        require(isinstance(channel,(int,np.integer)) and not isinstance(channel,bool),
                'channel must be mean or zero-based integer.')
        require(0<=int(channel)<nch,'Requested channel is out of range.')
        mono=x if x.ndim==1 else x[:,int(channel)]
        policy=f'channel {int(channel)}'
    return np.ascontiguousarray(mono),{
        'source_sample_rate':int(rate),'source_channels':nch,'source_dtype':dtype,
        'pcm_conversion':conversion,'channel_policy':policy,'source_samples':len(mono),
        'source_seconds':len(mono)/rate,'gain_normalized':False,'DC_removed':False,
        'source_peak':float(np.max(np.abs(x))),
    }


def metrics(x: np.ndarray) -> dict[str,float]:
    x=np.asarray(x,dtype=np.float64)
    rms=float(np.sqrt(np.mean(x*x)))
    peak=float(np.max(np.abs(x)))
    db=lambda z:20.0*math.log10(max(float(z),1e-8))
    return {'rms_dbfs':db(rms),'rms':rms,'peak_dbfs':db(peak),'peak':peak,
            'dc_offset':float(x.mean()),'zero_fraction':float(np.mean(x==0)),
            'near_full_scale_fraction':float(np.mean(np.abs(x)>=.999))}


@dataclass
class PreparedWindow:
    samples: np.ndarray                 # float64: retain AQ2 RMS precision
    metadata: dict[str,Any]
    flatline_reason: str | None = None


def prepare_window(audio: Any, sample_rate: int, channel: str | int = 'mean') -> PreparedWindow:
    mono,meta=native_mono(audio,sample_rate,channel)
    expected=5*int(sample_rate)
    if len(mono)<expected:
        raise IncompleteWindow(f'Need exactly 5 seconds: got {len(mono)}/{expected} source samples; no zero padding.')
    require(len(mono)==expected,'More than 5 seconds supplied. Select a window or use a stream buffer.')
    flat=None
    if mono.max()==mono.min():
        flat='DIGITAL_ZERO_NO_EVIDENCE' if float(mono[0])==0 else 'FLATLINE_NO_EVIDENCE'
    if int(sample_rate)!=SR and flat is None:
        from scipy.signal import resample_poly
        divisor=math.gcd(int(sample_rate),SR)
        x=resample_poly(mono,SR//divisor,int(sample_rate)//divisor).astype(np.float64)
    elif flat is not None:
        # Preserve exact flatline status; do not turn a constant into resampler boundary ringing.
        x=np.full(SAMPLES,float(mono[0]),dtype=np.float64)
    else:
        x=mono
    require(x.shape==(SAMPLES,) and np.isfinite(x).all(),'Resampled shape/nonfinite mismatch.')
    if flat is None and x.max()==x.min():
        flat='DIGITAL_ZERO_NO_EVIDENCE' if x[0]==0 else 'FLATLINE_NO_EVIDENCE'
    meta.update(sample_rate=SR,window_samples=SAMPLES,window_seconds=5.0,
                resampled=int(sample_rate)!=SR,post_resample_peak_above_full_scale=bool(np.max(np.abs(x))>1.000001),
                waveform_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest())
    return PreparedWindow(np.ascontiguousarray(x),meta,flat)


def read_file_window(path: str | Path, start_sec: float = 0.0) -> tuple[np.ndarray,int,dict]:
    """Read one explicit 5-second interval; never score an entire long file as 5s."""
    import soundfile as sf
    p=Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    start=finite_scalar(start_sec,'start_sec')
    require(start>=0,'start_sec must be nonnegative.')
    with sf.SoundFile(str(p),'r') as f:
        sr=int(f.samplerate); offset=int(round(start*sr)); count=5*sr
        if offset+count>len(f):
            raise IncompleteWindow(f'File interval is shorter than 5s: start={start}, total={len(f)/sr:.6f}s.')
        f.seek(offset)
        x=f.read(count,dtype='float64',always_2d=True)
        meta={'audio_path':str(p),'file_subtype':f.subtype,'file_seconds':len(f)/sr,
              'interval_start_sec':offset/sr,'interval_end_sec':(offset+count)/sr}
    return x,sr,meta


# =============================================================================
# Quality runtime; no emotion classifier is inferred from quality scores
# =============================================================================

class AudioQualityV1:
    def __init__(self, *, model_path: str | Path | None = None,
                 calibration: str | Path | None = None, assets_dir: str | Path | None = None,
                 candidate: str = 'ovrl_level_cap', threads: int = 2,
                 stale_after_sec: float = 1.0, quality_error_policy: str = 'exclude'):
        require(isinstance(threads,int) and threads>0,'threads must be positive.')
        self.stale_after_sec=finite_scalar(stale_after_sec,'stale_after_sec')
        require(self.stale_after_sec>0,'stale_after_sec must be positive.')
        require(quality_error_policy in ('exclude','raise'),'quality_error_policy must be exclude or raise.')
        self.quality_error_policy=quality_error_policy
        mp,cp,self.calibrator,self.identity=verified_assets(model_path,calibration,assets_dir,candidate)
        self.scorer=DNSMOSP835(mp,threads=threads)
        self.identity['scorer']=self.scorer.info
        self.identity['runtime_policy'].update(stale_after_sec=self.stale_after_sec,
                                              quality_model_error_policy=quality_error_policy)
        self.clock=time.monotonic
        self._score_lock=threading.Lock()

    def _base(self, window_id: str, live: bool, newest: float | None) -> dict[str,Any]:
        # Invalid timestamps must remain explicit errors, not NaN/Inf in JSON.
        try:
            newest = finite_scalar(newest, 'newest_sample_monotonic') if newest is not None else None
        except InputContractError:
            newest = None
        return {'module':'Audio Quality V1','version':VERSION,'window_id':window_id,
                'window_seconds':5.0,'sample_rate':SR,'candidate':self.calibrator.candidate,
                'router_tau':TAU,'live':bool(live),'newest_sample_monotonic':newest,
                'window_start_monotonic':None if newest is None else newest-5.0,
                'decision_monotonic':self.clock(),'freshness_checked':bool(live),
                'quality_is_fusion_weight':False,'audio_weight':None,
                'audio_weight_source':'computed only by AF4CBridge, not by this quality model',
                'dnsmos':None,'signal_metrics':None,'preprocessing':None,
                'model_sha256':self.identity['model_sha256'],
                'calibration_sha256':self.identity['calibration_sha256']}

    def unavailable(self, reason: str, *, window_id: str | None = None, status: str = 'UNAVAILABLE',
                    live: bool = False, newest_sample_monotonic: float | None = None,
                    error: str | None = None, payload_available: bool = False) -> dict[str,Any]:
        require(status in ('UNAVAILABLE','BUFFERING','ERROR'),'Unknown unavailable status.')
        out=self._base(audio_id(window_id),live,newest_sample_monotonic)
        out.update(status=status,quality_state=status,reason=reason,error=error,
                   payload_available=bool(payload_available),audio_available=False,quality_valid=False,
                   q_audio=0.0,q_mos=None,q_level_cap=None,level_cap_active=False,
                   q_audio_origin='FAIL_CLOSED_GUARD_NOT_MEASURED' if status=='ERROR' else 'NO_CURRENT_EVIDENCE',
                   quality_model_used=False,elapsed_seconds=0.0,
                   fusion_audio_input={'audio_probs':[.2]*5,'q_audio':0.0,'audio_available':False})
        return out

    def _freshness_error(self, live: bool, newest: float | None) -> str | None:
        if not live:
            return None
        try:
            newest=finite_scalar(newest,'newest_sample_monotonic')
        except InputContractError:
            return 'MISSING_OR_INVALID_CAPTURE_TIMESTAMP'
        age=self.clock()-newest
        if age < -0.05:
            return 'CAPTURE_TIMESTAMP_IN_FUTURE_OR_WRONG_CLOCK'
        if age>self.stale_after_sec:
            return 'STALE_AUDIO_INPUT'
        return None

    def assess_array(self, audio: Any, sample_rate: int = SR, *, window_id: str | None = None,
                     channel: str | int = 'mean', live: bool = False,
                     newest_sample_monotonic: float | None = None,
                     capture_error: str | bool | None = None, discontinuity: bool = False) -> dict[str,Any]:
        wid=audio_id(window_id); t0=self.clock()
        live=strict_bool(live,'live')
        discontinuity=strict_bool(discontinuity,'discontinuity')
        if capture_error:
            return self.unavailable('CAPTURE_BACKEND_ERROR',window_id=wid,live=live,
                                    newest_sample_monotonic=newest_sample_monotonic,error=str(capture_error))
        if discontinuity:
            return self.unavailable('DISCONTINUOUS_AUDIO_WINDOW',window_id=wid,live=live,
                                    newest_sample_monotonic=newest_sample_monotonic)
        reason=self._freshness_error(live,newest_sample_monotonic)
        if reason:
            return self.unavailable(reason,window_id=wid,live=live,newest_sample_monotonic=newest_sample_monotonic)
        if audio is None:
            return self.unavailable('NO_AUDIO_SAMPLES',window_id=wid,live=live,
                                    newest_sample_monotonic=newest_sample_monotonic)
        try:
            if np.asarray(audio).size==0:
                return self.unavailable('NO_AUDIO_SAMPLES',window_id=wid,live=live,
                                        newest_sample_monotonic=newest_sample_monotonic)
            prep=prepare_window(audio,sample_rate,channel)
        except IncompleteWindow as exc:
            return self.unavailable('INCOMPLETE_WINDOW',window_id=wid,status='BUFFERING',live=live,
                                    newest_sample_monotonic=newest_sample_monotonic,error=str(exc))
        except (ValueError,TypeError) as exc:
            return self.unavailable('INVALID_AUDIO_SAMPLES',window_id=wid,status='ERROR',live=live,
                                    newest_sample_monotonic=newest_sample_monotonic,error=str(exc))
        return self._assess_prepared(prep,wid,live,newest_sample_monotonic,t0)

    def _assess_prepared(self, prep: PreparedWindow, wid: str, live: bool,
                         newest: float | None, t0: float) -> dict[str,Any]:
        signal=metrics(prep.samples)
        if prep.flatline_reason:
            out=self.unavailable(prep.flatline_reason,window_id=wid,live=live,newest_sample_monotonic=newest)
            out.update(signal_metrics=signal,preprocessing=prep.metadata)
            return out
        try:
            with self._score_lock:
                mos=self.scorer.score(prep.samples)
            require(all(math.isfinite(float(mos[k])) for k in MOS_COLUMNS),'Nonfinite/missing DNSMOS outputs.')
            q=self.calibrator.assess_scores(payload_available=True,ovrl=mos['OVRL'],rms_dbfs=signal['rms_dbfs'])
        except Exception as exc:
            if self.quality_error_policy=='raise':
                raise QualityModelError('DNSMOS or calibration failed for valid input.') from exc
            out=self.unavailable('QUALITY_MODEL_ERROR',window_id=wid,status='ERROR',live=live,
                                 newest_sample_monotonic=newest,error=repr(exc),payload_available=True)
            out.update(quality_model_used=True,signal_metrics=signal,preprocessing=prep.metadata,
                       elapsed_seconds=self.clock()-t0)
            return out
        out=self._base(wid,live,newest)
        out.update(q,status='OK',quality_valid=True,payload_available=True,dnsmos=mos,
                   signal_metrics=signal,preprocessing=prep.metadata,error=None,
                   q_audio_origin='AQ3_CALIBRATION',elapsed_seconds=self.clock()-t0,
                   fusion_audio_input=None)
        # A completed slow inference must not rejuvenate an old captured window.
        reason=self._freshness_error(live,newest)
        if reason:
            rejected=self.unavailable(reason,window_id=wid,live=live,newest_sample_monotonic=newest)
            rejected.update(dnsmos=mos,signal_metrics=signal,preprocessing=prep.metadata,
                            discarded_q_audio=out['q_audio'],elapsed_seconds=self.clock()-t0)
            return rejected
        return json_safe(out)

    def assess_file(self, path: str | Path, *, start_sec: float = 0.0,
                    window_id: str | None = None, channel: str | int = 'mean') -> dict[str,Any]:
        wid=audio_id(window_id)
        try:
            x,sr,meta=read_file_window(path,start_sec)
        except FileNotFoundError as exc:
            return self.unavailable('AUDIO_FILE_MISSING',window_id=wid,error=str(exc))
        except IncompleteWindow as exc:
            return self.unavailable('INCOMPLETE_FILE_WINDOW',window_id=wid,status='BUFFERING',error=str(exc))
        except Exception as exc:
            return self.unavailable('AUDIO_DECODE_ERROR',window_id=wid,status='ERROR',error=repr(exc))
        out=self.assess_array(x,sr,window_id=wid,channel=channel)
        out['file_input']=meta
        return out

    def assess_pcm_bytes(self, data: bytes | bytearray | memoryview | None, *, sample_rate: int,
                         sample_format: str, channels: int = 1, **kwargs: Any) -> dict[str,Any]:
        formats={'int16_le':'<i2','int32_le':'<i4','float32_le':'<f4','uint8':'u1'}
        require(sample_format in formats,'Explicit sample_format required: int16_le/int32_le/float32_le/uint8.')
        require(isinstance(channels,int) and 1<=channels<=32,'channels must be 1..32.')
        if data is None or len(data)==0:
            return self.assess_array(None,sample_rate,**kwargs)
        dt=np.dtype(formats[sample_format])
        require(len(data)%(dt.itemsize*channels)==0,'PCM byte count not divisible by sample/channel frame size.')
        x=np.frombuffer(data,dtype=dt).reshape(-1,channels)
        return self.assess_array(x,sample_rate,**kwargs)

    def make_fusion_input(self, assessment: Mapping[str,Any], audio_probs: Any, *,
                          prediction_window_id: str | None, classifier_ok: bool = True,
                          prediction_window_seconds: float = 5.0) -> dict[str,Any]:
        require(assessment.get('version')==VERSION,'Unexpected assessment schema/version.')
        require(assessment.get('calibration_sha256')==self.identity['calibration_sha256'],
                'Assessment came from different calibration assets.')
        classifier=strict_bool(classifier_ok,'classifier_ok')
        available=strict_bool(assessment.get('audio_available'),'audio_available')
        if not available or not classifier:
            return {'audio_probs':[.2]*5,'q_audio':0.0,'audio_available':False}
        require(assessment.get('quality_valid') is True and assessment.get('status')=='OK',
                'Cannot use an unverified quality result as available.')
        require(prediction_window_id==assessment['window_id'],'Audio probability / quality window_id mismatch.')
        require(finite_scalar(prediction_window_seconds,'prediction_window_seconds')==5.0,
                '5s calibration cannot silently be attached to 20s trial probabilities.')
        if self._freshness_error(bool(assessment.get('live')),assessment.get('newest_sample_monotonic')):
            return {'audio_probs':[.2]*5,'q_audio':0.0,'audio_available':False}
        return self.calibrator.make_fusion_input(audio_probs,payload_available=True,
            ovrl=assessment['dnsmos']['OVRL'],rms_dbfs=assessment['signal_metrics']['rms_dbfs'])

    def process_array(self, audio: Any, sample_rate: int, *, context: Mapping[str,Any],
                      bridge: 'AF4CBridge | None' = None, **kwargs: Any) -> dict[str,Any]:
        """One call from a raw current window to quality, safe payload, and optional actual fusion."""
        require('window_id' in context,'Context needs explicit current window_id.')
        require('window_id' not in kwargs,'Set window_id in context, not twice.')
        assessment=self.assess_array(audio,sample_rate,window_id=context['window_id'],**kwargs)
        return self._process_result(assessment,context,bridge)

    def process_file(self, path: str | Path, *, context: Mapping[str,Any],
                     bridge: 'AF4CBridge | None' = None, start_sec: float = 0.0,
                     channel: str | int = 'mean') -> dict[str,Any]:
        require('window_id' in context,'Context needs explicit current window_id.')
        assessment=self.assess_file(path,start_sec=start_sec,window_id=context['window_id'],channel=channel)
        return self._process_result(assessment,context,bridge)

    def _process_result(self, assessment: dict, context: Mapping[str,Any],
                        bridge: 'AF4CBridge | None') -> dict[str,Any]:
        sample,kwargs=build_fusion_input(self,assessment,context)
        out={'audio_quality':assessment,'fusion_audio_input':{k:kwargs[k] for k in ('audio_probs','q_audio','audio_available')},
             'fusion_input_json':sample,'fusion':None,'audio_weight':None}
        if bridge is not None:
            out['fusion']=bridge.predict(self,assessment,context)
            out['audio_weight']=out['fusion']['audio_weight']
        return out
# =============================================================================
# Exact frozen-fusion adapter: q != alpha != total attribution
# =============================================================================

def build_fusion_input(runtime: AudioQualityV1, assessment: Mapping[str,Any],
                       context: Mapping[str,Any]) -> tuple[dict,dict]:
    require(isinstance(context,Mapping),'Fusion context must be a JSON object/mapping.')
    require(context.get('window_id')==assessment['window_id'],'Fusion context and audio window_id differ.')
    require(context.get('window_seconds')==5.0,'Explicit window_seconds=5.0 required for this online adapter.')
    require(context.get('class_order')==EMOTIONS,'Explicit class_order must match frozen EAV class order.')
    av=context.get('available'); quality=context.get('quality')
    require(isinstance(av,Mapping) and set(MODALITIES)<=set(av),'All three availability flags must be explicit.')
    require(isinstance(quality,Mapping) and {'eeg','video'}<=set(quality),
            'EEG/Video qualities must be explicit; no default healthy values.')
    # Audio quality from context is deliberately not used, preventing a stale
    # JSON q_audio from overriding the current DNSMOS result.
    a=runtime.make_fusion_input(assessment,context.get('audio_probs'),
                              prediction_window_id=context['window_id'],
                              prediction_window_seconds=context['window_seconds'],
                              classifier_ok=strict_bool(av['audio'],'available.audio'))
    probs={'audio':a['audio_probs']}; qs={'audio':a['q_audio']}; masks={'audio':a['audio_available']}
    for m in ('eeg','video'):
        masks[m]=strict_bool(av[m],f'available.{m}')
        if not masks[m]:
            probs[m]=[.2]*5; qs[m]=0.0
        else:
            probs[m]=normalize_probs(context.get(m+'_probs'),m+'_probs')
            q=finite_scalar(quality[m],f'quality.{m}')
            require(0<=q<=1,f'quality.{m} outside [0,1].')
            qs[m]=q
    sample={m+'_probs':probs[m] for m in MODALITIES}
    sample.update(quality=qs,available={m:int(masks[m]) for m in MODALITIES})
    kwargs={m+'_probs':probs[m] for m in MODALITIES}
    kwargs.update({f'q_{m}':qs[m] for m in MODALITIES})
    kwargs.update({f'{m}_available':masks[m] for m in MODALITIES})
    return sample,kwargs


def import_fusion_script(path: str | Path) -> Any:
    """Import the user-provided trusted local implementation; never run main()."""
    p=Path(path).expanduser().resolve()
    require(p.is_file(),f'Fusion script not found: {p}')
    name='_eav_frozen_fusion_'+hashlib.sha256(str(p).encode()).hexdigest()[:16]
    if name in sys.modules:
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,str(p))
    require(spec is not None and spec.loader is not None,'Cannot create fusion import spec.')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    for symbol in ('FinalAF4CSystem','ROUTER_TAU','EMOTIONS','MODALITIES'):
        require(hasattr(mod,symbol),f'Fusion script missing {symbol}.')
    require(float(mod.ROUTER_TAU)==TAU and list(mod.EMOTIONS)==EMOTIONS and list(mod.MODALITIES)==MODALITIES,
            'Frozen fusion threshold/class/modality order mismatch.')
    return mod


class AF4CBridge:
    """Use the supplied frozen network; expose its real adaptive-branch weights.

    F4 has no scalar mixture weight per modality. AF4-B alpha and effective
    weights describe only its adaptive branch; gamma mixes probabilities with
    the F4 branch in log space. No invented total-contribution percentage.
    """
    def __init__(self, system: Any, identity: Mapping[str,Any] | None = None):
        require(callable(getattr(system,'predict_batch',None)),'Expected FinalAF4CSystem.predict_batch.')
        require(float(getattr(system,'tau',float('nan')))==TAU,'Fusion must retain tau=0.80.')
        self.system=system
        self.identity=dict(identity or {'source':'caller-provided loaded FinalAF4CSystem'})
        self._lock=threading.Lock()

    @classmethod
    def from_paths(cls, fusion_script: str | Path, f4_checkpoint: str | Path,
                   af4b_checkpoint: str | Path, device: str = 'cuda') -> 'AF4CBridge':
        mod=import_fusion_script(fusion_script)
        f4=Path(f4_checkpoint).expanduser().resolve(); b=Path(af4b_checkpoint).expanduser().resolve()
        require(f4.is_file() and b.is_file(),'Both actual frozen fusion checkpoints are required.')
        require(device in ('cuda','cpu'),'Fusion device must be cpu or cuda.')
        sysmodel=mod.FinalAF4CSystem(f4_checkpoint=f4,af4b_checkpoint=b,
                                    device=mod.choose_device(device),batch_size=1,tau=TAU)
        return cls(sysmodel,{'fusion_script':str(Path(fusion_script).resolve()),
                            'fusion_script_sha256':digest_file(Path(fusion_script)),
                            'f4_checkpoint':str(f4),'f4_sha256':digest_file(f4),
                            'af4b_checkpoint':str(b),'af4b_sha256':digest_file(b),
                            'mode':'predict_only','benchmark_data_read':False})

    def predict(self, runtime: AudioQualityV1, assessment: Mapping[str,Any],
                context: Mapping[str,Any]) -> dict[str,Any]:
        sample,_=build_fusion_input(runtime,assessment,context)
        return self.predict_sample(sample,window_id=assessment['window_id'])

    def predict_sample(self, sample: Mapping[str,Any], *, window_id: str) -> dict[str,Any]:
        # Validate/canonicalize again even when called directly, before F4 sees x15.
        ps=[]; qs=[]; masks=[]
        for m in MODALITIES:
            a=strict_bool(sample['available'][m],m+' availability')
            if a:
                p=normalize_probs(sample[m+'_probs'],m+'_probs')
                q=finite_scalar(sample['quality'][m],m+' quality')
                require(0<=q<=1,'Quality must be in [0,1].')
            else:
                p=[.2]*5; q=0.0
            ps.extend(p); qs.append(q); masks.append(float(a))
        x=np.asarray([ps],dtype=np.float32); q=np.asarray([qs],dtype=np.float32)
        mask=np.asarray([masks],dtype=np.float32)
        with self._lock:
            r=self.system.predict_batch(x,q,mask)
        route=str(np.asarray(r['route']).reshape(-1)[0])
        state=str(np.asarray(r['system_state']).reshape(-1)[0])
        no_decision=bool(np.asarray(r['router_output']['no_decision']).reshape(-1)[0]>.5)
        final=np.asarray(r['router_output']['final_probs'],dtype=float)[0]
        alpha=np.asarray(r['af4b_output']['alpha'],dtype=float)[0]
        eff=np.asarray(r['af4b_output']['effective_weights'],dtype=float)[0]
        gamma=float(np.asarray(r['af4b_output']['gamma']).reshape(-1)[0])
        require(alpha.shape==(3,) and eff.shape==(3,5),'Unexpected fusion weight shapes.')
        require(np.isfinite(alpha).all() and np.isfinite(eff).all() and math.isfinite(gamma),
                'Fusion returned nonfinite adaptive weights.')
        if not masks[1] and not no_decision:
            require(abs(float(alpha[1]))<=1e-7 and np.max(np.abs(eff[1]))<=1e-7,
                    'Audio mask failed: actual AF4-B Audio adaptive weights are nonzero.')
        final=normalize_probs(final,'final fusion probabilities')
        active=route=='AF4-B' and not no_decision
        reason=('NO_DECISION_NO_ACTIVE_EVIDENCE' if no_decision else
                'F4_ACTIVE_NO_EXPLICIT_MODALITY_WEIGHT' if route=='F4' else 'AF4B_ADAPTIVE_BRANCH_WEIGHT')
        return {'window_id':window_id,'window_seconds':5.0,'system_state':state,'route':route,
                'no_decision':no_decision,'router_tau':TAU,
                'final':{'emotion':'NO_DECISION' if no_decision else EMOTIONS[int(np.argmax(final))],
                         'confidence':None if no_decision else float(max(final)),'probabilities':final},
                'audio_available':bool(masks[1]),'q_audio':float(qs[1]),
                'audio_weight':float(alpha[1]) if active else None,
                'audio_weight_kind':reason,'audio_weight_is_total_attribution':False,
                'af4b_alpha_diagnostic':dict(zip(MODALITIES,alpha.tolist())),
                'af4b_audio_effective_weights_by_class':dict(zip(EMOTIONS,eff[1].tolist())) if active else None,
                'af4b_gamma_diagnostic':gamma,'adaptive_branch_active':active,
                'input_canonicalized':True,'fusion_input':{
                    **{m+'_probs':ps[5*i:5*(i+1)] for i,m in enumerate(MODALITIES)},
                    'quality':dict(zip(MODALITIES,qs)),
                    'available':dict(zip(MODALITIES,[int(v) for v in masks]))}}

    def contract_preflight(self) -> dict[str,Any]:
        """Real loaded networks on synthetic probabilities; not an accuracy test."""
        p=[.05,.15,.2,.5,.1]
        def sample(a: bool, audio_p: Any, qa: float, all_missing: bool=False) -> dict:
            return {'eeg_probs':p,'audio_probs':audio_p,'video_probs':p,
                    'quality':{'eeg':1.,'audio':qa,'video':1.},
                    'available':{'eeg':not all_missing,'audio':a,'video':not all_missing}}
        h=self.predict_sample(sample(True,p,1.0),window_id='preflight')
        d=self.predict_sample(sample(True,p,.4),window_id='preflight')
        m1=self.predict_sample(sample(False,[1,0,0,0,0],0.),window_id='preflight')
        m2=self.predict_sample(sample(False,[0,0,0,0,1],0.),window_id='preflight')
        missing=self.predict_sample(sample(False,[float('nan')]*5,0.,True),window_id='preflight')
        back=self.predict_sample(sample(True,p,1.),window_id='preflight')
        checks={'healthy_F4':h['route']=='F4','degraded_AF4B':d['route']=='AF4-B',
                'missing_audio_gate_zero':m1['audio_weight']==0.,
                'missing_audio_class_weights_zero':all(v==0 for v in m1['af4b_audio_effective_weights_by_class'].values()),
                'stale_audio_content_invariant':np.allclose(m1['final']['probabilities'],m2['final']['probabilities'],atol=1e-7),
                'all_missing_no_decision':missing['no_decision'] and missing['audio_weight'] is None,
                'restored_same_output':np.allclose(h['final']['probabilities'],back['final']['probabilities'],atol=1e-7)}
        require(all(checks.values()),f'Frozen fusion preflight failed: {checks}')
        return {'status':'PASS','checks':checks,'synthetic_probabilities':True,
                'accuracy_evaluated':False,'identity':self.identity}


# =============================================================================
# Native-rate streaming window buffer. No neural inference in the callback.
# =============================================================================

class AudioWindowBuffer:
    """Bounded latest-5s buffer. Clear on gaps/faults; recover with 5s fresh data.

    Call push()/poll() from ONE worker, not the realtime device callback.
    End timestamps must be from the same time.monotonic clock as the runtime.
    hop_seconds is the update cadence; the analysed window remains exactly 5s.
    """
    def __init__(self, runtime: AudioQualityV1, sample_rate: int, *, hop_seconds: float = 1.0,
                 channel: str | int = 'mean', continuity_tolerance_sec: float = .10):
        rate=finite_scalar(sample_rate,'sample_rate')
        require(rate.is_integer() and 8000<=rate<=192000,'Unsupported stream sample rate.')
        self.runtime=runtime; self.sr=int(rate); self.channel=channel
        self.hop=finite_scalar(hop_seconds,'hop_seconds')
        require(0<self.hop<=5,'hop_seconds must be in (0,5].')
        self.tolerance=finite_scalar(continuity_tolerance_sec,'continuity_tolerance_sec')
        require(self.tolerance>=0,'continuity tolerance must be nonnegative.')
        self.prefix='stream-'+uuid.uuid4().hex
        self.sequence=0
        self.reset()

    def reset(self) -> None:
        self.chunks=collections.deque()
        self.n=0
        self.newest=None
        self.last_emitted_end=None
        self.ever_scored=False

    def _event(self, reason: str, *, status: str='UNAVAILABLE',error: str | None=None) -> dict:
        self.sequence+=1
        return self.runtime.unavailable(reason,window_id=f'{self.prefix}-{self.sequence}',status=status,
                                        live=True,newest_sample_monotonic=self.newest,error=error)

    def push(self, audio: Any, newest_sample_monotonic: float, *,
             capture_error: str | bool | None=None, discontinuity: bool=False) -> dict | None:
        if capture_error or discontinuity:
            self.reset()
            return self._event('CAPTURE_BACKEND_ERROR' if capture_error else 'CAPTURE_DISCONTINUITY',
                               error=str(capture_error) if capture_error else None)
        try:
            end=finite_scalar(newest_sample_monotonic,'newest_sample_monotonic')
            fresh=self.runtime._freshness_error(True,end)
            require(fresh is None,fresh or 'stale')
            x,_=native_mono(audio,self.sr,self.channel)
        except Exception as exc:
            self.reset()
            return self._event('INVALID_OR_STALE_STREAM_CHUNK',status='ERROR',error=str(exc))
        gap=None
        if self.newest is not None:
            start=end-len(x)/self.sr
            if end<=self.newest or abs(start-self.newest)>self.tolerance:
                gap='CAPTURE_TIMESTAMP_DISCONTINUITY'
                self.reset()
        self.newest=end
        self.chunks.append(x.copy());self.n+=len(x)
        target=5*self.sr
        while self.n>target:
            excess=self.n-target
            first=self.chunks.popleft()
            if len(first)<=excess:
                self.n-=len(first)
            else:
                self.chunks.appendleft(first[excess:]);self.n-=excess
        if gap:
            return self._event(gap)
        return None

    def poll(self) -> dict | None:
        reason=self.runtime._freshness_error(True,self.newest) if self.newest is not None else None
        if reason:
            event=self._event(reason)
            self.reset()
            return event
        if self.n<5*self.sr:
            return None
        if self.last_emitted_end is not None and self.newest-self.last_emitted_end<self.hop-1/self.sr:
            return None
        x=np.concatenate(list(self.chunks))
        self.sequence+=1;wid=f'{self.prefix}-{self.sequence}'
        self.last_emitted_end=self.newest
        self.ever_scored=True
        out=self.runtime.assess_array(x,self.sr,window_id=wid,live=True,newest_sample_monotonic=self.newest)
        out['stream_hop_seconds']=self.hop
        return out


def microphone_results(runtime: AudioQualityV1, *, device: str | int | None=None,
                       sample_rate: int=48000, channels: int=1, channel: str | int='mean',
                       hop_seconds: float=1.0, duration_seconds: float=0., reconnect_seconds: float=2.0):
    """Opt-in sounddevice capture. Bounded queue; no model calls in callback.

    This Python implementation is soft realtime, not a hard-realtime guarantee.
    Device and queue discontinuities invalidate buffered evidence. Retrying opens
    the configured device again; physical OS/device reconnection is not promised.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError('Microphone mode needs sounddevice: python -m pip install sounddevice') from exc
    require(isinstance(channels,int) and 1<=channels<=32,'channels must be 1..32.')
    require(math.isfinite(duration_seconds) and duration_seconds>=0,'Invalid duration.')
    require(math.isfinite(reconnect_seconds) and reconnect_seconds>0,'Invalid reconnect delay.')
    started=runtime.clock()
    running=lambda:duration_seconds==0 or runtime.clock()-started<duration_seconds
    while running():
        buffer=AudioWindowBuffer(runtime,sample_rate,hop_seconds=hop_seconds,channel=channel)
        messages=queue.Queue(maxsize=128)
        dropped=threading.Event()
        latest_callback=[None]
        def callback(indata,frames,times,status):
            now=time.monotonic()
            # PortAudio times are in a DIFFERENT clock; map their relative age
            # into monotonic, never compare an ADC timestamp directly with it.
            try:
                adc_end=float(times.inputBufferAdcTime)+frames/sample_rate
                delay=float(times.currentTime)-adc_end
                if not math.isfinite(delay) or float(times.inputBufferAdcTime)<=0 or delay < -.10:
                    item=(None,None,'INVALID_CAPTURE_CLOCK')
                else:
                    end=now-max(0.,delay)
                    item=(indata.copy(),end,str(status) if status else None)
                latest_callback[0]=now
                messages.put_nowait(item)
            except queue.Full:
                dropped.set()
            except Exception:
                dropped.set()
        try:
            with sd.InputStream(device=device,samplerate=sample_rate,channels=channels,
                                dtype='float32',callback=callback,blocksize=0) as stream:
                opened=runtime.clock()
                yield runtime.unavailable('WAITING_FOR_FIVE_SECONDS',status='BUFFERING',live=True)
                while running():
                    if not stream.active:
                        raise RuntimeError('Capture stream became inactive.')
                    if dropped.is_set():
                        dropped.clear();buffer.reset()
                        while True:
                            try: messages.get_nowait()
                            except queue.Empty: break
                        yield runtime.unavailable('CAPTURE_QUEUE_OVERFLOW',live=True)
                        continue
                    try:
                        item=messages.get(timeout=.05)
                    except queue.Empty:
                        last=latest_callback[0] or opened
                        if runtime.clock()-last>runtime.stale_after_sec:
                            raise RuntimeError('No fresh capture callback within configured timeout.')
                        result=buffer.poll()
                        if result is not None: yield result
                        continue
                    items=[item]
                    while True:
                        try:items.append(messages.get_nowait())
                        except queue.Empty:break
                    for x,end,error in items:
                        event=buffer.push(x,end if end is not None else runtime.clock(),capture_error=error)
                        if event is not None:yield event
                    result=buffer.poll()
                    if result is not None:yield result
        except (KeyboardInterrupt,GeneratorExit,QualityModelError):
            # In explicit raise mode a quality-model fault must not be relabelled
            # as a microphone failure or trigger device reconnect attempts.
            raise
        except Exception as exc:
            yield runtime.unavailable('MICROPHONE_CAPTURE_ERROR',live=True,error=repr(exc))
            deadline=runtime.clock()+reconnect_seconds
            while running() and runtime.clock()<deadline:
                time.sleep(.05)

def microphone_pipeline(runtime: AudioQualityV1, context_provider: Callable[[dict],Mapping[str,Any]],
                        bridge: AF4CBridge, **capture_options: Any):
    """Robot integration: synchronizer supplies fresh same-window emotion outputs.

    context_provider is user-owned; it must never recycle a previous window's
    probabilities. It may explicitly mark any unavailable classifier false.
    """
    for assessment in microphone_results(runtime,**capture_options):
        context=context_provider(assessment)
        yield runtime._process_result(assessment,context,bridge)


# =============================================================================
# Self tests and command line
# =============================================================================

def run_self_test() -> dict[str,Any]:
    """Local algorithm/adapter tests, WITHOUT ONNX, devices or trained checkpoints."""
    g={'feature':'OVRL','kind':'increasing_piecewise_linear','out_of_bounds':'clip',
       'x':[1.,2.,3.,4.],'y':[.4,.6,.8,1.]}
    h={**g,'feature':'rms_dbfs','x':[-80.,-60.,-40.,-20.]}
    fixture={'schema':SCHEMA,'router_tau':TAU,'formal_test_used':False,
             'artifact_state':'CANDIDATES_NOT_SELECTED','curves':{'ovrl':g,'level':h},
             'input_contract':{'mos_column':'OVRL','level_column':'rms_dbfs',
                               'sample_rate':16000,'window_samples':80000}}
    cal=AudioQualityCalibrator(fixture,candidate='ovrl_level_cap')
    t=np.arange(SAMPLES)/SR
    x=.02*np.sin(2*np.pi*220*t)
    checks={}
    checks['AQ2_5s_repeat_has_one_segment']=len(official_segments(x))==1
    checks['AQ2_segment_length']=official_segments(x)[0].shape==(DNS_SAMPLES,)
    checks['AQ2_repeat_not_zero_padding']=np.allclose(official_segments(x)[0][SAMPLES:],x[:DNS_SAMPLES-SAMPLES])
    checks['RMS_gain_preserved']=abs((metrics(x*.001)['rms_dbfs']-metrics(x)['rms_dbfs'])+60)<1e-10
    checks['portable_cap']=cal.assess_scores(payload_available=True,ovrl=4.,rms_dbfs=-80.)['q_audio']==.4
    checks['missing_zero']=cal.assess_scores(payload_available=False)['q_audio']==0.
    checks['old_probability_cleared']=cal.make_fusion_input([1,0,0,0,0],payload_available=False)==cal.make_fusion_input([0,0,0,0,1],payload_available=False)
    checks['classifier_error_cleared']=cal.make_fusion_input(None,payload_available=True,classifier_ok=False)['audio_available'] is False
    checks['low_level_not_missing']=cal.assess_scores(payload_available=True,ovrl=3.,rms_dbfs=-140.)['audio_available'] is True
    checks['signed_PCM_conversion']=native_mono(np.array([-32768,0,16384],dtype=np.int16),SR)[0].tolist()==[-1.,0.,.5]
    checks['source_not_modified']=np.array_equal(prepare_window(x,SR).samples,x)
    checks['digital_zero']=prepare_window(np.zeros(SAMPLES),SR).flatline_reason=='DIGITAL_ZERO_NO_EVIDENCE'
    checks['constant_DC']=prepare_window(np.ones(SAMPLES)*.02,SR).flatline_reason=='FLATLINE_NO_EVIDENCE'
    rejected=False
    try:prepare_window(x[:-1],SR)
    except IncompleteWindow:rejected=True
    checks['short_input_not_padded']=rejected
    require(all(checks.values()),f'Self-test failed: {checks}')
    return {'status':'PASS','n_checks':len(checks),'checks':checks,
            'fixture_calibration':True,'actual_DNSMOS_forward':False,
            'actual_fusion_forward':False,'microphone_accessed':False}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Audio Quality V1: raw audio -> DNSMOS + AQ3 -> frozen AF4-C (predict only).')
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--audio',help='WAV/FLAC or another soundfile-supported file; select a 5-second interval.')
    modes.add_argument('--mic',action='store_true',help='Explicitly capture the local microphone; Ctrl+C stops.')
    modes.add_argument('--list-devices',action='store_true',help='List local sounddevice devices; no recording.')
    modes.add_argument('--self-test',action='store_true',help='Offline local checks without trained model files.')
    modes.add_argument('--check-assets',action='store_true',help='Check calibration/model identity without importing ONNX Runtime.')
    modes.add_argument('--preflight',action='store_true',help='Load real ONNX and check finite output; no EAV benchmark.')
    p.add_argument('--assets-dir',help='Directory containing sig_bak_ovr.onnx and audio_quality_calibration.json.')
    p.add_argument('--model-path')
    p.add_argument('--calibration')
    p.add_argument('--candidate',choices=CANDIDATES,default='ovrl_level_cap')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--quality-error-policy',choices=('exclude','raise'),default='exclude')
    p.add_argument('--stale-after-sec',type=float,default=1.0,help='Age of newest sample, NOT age of window start; engineering timeout.')
    p.add_argument('--start-sec',type=float,default=0.,help='File interval begins here; exactly 5 seconds are read.')
    p.add_argument('--window-id',help='Explicit synchronization ID for the selected window.')
    p.add_argument('--channel',default='mean',help='mean or zero-based input channel index.')
    p.add_argument('--context-json','--fusion-input-json',dest='context_json',help='Current same-window probabilities/EEG+Video quality and masks.')
    p.add_argument('--fusion-script',help='Exact trusted local frozen fusion .py; imported, main() is never called.')
    p.add_argument('--f4-checkpoint',help='Actual frozen F4 checkpoint file, not its parent directory.')
    p.add_argument('--af4b-checkpoint',help='Actual frozen AF4-B checkpoint file.')
    p.add_argument('--fusion-device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--output','--json-output',dest='output',help='Atomic JSON output (file/preflight modes); NDJSON in mic mode.')
    p.add_argument('--export-fusion-input',help='Write the exact runtime JSON accepted by the supplied fusion --mode predict.')
    p.add_argument('--mic-device',help='sounddevice device index or name.')
    p.add_argument('--sample-rate',type=int,default=48000,help='Microphone native/requested sample rate.')
    p.add_argument('--channels',type=int,default=1,help='Microphone channel count.')
    p.add_argument('--hop-sec',type=float,default=1.,help='Mic quality update cadence; window remains 5 seconds.')
    p.add_argument('--duration-sec',type=float,default=0.,help='Microphone run limit; 0 means until Ctrl+C.')
    p.add_argument('--reconnect-sec',type=float,default=2.,help='Delay before retrying a failed capture stream.')
    return p.parse_args()


def _guard_output_paths(args: argparse.Namespace, identity: Mapping[str,Any] | None = None) -> None:
    protected=[]
    for s in (args.audio,args.context_json,args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint,
              str(Path(__file__)),*(str(v) for k,v in (identity or {}).items() if k.endswith('_path'))):
        if s: protected.append(Path(s).expanduser().resolve())
    outpaths=[]
    for value in (args.output,args.export_fusion_input):
        if value:
            path=Path(value).expanduser().resolve()
            require(path not in protected,'Output cannot overwrite an input/model/script asset.')
            require(path.name not in (MODEL_NAME,CALIBRATION_NAME),
                    'A report must not overwrite a standard deployment model/calibration asset.')
            outpaths.append(path)
    require(len(set(outpaths))==len(outpaths),'Report and fusion-input outputs must be distinct.')


def _print_assessment(r: Mapping[str,Any]) -> None:
    print(f"Window                    : {r['window_id']}")
    print(f"Status / quality state    : {r['status']} / {r['quality_state']}")
    print(f"Audio available           : {r['audio_available']}")
    print(f"q_audio                   : {r['q_audio']:.6f} (quality, NOT final weight)")
    if r.get('dnsmos'):
        d=r['dnsmos'];print(f"DNSMOS SIG / BAK / OVRL    : {d['SIG']:.4f} / {d['BAK']:.4f} / {d['OVRL']:.4f}")
    if r.get('signal_metrics'):
        print(f"RMS dBFS                  : {r['signal_metrics']['rms_dbfs']:.4f}")
    if r.get('q_mos') is not None:
        print(f"q_mos / q_level_cap       : {r['q_mos']} / {r.get('q_level_cap')}")
    print(f"Reason                    : {r['reason']}")
    if r.get('error'):print(f"Error                     : {r['error']}")


def main() -> int:
    args=parse_args()
    if args.self_test:
        report=run_self_test();_guard_output_paths(args)
        print(json.dumps(report,indent=2))
        if args.output:atomic_json(Path(args.output),report)
        return 0
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices());return 0
    if args.channel!='mean':
        try:args.channel=int(args.channel)
        except ValueError as exc:raise InputContractError('--channel must be mean or an integer.') from exc
    if args.check_assets:
        _,_,_,identity=verified_assets(args.model_path,args.calibration,args.assets_dir,args.candidate)
        _guard_output_paths(args,identity)
        report={'status':'PASS','scope':'assets and hash only; not model inference','identity':identity}
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.output:atomic_json(Path(args.output),report)
        return 0
    has_fusion=any([args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint])
    if has_fusion:
        require(all([args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint]),
                '--fusion-script, --f4-checkpoint and --af4b-checkpoint must be specified together.')
        require(args.preflight or (args.audio and args.context_json),
                'File fusion requires --context-json. Live fusion uses microphone_pipeline() with a synchronizer, not a stale static JSON.')
    require(not (args.mic and (args.context_json or args.export_fusion_input)),
            'Do not repeatedly reuse static probability JSON for microphone windows.')
    require(not args.export_fusion_input or bool(args.audio and args.context_json),
            '--export-fusion-input requires --audio and --context-json.')
    print('='*100)
    print('AUDIO QUALITY V1 — DNSMOS + AQ3 + FROZEN AF4-C ADAPTER')
    runtime=AudioQualityV1(model_path=args.model_path,calibration=args.calibration,assets_dir=args.assets_dir,
                          candidate=args.candidate,threads=args.threads,stale_after_sec=args.stale_after_sec,
                          quality_error_policy=args.quality_error_policy)
    _guard_output_paths(args,runtime.identity)
    print(f"DNSMOS                    : {runtime.identity['model_path']}")
    print(f"Calibration               : {runtime.identity['calibration_path']}")
    print(f"Candidate                 : {args.candidate} | tau={TAU:.2f}")
    bridge=AF4CBridge.from_paths(args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint,
                                args.fusion_device) if has_fusion else None
    if args.preflight:
        report={'status':'PASS','scope':'real DNSMOS load/finite dummy forward and software self-tests',
                'identity':runtime.identity,'self_test':run_self_test(),
                'actual_DNSMOS_load_forward':True,'actual_microphone_test':False,
                'fusion_preflight':bridge.contract_preflight() if bridge else None,
                'quality_accuracy_tested':False}
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.output:atomic_json(Path(args.output),report)
        return 0
    if args.mic:
        device=args.mic_device
        if device and device.isdecimal():device=int(device)
        output_file=None
        try:
            if args.output:
                out=Path(args.output).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
                # Do not accidentally overwrite a prior run's event log.
                output_file=out.open('x',encoding='utf-8')
            for r in microphone_results(runtime,device=device,sample_rate=args.sample_rate,channels=args.channels,
                    channel=args.channel,hop_seconds=args.hop_sec,duration_seconds=args.duration_sec,
                    reconnect_seconds=args.reconnect_sec):
                _print_assessment(r)
                if output_file:
                    output_file.write(json.dumps(json_safe(r),ensure_ascii=False,allow_nan=False)+'\n');output_file.flush()
        finally:
            if output_file:output_file.close()
        return 0
    if args.context_json:
        context=read_json(Path(args.context_json).expanduser().resolve())
        if args.window_id:
            require(context.get('window_id')==args.window_id,'--window-id and context.window_id differ.')
        report=runtime.process_file(args.audio,context=context,bridge=bridge,start_sec=args.start_sec,channel=args.channel)
        if args.export_fusion_input:
            atomic_json(Path(args.export_fusion_input),report['fusion_input_json'])
    else:
        r=runtime.assess_file(args.audio,start_sec=args.start_sec,window_id=args.window_id,channel=args.channel)
        report={'audio_quality':r,'fusion_audio_input':r.get('fusion_audio_input'),
                'fusion':None,'audio_weight':None}
    report['model_identity']=runtime.identity
    _print_assessment(report['audio_quality'])
    if report.get('fusion_audio_input'):
        print('Fusion Audio input        : '+json.dumps(report['fusion_audio_input']))
    if report.get('fusion'):
        f=report['fusion']
        print(f"Fusion route/state        : {f['route']} / {f['system_state']}")
        print(f"Audio weight              : {f['audio_weight']} | {f['audio_weight_kind']}")
        print(f"Final emotion/confidence  : {f['final']['emotion']} / {f['final']['confidence']}")
    else:
        print('Audio final weight        : NOT COMPUTED (requires frozen fusion and current emotion probabilities)')
    if args.output:
        atomic_json(Path(args.output),report);print(f'Output                    : {Path(args.output).resolve()}')
    print('='*100)
    # A missing input is a valid sensor state. A software/model error is not.
    return 2 if report['audio_quality']['status']=='ERROR' else 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped by user. No frozen model or input audio was modified.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'\nAUDIO QUALITY V1 ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2)
