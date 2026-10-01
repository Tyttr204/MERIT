#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Audio Emotion A1 — frozen local emotion2vec+ large + EAV five-class head.

Single runtime file; the A1 training script is NOT imported or executed.
The emotion2vec architecture still comes from the installed FunASR package.
A complete LOCAL emotion2vec+ large model snapshot and the trained EAV head
are required. There is no model download, fitting, VAD, denoising or quality
calibration in this script. No manifests or automatic TEST evaluation are run.

Frozen A1 feature contract (from the supplied training code):
  16 kHz mono / 80,000 samples / one 5-s window
  FunASR generate(granularity='utterance', extract_embedding=True)
  raw 1024-D feats + original 9 scores, in the original label order
  NO extra embedding L2 normalization and NO extra score softmax
  LayerNorm -> Linear -> GELU -> Dropout -> Linear(5)
  Neutral, Sadness, Anger, Happiness, Calmness
Both backbone and EAV head are eval/frozen. The head is evaluated in FP32.
The snapshot's OWN normalize setting remains intact; this is distinct from
external gain/loudness normalization (which is NOT added). Never send a
backbone-normalized waveform to the DNSMOS/level branch.

Asset layout:
  audio_emotion_a1_deployment.py
  best_a1_embedding_scores_validation_selected.pt
  emotion2vec_plus_large/{model.pt,config.yaml,configuration.json,tokens.txt,...}
  train_feature_cache_info.json  # or val_feature_cache_info.json
The small A1 feature metadata pins the 9-score order. Alternatively export
'audio_emotion_a1_contract.json' once and carry that instead. The export pins
actual local files; it does not prove historical identity to unrecorded weights.

Commands:
  python audio_emotion_a1_deployment.py --self-test
  python audio_emotion_a1_deployment.py --check-assets --assets-dir .\audio
  python audio_emotion_a1_deployment.py --check-assets --a1-run "...\stagea1_..."
  python audio_emotion_a1_deployment.py --check-assets --assets-dir .\audio \
      --export-contract .\audio\audio_emotion_a1_contract.json
  python audio_emotion_a1_deployment.py --assets-dir .\audio --preflight
  python audio_emotion_a1_deployment.py --assets-dir .\audio --audio window.wav \
      --window-id capture_001 --output prediction.json

Python:
  head = AudioEmotionA1(assets_dir='audio')
  r = head.predict_array(samples, sample_rate=16000, window_id='capture_001')
  # r['audio_probs'], r['classifier_ok'], r['audio_available'], r['confidence']

Missing input / flatline / stale input returns a uniform placeholder with
classifier_ok=False and audio_available=False. This is NOT a prediction of
Neutral or proof of hardware failure. A model error is status=ERROR, separate
from input loss. Input recovery is re-evaluated without using cached emotion
probabilities. Small nonconstant audio is NOT hard rejected by its level.

The optional process_with_quality_array/file helpers join this head to the
EXISTING audio_quality_v1_deployment.py and, optionally, its AF4CBridge. Those
modules and trained fusion checkpoints are not required for emotion-only use.
q_audio and actual fusion weights are never inferred from head confidence.

Default backend transport is a private temporary FLOAT WAV (no extra PCM-16
quantization) to preserve the original A1 path-based FunASR entry. Files are
removed after each request. '--backend-input array' is an explicit alternative;
'--compare-backend-inputs --audio ...' measures parity on the installed backend.
No raw audio/features are persistently stored unless the caller saves them.

Local-path loading, disable_update=True and trust_remote_code=False avoid
requested hub downloads/remote code. They are NOT a network sandbox: validate
the prepared environment with outbound access disabled before offline shipping.
Only load locally trusted PyTorch/model files. Dependencies can execute code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = 'AUDIO-EMOTION-A1.1'
CONTRACT_SCHEMA = 'eav.audio_emotion.a1.runtime.v1'
CONTRACT_NAME = 'audio_emotion_a1_contract.json'
HEAD_NAME = 'best_a1_embedding_scores_validation_selected.pt'
MODEL_DIR_NAME = 'emotion2vec_plus_large'
EMOTIONS = ['Neutral', 'Sadness', 'Anger', 'Happiness', 'Calmness']
SR, SAMPLES, WINDOW_SECONDS = 16000, 80000, 5.0
EMBED_DIM, SCORE_DIM = 1024, 9
CANDIDATES = {'embedding': 1024, 'embedding_scores': 1033}
# Only an optional fallback used with an explicit command-line opt-in. It does
# not assert that a historical run's nine labels were independently recovered.
OFFICIAL_LABELS = ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy',
                   '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']


class InputContractError(ValueError):
    """Input or artifact contract is invalid; never guess a class order."""


class IncompleteWindow(InputContractError):
    pass


class ModelContractError(RuntimeError):
    """The frozen head/backbone interface differs from the recorded A1 contract."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise InputContractError(message)


def json_safe(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, Path):
        return str(x)
    return x


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(value, dict), f'JSON must be an object: {path}')
    return value


def atomic_json(path: Path, obj: Any, *, overwrite: bool = False) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists; choose a new name: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(json_safe(obj), ensure_ascii=False,
                                       indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(obj), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise InputContractError(f'{name} must be bool or integer 0/1.')


def finite_scalar(value: Any, name: str) -> float:
    require(not isinstance(value, (bool, np.bool_)), f'{name} cannot be boolean.')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise InputContractError(f'{name} must be finite.') from exc
    require(math.isfinite(value), f'{name} must be finite.')
    return value


def exact_int(value: Any, name: str, low: int, high: int) -> int:
    x = finite_scalar(value, name)
    require(x.is_integer() and low <= x <= high, f'{name} must be an integer in [{low},{high}].')
    return int(x)


def audio_id(value: str | None) -> str:
    if value is None:
        return 'audio-' + uuid.uuid4().hex
    require(isinstance(value, str) and bool(value.strip()), 'window_id must be a nonempty string.')
    return value.strip()


def normalize_probs(value: Any) -> list[float]:
    p = np.asarray(value, dtype=np.float64)
    require(p.shape == (5,) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
            'Expected five current class probabilities, not logits.')
    require(abs(float(p.sum()) - 1) <= 1e-3, 'Probability sum differs from one.')
    return (p / p.sum()).tolist()


# The following native conversion/window helpers preserve the same numerical
# implementation as the previously delivered Audio Quality V1 runtime. Keeping
# the two branches on the same raw window avoids quality/probability mismatch.


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


def pcm_array(data: bytes | bytearray | memoryview, sample_format: str, channels: int) -> np.ndarray:
    formats = {'int16_le': '<i2', 'int32_le': '<i4', 'float32_le': '<f4',
               'float64_le': '<f8', 'uint8': 'u1', 'int24_le': None}
    require(sample_format in formats, f'Unknown PCM format: {sample_format}')
    channels = exact_int(channels, 'channels', 1, 32)
    require(isinstance(data, (bytes, bytearray, memoryview)), 'PCM input must be bytes-like.')
    raw = memoryview(data).tobytes()
    width = 3 if sample_format == 'int24_le' else np.dtype(formats[sample_format]).itemsize
    require(len(raw) % (width * channels) == 0, 'Incomplete PCM sample/channel frame.')
    if sample_format == 'int24_le':
        z = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = z[:, 0] | (z[:, 1] << 8) | (z[:, 2] << 16)
        # Packed little-endian 24-bit -> signed full-scale FLOAT, not int32 /2^31.
        v = ((v ^ 0x800000) - 0x800000).astype(np.float64) / 8388608.0
    else:
        v = np.frombuffer(raw, dtype=np.dtype(formats[sample_format]))
    return v.reshape(-1, channels)


# =============================================================================
# Asset discovery and recorded score order
# =============================================================================

def existing_path(value: str | Path, *, directory: bool = False) -> Path:
    text = str(value)
    require(not re.match(r'^[a-z]+://', text, re.I), 'Only local asset/audio paths are allowed.')
    if os.name != 'nt' and PureWindowsPath(text).is_absolute():
        raise FileNotFoundError(f'Windows path on non-Windows host; provide its relocated local path: {text}')
    p = Path(value).expanduser().resolve()
    if not (p.is_dir() if directory else p.is_file()):
        raise FileNotFoundError(p)
    return p


def unique_paths(values: Sequence[Path]) -> list[Path]:
    return list(dict.fromkeys(p.expanduser().resolve() for p in values))


def score_labels(value: Any) -> list[str]:
    require(isinstance(value, (list, tuple)) and len(value) == SCORE_DIM,
            'Expected an ordered list of nine original emotion2vec score labels.')
    require(all(isinstance(x, str) and x and x == x.strip() for x in value),
            'Score labels must be nonempty exact strings.')
    require(len(set(value)) == SCORE_DIM, 'Duplicate official score labels.')
    return list(value)


def safe_checkpoint_load(path: Path, allow_unsafe_pickle: bool = False) -> Mapping[str, Any]:
    import torch
    # Never silently switch to unrestricted pickle after a safe loader failure.
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=not allow_unsafe_pickle)
    except Exception as exc:
        raise ModelContractError('Cannot load the local A1 checkpoint. Default uses weights_only=True. '
            'Only for a reviewed legacy checkpoint, --allow-unsafe-checkpoint enables pickle. '
            f'Original error: {type(exc).__name__}: {exc}') from exc
    if not isinstance(ckpt, Mapping):
        raise ModelContractError('A1 checkpoint must be a metadata/state dictionary.')
    return ckpt


def inspect_head(path: Path, expected_candidate: str, allow_unsafe_pickle: bool = False) -> tuple[dict, Mapping]:
    import torch
    ckpt = safe_checkpoint_load(path, allow_unsafe_pickle)
    require(str(ckpt.get('stage', '')).upper() == 'A1', 'Head checkpoint is not stage A1.')
    candidate = str(ckpt.get('candidate', ''))
    require(candidate in CANDIDATES, 'Unsupported A1 feature candidate.')
    require(expected_candidate == 'auto' or candidate == expected_candidate,
            f'Head candidate is {candidate!r}, not requested {expected_candidate!r}.')
    require(ckpt.get('classes') == EMOTIONS, 'Frozen EAV class order does not match.')
    input_dim = exact_int(ckpt.get('input_dim'), 'head.input_dim', 1, 10000)
    require(input_dim == CANDIDATES[candidate], 'A1 candidate and input_dim disagree.')
    hidden = exact_int(ckpt.get('hidden_dim'), 'head.hidden_dim', 1, 8192)
    dropout = finite_scalar(ckpt.get('dropout'), 'head.dropout')
    require(0 <= dropout < 1, 'Invalid head dropout.')
    state = ckpt.get('model_state_dict')
    require(isinstance(state, Mapping), 'Missing model_state_dict.')
    expected_shapes = {
        'net.0.weight': (input_dim,), 'net.0.bias': (input_dim,),
        'net.1.weight': (hidden, input_dim), 'net.1.bias': (hidden,),
        'net.4.weight': (5, hidden), 'net.4.bias': (5,),
    }
    require(set(state) == set(expected_shapes), 'A1 head tensor keys differ from the frozen LayerNorm/MLP.')
    for key, shape in expected_shapes.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                f'Bad tensor shape/type: {key}, expected {shape}.')
        require(value.is_floating_point() and torch.isfinite(value).all().item(), f'Invalid tensor values: {key}.')
    info = {'path': str(path), 'sha256': digest_file(path), 'stage': 'A1', 'candidate': candidate,
            'classes': EMOTIONS, 'input_dim': input_dim, 'hidden_dim': hidden,
            'dropout': dropout, 'epoch': ckpt.get('epoch'),
            'selection_metric': ckpt.get('selection_metric'), 'selection_score': ckpt.get('selection_score')}
    return info, state


def build_head(input_dim: int, hidden_dim: int, dropout: float) -> Any:
    import torch.nn as nn
    class AudioEmotionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 5))
        def forward(self, x: Any) -> Any:
            return self.net(x)
    return AudioEmotionHead()


def local_model_inventory(directory: Path) -> dict:
    """Audit complete official snapshot, never replace it with a guessed model."""
    import yaml
    config_path = directory / 'config.yaml'
    weight_path = directory / 'model.pt'
    token_path = directory / 'tokens.txt'
    for p in (config_path, weight_path, token_path):
        if not p.is_file():
            raise FileNotFoundError(f'Incomplete local emotion2vec snapshot: {p}')
    require(weight_path.stat().st_size > 100_000, 'model.pt is too small (possible LFS pointer/error page).')
    with weight_path.open('rb') as f:
        prefix = f.read(160).lstrip().lower()
    require(not prefix.startswith((b'<', b'version https://git-lfs')), 'Not actual model.pt weights.')
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    require(isinstance(cfg, dict) and cfg.get('model') == 'Emotion2vec', 'Expected official Emotion2vec model config.')
    mc = cfg.get('model_conf', {})
    require(mc.get('embed_dim') == EMBED_DIM, 'Expected emotion2vec+ large 1024-D configuration.')
    require(isinstance(mc.get('normalize'), bool), 'Missing original model_conf.normalize; do not guess it.')
    for field in ('vad_model', 'punc_model', 'spk_model', 'remote_code'):
        require(not cfg.get(field), f'Unexpected auxiliary/remote module in config: {field}')
    require(cfg.get('trust_remote_code', False) is False, 'Remote code is not allowed in this runtime.')
    labels = token_path.read_text(encoding='utf-8-sig').splitlines()
    labels = score_labels([x for x in labels if not x.startswith('unuse')])
    conf_path = directory / 'configuration.json'
    if conf_path.is_file():
        conf = read_json(conf_path)
        metas = conf.get('file_path_metas')
        require(isinstance(metas, dict), 'configuration.json lacks file_path_metas.')
        require(metas.get('config') == 'config.yaml' and metas.get('init_param') == 'model.pt',
                'Nonstandard snapshot config/weight paths: use the original complete A1 model package.')
        tok = metas.get('tokenizer_conf', {}).get('token_list')
        require(tok == 'tokens.txt', 'Expected configuration.json to reference tokens.txt.')
    files = [weight_path, config_path, token_path]
    if conf_path.is_file():
        files.append(conf_path)
    identities = {p.name: {'sha256': digest_file(p), 'bytes': p.stat().st_size} for p in files}
    return {'path': str(directory), 'files': identities, 'official_score_labels': labels,
            'model_class': 'Emotion2vec', 'embedding_dim': EMBED_DIM,
            'backbone_internal_normalize': mc['normalize'],
            'external_gain_normalization_added': False,
            'scope_map_from_config': cfg.get('scope_map'),
            'model_identity_fingerprint': fingerprint(identities)}


def _find_run(roots: Sequence[Path], expected_candidate: str) -> Path | None:
    names = [HEAD_NAME] if expected_candidate != 'embedding' else ['best_a1_embedding_validation_selected.pt']
    good = []
    for root in roots:
        for run in (root / 'EAV_dataset' / 'models' / 'audio').glob('stagea1_frozen_emotion2vecplus_large_*'):
            if any((run / n).is_file() for n in names):
                good.append(run.resolve())
    good = unique_paths(good)
    if len(good) > 1:
        raise InputContractError('Multiple A1 runs; pin --a1-run. Candidates:\n' + '\n'.join(map(str, good)))
    return good[0] if good else None


def resolve_assets(*, assets_dir: str | Path | None = None, a1_run: str | Path | None = None,
                   model_dir: str | Path | None = None, head_checkpoint: str | Path | None = None,
                   feature_info: str | Path | None = None, contract: str | Path | None = None,
                   expected_candidate: str = 'embedding_scores', allow_unverified_score_order: bool = False,
                   allow_unsafe_checkpoint: bool = False) -> tuple[dict, Mapping]:
    require(expected_candidate in (*CANDIDATES, 'auto'), 'Invalid expected_candidate.')
    base = existing_path(assets_dir, directory=True) if assets_dir else Path(__file__).resolve().parent
    run = existing_path(a1_run, directory=True) if a1_run else None
    roots = [base] if assets_dir else unique_paths([base, Path.cwd(), Path(__file__).resolve().parent])
    primary = [run] if run else roots
    default_name = 'best_a1_embedding_validation_selected.pt' if expected_candidate == 'embedding' else HEAD_NAME
    if head_checkpoint:
        head_path = existing_path(head_checkpoint)
    else:
        hits = [p / default_name for p in primary if (p / default_name).is_file()]
        if not hits and run is None:
            run = _find_run(roots, expected_candidate)
            if run:
                hits = [run / default_name]
        if not hits:
            raise FileNotFoundError('Cannot find frozen A1 head. Place it beside the script, '
                                    'or use --a1-run / --head-checkpoint.')
        head_path = existing_path(hits[0])
    head_info, state = inspect_head(head_path, expected_candidate, allow_unsafe_checkpoint)
    # A selection file, when present, must agree; never silently deploy the runner-up.
    selection_path = head_path.parent / 'validation_model_selection.json'
    if selection_path.is_file():
        sel = read_json(selection_path)
        require(sel.get('selected_candidate') == head_info['candidate'],
                'Provided head differs from validation_model_selection.json selected_candidate.')
    cp = existing_path(contract) if contract else None
    if cp is None:
        cps = unique_paths([base / CONTRACT_NAME, head_path.parent / CONTRACT_NAME])
        cp = next((p for p in cps if p.is_file()), None)
    recorded_contract = read_json(cp) if cp else None
    sources: list[dict] = []
    if recorded_contract:
        require(recorded_contract.get('schema') == CONTRACT_SCHEMA, 'Wrong runtime contract schema.')
        require(recorded_contract.get('class_order') == EMOTIONS, 'Runtime contract EAV classes changed.')
        require(recorded_contract.get('head', {}).get('sha256') == head_info['sha256'], 'A1 head hash differs from contract.')
        require(recorded_contract.get('head', {}).get('candidate') == head_info['candidate'], 'Contract candidate mismatch.')
        require(recorded_contract.get('feature_policy') == 'raw_embedding_then_original_scores_no_extra_normalization',
                'Runtime feature policy mismatch.')
        ic = recorded_contract.get('input_contract', {})
        require(ic.get('sample_rate') == SR and ic.get('window_samples') == SAMPLES and
                ic.get('window_seconds') == 5.0 and ic.get('granularity') == 'utterance' and
                ic.get('extract_embedding') is True, 'Runtime input geometry/feature extraction contract mismatch.')
        sources.append({'path': str(cp), 'sha256': digest_file(cp),
                        'labels': score_labels(recorded_contract.get('official_score_labels')),
                        'kind': 'runtime_contract'})
    meta_objects = []
    if feature_info:
        meta_paths = [existing_path(feature_info)]
    else:
        meta_paths = unique_paths([d / n for d in (head_path.parent, base)
                                  for n in ('train_feature_cache_info.json', 'val_feature_cache_info.json')])
        meta_paths = [p for p in meta_paths if p.is_file()]
    for p in meta_paths:
        obj = read_json(p)
        require(obj.get('granularity') == 'utterance' and obj.get('extract_embedding') is True,
                f'{p.name}: original utterance/extract_embedding contract missing.')
        require(obj.get('embedding_shape', [None, None])[-1] == EMBED_DIM and
                obj.get('score_shape', [None, None])[-1] == SCORE_DIM, 'Feature-cache dimensions differ from A1.')
        labs = score_labels(obj.get('official_score_labels'))
        sources.append({'path': str(p), 'sha256': digest_file(p), 'labels': labs, 'kind': 'A1_feature_metadata'})
        meta_objects.append(obj)
    if sources:
        labels = sources[0]['labels']
        require(all(s['labels'] == labels for s in sources), 'A1 metadata/contract nine-score orders disagree.')
        order_verified = (recorded_contract.get('training_score_order_verified', False) if recorded_contract
                          else True)
        if any(s['kind'] == 'A1_feature_metadata' for s in sources):
            order_verified = True
    else:
        require(allow_unverified_score_order,
                'Missing original 9-score order. Copy train_feature_cache_info.json or '
                'val_feature_cache_info.json from the A1 run; or use a previously exported '
                'audio_emotion_a1_contract.json. --allow-unverified-score-order is an explicitly '
                'UNVERIFIED compatibility mode, not the normal deployment path.')
        labels, order_verified = OFFICIAL_LABELS.copy(), False
    if model_dir:
        md = existing_path(model_dir, directory=True)
    else:
        paths = [base / MODEL_DIR_NAME, head_path.parent / MODEL_DIR_NAME]
        if recorded_contract:
            # Absolute provenance is only a fallback; portable sibling layout wins.
            paths.append(Path(recorded_contract.get('backbone', {}).get('path', '__missing__')))
        for obj in meta_objects:
            mp = obj.get('model', {}).get('model_path')
            if mp:
                paths.append(Path(mp))
        present = []
        for p in paths:
            if p.is_dir() and (p / 'config.yaml').is_file():
                present.append(p.resolve())
        if not present:
            raise FileNotFoundError('Complete local emotion2vec_plus_large model directory not found. '
                                    'Use --model-dir with the ORIGINAL A1 backbone snapshot, not a hub ID.')
        md = present[0]
    backbone = local_model_inventory(md)
    require(backbone['official_score_labels'] == labels,
            'Local tokens.txt order differs from recorded A1 score order. No automatic reordering is performed.')
    if recorded_contract:
        require(recorded_contract.get('backbone', {}).get('files') == backbone['files'],
                'Local emotion2vec files do not match runtime contract hashes.')
    info = {'schema': CONTRACT_SCHEMA, 'runtime_version': VERSION, 'class_order': EMOTIONS,
            'head': head_info, 'backbone': backbone, 'official_score_labels': labels,
            'training_score_order_verified': bool(order_verified), 'score_order_sources': sources,
            'feature_policy': 'raw_embedding_then_original_scores_no_extra_normalization',
            'input_contract': {'sample_rate': SR, 'window_samples': SAMPLES, 'window_seconds': 5.0,
                               'granularity': 'utterance', 'extract_embedding': True,
                               'external_gain_normalization': False, 'extra_score_softmax': False,
                               'embedding_l2_normalization': False},
            'packages_at_audit': {n: package_version(n) for n in
                                  ('numpy', 'torch', 'torchaudio', 'funasr', 'soundfile', 'scipy', 'PyYAML')},
            'historical_backbone_hash_recorded_by_original_training': False,
            'identity_note': 'Hashes pin the current supplied local artifacts; original A1 feature '
                             'metadata recorded model_path, not its checkpoint hash. Use the preserved A1 snapshot.',
            'training_performed': False, 'evaluation_performed': False}
    info['asset_fingerprint'] = fingerprint({'head': head_info['sha256'], 'backbone': backbone['files'],
                                           'labels': labels, 'feature_policy': info['feature_policy']})
    return info, state


def choose_device(name: str) -> Any:
    import torch
    require(name == 'cpu' or bool(re.fullmatch(r'cuda(?::\d+)?', name)), 'device must be cpu/cuda/cuda:N.')
    if name.startswith('cuda'):
        require(torch.cuda.is_available(), 'CUDA requested but unavailable; activate the original .venv-video.')
        index = int(name.split(':')[1]) if ':' in name else 0
        require(index < torch.cuda.device_count(), 'CUDA device index does not exist.')
        return torch.device('cuda', index)
    return torch.device('cpu')


# =============================================================================
# Exact original A1 feature semantics and local FunASR runner
# =============================================================================

def parse_emotion2vec_item(item: Any, expected_labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not isinstance(item, Mapping) or 'feats' not in item:
        raise ModelContractError("emotion2vec output must include 'feats'; extract_embedding=True is required.")
    feat = np.asarray(item['feats'], dtype=np.float32).reshape(-1)
    scores = np.asarray(item.get('scores', []), dtype=np.float32).reshape(-1)
    labels = [str(v) for v in item.get('labels', [])]
    if feat.shape != (EMBED_DIM,) or not np.isfinite(feat).all():
        raise ModelContractError(f'Expected finite 1024-D utterance embedding; got {feat.shape}.')
    if scores.shape != (SCORE_DIM,) or not np.isfinite(scores).all():
        raise ModelContractError(f'Expected finite original nine scores; got {scores.shape}.')
    if labels != list(expected_labels):
        raise ModelContractError(f'Nine-score label ORDER mismatch.\nExpected: {list(expected_labels)}\nGot: {labels}')
    # Check, but NEVER transform scores. A1 concatenated the returned values.
    if np.any(scores < 0) or np.any(scores > 1) or abs(float(scores.sum()) - 1.0) > 1e-3:
        raise ModelContractError('Official nine scores are not probabilities; refusing to invent a softmax.')
    return np.ascontiguousarray(feat), np.ascontiguousarray(scores), labels


def make_candidate_features(candidate: str, feats: np.ndarray, scores: np.ndarray) -> np.ndarray:
    e, s = np.asarray(feats, dtype=np.float32), np.asarray(scores, dtype=np.float32)
    require(e.shape == (EMBED_DIM,) and s.shape == (SCORE_DIM,), 'Invalid A1 feature dimensions.')
    require(np.isfinite(e).all() and np.isfinite(s).all(), 'Nonfinite A1 features.')
    require(candidate in CANDIDATES, 'Unknown A1 candidate.')
    return np.ascontiguousarray(e if candidate == 'embedding' else np.concatenate([e, s]), dtype=np.float32)


def single_result(raw: Any) -> Mapping:
    if isinstance(raw, tuple):
        raw = raw[0] if raw else []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], Mapping):
        raise ModelContractError('Expected exactly one FunASR item for one 5-second window.')
    return raw[0]


class FunASRLocalBackbone:
    """Installed built-in Emotion2vec, complete audited snapshot; never a hub alias."""
    def __init__(self, inventory: Mapping, device: Any, *, cpu_threads: int = 4,
                 backend_input: str = 'wav', temp_dir: str | Path | None = None):
        import torch
        require(backend_input in ('wav', 'array'), 'backend_input must be wav or array.')
        cpu_threads = exact_int(cpu_threads, 'cpu_threads', 1, 128)
        self.inventory = dict(inventory)
        self.device = device
        self.backend_input = backend_input
        self.temp_dir = None if temp_dir is None else existing_path(temp_dir, directory=True)
        # This is not advertised as an air-gap/network sandbox. Local assets,
        # no remote-code trust and no update checks are the requested behavior.
        try:
            import funasr
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError('Cannot import the original FunASR environment. Use the same .venv-video '
                               'that ran A1; do not blindly upgrade Torch. ' + repr(exc)) from exc
        root = existing_path(inventory['path'], directory=True)
        self.wrapper = AutoModel(model=str(root), model_path=str(root), hub='ms',
                                 init_param=str(root / 'model.pt'),
                                 device=str(device), disable_update=True, disable_pbar=True,
                                 check_latest=False, trust_remote_code=False,
                                 ncpu=cpu_threads, batch_size=1, fp16=False, bf16=False)
        net = getattr(self.wrapper, 'model', None)
        if not isinstance(net, torch.nn.Module):
            raise ModelContractError('FunASR did not expose a torch model; cannot verify frozen state.')
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        params = list(net.parameters())
        require(bool(params), 'Backbone has no parameters.')
        require(all(p.device == device for p in params), 'FunASR silently selected a different device.')
        require(all(not p.requires_grad for p in params), 'Backbone is not frozen.')
        proj = getattr(net, 'proj', None)
        require(proj is not None and getattr(proj, 'in_features', None) == EMBED_DIM,
                'Loaded backbone has no compatible official classification projection.')
        cfg = getattr(net, 'cfg', None)
        try:
            internal_norm = bool(cfg.normalize)
        except Exception as exc:
            raise ModelContractError('Cannot verify original backbone normalization setting.') from exc
        require(internal_norm == inventory['backbone_internal_normalize'], 'Backbone config normalize mismatch.')
        kw = getattr(self.wrapper, 'kwargs', {})
        init_path = kw.get('init_param') if isinstance(kw, Mapping) else None
        require(init_path is not None and Path(str(init_path)).resolve() == (root / 'model.pt').resolve(),
                'FunASR resolved an unexpected or missing init_param; refusing potentially unloaded weights.')
        require(not any(getattr(self.wrapper, n, None) is not None for n in ('vad_model', 'punc_model', 'spk_model')),
                'Unexpected auxiliary model; original A1 did not use VAD/punctuation/speaker pipelines.')
        self.info = {'funasr_version': getattr(funasr, '__version__', package_version('funasr')),
                     'device': str(device), 'backend_input': backend_input,
                     'init_param': str(init_path), 'parameters': sum(p.numel() for p in params),
                     'trainable_parameters': 0, 'internal_normalize_preserved': internal_norm,
                     'requested_downloads': False, 'trust_remote_code': False,
                     'real_backbone_loaded': True}

    def generate(self, samples: np.ndarray, *, transport: str | None = None) -> Mapping:
        import torch
        mode = self.backend_input if transport is None else transport
        require(mode in ('wav', 'array'), 'Invalid transport.')
        x = np.asarray(samples, dtype=np.float32)
        require(x.shape == (SAMPLES,) and np.isfinite(x).all(), 'Backbone requires 80000 finite mono samples.')
        with torch.inference_mode():
            if mode == 'array':
                # Copy prevents backend code from modifying the quality branch's input.
                raw = self.wrapper.generate(input=x.copy(), granularity='utterance', extract_embedding=True,
                                            fs=SR, cache={}, batch_size=1)
            else:
                import soundfile as sf
                with tempfile.TemporaryDirectory(prefix='eav_a1_', dir=self.temp_dir) as directory:
                    p = Path(directory) / 'window.wav'
                    # FLOAT preserves all float32 model-input samples exactly. No PCM16
                    # round trip, peak normalization, external waveform layer_norm, or VAD.
                    sf.write(str(p), x, SR, subtype='FLOAT', format='WAV')
                    raw = self.wrapper.generate(input=[str(p)], granularity='utterance', extract_embedding=True,
                                                fs=SR, cache={}, batch_size=1)
        return single_result(raw)


# =============================================================================
# Public deployment class
# =============================================================================

class AudioEmotionA1:
    def __init__(self, *, assets_dir: str | Path | None = None, a1_run: str | Path | None = None,
                 model_dir: str | Path | None = None, head_checkpoint: str | Path | None = None,
                 feature_info: str | Path | None = None, contract: str | Path | None = None,
                 expected_candidate: str = 'embedding_scores', device: str = 'cuda',
                 backend_input: str = 'wav', cpu_threads: int = 4,
                 temp_dir: str | Path | None = None, stale_after_sec: float = 1.0,
                 error_policy: str = 'exclude', allow_unverified_score_order: bool = False,
                 allow_unsafe_checkpoint: bool = False):
        import torch
        self._setup_runtime(stale_after_sec, error_policy)
        self.identity, state = resolve_assets(assets_dir=assets_dir, a1_run=a1_run, model_dir=model_dir,
            head_checkpoint=head_checkpoint, feature_info=feature_info, contract=contract,
            expected_candidate=expected_candidate, allow_unverified_score_order=allow_unverified_score_order,
            allow_unsafe_checkpoint=allow_unsafe_checkpoint)
        self.device = choose_device(device)
        hi = self.identity['head']
        self.head = build_head(hi['input_dim'], hi['hidden_dim'], hi['dropout'])
        self.head.load_state_dict(state, strict=True)
        self.head = self.head.float().to(self.device).eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.backbone = FunASRLocalBackbone(self.identity['backbone'], self.device,
            cpu_threads=cpu_threads, backend_input=backend_input, temp_dir=temp_dir)
        self.identity['runtime_backbone'] = self.backbone.info
        self.identity['head_precision'] = 'float32_no_autocast'
        with torch.inference_mode():
            p = self.head(torch.zeros((1, hi['input_dim']), device=self.device))
            require(p.shape == (1, 5) and torch.isfinite(p).all().item(), 'A1 head forward preflight failed.')

    def _setup_runtime(self, stale_after_sec: float, error_policy: str) -> None:
        self.stale_after_sec = finite_scalar(stale_after_sec, 'stale_after_sec')
        require(self.stale_after_sec > 0, 'stale_after_sec must be positive.')
        require(error_policy in ('exclude', 'raise'), 'error_policy must be exclude or raise.')
        self.error_policy = error_policy
        self.clock = time.monotonic
        self._inference_lock = threading.Lock()

    def _freshness_reason(self, live: bool, newest: Any) -> str | None:
        if not live:
            return None
        try:
            newest = finite_scalar(newest, 'newest_sample_monotonic')
        except InputContractError:
            return 'MISSING_OR_INVALID_SAMPLE_TIMESTAMP'
        age = self.clock() - newest
        if age < -0.01:
            return 'FUTURE_SAMPLE_TIMESTAMP'
        if age > self.stale_after_sec:
            return 'STALE_AUDIO_INPUT'
        return None

    def _base(self, window_id: str, live: bool, newest: Any) -> dict[str, Any]:
        stamp = None
        try:
            stamp = finite_scalar(newest, 'newest_sample_monotonic') if newest is not None else None
        except InputContractError:
            pass
        return {'module': 'Audio Emotion A1', 'version': VERSION, 'window_id': window_id,
                'window_seconds': 5.0, 'sample_rate': SR, 'class_order': EMOTIONS.copy(),
                'head_candidate': self.identity['head']['candidate'],
                'asset_fingerprint': self.identity['asset_fingerprint'],
                'training_score_order_verified': self.identity['training_score_order_verified'],
                'live': live, 'newest_sample_monotonic': stamp,
                'window_start_monotonic': None if stamp is None else stamp - 5.0,
                'decision_monotonic': self.clock(), 'freshness_checked': live,
                'quality_computed': False, 'fusion_weight_computed': False,
                'confidence_definition': 'maximum EAV five-class probability, NOT quality or calibrated correctness'}

    def unavailable(self, reason: str, *, window_id: str | None = None, status: str = 'UNAVAILABLE',
                    live: bool = False, newest_sample_monotonic: float | None = None,
                    payload_available: bool = False, error: str | None = None) -> dict:
        require(status in ('UNAVAILABLE', 'ERROR', 'BUFFERING'), 'Invalid unavailable state.')
        out = self._base(audio_id(window_id), bool(live), newest_sample_monotonic)
        out.update(status=status, reason=reason, error=error, payload_available=bool(payload_available),
                   classifier_ok=False, audio_available=False, audio_probs=[0.2] * 5,
                   pred_label_id=None, pred_emotion='NO_AUDIO_EVIDENCE', confidence=None,
                   probabilities_are_placeholder=True, inference_performed=False,
                   preprocessing=None, elapsed_seconds=0.0)
        return out

    def _failure(self, exc: Exception, reason: str, wid: str, live: bool, newest: Any,
                 *, payload_available: bool = False) -> dict:
        if self.error_policy == 'raise':
            raise exc
        return self.unavailable(reason, window_id=wid, status='ERROR', live=live,
            newest_sample_monotonic=newest, payload_available=payload_available,
            error=f'{type(exc).__name__}: {exc}')

    def _infer(self, samples: np.ndarray, transport: str | None = None) -> tuple[np.ndarray, dict]:
        import torch
        # This shared lock covers the stateful FunASR wrapper and head forward.
        with self._inference_lock, torch.inference_mode():
            self.head.eval()
            item = self.backbone.generate(samples, transport=transport)
            feat, scores, labels = parse_emotion2vec_item(item, self.identity['official_score_labels'])
            x = make_candidate_features(self.identity['head']['candidate'], feat, scores)
            logits = self.head(torch.from_numpy(x[None, :]).to(self.device)).float().cpu().numpy()
        require(logits.shape == (1, 5) and np.isfinite(logits).all(), 'A1 returned invalid logits.')
        # Match original A1 softmax_np; keep the head feature path in float32.
        shifted = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(shifted)
        p = p / p.sum(axis=1, keepdims=True)
        return p[0], {'embedding_dim': len(feat), 'embedding_norm': float(np.linalg.norm(feat)),
                      'official_score_labels': labels, 'official_scores': scores.tolist(),
                      'head_input_dim': len(x), 'extra_feature_normalization': False,
                      'head_logits': logits[0].tolist(),
                      'backend_input': transport or self.backbone.backend_input}

    def predict_array(self, audio: Any, sample_rate: int = SR, *, window_id: str | None = None,
                      channel: str | int = 'mean', live: bool = False,
                      newest_sample_monotonic: float | None = None,
                      capture_ok: bool = True, continuous: bool = True) -> dict[str, Any]:
        wid = audio_id(window_id)
        live = strict_bool(live, 'live')
        t0 = self.clock()
        common = dict(window_id=wid, live=live, newest_sample_monotonic=newest_sample_monotonic)
        if not strict_bool(capture_ok, 'capture_ok'):
            return self.unavailable('CAPTURE_BACKEND_ERROR', **common)
        if not strict_bool(continuous, 'continuous'):
            return self.unavailable('DISCONTINUOUS_AUDIO_WINDOW', **common)
        reason = self._freshness_reason(live, newest_sample_monotonic)
        if reason:
            return self.unavailable(reason, **common)
        if audio is None:
            return self.unavailable('NO_AUDIO_SAMPLES', **common)
        try:
            if np.asarray(audio).size == 0:
                return self.unavailable('NO_AUDIO_SAMPLES', **common)
            prep = prepare_window(audio, sample_rate, channel)
        except IncompleteWindow as exc:
            return self.unavailable('INCOMPLETE_AUDIO_WINDOW', status='BUFFERING', error=str(exc), **common)
        except Exception as exc:
            return self._failure(exc, 'INVALID_AUDIO_SAMPLES', wid, live, newest_sample_monotonic)
        if prep.flatline_reason:
            out = self.unavailable(prep.flatline_reason, **common)
            out['preprocessing'] = prep.metadata
            return out
        # Do NOT reject nonflat low-level/noisy audio here: DNSMOS/AQ3 handles quality.
        try:
            p, diagnostics = self._infer(prep.samples)
        except Exception as exc:
            out = self._failure(exc, 'A1_INFERENCE_ERROR', wid, live, newest_sample_monotonic,
                                payload_available=True)
            out.update(preprocessing=prep.metadata, elapsed_seconds=self.clock() - t0,
                       inference_performed=True)
            return out
        reason = self._freshness_reason(live, newest_sample_monotonic)
        if reason:
            out = self.unavailable(reason, **common)
            out.update(preprocessing=prep.metadata, inference_performed=True,
                       inference_result_discarded=True, elapsed_seconds=self.clock() - t0)
            return out
        probs = normalize_probs(p)
        pred = int(np.argmax(probs))
        out = self._base(wid, live, newest_sample_monotonic)
        out.update(status='OK', reason='CURRENT_AUDIO_CLASSIFIED', error=None,
                   payload_available=True, classifier_ok=True, audio_available=True,
                   audio_probs=probs, pred_label_id=pred, pred_emotion=EMOTIONS[pred],
                   confidence=probs[pred], probabilities_are_placeholder=False,
                   preprocessing=prep.metadata, feature_diagnostics=diagnostics,
                   inference_performed=True, elapsed_seconds=self.clock() - t0)
        return json_safe(out)

    def predict_file(self, path: str | Path, *, start_sec: float = 0.0,
                     window_id: str | None = None, channel: str | int = 'mean') -> dict:
        wid = audio_id(window_id)
        try:
            # Input is an explicit 5-s interval, not the entire file treated as 5s.
            x, sr, meta = read_file_window(path, start_sec)
        except FileNotFoundError as exc:
            return self.unavailable('AUDIO_FILE_MISSING', window_id=wid, error=str(exc))
        except IncompleteWindow as exc:
            return self.unavailable('INCOMPLETE_FILE_WINDOW', window_id=wid, status='BUFFERING', error=str(exc))
        except Exception as exc:
            return self._failure(exc, 'AUDIO_DECODE_ERROR', wid, False, None)
        out = self.predict_array(x, sr, window_id=wid, channel=channel)
        out['file_input'] = meta
        return out

    def predict(self, audio_path: str | Path, **kwargs: Any) -> dict:
        """Convenience alias for a local audio file; does not decode video."""
        return self.predict_file(audio_path, **kwargs)

    def predict_pcm_bytes(self, data: bytes | bytearray | memoryview | None, *, sample_rate: int,
                          sample_format: str, channels: int = 1, **kwargs: Any) -> dict:
        if data is None or len(data) == 0:
            return self.predict_array(None, sample_rate, **kwargs)
        x = pcm_array(data, sample_format, channels)
        return self.predict_array(x, sample_rate, **kwargs)

    def predict_many(self, paths: Sequence[str | Path]) -> list[dict]:
        """Sequential, independent windows; model is initialized only once."""
        return [self.predict_file(p) for p in paths]

    def current_classifier_ok(self, prediction: Mapping) -> bool:
        require(prediction.get('version') == VERSION and
                prediction.get('asset_fingerprint') == self.identity['asset_fingerprint'],
                'Prediction comes from another head/runtime.')
        return (prediction.get('status') == 'OK' and prediction.get('classifier_ok') is True
                and prediction.get('audio_available') is True
                and self._freshness_reason(bool(prediction.get('live')),
                                           prediction.get('newest_sample_monotonic')) is None)

    def make_quality_fusion_input(self, prediction: Mapping, quality_runtime: Any,
                                  assessment: Mapping) -> dict:
        """Join actual current probabilities to the already-delivered Quality V1 API."""
        require(prediction.get('window_id') == assessment.get('window_id'), 'Emotion/quality window IDs differ.')
        require(prediction.get('window_seconds') == assessment.get('window_seconds') == 5.0,
                'Emotion and quality must describe the same five-second interval.')
        require(bool(prediction.get('live')) == bool(assessment.get('live')), 'Live/offline domains differ.')
        if prediction.get('live'):
            require(prediction.get('newest_sample_monotonic') == assessment.get('newest_sample_monotonic'),
                    'Emotion/quality capture timestamps differ.')
        # Both preprocessors hash the SAME float64 post-resample waveform.
        pp, qp = prediction.get('preprocessing'), assessment.get('preprocessing')
        if pp and qp:
            require(pp.get('waveform_sha256') == qp.get('waveform_sha256'),
                    'Same ID but different actual emotion/quality waveforms. Refusing fusion.')
        ok = self.current_classifier_ok(prediction)
        return quality_runtime.make_fusion_input(assessment, prediction.get('audio_probs'),
            prediction_window_id=prediction['window_id'], classifier_ok=ok,
            prediction_window_seconds=5.0)

    def process_with_quality_array(self, audio: Any, sample_rate: int, *, quality_runtime: Any,
                                   window_id: str | None = None, context: Mapping | None = None,
                                   bridge: Any = None, channel: str | int = 'mean', live: bool = False,
                                   newest_sample_monotonic: float | None = None,
                                   capture_ok: bool = True, continuous: bool = True) -> dict:
        """One snapshot -> existing DNSMOS quality -> A1 -> optional actual AF4-C.

        Other-modality context is explicit and belongs to the same window. Its
        old Audio probability/quality fields are overwritten, never trusted.
        This does not start a second microphone or run a fusion evaluation.
        """
        require(context is None or isinstance(context, Mapping), 'context must be a mapping.')
        wid = audio_id(window_id or (context.get('window_id') if context else None))
        if context is not None:
            require(context.get('window_id') == wid and context.get('window_seconds') == 5.0,
                    'Fusion context needs the same explicit window_id and window_seconds=5.0.')
            require(context.get('class_order') == EMOTIONS, 'Fusion context class order mismatch.')
        snapshot = None if audio is None else np.array(audio, copy=True)
        assessment = quality_runtime.assess_array(snapshot, sample_rate, window_id=wid, channel=channel,
            live=live, newest_sample_monotonic=newest_sample_monotonic,
            capture_error=None if strict_bool(capture_ok, 'capture_ok') else 'CAPTURE_BACKEND_ERROR',
            discontinuity=not strict_bool(continuous, 'continuous'))
        if not assessment.get('audio_available'):
            prediction = self.unavailable('QUALITY_BRANCH_EXCLUDED_CURRENT_INPUT', window_id=wid,
                status='ERROR' if assessment.get('status') == 'ERROR' else
                       'BUFFERING' if assessment.get('status') == 'BUFFERING' else 'UNAVAILABLE',
                live=live, newest_sample_monotonic=newest_sample_monotonic,
                payload_available=bool(assessment.get('payload_available')), error=assessment.get('error'))
        else:
            prediction = self.predict_array(snapshot, sample_rate, window_id=wid, channel=channel,
                live=live, newest_sample_monotonic=newest_sample_monotonic,
                capture_ok=capture_ok, continuous=continuous)
        payload = self.make_quality_fusion_input(prediction, quality_runtime, assessment)
        out = {'audio_emotion': prediction, 'audio_quality': assessment,
               'fusion_audio_input': payload, 'audio_weight': None, 'fusion': None}
        if bridge is not None:
            require(context is not None, 'Actual fusion requires current EEG/Video context.')
            ctx = copy.deepcopy(dict(context))
            av = dict(ctx.get('available', {}))
            av['audio'] = payload['audio_available']
            ctx.update(audio_probs=payload['audio_probs'], available=av)
            # AF4CBridge rechecks quality freshness and handles absent modalities.
            out['fusion'] = bridge.predict(quality_runtime, assessment, ctx)
            out['audio_weight'] = out['fusion'].get('audio_weight')
            used = out['fusion']['fusion_input']
            out['fusion_audio_input'] = {'audio_probs': used['audio_probs'],
                'q_audio': used['quality']['audio'], 'audio_available': bool(used['available']['audio'])}
        return json_safe(out)

    def process_with_quality_file(self, path: str | Path, *, quality_runtime: Any,
                                  start_sec: float = 0.0, window_id: str | None = None,
                                  context: Mapping | None = None, bridge: Any = None,
                                  channel: str | int = 'mean') -> dict:
        require(context is None or isinstance(context, Mapping), 'context must be a mapping.')
        wid = audio_id(window_id or (context.get('window_id') if context else None))
        # Decode once: both branches see identical actual samples, not two reads
        # of a file that might be replaced between calls.
        try:
            x, sr, meta = read_file_window(path, start_sec)
        except Exception as exc:
            if self.error_policy == 'raise' and not isinstance(exc, (FileNotFoundError, IncompleteWindow)):
                raise
            # No second read after a failed decode: both branches share the same
            # failure even if the underlying file changes during the call.
            status = 'UNAVAILABLE' if isinstance(exc, FileNotFoundError) else 'BUFFERING' if isinstance(exc, IncompleteWindow) else 'ERROR'
            reason = 'AUDIO_FILE_MISSING' if isinstance(exc, FileNotFoundError) else 'INCOMPLETE_FILE_WINDOW' if isinstance(exc, IncompleteWindow) else 'AUDIO_DECODE_ERROR'
            prediction = self.unavailable(reason, window_id=wid, status=status, error=str(exc))
            assessment = quality_runtime.unavailable(reason, window_id=wid, status=status, error=str(exc))
            payload = self.make_quality_fusion_input(prediction, quality_runtime, assessment)
            out = {'audio_emotion': prediction, 'audio_quality': assessment,
                   'fusion_audio_input': payload, 'audio_weight': None, 'fusion': None}
            if bridge is not None:
                require(context is not None, 'Fusion needs current other-modality context.')
                ctx = copy.deepcopy(dict(context))
                ctx.update(audio_probs=payload['audio_probs'])
                ctx['available'] = dict(ctx.get('available', {}), audio=payload['audio_available'])
                out['fusion'] = bridge.predict(quality_runtime, assessment, ctx)
                out['audio_weight'] = out['fusion'].get('audio_weight')
                used = out['fusion']['fusion_input']
                out['fusion_audio_input'] = {'audio_probs': used['audio_probs'],
                    'q_audio': used['quality']['audio'], 'audio_available': bool(used['available']['audio'])}
            return out
        out = self.process_with_quality_array(x, sr, quality_runtime=quality_runtime,
            window_id=wid, context=context, bridge=bridge, channel=channel)
        out['file_input'] = meta
        return out

    def preflight(self, audio: str | Path | None = None, start_sec: float = 0.0) -> dict:
        if audio is not None:
            result = self.predict_file(audio, start_sec=start_sec, window_id='a1_preflight')
            kind = 'USER_AUDIO_FORWARD'
        else:
            t = np.arange(SAMPLES, dtype=np.float64) / SR
            x = .025 * np.sin(2*np.pi*220*t) + .008 * np.sin(2*np.pi*431*t)
            result = self.predict_array(x, SR, window_id='a1_shape_probe')
            kind = 'SYNTHETIC_SHAPE_PROBE_NOT_ACCURACY'
        return {'status': 'PASS' if result['status'] == 'OK' else 'FAIL', 'scope': kind,
                'real_local_backbone_loaded': True, 'real_local_head_loaded': True,
                'real_audio_accuracy_validated': False, 'result': result, 'identity': self.identity}

    def compare_backend_inputs(self, path: str | Path, start_sec: float = 0.0,
                               atol: float = 1e-5) -> dict:
        raw, sr, _ = read_file_window(path, start_sec)
        prep = prepare_window(raw, sr)
        require(prep.flatline_reason is None, 'Use a nonflat reference WAV.')
        pw, dw = self._infer(prep.samples, transport='wav')
        pa, da = self._infer(prep.samples, transport='array')
        error = float(np.max(np.abs(pw-pa)))
        return {'scope': 'SAME_WAVEFORM_LOCAL_BACKEND_TRANSPORT_PARITY',
                'status': 'PASS' if error <= atol else 'REVIEW', 'max_abs_probability_difference': error,
                'atol': atol, 'argmax_agreement': int(np.argmax(pw)) == int(np.argmax(pa)),
                'wav_probs': pw.tolist(), 'array_probs': pa.tolist(),
                'embedding_norm_wav': dw['embedding_norm'], 'embedding_norm_array': da['embedding_norm'],
                'no_transport_policy_changed': True}


# =============================================================================
# Asset-free basic checks and CLI
# =============================================================================

def run_self_test() -> dict:
    checks = {}
    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    t = np.arange(SAMPLES)/SR
    x = .03*np.sin(2*np.pi*300*t)
    p = prepare_window(x, SR)
    check('5s_16k_shape', p.samples.shape == (SAMPLES,))
    check('no_external_gain_change', np.array_equal(x, p.samples))
    check('low_nonflat_remains_scorable', prepare_window(x*1e-7, SR).flatline_reason is None)
    check('digital_zero', prepare_window(np.zeros(SAMPLES), SR).flatline_reason == 'DIGITAL_ZERO_NO_EVIDENCE')
    check('constant_not_resampler_ringing', prepare_window(np.ones(240000)*.1, 48000).flatline_reason == 'FLATLINE_NO_EVIDENCE')
    for name, bad in [('short_rejected', x[:-1]), ('long_rejected', np.r_[x, 0]), ('nan_rejected', x*np.nan)]:
        caught = False
        try:
            prepare_window(bad, SR)
        except (InputContractError, IncompleteWindow):
            caught = True
        check(name, caught)
    pcm = np.array([-32768, 0, 32767], dtype='<i2')
    xx, _ = native_mono(pcm, SR)
    check('int16_scaling', np.array_equal(xx, [-1, 0, 32767/32768]))
    check('pcm24_sign_extension', np.allclose(pcm_array(bytes([0,0,128, 255,255,127]), 'int24_le', 1)[:,0],
                                            [-1, 8388607/8388608]))
    feat = np.arange(1024, dtype=np.float32)*.013
    scores = np.arange(1,10, dtype=np.float32)/45
    e,s,_ = parse_emotion2vec_item({'feats':feat,'scores':scores,'labels':OFFICIAL_LABELS}, OFFICIAL_LABELS)
    es = make_candidate_features('embedding_scores',e,s)
    check('1033_features', es.shape == (1033,))
    check('no_l2_or_extra_softmax', np.array_equal(es[:1024],feat) and np.array_equal(es[1024:],scores))
    caught = False
    try:
        parse_emotion2vec_item({'feats':feat,'scores':scores,'labels':OFFICIAL_LABELS[::-1]}, OFFICIAL_LABELS)
    except ModelContractError:
        caught = True
    check('changed_label_order_rejected', caught)
    check('embedding_only_shape', make_candidate_features('embedding',e,s).shape == (1024,))
    report = {'status':'PASS','n_checks':len(checks),'checks':checks,
              'real_emotion2vec_or_head_loaded':False,'scope':'PURE_LOGIC_ONLY'}
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Frozen local A1 audio emotion inference; quality remains a separate module.')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--self-test', action='store_true', help='Asset-free numerical tests (no FunASR required).')
    modes.add_argument('--check-assets', action='store_true', help='Audit head/config/tokens/hashes; no backbone inference.')
    modes.add_argument('--preflight', action='store_true', help='Load real local models; use --audio or synthetic shape probe.')
    modes.add_argument('--compare-backend-inputs', action='store_true', help='Requires --audio; compare FLOAT WAV vs array input.')
    p.add_argument('--audio', help='Local audio file; reads one explicit 5-second interval.')
    p.add_argument('--start-sec', type=float, default=0.0)
    p.add_argument('--window-id')
    p.add_argument('--assets-dir')
    p.add_argument('--a1-run', help='Original A1 output directory; reads only head/feature metadata/selection JSON.')
    p.add_argument('--model-dir', help='Complete LOCAL emotion2vec_plus_large directory, not a hub ID.')
    p.add_argument('--head-checkpoint')
    p.add_argument('--feature-info', help='Original train_feature_cache_info.json or val_feature_cache_info.json.')
    p.add_argument('--contract', help='Previously exported audio_emotion_a1_contract.json.')
    p.add_argument('--export-contract', help='Write small pinned contract to a NEW path; usable with --check-assets.')
    p.add_argument('--expected-candidate', choices=['embedding_scores','embedding','auto'], default='embedding_scores')
    p.add_argument('--device', default='cuda')
    p.add_argument('--backend-input', choices=['wav','array'], default='wav')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--temp-dir')
    p.add_argument('--stale-after-sec', type=float, default=1.0)
    p.add_argument('--error-policy', choices=['exclude','raise'], default='exclude')
    p.add_argument('--allow-unverified-score-order', action='store_true', help='Explicit UNVERIFIED fallback; prefer original feature metadata.')
    p.add_argument('--allow-unsafe-checkpoint', action='store_true', help='Unrestricted pickle ONLY for a locally trusted legacy checkpoint.')
    p.add_argument('--output', help='NEW JSON path; never overwrite an existing artifact/audio file.')
    p.add_argument('--with-quality', action='store_true', help='Run the existing sibling audio_quality_v1_deployment.py too.')
    p.add_argument('--quality-assets-dir')
    p.add_argument('--quality-model-path')
    p.add_argument('--quality-calibration')
    p.add_argument('--quality-candidate', choices=['ovrl_only','ovrl_level_cap'], default='ovrl_level_cap')
    p.add_argument('--context-json', help='Same-window EEG/Video data; Audio entries are replaced by new A1 output.')
    p.add_argument('--fusion-script')
    p.add_argument('--f4-checkpoint')
    p.add_argument('--af4b-checkpoint')
    args = p.parse_args()
    if not any((args.self_test,args.check_assets,args.preflight,args.compare_backend_inputs,args.audio)):
        p.print_help()
        p.exit(0)
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        report = run_self_test()
        if args.output:
            atomic_json(Path(args.output), report)
        return 0
    options = {k:getattr(args,k) for k in ('assets_dir','a1_run','model_dir','head_checkpoint','feature_info',
        'contract','expected_candidate','allow_unverified_score_order','allow_unsafe_checkpoint')}
    if args.output:
        require(not Path(args.output).expanduser().exists(), 'Output exists; use a new JSON path.')
    if args.export_contract:
        require(not Path(args.export_contract).expanduser().exists(), 'Contract exists; do not overwrite frozen assets.')
    print('='*105)
    print('AUDIO EMOTION A1 — LOCAL FROZEN DEPLOYMENT')
    print('='*105)
    if args.check_assets:
        identity, _ = resolve_assets(**options)
        report = {'status':'PASS','scope':'ASSET_AUDIT_ONLY_NO_BACKBONE_FORWARD','identity':identity}
        if args.export_contract:
            atomic_json(Path(args.export_contract), identity)
        if args.output:
            atomic_json(Path(args.output), report)
        print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
        return 0
    if args.compare_backend_inputs:
        require(bool(args.audio), '--compare-backend-inputs needs --audio.')
    if args.with_quality:
        require(bool(args.audio) and not args.preflight and not args.compare_backend_inputs,
                '--with-quality is for --audio runtime. Check each real model preflight separately.')
    runtime = AudioEmotionA1(**options, device=args.device, backend_input=args.backend_input,
        cpu_threads=args.cpu_threads, temp_dir=args.temp_dir, stale_after_sec=args.stale_after_sec,
        error_policy=args.error_policy)
    print(f'Head checkpoint           : {runtime.identity["head"]["path"]}')
    print(f'Head candidate / input    : {runtime.identity["head"]["candidate"]} / {runtime.identity["head"]["input_dim"]}')
    print(f'Local backbone            : {runtime.identity["backbone"]["path"]}')
    print(f'Device                    : {runtime.device}')
    print(f'A1 score order recorded   : {runtime.identity["training_score_order_verified"]}')
    print('Checkpoint strict load    : PASS (A1 head); FunASR local backbone loaded')
    if args.export_contract:
        atomic_json(Path(args.export_contract), runtime.identity)
    if args.preflight:
        report = runtime.preflight(args.audio,args.start_sec)
        print(f'A1 DEPLOYMENT PREFLIGHT    : {report["status"]} | {report["scope"]}')
        code = 0 if report['status']=='PASS' else 2
    elif args.compare_backend_inputs:
        report = runtime.compare_backend_inputs(args.audio,args.start_sec)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        code = 0 if report['status']=='PASS' else 2
    elif args.with_quality:
        from audio_quality_v1_deployment import AudioQualityV1, AF4CBridge
        quality = AudioQualityV1(assets_dir=args.quality_assets_dir or args.assets_dir,
            model_path=args.quality_model_path,calibration=args.quality_calibration,
            candidate=args.quality_candidate,stale_after_sec=args.stale_after_sec)
        context = read_json(existing_path(args.context_json)) if args.context_json else None
        bridge = None
        any_fusion = any([args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint])
        if any_fusion:
            require(all([args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint,context]),
                    'Actual fusion needs --fusion-script, --f4-checkpoint, --af4b-checkpoint, --context-json.')
            bridge = AF4CBridge.from_paths(args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint,device=args.device)
        report = runtime.process_with_quality_file(args.audio,quality_runtime=quality,start_sec=args.start_sec,
            window_id=args.window_id,context=context,bridge=bridge)
        print(json.dumps(json_safe(report),ensure_ascii=False,indent=2))
        code = 2 if any(report[k].get('status')=='ERROR' for k in ('audio_emotion','audio_quality')) else 0
    else:
        require(bool(args.audio), 'Use --audio for a real prediction.')
        result = runtime.predict_file(args.audio,start_sec=args.start_sec,window_id=args.window_id)
        report = {'result':result,'model_identity':runtime.identity}
        print(f'Status                    : {result["status"]}')
        print(f'Classifier available      : {result["audio_available"]}')
        print(f'Prediction                : {result["pred_emotion"]}')
        print(f'Confidence                : {result["confidence"]}')
        print(f'Audio probabilities       : {result["audio_probs"]}')
        print(f'Reason                    : {result["reason"]}')
        print('q_audio / fusion weight   : NOT computed by emotion-only head')
        if result.get('error'):
            print(f'Error                     : {result["error"]}')
        code = 2 if result['status']=='ERROR' else 0
    if args.output:
        atomic_json(Path(args.output),report)
        print(f'Output                    : {Path(args.output).resolve()}')
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; no model changed.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'\nA1 DEPLOYMENT ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
