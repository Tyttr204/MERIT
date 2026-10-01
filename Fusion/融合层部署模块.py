#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Frozen AF4-C deployment: probabilities + quality + availability -> emotion.

The FrozenF4Core/AdaptiveF4B classes and numerical inference/router functions
are copied from the user's reviewed standalone fusion source (AST-equivalent).
No new model, no changed tau/beta, no training/evaluate/data-discovery entry.
Core assets: this file and the SAME two frozen checkpoints. A pinned manifest
is optional but recommended. No old source, F1 arrays, or selection JSON needed.

BOUNDARIES:
* q != confidence != total modality contribution. alpha/effective_weights are
  the AF4-B adaptive branch, not attribution through its F4 log-residual path.
* Missing modalities are canonicalized BEFORE BOTH F4 calls, including batch
  and legacy APIs: uniform probabilities, q=0, mask=0. Never retain stale values.
* All quality/mask fields must be explicit. Invalid numerical contracts raise.
* predict_batch/predict_one are numerical compatibility APIs: they CANNOT check
  times. Robot code should use predict_sample(live=True)/predict_packets and,
  for asynchronously produced heads, FusionWindowCoordinator.
* Live packets use actual capture times in the coordinator's monotonic clock
  domain. Never relabel an old prediction as new or stamp it with inference time.
* 5s is an online integration mode, not the original trial benchmark. 20s inputs
  require an externally established quality aggregation. No q smoothing here.
* Neither classifiers, quality detectors, device drivers nor robot actions are
  executed. A valid prediction is NOT permission to move an arm.

Runtime dependencies: numpy, torch only. Use the existing repaired environment.
  python -X utf8 fusion_af4c_deployment.py --self-test
  python -X utf8 fusion_af4c_deployment.py --check-assets --assets-dir .
  python -X utf8 fusion_af4c_deployment.py --preflight --assets-dir . --device cuda
  python -X utf8 fusion_af4c_deployment.py --input-json example_offline.json \
      --assets-dir . --output result.json
No operation defaults to evaluation. With no arguments this prints help.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import os
import platform
import sys
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError("Use the existing verified environment with NumPy and Torch; "
                      "do not upgrade the multimodal stack automatically.") from exc

VERSION = "FUSION-AF4C-DEPLOYMENT.1.0"
INPUT_SCHEMA = "eav.af4c.runtime.input.v1"
PACKET_SCHEMA = "eav.af4c.modality.packet.v1"
MANIFEST_SCHEMA = "eav.af4c.deployment.manifest.v1"
EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
MODALITIES = ["eeg", "audio", "video"]
NUM_CLASSES, NUM_MODALITIES, INPUT_DIM = 5, 3, 15
F4_HIDDEN_DEFAULT = 32
ROUTER_TAU = 0.80
EXPECTED_AF4B_CANDIDATE = "af4b_qbeta_200"
F4_NAME = "best_validation_selected.pt"
AF4B_NAME = "best_validation_robustness.pt"
MANIFEST_NAME = "fusion_deployment_manifest.json"
REVIEWED_HASHES = {
    "f4": "3baee8225f871591f1f28da39717553c8d7a2fcc879dcd6de9297d485629d3d1",
    "af4b": "de1056dd1a2e31778443a47cc8229911ce573f5be3d63886d10d22382dfb088b",
}
SOURCE_FUSION_SHA256 = "ba4a1b315a415c5236b38150f0603ad1a0b62d0515bf9790033b6fb304758fd6"
F4_MAPPING = {"norm.weight": "net.0.weight", "norm.bias": "net.0.bias",
              "fc1.weight": "net.1.weight", "fc1.bias": "net.1.bias",
              "fc2.weight": "net.4.weight", "fc2.bias": "net.4.bias"}
PROBABILITY_ATOL = 1e-5
INFERENCE_WARNING = ("Frozen checkpoints were developed using trial-level probabilities. "
    "5-second online performance and 20-second quality aggregation are not validated by this wrapper.")
WEIGHT_WARNING = "AF4-B adaptive-branch weights only; NOT total attribution through F4 residual."


class InputContractError(ValueError):
    """Malformed metadata/probabilities/quality. Never silently invent health."""
class AssetContractError(RuntimeError):
    """Wrong, incomplete or mismatched frozen assets."""
class FusionInferenceError(RuntimeError):
    """Inference failure. Not a sensor fault or a valid emotion."""
class WindowContractError(InputContractError):
    """Wrong session, window, timing or clock domain."""
class DuplicateWindowError(WindowContractError):
    pass


def require(ok: Any, message: str, error: type[Exception] = InputContractError) -> None:
    if not ok:
        raise error(message)


def strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
        return bool(value)
    raise InputContractError(f"{name}: explicit boolean or numeric 0/1 required, not {value!r}")


def finite_number(value: Any, name: str) -> float:
    require(isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)), f"{name}: numeric scalar required")
    v = float(value)
    require(math.isfinite(v), f"{name}: NaN/Inf are not allowed")
    return v


def text_id(value: Any, name: str) -> str:
    require(isinstance(value, str) and value.strip() == value and 0 < len(value) <= 256,
            f"{name}: nonempty, trimmed string (<=256 characters) required")
    return value


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), "Refusing to serialize NaN/Inf output")
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: str | Path) -> dict:
    def pairs(xs):
        d = {}
        for k, v in xs:
            require(k not in d, f"Duplicate JSON key: {k}")
            d[k] = v
        return d
    def invalid(v):
        raise InputContractError(f"Non-standard JSON constant: {v}")
    p = Path(path)
    require(p.is_file(), f"JSON file not found: {p}")
    require(p.stat().st_size <= 32*1024*1024, "JSON file exceeds 32 MiB")
    obj = json.loads(p.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(obj, dict), "JSON root must be an object")
    return obj


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    p = Path(path).expanduser().resolve()
    require(overwrite or not p.exists(), f"Output exists (not overwritten): {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    # Exclusive create prevents accidental model/source overwrite by a typo.
    with p.open("w" if overwrite else "x", encoding="utf-8") as f:
        f.write(text)


def choose_device(name: str | torch.device) -> torch.device:
    if str(name) == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.device(name)
    require(d.type in ("cpu", "cuda"), "Only cpu/cuda supported in this checked implementation")
    if d.type == "cuda":
        require(torch.cuda.is_available(), "CUDA explicitly requested but unavailable; no CPU fallback")
        if d.index is not None:
            require(d.index < torch.cuda.device_count(), "CUDA device index unavailable")
    return d


def validated_probability(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    require(raw.dtype.kind in "iuf" and raw.shape == (5,), f"{name}: need real [5] probabilities, not logits")
    p = raw.astype(np.float64, copy=True)
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
            f"{name}: probabilities must be finite in [0,1]")
    require(abs(float(p.sum()) - 1.) <= PROBABILITY_ATOL,
            f"{name}: sum must be 1 (tolerance={PROBABILITY_ATOL}), not arbitrary scores")
    return p.astype(np.float32)  # preserve original FP32 input, do not recalibrate


def canonical_arrays(x15: Any, quality: Any, mask: Any, *, max_rows: int = 10000
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x15)
    q0, m0 = np.asarray(quality), np.asarray(mask)
    require(x.ndim == 2 and x.shape[1] == 15 and 0 < len(x) <= max_rows, "Expected nonempty x15 [N,15]")
    require(x.dtype.kind in "iuf", "x15 must be numeric, not string/object/complex/bool")
    require(q0.shape == m0.shape == (len(x),3), "quality and mask must both be [N,3]")
    require(q0.dtype.kind in "iuf", "quality must be real numeric, not bool/string")
    q = np.array(q0, dtype=np.float64, copy=True)
    require(np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)),
            "All quality entries must be explicit finite [0,1]; no clipping of invalid values")
    require(m0.dtype.kind in "biuf" and np.isfinite(m0).all() and np.isin(m0, [0,1]).all(),
            "mask must be exact boolean/0/1, not 0.6, NaN or string")
    m = np.array(m0, dtype=np.float32, copy=True)
    out = np.array(x, dtype=np.float64, copy=True).reshape(-1,3,5)
    for i in range(len(out)):
        for j, name in enumerate(MODALITIES):
            if m[i,j] == 0:
                out[i,j] = 0.2  # discard even NaN/old scores on a declared missing modality
                q[i,j] = 0.
            else:
                out[i,j] = validated_probability(out[i,j], f"{name}[{i}]")
    return np.ascontiguousarray(out.reshape(-1,15),dtype=np.float32), q.astype(np.float32), m


def sample_arrays(sample: Mapping[str, Any]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    require(isinstance(sample, Mapping), "Sample must be a mapping")
    qa, av = sample.get("quality"), sample.get("available")
    require(isinstance(qa, Mapping) and set(qa) == set(MODALITIES), "All three quality fields required")
    require(isinstance(av, Mapping) and set(av) == set(MODALITIES), "All three availability fields required")
    ps, qs, masks = [], [], []
    for m in MODALITIES:
        a = strict_bool(av[m], "available."+m)
        q = finite_number(qa[m], "quality."+m)
        require(0 <= q <= 1, "Quality must be in [0,1]")
        p = validated_probability(sample.get(m+"_probs"), m+"_probs") if a else np.full(5,.2)
        ps.extend(p.tolist()); qs.append(q); masks.append(a)
    return canonical_arrays([ps],[qs],[masks])


# ---------------------------------------------------------------------------
# Original frozen numerical core, formatting only. Do not alter network maths.
# ---------------------------------------------------------------------------
def ensure_probs(name: str, probs: np.ndarray) -> None:
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise RuntimeError(f'{name}: expected [N,5], got {probs.shape}')
    if not np.isfinite(probs).all():
        raise RuntimeError(f'{name}: NaN/Inf detected.')
    if np.min(probs) < -1e-06:
        raise RuntimeError(f'{name}: negative probability detected.')
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-05, rtol=1e-05):
        raise RuntimeError(f'{name}: probability rows do not sum to 1.')

class FrozenF4Core(nn.Module):

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(INPUT_DIM)
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, NUM_CLASSES)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x15: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.act(self.fc1(self.norm(x15)))
        logits = self.fc2(hidden)
        probs = torch.softmax(logits, dim=1)
        return {'hidden': hidden, 'logits': logits, 'probs': probs}

@torch.inference_mode()
def infer_f4_core(model: FrozenF4Core, x15: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x15), batch_size):
        xb = torch.from_numpy(np.asarray(x15[start:start + batch_size], dtype=np.float32)).to(device)
        outputs.append(model(xb)['probs'].float().cpu().numpy())
    probs = np.concatenate(outputs, axis=0).astype(np.float32)
    ensure_probs('F4 output', probs)
    return probs

class AdaptiveF4B(nn.Module):

    def __init__(self, frozen_f4: FrozenF4Core, hidden: int, quality_beta: float, residual_bias_init: float=-3.0) -> None:
        super().__init__()
        self.f4 = frozen_f4
        self.hidden = int(hidden)
        self.quality_beta = float(quality_beta)
        context_dim = hidden + NUM_MODALITIES + NUM_MODALITIES
        self.modality_gate = nn.Linear(context_dim, NUM_MODALITIES)
        self.class_gate = nn.Linear(context_dim, NUM_MODALITIES * NUM_CLASSES)
        self.residual_gate = nn.Linear(context_dim, 1)
        nn.init.zeros_(self.modality_gate.weight)
        nn.init.zeros_(self.modality_gate.bias)
        nn.init.zeros_(self.class_gate.weight)
        nn.init.zeros_(self.class_gate.bias)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, residual_bias_init)
        for parameter in self.f4.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool=True):
        super().train(mode)
        self.f4.eval()
        return self

    def forward(self, x15: torch.Tensor, quality: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x15.ndim != 2 or x15.shape[1] != INPUT_DIM:
            raise RuntimeError(f'x15 must be [B,15], got {tuple(x15.shape)}')
        if quality.shape != (len(x15), NUM_MODALITIES):
            raise RuntimeError(f'quality shape={tuple(quality.shape)}')
        if mask.shape != (len(x15), NUM_MODALITIES):
            raise RuntimeError(f'mask shape={tuple(mask.shape)}')
        quality = torch.clamp(quality, 0.0, 1.0)
        mask = (mask > 0.5).to(x15.dtype)
        quality = quality * mask
        modality_probs = x15.view(-1, NUM_MODALITIES, NUM_CLASSES)
        with torch.no_grad():
            f4_out = self.f4(x15)
            hidden = f4_out['hidden']
            p_f4 = f4_out['probs']
        context = torch.cat([hidden, quality, mask], dim=1)
        eps = 1e-06
        raw_modality_logits = self.modality_gate(context)
        quality_prior = self.quality_beta * torch.log(quality.clamp_min(eps))
        modality_logits = raw_modality_logits + quality_prior
        available = mask > 0.5
        available_count = available.sum(dim=1)
        all_missing = available_count == 0
        safe_modality_logits = modality_logits.masked_fill(~available, -10000.0)
        if all_missing.any():
            safe_modality_logits = safe_modality_logits.clone()
            safe_modality_logits[all_missing] = 0.0
        alpha = torch.softmax(safe_modality_logits, dim=1)
        alpha = alpha * mask
        alpha_sum = alpha.sum(dim=1, keepdim=True)
        alpha = torch.where(alpha_sum > 0, alpha / alpha_sum.clamp_min(eps), torch.full_like(alpha, 1.0 / NUM_MODALITIES))
        class_logits = self.class_gate(context).view(-1, NUM_MODALITIES, NUM_CLASSES)
        class_reliability = torch.sigmoid(class_logits)
        effective_logits = torch.log(alpha[:, :, None].clamp_min(eps)) + torch.log(class_reliability.clamp_min(eps))
        class_available = available[:, :, None].expand(-1, -1, NUM_CLASSES)
        effective_logits = effective_logits.masked_fill(~class_available, -10000.0)
        if all_missing.any():
            effective_logits = effective_logits.clone()
            effective_logits[all_missing] = 0.0
        effective_weights = torch.softmax(effective_logits, dim=1)
        effective_weights = effective_weights * class_available.to(effective_weights.dtype)
        weight_sum = effective_weights.sum(dim=1, keepdim=True)
        effective_weights = torch.where(weight_sum > 0, effective_weights / weight_sum.clamp_min(eps), torch.full_like(effective_weights, 1.0 / NUM_MODALITIES))
        adaptive_score = (effective_weights * modality_probs).sum(dim=1)
        adaptive_probs = adaptive_score / adaptive_score.sum(dim=1, keepdim=True).clamp_min(eps)
        gamma = torch.sigmoid(self.residual_gate(context).squeeze(-1))
        log_final = (1.0 - gamma[:, None]) * torch.log(p_f4.clamp_min(eps)) + gamma[:, None] * torch.log(adaptive_probs.clamp_min(eps))
        final_probs = torch.softmax(log_final, dim=1)
        no_decision = all_missing.to(final_probs.dtype)
        if all_missing.any():
            uniform = torch.full((int(all_missing.sum().item()), NUM_CLASSES), 1.0 / NUM_CLASSES, device=final_probs.device, dtype=final_probs.dtype)
            final_probs = final_probs.clone()
            adaptive_probs = adaptive_probs.clone()
            gamma = gamma.clone()
            final_probs[all_missing] = uniform
            adaptive_probs[all_missing] = uniform
            gamma[all_missing] = 1.0
        return {'f4_probs': p_f4, 'adaptive_probs': adaptive_probs, 'final_probs': final_probs, 'alpha': alpha, 'class_reliability': class_reliability, 'effective_weights': effective_weights, 'gamma': gamma, 'no_decision': no_decision, 'available_count': available_count.to(final_probs.dtype)}

@torch.inference_mode()
def infer_af4b(model: AdaptiveF4B, x15: np.ndarray, quality: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
    model.eval()
    keys = ['f4_probs', 'adaptive_probs', 'final_probs', 'alpha', 'class_reliability', 'effective_weights', 'gamma', 'no_decision', 'available_count']
    parts = {key: [] for key in keys}
    for start in range(0, len(x15), batch_size):
        xb = torch.from_numpy(np.asarray(x15[start:start + batch_size], dtype=np.float32)).to(device)
        qb = torch.from_numpy(np.asarray(quality[start:start + batch_size], dtype=np.float32)).to(device)
        mb = torch.from_numpy(np.asarray(mask[start:start + batch_size], dtype=np.float32)).to(device)
        out = model(xb, qb, mb)
        for key in keys:
            parts[key].append(out[key].float().cpu().numpy())
    result = {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in parts.items()}
    ensure_probs('AF4-B final', result['final_probs'])
    return result

def modality_predictions_and_confidence(x15: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    x15 = np.asarray(x15, dtype=np.float32)
    if x15.ndim != 2 or x15.shape[1] != INPUT_DIM:
        raise RuntimeError(f'Expected [N,15], got {x15.shape}')
    probs3 = x15.reshape(-1, NUM_MODALITIES, NUM_CLASSES)
    result = {}
    for m, modality in enumerate(MODALITIES):
        probs = probs3[:, m, :]
        pred = probs.argmax(axis=1)
        result[modality] = {'probs': probs.astype(np.float32), 'pred_label_id': pred.astype(np.int64), 'pred_emotion': np.asarray([EMOTIONS[int(i)] for i in pred], dtype=object), 'confidence': probs.max(axis=1).astype(np.float32)}
    return result

def final_prediction_and_confidence(probs: np.ndarray) -> Dict[str, np.ndarray]:
    ensure_probs('final confidence input', probs)
    probs = np.asarray(probs, dtype=np.float32)
    pred = probs.argmax(axis=1)
    return {'pred_label_id': pred.astype(np.int64), 'pred_emotion': np.asarray([EMOTIONS[int(i)] for i in pred], dtype=object), 'confidence': probs.max(axis=1).astype(np.float32)}

def compute_degraded_mask(quality: np.ndarray, mask: np.ndarray, tau: float=ROUTER_TAU) -> np.ndarray:
    quality = np.asarray(quality, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if quality.shape != mask.shape:
        raise RuntimeError('quality/mask shape mismatch.')
    if quality.ndim != 2 or quality.shape[1] != NUM_MODALITIES:
        raise RuntimeError(f'Expected quality/mask [N,3], got {quality.shape}')
    any_unavailable = (mask < 0.5).any(axis=1)
    low_quality = quality.min(axis=1) < float(tau)
    return any_unavailable | low_quality

def route_predictions(f4_probs: np.ndarray, af4b_output: Dict[str, np.ndarray], quality: np.ndarray, mask: np.ndarray, tau: float=ROUTER_TAU) -> Dict[str, np.ndarray]:
    ensure_probs('router F4', f4_probs)
    ensure_probs('router AF4-B', af4b_output['final_probs'])
    degraded = compute_degraded_mask(quality, mask, tau)
    final_probs = np.asarray(f4_probs, dtype=np.float32).copy()
    final_probs[degraded] = af4b_output['final_probs'][degraded]
    no_decision = np.zeros(len(final_probs), dtype=np.float32)
    no_decision[degraded] = af4b_output['no_decision'][degraded]
    ensure_probs('router final', final_probs)
    return {'final_probs': final_probs, 'use_af4b': degraded.astype(np.float32), 'use_f4': (~degraded).astype(np.float32), 'no_decision': no_decision}


# ---------------------------------------------------------------------------
# Pinned asset loading and compatibility numerical API
# ---------------------------------------------------------------------------

def resolve_assets(assets_dir: str | Path | None = None,
                   f4_checkpoint: str | Path | None = None,
                   af4b_checkpoint: str | Path | None = None) -> tuple[Path,Path]:
    require((f4_checkpoint is None) == (af4b_checkpoint is None),
            "Supply BOTH checkpoint paths, or --assets-dir", AssetContractError)
    if f4_checkpoint is not None:
        require(assets_dir is None, "Use explicit checkpoint pair OR assets_dir", AssetContractError)
        pair = (Path(f4_checkpoint), Path(af4b_checkpoint))
    else:
        if assets_dir is not None:
            directory = Path(assets_dir).expanduser().resolve()
        else:
            roots = list(dict.fromkeys([Path(__file__).resolve().parent, Path.cwd().resolve()]))
            found = [r for r in roots if (r/F4_NAME).is_file() and (r/AF4B_NAME).is_file()]
            require(len(found) == 1, "No unambiguous flat asset directory; use --assets-dir "
                    "or both explicit paths. No recursive/latest-run discovery.", AssetContractError)
            directory = found[0]
        pair = (directory/F4_NAME, directory/AF4B_NAME)
    pair = tuple(p.expanduser().resolve() for p in pair)
    require(pair[0] != pair[1], "F4 and AF4-B cannot be the same file", AssetContractError)
    require(all(p.is_file() for p in pair), "Missing frozen checkpoint: "+str(pair), AssetContractError)
    return pair


def restricted_checkpoint(path: Path, role: str) -> tuple[dict,dict]:
    require(0 < path.stat().st_size <= 64*1024*1024, f"{role}: invalid checkpoint size", AssetContractError)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    require(sha == REVIEWED_HASHES[role],
            f"{role}: checkpoint SHA256 differs from the reviewed frozen pair. "
            "No replacement/automatic model selection is allowed.", AssetContractError)
    try:
        obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AssetContractError(f"{role}: restricted checkpoint load failed; no unsafe fallback: {exc}") from exc
    require(isinstance(obj, dict) and isinstance(obj.get("model_state_dict"), Mapping),
            f"{role}: missing model_state_dict", AssetContractError)
    for key, val in obj["model_state_dict"].items():
        require(isinstance(key,str) and isinstance(val,torch.Tensor) and
                val.layout == torch.strided and val.dtype == torch.float32 and torch.isfinite(val).all().item(),
                f"{role}: invalid/nonfinite/non-FP32 state {key!r}", AssetContractError)
    return obj, {"file": str(path), "sha256":sha, "bytes":len(data),
                 "stage":obj.get("stage"), "candidate":obj.get("candidate"),
                 "epoch":obj.get("epoch"), "seed":obj.get("seed")}


def state_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_pair(f4_path: Path, af4b_path: Path, device: torch.device):
    fc, fi = restricted_checkpoint(f4_path,"f4")
    bc, bi = restricted_checkpoint(af4b_path,"af4b")
    require(fc.get("stage") == "F4" and fc.get("candidate") == "mlp_stacker",
            "F4 identity mismatch", AssetContractError)
    require(bc.get("stage") == "AF4-B" and bc.get("candidate") == EXPECTED_AF4B_CANDIDATE,
            "AF4-B identity mismatch", AssetContractError)
    require(fc.get("mlp_hidden") == bc.get("f4_hidden") == 32 and bc.get("quality_beta") == 2.,
            "Frozen hidden/beta mismatch", AssetContractError)
    require(set(fc["model_state_dict"]) == set(F4_MAPPING.values()),
            "Unexpected F4 tensor set", AssetContractError)
    mapped = {k: fc["model_state_dict"][v] for k,v in F4_MAPPING.items()}
    for k,v in mapped.items():
        b = bc["model_state_dict"].get("f4."+k)
        require(isinstance(b,torch.Tensor) and v.shape == b.shape and torch.equal(v,b),
                "Independent F4 differs from AF4-B's embedded F4: "+k, AssetContractError)
    # Model constructors initialize temporary weights; restore caller RNG afterwards.
    with torch.random.fork_rng(devices=[]):
        f4 = FrozenF4Core(hidden=32)
        inner = FrozenF4Core(hidden=32)
        f4.load_state_dict(mapped, strict=True)
        inner.load_state_dict(mapped, strict=True)
        af4b = AdaptiveF4B(inner, hidden=32, quality_beta=2.,
                           residual_bias_init=float(bc["residual_bias_init"]))
        expected = af4b.state_dict()
        require(set(expected) == set(bc["model_state_dict"]), "AF4-B tensor set mismatch", AssetContractError)
        require(all(expected[k].shape == v.shape for k,v in bc["model_state_dict"].items()),
                "AF4-B tensor shape mismatch", AssetContractError)
        af4b.load_state_dict(bc["model_state_dict"], strict=True)
    for model in (f4,af4b):
        model.to(device=device,dtype=torch.float32).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    identity = {"f4":fi, "af4b":bi,
                "embedded_f4_exact_match":True,
                "embedded_f4_tensors_compared":len(mapped),
                "f4_parameters":sum(p.numel() for p in f4.parameters()),
                "af4b_parameters_including_f4":sum(p.numel() for p in af4b.parameters()),
                "source_fusion_sha256":SOURCE_FUSION_SHA256,
                "runtime_script_sha256":sha256_file(Path(__file__)),
                "torch":str(torch.__version__), "numpy":np.__version__,
                "device":str(device),
                "checkpoint_training_input":fc.get("input"),
                "weights_only":True, "training_performed":False,
                "benchmark_arrays_read":False}
    return f4,af4b,fc,bc,identity


def validate_manifest(manifest: Mapping[str,Any], identity: Mapping[str,Any]) -> None:
    require(manifest.get("schema") == MANIFEST_SCHEMA, "Wrong deployment manifest schema", AssetContractError)
    require(manifest.get("router_tau") == ROUTER_TAU and manifest.get("class_order") == EMOTIONS and
            manifest.get("modality_order") == MODALITIES, "Manifest frozen protocol mismatch", AssetContractError)
    require(manifest.get("runtime_script_sha256") == identity["runtime_script_sha256"],
            "Runtime script changed since manifest export", AssetContractError)
    for m in ("f4","af4b"):
        require(manifest.get("checkpoints",{}).get(m,{}).get("sha256") == identity[m]["sha256"],
                m+" manifest checkpoint mismatch", AssetContractError)
    require(manifest.get("source_fusion_sha256") == SOURCE_FUSION_SHA256,
            "Manifest source lineage mismatch", AssetContractError)


class FinalAF4CSystem:
    """Same numeric API as original, with strict validation and missing clearing.

    predict_batch/predict_one preserve the raw result layout for existing EEG
    and Audio bridges. They have NO temporal metadata and must not be mistaken
    for synchronization. Use predict_sample/predict_packets in the main runtime.
    """
    def __init__(self, f4_checkpoint: str | Path | None = None,
                 af4b_checkpoint: str | Path | None = None,
                 device: str | torch.device = "cuda", batch_size: int = 1024,
                 tau: float = ROUTER_TAU, *, assets_dir: str | Path | None = None,
                 manifest: str | Path | None = None, decision_window_seconds: float = 5.,
                 stale_after_sec: float = 1., alignment_tolerance_sec: float = .10,
                 future_tolerance_sec: float = .02, clock: Callable[[],float] = time.monotonic):
        require(finite_number(tau,"tau") == ROUTER_TAU, "Frozen tau is 0.80")
        require(type(batch_size) is int and 1 <= batch_size <= 10000, "batch_size must be 1..10000")
        self.tau = ROUTER_TAU
        self.device = choose_device(device); self.batch_size = batch_size
        self.decision_window_seconds = finite_number(decision_window_seconds,"decision_window_seconds")
        require(self.decision_window_seconds in (5.,20.), "Explicit supported window: 5 or 20 seconds")
        self.stale_after_sec = finite_number(stale_after_sec,"stale_after_sec")
        self.alignment_tolerance_sec = finite_number(alignment_tolerance_sec,"alignment_tolerance_sec")
        self.future_tolerance_sec = finite_number(future_tolerance_sec,"future_tolerance_sec")
        require(0 < self.stale_after_sec <= 30 and 0 <= self.alignment_tolerance_sec <= .5 and
                0 <= self.future_tolerance_sec <= .1, "Invalid temporal runtime policy")
        require(callable(clock), "clock must be callable")
        self.clock = clock; self._lock = threading.RLock()
        self.f4_checkpoint,self.af4b_checkpoint = resolve_assets(assets_dir,f4_checkpoint,af4b_checkpoint)
        self.f4,self.af4b,self.f4_ckpt,self.af4b_ckpt,self.identity = load_pair(
            self.f4_checkpoint,self.af4b_checkpoint,self.device)
        self._initial_model_hashes = (state_digest(self.f4),state_digest(self.af4b))
        candidate = Path(manifest).expanduser().resolve() if manifest is not None else None
        if candidate is None and self.f4_checkpoint.parent == self.af4b_checkpoint.parent:
            sidecar = self.f4_checkpoint.parent / MANIFEST_NAME
            if sidecar.is_file():
                candidate = sidecar
        self.manifest_path = candidate
        if candidate is not None:
            validate_manifest(read_json(candidate),self.identity)
        self.identity["manifest_verified"] = candidate is not None
        self.identity["manifest_path"] = str(candidate) if candidate else None

    def check_assets(self) -> dict:
        require((state_digest(self.f4),state_digest(self.af4b)) == self._initial_model_hashes,
                "Frozen model state was modified in memory", AssetContractError)
        require(all(not p.requires_grad for m in (self.f4,self.af4b) for p in m.parameters()),
                "Model no longer frozen", AssetContractError)
        require(not self.f4.training and not self.af4b.training, "Model not in eval mode", AssetContractError)
        return {"status":"PASS","identity":copy.deepcopy(self.identity),
                "router_tau":self.tau,"class_order":EMOTIONS.copy(),"modality_order":MODALITIES.copy(),
                "runtime_policy":self.runtime_policy(),"real_checkpoint_loaded":True,
                "forward_performed":False,"accuracy_evaluated":False}

    def runtime_policy(self) -> dict:
        return {"decision_window_seconds":self.decision_window_seconds,
                "stale_after_sec":self.stale_after_sec,
                "alignment_tolerance_sec":self.alignment_tolerance_sec,
                "future_tolerance_sec":self.future_tolerance_sec,
                "timing_limits_are_engineering_settings_not_learned":True,
                "quality_aggregation":"external, not implemented",
                "warning":INFERENCE_WARNING}

    def export_manifest(self, path: str | Path) -> dict:
        self.check_assets()
        obj = {"schema":MANIFEST_SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
               "version":VERSION,"source_fusion_sha256":SOURCE_FUSION_SHA256,
               "runtime_script_sha256":self.identity["runtime_script_sha256"],
               "router_tau":ROUTER_TAU,"class_order":EMOTIONS.copy(),"modality_order":MODALITIES.copy(),
               "checkpoints":{m:{"filename":Path(self.identity[m]["file"]).name,
                                 "sha256":self.identity[m]["sha256"],
                                 "bytes":self.identity[m]["bytes"]} for m in ("f4","af4b")},
               "runtime_policy_at_export":self.runtime_policy(),
               "missing_policy":"replace probabilities with uniform, q=0, mask=0 BEFORE both cores",
               "checkpoint_pair_checked":True,"pretrained_models_modified":False,
               "hardware_validation_performed":False,"accuracy_evaluated":False,
               "dependencies_at_export":{"torch":str(torch.__version__),"numpy":np.__version__},
               "dependency_versions_are_test_environment_not_upgrade_recommendations":True}
        write_json(path,obj)
        return obj

    def predict_batch(self, x15: Any, quality: Any, mask: Any) -> Dict[str,Any]:
        x,q,m = canonical_arrays(x15,quality,mask)
        with self._lock, torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            try:
                f4p = infer_f4_core(self.f4,x,self.device,self.batch_size)
                ao = infer_af4b(self.af4b,x,q,m,self.device,self.batch_size)
                routed = route_predictions(f4p,ao,q,m,self.tau)
            except Exception as exc:
                raise FusionInferenceError(f"Frozen forward failed, no emotion output: {exc}") from exc
        no = m.sum(axis=1) == 0
        require(np.array_equal(routed["no_decision"] > .5, no), "All-missing route violated", FusionInferenceError)
        if no.any():
            require(np.allclose(routed["final_probs"][no],.2,atol=1e-7,rtol=0),
                    "All-missing probabilities must be uniform", FusionInferenceError)
        # Validate learned branch health on every forward, not just preflight.
        for k in ("alpha","effective_weights","class_reliability","gamma"):
            require(np.isfinite(ao[k]).all(), "Nonfinite adaptive output "+k, FusionInferenceError)
        nonempty = ~no
        if nonempty.any():
            require(np.max(np.abs(ao["alpha"][nonempty]*(1-m[nonempty]))) <= 1e-7,
                    "Missing modality retained adaptive alpha", FusionInferenceError)
            require(np.max(np.abs(ao["effective_weights"][nonempty]*(1-m[nonempty,:,None]))) <= 1e-7,
                    "Missing modality retained class weight", FusionInferenceError)
        route = np.where(routed["use_af4b"] > .5, "AF4-B", "F4")
        result = {"x15":x,"quality":q,"mask":m,
                  "modality":modality_predictions_and_confidence(x),
                  "f4_probs":f4p,"f4_info":final_prediction_and_confidence(f4p),
                  "af4b_output":ao,"af4b_info":final_prediction_and_confidence(ao["final_probs"]),
                  "router_output":routed,"final_info":final_prediction_and_confidence(routed["final_probs"]),
                  "route":route,"system_state":np.where(no,"NO_DECISION",np.where(route=="F4","HEALTHY","DEGRADED")),
                  "input_validation":{"strict_numeric":True,"missing_canonicalized":True,
                                      "temporal_metadata_checked":False},
                  "runtime_version":VERSION}
        return result

    def predict_one(self, eeg_probs: Any, audio_probs: Any, video_probs: Any,
                    q_eeg: float, q_audio: float, q_video: float,
                    eeg_available: bool, audio_available: bool, video_available: bool) -> dict:
        # Deliberately no defaults for quality or availability.
        s = dict(eeg_probs=eeg_probs,audio_probs=audio_probs,video_probs=video_probs,
                 quality=dict(eeg=q_eeg,audio=q_audio,video=q_video),
                 available=dict(eeg=eeg_available,audio=audio_available,video=video_available))
        x,q,m = sample_arrays(s)
        return result_to_serializable_sample(self.predict_batch(x,q,m),0)

    def predict_sample(self, sample: Mapping[str,Any], *, expected_clock_id: str | None = None) -> dict:
        """Metadata-aware entry. Offline samples need explicit identity/geometry.
        Live input must include timing for EACH declared available modality.
        No hidden promotion of legacy dictionaries to time-validated packets.
        """
        require(isinstance(sample,Mapping), "Sample must be a mapping")
        s = copy.deepcopy(dict(sample))
        require(s.get("schema") == INPUT_SCHEMA, "Explicit runtime schema required")
        session = text_id(s.get("session_id"),"session_id")
        window = text_id(s.get("window_id"),"window_id")
        require(s.get("class_order") == EMOTIONS and s.get("modality_order") == MODALITIES,
                "Class and modality order must match the frozen model")
        seconds = finite_number(s.get("window_seconds"),"window_seconds")
        require(seconds == self.decision_window_seconds, "Sample/runtime window length mismatch", WindowContractError)
        live = strict_bool(s.get("live"),"live")
        for field in ("reasons","producer_evidence_ids"):
            if field in s:
                require(isinstance(s[field],Mapping) and set(s[field]) <= set(MODALITIES),
                        field+": expected a modality-keyed dictionary")
                for name,val in s[field].items():
                    text_id(val,field+"."+name)
        # Validate at entry before temporal filtering (malformed q is an error).
        x,q,m = sample_arrays(s)
        timed = validate_timing(s,self,expected_clock_id) if live or "timing" in s else None
        reasons = {name: [] for name in MODALITIES}
        for j,name in enumerate(MODALITIES):
            if not m[0,j]:
                reasons[name].append(str(s.get("reasons",{}).get(name,"DECLARED_UNAVAILABLE")))
        starts = finite_number(self.clock(),"clock") if live else None
        retries = 0
        while True:
            if live:
                now = finite_number(self.clock(),"clock")
                for j,name in enumerate(MODALITIES):
                    if m[0,j] and now - timed["modalities"][name]["newest_sample_monotonic"] > self.stale_after_sec:
                        m[0,j]=0; q[0,j]=0; x[0,j*5:(j+1)*5]=.2
                        reasons[name].append("STALE_CAPTURE_BEFORE_FUSION")
            raw = self.predict_batch(x,q,m)
            ended = finite_number(self.clock(),"clock") if live else None
            newly_stale = []
            if live:
                require(ended >= starts, "Clock moved backwards", WindowContractError)
                newly_stale = [j for j,name in enumerate(MODALITIES) if m[0,j] and
                    ended - timed["modalities"][name]["newest_sample_monotonic"] > self.stale_after_sec]
            if not newly_stale:
                break
            # At most 3 branches can expire. Recompute after each removal; never
            # report an output produced with evidence already expired at return.
            retries += 1
            require(retries <= 3, "Excess temporal recomputations", FusionInferenceError)
            for j in newly_stale:
                name=MODALITIES[j]; m[0,j]=0; q[0,j]=0; x[0,j*5:(j+1)*5]=.2
                reasons[name].append("STALE_CAPTURE_DURING_FUSION")
        out = result_to_serializable_sample(raw,0)
        out.update(session_id=session,window_id=window,window_seconds=seconds,live=live,
                   input_schema=INPUT_SCHEMA,asset_identity=copy.deepcopy(self.identity),
                   runtime_policy=self.runtime_policy(),warning=INFERENCE_WARNING,
                   producer_evidence_ids=copy.deepcopy(s.get("producer_evidence_ids",{})))
        out["input_validation"]["temporal_metadata_checked"] = timed is not None
        out["input_validation"]["freshness_checked"] = live
        out["input_validation"]["timing_evidence"] = ("PER_MODALITY_CAPTURE_METADATA" if timed else "OFFLINE_CALLER_DECLARATION_ONLY")
        out["temporal"] = {"recomputed_after_expiry":retries,"issued_monotonic":ended,
                           "clock_id":timed["clock_id"] if timed else None,
                           "valid_until_monotonic":min(
                               timed["modalities"][name]["newest_sample_monotonic"]+self.stale_after_sec
                               for j,name in enumerate(MODALITIES) if m[0,j]) if live and m.any() else None}
        for name in MODALITIES:
            out["modalities"][name]["unavailable_reasons"] = reasons[name]
        return out

    def predict_packets(self, packets: Mapping[str,Mapping[str,Any]], *,
                        session_id: str, window_id: str, window_end_monotonic: float | None = None,
                        clock_id: str | None = None, live: bool = True) -> dict:
        s = packets_to_sample(packets,session_id=session_id,window_id=window_id,
                              window_seconds=self.decision_window_seconds,live=live,
                              window_end_monotonic=window_end_monotonic,clock_id=clock_id)
        return self.predict_sample(s,expected_clock_id=clock_id)

    def preflight(self) -> dict:
        return run_preflight(self)


# Clear public name for the final runtime. Compatibility class remains available.
FusionAF4C = FinalAF4CSystem


def result_to_serializable_sample(result: Mapping[str,Any], index: int,
                                  *, include_diagnostics: bool = False) -> dict:
    no = bool(result["router_output"]["no_decision"][index] > .5)
    route = str(result["route"][index])
    active = route == "AF4-B" and not no
    probs = np.asarray(result["router_output"]["final_probs"][index],dtype=float)
    ensure_probs("serialized final",probs[None,:])
    a = result["af4b_output"]
    alpha = np.asarray(a["alpha"][index],dtype=float)
    effective = np.asarray(a["effective_weights"][index],dtype=float)
    weights = dict(zip(MODALITIES,alpha.tolist())) if active else None
    eff = {m:dict(zip(EMOTIONS,effective[j].tolist())) for j,m in enumerate(MODALITIES)} if active else None
    final = {"emotion":"NO_DECISION" if no else EMOTIONS[int(probs.argmax())],
             "confidence":None if no else float(probs.max()), "probabilities":probs.tolist(),
             "is_evidence":not no}
    sample = {"system_state":str(result["system_state"][index]),"route":route,
              "router_tau":ROUTER_TAU,"no_decision":no,"final":final,"modalities":{},
              "class_order":EMOTIONS.copy(),"modality_order":MODALITIES.copy(),
              "adaptive_branch_active":active,"modality_weights":weights,
              "class_effective_weights":eff,"weight_semantics":WEIGHT_WARNING,
              "q_is_fusion_weight":False, "confidence_is_calibrated_correctness":False,
              "input_validation":copy.deepcopy(result.get("input_validation",{})),
              "af4b":{"active":active,"alpha":weights,
                      "effective_weights":eff,"gamma":float(a["gamma"][index]) if active else None},
              "f4":{"active":route=="F4" and not no},
              "fusion_input":{"quality":{},"available":{}},
              "runtime_version":VERSION}
    for j,name in enumerate(MODALITIES):
        avail = bool(result["mask"][index,j])
        p = np.asarray(result["x15"][index,j*5:(j+1)*5],dtype=float)
        q = float(result["quality"][index,j])
        sample["modalities"][name] = {"available":avail,"quality":q,
              "emotion":EMOTIONS[int(p.argmax())] if avail else "UNAVAILABLE",
              "confidence":float(p.max()) if avail else None,
              "probabilities":p.tolist(),"probabilities_are_placeholder":not avail,
              "adaptive_weight":float(alpha[j]) if active else None}
        sample[name+"_weight"] = float(alpha[j]) if active else None
        sample["q_"+name] = q; sample[name+"_available"] = avail
        sample["fusion_input"][name+"_probs"] = p.tolist()
        sample["fusion_input"]["quality"][name] = q
        sample["fusion_input"]["available"][name] = avail
    if include_diagnostics:
        sample["diagnostics"] = {"not_active_contribution":True,
            "f4_probabilities":np.asarray(result["f4_probs"][index],float).tolist(),
            "af4b_probabilities":np.asarray(a["final_probs"][index],float).tolist(),
            "raw_af4b_alpha_including_all_missing_fallback":alpha.tolist(),
            "raw_af4b_gamma":float(a["gamma"][index])}
    return sample


# ---------------------------------------------------------------------------
# Same-window packet contract and caller-driven, bounded coordinator
# ---------------------------------------------------------------------------

def make_modality_packet(modality: str, *, session_id: str, window_id: str,
                         window_seconds: float, class_order: Sequence[str],
                         probabilities: Any, quality: float, available: bool,
                         timing: Mapping[str,Any] | None = None,
                         reason: str | None = None, evidence_id: str | None = None) -> dict:
    """Assemble an ALREADY paired classifier/quality result for one source span.
    This cannot independently prove that q/probabilities used identical raw data;
    the producing head must preserve its source identity instead of relabelling.
    timing uses physical capture boundaries and newest captured sample, not return
    time. For unavailable packets probabilities may be None/obsolete and are
    discarded immediately.
    """
    require(modality in MODALITIES,"Unknown modality")
    session = text_id(session_id,"session_id"); wid = text_id(window_id,"window_id")
    sec = finite_number(window_seconds,"window_seconds")
    require(sec in (5.,20.) and list(class_order) == EMOTIONS, "Invalid packet classes/window")
    av = strict_bool(available,"available")
    q = finite_number(quality,"quality"); require(0 <= q <= 1,"Packet quality outside [0,1]")
    p = validated_probability(probabilities,"probabilities").tolist() if av else [.2]*5
    if reason is not None:
        reason = text_id(reason,"reason")
    packet = {"schema":PACKET_SCHEMA,"modality":modality,"session_id":session,
              "window_id":wid,"window_seconds":sec,"class_order":EMOTIONS.copy(),
              "probabilities":p,"quality":q if av else 0.,"available":av,
              "reason":reason or ("CURRENT_EVIDENCE" if av else "DECLARED_UNAVAILABLE")}
    if timing is not None:
        require(isinstance(timing,Mapping),"Packet timing must be an object")
        packet["timing"] = copy.deepcopy(dict(timing))
    if evidence_id is not None:
        packet["evidence_id"] = text_id(evidence_id,"evidence_id")
    return packet


def packets_to_sample(packets: Mapping[str,Mapping[str,Any]], *, session_id: str,
                      window_id: str, window_seconds: float, live: bool,
                      window_end_monotonic: float | None = None,
                      clock_id: str | None = None) -> dict:
    require(isinstance(packets,Mapping) and set(packets) == set(MODALITIES),
            "Exactly three explicit packets required (use unavailable packets for missing heads)")
    session=text_id(session_id,"session_id"); wid=text_id(window_id,"window_id")
    sec=finite_number(window_seconds,"window_seconds")
    live=strict_bool(live,"live")
    out={"schema":INPUT_SCHEMA,"session_id":session,"window_id":wid,
         "window_seconds":sec,"class_order":EMOTIONS.copy(),"modality_order":MODALITIES.copy(),
         "live":live,"quality":{},"available":{},"reasons":{},"producer_evidence_ids":{}}
    tm={}
    for name in MODALITIES:
        p=packets[name]
        require(isinstance(p,Mapping) and p.get("schema")==PACKET_SCHEMA and p.get("modality")==name,
                "Invalid packet schema/modality")
        require(p.get("session_id")==session and p.get("window_id")==wid and
                p.get("window_seconds")==sec and p.get("class_order")==EMOTIONS,
                name+": wrong session/window/classes; do not reuse cached predictions",WindowContractError)
        a=strict_bool(p.get("available"),name+".available")
        out["available"][name]=a; out["quality"][name]=p.get("quality")
        out[name+"_probs"]=p.get("probabilities") if a else [.2]*5
        out["reasons"][name]=str(p.get("reason","DECLARED_UNAVAILABLE"))
        if "evidence_id" in p:
            out["producer_evidence_ids"][name]=text_id(p["evidence_id"],name+".evidence_id")
        if a and "timing" in p:
            require(isinstance(p["timing"],Mapping),"Invalid packet timing")
            tm[name]={**copy.deepcopy(dict(p["timing"])),
                      "window_id":p["window_id"],"window_seconds":p["window_seconds"],
                      "session_id":p["session_id"]}
    if live:
        out["timing"]={"clock_id":text_id(clock_id,"clock_id"),
                      "window_end_monotonic":finite_number(window_end_monotonic,"window_end_monotonic"),
                      "modalities":tm}
    elif tm:
        require(clock_id is not None and window_end_monotonic is not None,
                "Offline timed packets still need an explicit common clock/window end")
        out["timing"]={"clock_id":text_id(clock_id,"clock_id"),
                      "window_end_monotonic":finite_number(window_end_monotonic,"window_end_monotonic"),
                      "modalities":tm}
    sample_arrays(out)
    return out


def validate_timing(sample: Mapping[str,Any], system: FinalAF4CSystem,
                    expected_clock_id: str | None) -> dict:
    t=sample.get("timing")
    require(isinstance(t,Mapping),"Live input requires explicit per-modality capture timing",WindowContractError)
    cid=text_id(t.get("clock_id"),"timing.clock_id")
    if strict_bool(sample["live"],"live"):
        require(expected_clock_id is not None and cid==expected_clock_id,
                "Live caller must supply matching expected_clock_id; no clock-domain inference",
                WindowContractError)
    elif expected_clock_id is not None:
        require(cid==expected_clock_id,"Clock-domain mismatch",WindowContractError)
    end=finite_number(t.get("window_end_monotonic"),"decision window end")
    sec=system.decision_window_seconds; begin=end-sec
    mm=t.get("modalities")
    require(isinstance(mm,Mapping) and set(mm)<=set(MODALITIES),
            "Invalid per-modality timing object",WindowContractError)
    now=finite_number(system.clock(),"clock") if sample["live"] else None
    if now is not None:
        require(end <= now+system.future_tolerance_sec,"Decision window is in the future",WindowContractError)
    out={"clock_id":cid,"window_end_monotonic":end,"modalities":{}}
    tol=system.alignment_tolerance_sec
    for name in MODALITIES:
        if not strict_bool(sample["available"][name],name+".available"):
            continue
        a=mm.get(name)
        require(isinstance(a,Mapping),"Missing capture timing for available "+name,WindowContractError)
        require(a.get("clock_id")==cid and a.get("window_id")==sample["window_id"]
                and a.get("session_id")==sample["session_id"] and a.get("window_seconds")==sec,
                name+": clock/session/window mismatch",WindowContractError)
        start=finite_number(a.get("window_start_monotonic"),name+" window start")
        stop=finite_number(a.get("window_end_monotonic"),name+" window end")
        newest=finite_number(a.get("newest_sample_monotonic"),name+" latest capture")
        require(stop>start and abs((stop-start)-sec)<=tol and
                abs(start-begin)<=tol and abs(stop-end)<=tol,
                name+": physical window boundaries do not align",WindowContractError)
        require(start<=newest<=stop+system.future_tolerance_sec and stop-newest<=tol,
                name+": newest capture is inconsistent with declared window",WindowContractError)
        if now is not None:
            require(newest <= now+system.future_tolerance_sec,
                    name+": newest capture in the future",WindowContractError)
        out["modalities"][name]={"newest_sample_monotonic":newest,
                                 "window_start_monotonic":start,"window_end_monotonic":stop}
    return out


class FusionWindowCoordinator:
    """Bounded, caller-polled asynchronous-head collector, not a background job.

    begin_window() uses a strictly increasing window end (in one session).
    submit() receives each head at most once. finalize() yields None while
    waiting; on deadline absent heads become explicitly unavailable. No
    last-known probability reuse. A finalized slot is consumed even on forward
    error (no automatic repeated action). Create a NEW coordinator/session
    when acquisition restarts. Poll from the main loop, not the device callback.

    This does not acquire signals, execute heads, start threads or move a robot.
    At-most-once is in-process and per coordinator; persistent exactly-once
    actuator control must be supplied by the action layer.
    """
    def __init__(self, system: FinalAF4CSystem, *, session_id: str, clock_id: str,
                 deadline_sec: float = .75, max_pending: int = 32):
        self.system=system; self.session_id=text_id(session_id,"session_id")
        self.clock_id=text_id(clock_id,"clock_id")
        self.deadline_sec=finite_number(deadline_sec,"deadline_sec")
        require(0<=self.deadline_sec<=system.stale_after_sec,
                "deadline_sec must not exceed stale_after_sec")
        require(type(max_pending) is int and 1<=max_pending<=1024,"max_pending must be 1..1024")
        self.max_pending=max_pending; self._lock=threading.RLock()
        self._pending: OrderedDict[str,dict]=OrderedDict()
        self._last_end=-math.inf
        self._last_emitted_end=-math.inf
        self._closed: OrderedDict[str,str]=OrderedDict()
        self._used_ids:set[str]=set()

    def begin_window(self, window_id: str, *, window_end_monotonic: float) -> dict:
        wid=text_id(window_id,"window_id")
        end=finite_number(window_end_monotonic,"window_end_monotonic")
        with self._lock:
            require(wid not in self._used_ids,"Window ID reused in this session",DuplicateWindowError)
            require(end>self._last_end,"Window ends must increase; no replay",WindowContractError)
            require(end<=self.system.clock()+self.system.future_tolerance_sec,
                    "Cannot begin future capture window",WindowContractError)
            require(len(self._pending)<self.max_pending,"Pending queue full; finalize/expire old windows")
            # Sequence identities retained for one session only. Caps bound memory;
            # reset with a NEW session rather than silently allowing ID reuse.
            require(len(self._used_ids)<100000,"Session identity limit reached; start a new coordinator/session")
            self._used_ids.add(wid); self._last_end=end
            item={"window_id":wid,"window_end_monotonic":end,"deadline":end+self.deadline_sec,"packets":{}}
            self._pending[wid]=item
            return {"session_id":self.session_id,"window_id":wid,
                    "window_seconds":self.system.decision_window_seconds,
                    "clock_id":self.clock_id,"deadline_monotonic":item["deadline"]}

    def submit(self, packet: Mapping[str,Any]) -> None:
        p=copy.deepcopy(dict(packet))
        require(p.get("schema")==PACKET_SCHEMA and p.get("session_id")==self.session_id,
                "Packet from wrong schema/session",WindowContractError)
        wid=text_id(p.get("window_id"),"window_id"); name=p.get("modality")
        require(name in MODALITIES,"Unknown modality")
        require(p.get("window_seconds")==self.system.decision_window_seconds and
                p.get("class_order")==EMOTIONS,"Packet geometry/order mismatch",WindowContractError)
        with self._lock:
            require(wid in self._pending,"No open window; late/replayed result refused",WindowContractError)
            row=self._pending[wid]
            require(name not in row["packets"],"Duplicate modality in one window",DuplicateWindowError)
            require(self.system.clock()<=row["deadline"],"Packet arrived after window deadline",WindowContractError)
            # Numerical contracts checked immediately, even on a partial collection.
            trial={m:make_modality_packet(m,session_id=self.session_id,window_id=wid,
                   window_seconds=self.system.decision_window_seconds,class_order=EMOTIONS,
                   probabilities=None,quality=0.,available=False) for m in MODALITIES}
            trial[name]=p
            s=packets_to_sample(trial,session_id=self.session_id,window_id=wid,
                    window_seconds=self.system.decision_window_seconds,live=True,
                    window_end_monotonic=row["window_end_monotonic"],clock_id=self.clock_id)
            validate_timing(s,self.system,self.clock_id)
            row["packets"][name]=p

    def finalize(self, window_id: str) -> dict | None:
        wid=text_id(window_id,"window_id")
        with self._lock:
            require(wid in self._pending,"No open window; never finalize twice",DuplicateWindowError)
            item=self._pending[wid]
            if len(item["packets"])<3 and self.system.clock()<item["deadline"]:
                return None
            self._pending.pop(wid)
            self._closed[wid]="INFERENCE_STARTED"
            while len(self._closed)>256:
                self._closed.popitem(last=False)
        absent=[m for m in MODALITIES if m not in item["packets"]]
        packets=item["packets"]
        for m in absent:
            packets[m]=make_modality_packet(m,session_id=self.session_id,window_id=wid,
                    window_seconds=self.system.decision_window_seconds,class_order=EMOTIONS,
                    probabilities=None,quality=0.,available=False,reason="HEAD_DEADLINE_MISSED")
        try:
            result=self.system.predict_packets(packets,session_id=self.session_id,window_id=wid,
                      window_end_monotonic=item["window_end_monotonic"],clock_id=self.clock_id,live=True)
        except Exception:
            with self._lock:
                self._closed[wid]="FAILED_NO_AUTOMATIC_RETRY"
            raise
        with self._lock:
            self._closed[wid]="FINALIZED"
        with self._lock:
            publishable = item["window_end_monotonic"] > self._last_emitted_end
            if publishable:
                self._last_emitted_end = item["window_end_monotonic"]
        # A late older calculation must not replace a newer state in the UI or
        # action policy. This is an explicit publication flag, not model retuning.
        result["publishable"] = publishable
        result["publication_reason"] = "LATEST_COMPLETED_WINDOW" if publishable else "SUPERSEDED_BY_NEWER_WINDOW"
        result["coordinator"]={"finalized_once":True,"session_id":self.session_id,
            "deadline_monotonic":item["deadline"],"heads_missing_at_deadline":absent,
            "last_known_predictions_reused":False}
        return result

    def poll(self) -> list[dict]:
        # Explicit foreground operation. Call again from the application's loop.
        with self._lock:
            ready=[wid for wid,x in self._pending.items() if len(x["packets"])==3 or self.system.clock()>=x["deadline"]]
        return [self.finalize(wid) for wid in ready]

    def invalidate_pending(self, reason: str="CAPTURE_SESSION_RESET") -> list[str]:
        reason=text_id(reason,"reason")
        with self._lock:
            names=list(self._pending)
            self._pending.clear()
            for wid in names:
                self._closed[wid]=reason
            while len(self._closed)>256:
                self._closed.popitem(last=False)
        return names


# ---------------------------------------------------------------------------
# Local checks and prediction-only CLI
# ---------------------------------------------------------------------------

def offline_example() -> dict:
    return {"schema":INPUT_SCHEMA,"session_id":"OFFLINE_EXAMPLE_NOT_SENSOR_DATA",
            "window_id":"example_001","window_seconds":5.0,"live":False,
            "class_order":EMOTIONS.copy(),"modality_order":MODALITIES.copy(),
            "eeg_probs":[.1,.2,.3,.2,.2], "audio_probs":[.1,.1,.6,.1,.1],
            "video_probs":[.1,.1,.2,.5,.1],
            "quality":{"eeg":.92,"audio":.45,"video":.88},
            "available":{"eeg":True,"audio":True,"video":True}}


def run_self_test() -> dict:
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        require(condition,"Self-test failed: "+name)
    def reject(name,fn):
        try:
            fn()
        except (InputContractError,AssetContractError):
            checks[name]=True; return
        raise AssertionError("Invalid input was accepted: "+name)
    s=offline_example()
    x,q,m=sample_arrays(s)
    check("valid_numeric_shape",x.shape==(1,15) and q.shape==m.shape==(1,3))
    check("input_not_modified",s==offline_example())
    reject("nonbinary_mask_rejected",lambda:canonical_arrays(x,q,[[1,.6,1]]))
    reject("nan_quality_rejected",lambda:canonical_arrays(x,[[1,np.nan,1]],m))
    reject("infinite_quality_rejected",lambda:canonical_arrays(x,[[1,np.inf,1]],m))
    reject("out_of_range_quality_rejected",lambda:canonical_arrays(x,[[1,2,1]],m))
    reject("missing_quality_not_defaulted",lambda:sample_arrays({**s,"quality":{"eeg":1,"audio":1}}))
    reject("missing_mask_not_defaulted",lambda:sample_arrays({**s,"available":{}}))
    reject("logits_not_probabilities",lambda:validated_probability([1,2,3,4,5],"p"))
    reject("nested_probs_not_flattened",lambda:validated_probability([[.2]*5],"p"))
    reject("negative_probability_rejected",lambda:validated_probability([-.1,.2,.3,.3,.3],"p"))
    reject("string_bool_rejected",lambda:strict_bool("false","available"))
    for j,name in enumerate(MODALITIES):
        bad=x.copy(); bad[0,j*5:(j+1)*5]=np.nan
        mm=m.copy(); mm[0,j]=0
        clean,qq,_=canonical_arrays(bad,q,mm)
        check("missing_"+name+"_clears_nan",np.all(clean[0,j*5:(j+1)*5]==np.float32(.2)) and qq[0,j]==0)
    low=q.copy(); low[0,0]=.4
    check("low_quality_not_hard_missing",canonical_arrays(x,low,m)[2][0,0]==1)
    rejected = {**s,"audio_probs":None,"available":{**s["available"],"audio":False}}
    check("missing_none_supported",sample_arrays(rejected)[0].shape==(1,15))
    reject("empty_batch_rejected",lambda:canonical_arrays(np.empty((0,15)),np.empty((0,3)),np.empty((0,3))))
    allmissing=canonical_arrays(x,q,np.zeros((1,3)))
    check("all_missing_canonical",np.all(allmissing[0]==np.float32(.2)) and not allmissing[1].any())
    check("tau_equal_healthy",not compute_degraded_mask(np.array([[.8,.8,.8]],np.float32),m)[0])
    check("tau_below_degraded",compute_degraded_mask(np.array([[.799,.9,.9]],np.float32),m)[0])
    check("tau_zero_available_degraded",compute_degraded_mask(np.array([[0,1,1]],np.float32),m)[0])
    return {"status":"PASS","n_checks":len(checks),"checks":checks,
            "real_checkpoint_loaded":False,"sensor_models_used":False,"hardware_tested":False}


def run_preflight(system: FinalAF4CSystem) -> dict:
    checks={}; rng=np.random.default_rng(20260919)
    initial=(state_digest(system.f4),state_digest(system.af4b))
    def check(name,cond):
        checks[name]=bool(cond)
        require(cond,"Preflight failed: "+name,FusionInferenceError)
    base=rng.dirichlet(np.ones(5),size=(64,3)).astype(np.float32).reshape(64,15)
    original_input=base.copy()
    ones=np.ones((64,3),np.float32)
    h=system.predict_batch(base,ones,ones)
    check("healthy_is_exact_F4",np.array_equal(h["router_output"]["final_probs"],h["f4_probs"]))
    check("all_healthy_route_F4",np.all(h["route"]=="F4"))
    max_missing_difference={}
    for bits in itertools.product((0,1),repeat=3):
        mask=np.tile(np.array(bits,np.float32),(64,1))
        qs=np.full((64,3),.6,np.float32)
        r=system.predict_batch(base,qs,mask)
        tag="".join(map(str,bits))
        check("mask_"+tag+"_finite",np.isfinite(r["router_output"]["final_probs"]).all())
        check("mask_"+tag+"_no_decision",np.all((r["system_state"]=="NO_DECISION")==not_any(bits)))
        if any(bits):
            check("mask_"+tag+"_alpha_zero",np.max(np.abs(r["af4b_output"]["alpha"]*(1-mask)))<=1e-7)
            check("mask_"+tag+"_class_zero",np.max(np.abs(r["af4b_output"]["effective_weights"]*(1-mask[:,:,None])))<=1e-7)
        else:
            o=result_to_serializable_sample(r,0)
            check("all_missing_semantics",o["final"]["confidence"] is None and o["modality_weights"] is None
                  and all(z["emotion"]=="UNAVAILABLE" for z in o["modalities"].values()))
    for j,name in enumerate(MODALITIES):
        mask=ones.copy(); mask[:,j]=0
        old_a=base.copy(); old_b=base.copy()
        old_a[:,j*5:(j+1)*5]=np.tile(np.eye(5,dtype=np.float32)[0],(64,1))
        old_b[:,j*5:(j+1)*5]=rng.dirichlet(np.ones(5),size=64).astype(np.float32)
        r1=system.predict_batch(old_a,ones,mask)
        r2=system.predict_batch(old_b,ones,mask)
        delta=float(np.max(np.abs(r1["router_output"]["final_probs"]-r2["router_output"]["final_probs"])))
        max_missing_difference[name]=delta
        check(name+"_old_content_invariant",delta==0.)
        for val,expected in ((1.,False),(.8,False),(.7999,True),(0.,True)):
            qq=ones.copy(); qq[:,j]=val
            rr=system.predict_batch(base,qq,ones)
            check(name+"_threshold_"+str(val),np.all((rr["route"]=="AF4-B")==expected))
    after=system.predict_batch(base,ones,ones)
    check("restored_identical_output",np.array_equal(after["router_output"]["final_probs"],h["router_output"]["final_probs"]))
    check("source_inputs_unmodified",np.array_equal(base,original_input))
    check("models_unchanged",initial==(state_digest(system.f4),state_digest(system.af4b)))
    check("models_eval_and_frozen",not system.f4.training and not system.af4b.training and
          all(not p.requires_grad for model in (system.f4,system.af4b) for p in model.parameters()))
    out={"status":"PASS","n_checks":len(checks),"checks":checks,
         "missing_content_max_abs_change":max_missing_difference,
         "asset_identity":system.identity,"real_uploaded_checkpoint_forward":True,
         "synthetic_probabilities_only":True,"accuracy_evaluated":False,
         "raw_sensors_or_quality_models_used":False,"hardware_tested":False,
         "formal_test_data_read":False,"runtime_policy":system.runtime_policy()}
    return out


def not_any(bits: Sequence[int]) -> bool:
    return not any(bits)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Frozen AF4-C prediction-only deployment; no training/evaluate",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version",action="version",version=VERSION)
    action=p.add_mutually_exclusive_group()
    action.add_argument("--self-test",action="store_true",help="No weights required")
    action.add_argument("--check-assets",action="store_true")
    action.add_argument("--preflight",action="store_true",help="Loaded real weights + synthetic probability probes")
    action.add_argument("--input-json",help="Single sample or {'samples': [...]} using explicit metadata")
    action.add_argument("--write-example",metavar="PATH",help="Write clearly labelled OFFLINE synthetic example")
    p.add_argument("--assets-dir",help="Flat directory with the two checkpoints")
    p.add_argument("--f4-checkpoint"); p.add_argument("--af4b-checkpoint")
    p.add_argument("--manifest",help="Optional deployment identity manifest (sidecar also auto-checked)")
    p.add_argument("--export-manifest",metavar="PATH",help="Export only after successful asset/preflight checks")
    p.add_argument("--device",default="cuda",help="cpu, cuda, cuda:N, or explicit auto")
    p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--window-seconds",type=float,choices=[5.,20.],default=5.)
    p.add_argument("--stale-after-sec",type=float,default=1.)
    p.add_argument("--alignment-tolerance-sec",type=float,default=.1)
    p.add_argument("--future-tolerance-sec",type=float,default=.02)
    p.add_argument("--live-clock-id",help="Expected local monotonic clock identity for a live JSON input")
    p.add_argument("--output",help="UTF-8 JSON; never overwrites existing files")
    args=p.parse_args(argv)
    if not any((args.self_test,args.check_assets,args.preflight,args.input_json,args.write_example)):
        p.print_help(); args.help_only=True
    else:
        args.help_only=False
    require(not args.export_manifest or args.check_assets or args.preflight,
            "--export-manifest requires --check-assets or --preflight")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    # This configures this CLI's text streams only, not the parent/global locale.
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="backslashreplace")
    args=None
    try:
        args=parse_args(argv)
        if args.help_only:
            return 0
        if args.write_example:
            write_json(args.write_example,offline_example())
            print("OFFLINE EXAMPLE WRITTEN:",args.write_example)
            return 0
        if args.self_test:
            report=run_self_test()
        else:
            # Check destination collisions BEFORE expensive work.
            inputs=[p for p in (args.f4_checkpoint,args.af4b_checkpoint,args.input_json,args.manifest,__file__) if p]
            for output in (args.output,args.export_manifest):
                if output:
                    dest=Path(output).expanduser().resolve()
                    require(not dest.exists(),"Output exists; use a new filename: "+str(dest))
                    require(dest not in [Path(p).expanduser().resolve() for p in inputs],
                            "Output cannot overwrite an input or source file")
            system=FusionAF4C(assets_dir=args.assets_dir,f4_checkpoint=args.f4_checkpoint,
                   af4b_checkpoint=args.af4b_checkpoint,device=args.device,batch_size=args.batch_size,
                   manifest=args.manifest,decision_window_seconds=args.window_seconds,
                   stale_after_sec=args.stale_after_sec,alignment_tolerance_sec=args.alignment_tolerance_sec,
                   future_tolerance_sec=args.future_tolerance_sec)
            if args.check_assets:
                report=system.check_assets()
            elif args.preflight:
                report=system.preflight()
            else:
                obj=read_json(args.input_json)
                samples=obj.get("samples")
                if samples is None:
                    samples=[obj]; single=True
                else:
                    require(set(obj)=={"samples"} and isinstance(samples,list) and 0<len(samples)<=10000,
                            "Batch root must contain only a nonempty samples list (max 10000)")
                    single=False
                results=[system.predict_sample(s,expected_clock_id=args.live_clock_id) for s in samples]
                report=results[0] if single else {"samples":results,"count":len(results)}
            if args.export_manifest:
                system.export_manifest(args.export_manifest)
        if args.output:
            write_json(args.output,report)
        if args.check_assets or args.preflight or args.self_test:
            print("FUSION AF4-C STATUS :",report["status"])
            if "n_checks" in report: print("Checks passed       :",report["n_checks"])
            if not args.self_test:
                print("Real weights        : LOADED (UNCHANGED)")
                print("Router tau          : 0.80")
                print("Benchmark data      : NOT READ")
        elif args.input_json:
            for i,item in enumerate([report] if "samples" not in report else report["samples"],1):
                print(f"Sample {i}: {item['system_state']} | {item['route']} | "
                      f"{item['final']['emotion']} | confidence={item['final']['confidence']}")
        if args.output: print("REPORT:",str(Path(args.output).resolve()))
        if args.export_manifest: print("MANIFEST:",str(Path(args.export_manifest).resolve()))
        return 0
    except KeyboardInterrupt:
        print("INTERRUPTED: no successful prediction claimed",file=sys.stderr); return 130
    except Exception as exc:
        print(f"FUSION AF4-C ERROR: {type(exc).__name__}: {exc}",file=sys.stderr)
        print("No training, automatic installation or broader model replacement attempted.",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
