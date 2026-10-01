#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""EEG Emotion E4 V1 -- standalone, frozen 5-class deployment.

SOURCE: the supplied EEG头训练.py, actual E4-fast network implementation.
No training/validation loop, data discovery, automatic downloads or installs.
Required: best_multiscale_dilated_tcn_eegnet_validation_selected.pt and the
PAIRED train_channel_normalization.json. No external pretrained backbone.

Input: one pre-E4 500-Hz, 5-second, 30-electrode window, [30,2500].
Original pipeline: preserve source float32/float64 -> resample_poly(2,5)
-> float32 -> frozen TRAIN channel mean/std -> [1,1,30,1000] -> E4 logits
-> original float64 numpy softmax. FP32 eval; no AMP/extra filtering/ICA.
The actual network is temporal stems -> spatial + pool(4) -> per-scale TCN
-> separable + pool(8) -> flatten -> classifier (NOT pre-spatial TCN).

Scale policy:
  input_unit='training_native' is an explicit declaration that values already
  use the training MAT's numeric scale. It does NOT establish a physical unit.
  For new V/mV/uV inputs, BOTH input and training voltage scales must be known.
  Do not feed SI volts directly to a model trained on unconverted microvolt
  numbers. No amplitude-based unit inference is made. ADC offsets/reference
  changes must be handled by a separately verified acquisition adapter.

The emotion module does NOT estimate q_eeg or a fusion contribution weight.
Optional process_with_quality_array() shares the SAME original source window
with the supplied eeg_quality_v1_deployment.py and calls its safe AF4-C bridge.
Invalid/missing/stale input returns uniform PLACEHOLDERS, no emotion label and
no confidence. Runtime inference errors raise by default; explicit protective
'unavailable' policy is labelled ERROR, never a detected hardware fault.
All available outputs require current, matching source identity for fusion.

Inputs: numpy arrays, .npy/.npz/classic .mat; explicit one-window VAL replay.
No 200-Hz/standardized input route is silently substituted. The optional
20-second emotion-only API averages four window softmax outputs; it does NOT
select a 20-second quality aggregation or run the 5-second fusion bridge.

Existing repaired .venv-video; do not upgrade dependencies:
  python -X utf8 eeg_emotion_e4_deployment.py --self-test
  python -X utf8 eeg_emotion_e4_deployment.py --check-assets --e4-run PATH
  python -X utf8 eeg_emotion_e4_deployment.py --preflight --e4-run PATH
  python -X utf8 eeg_emotion_e4_deployment.py --assets-dir . --eeg window.npy \
    --assume-e4-channel-order --input-unit training_native --window-id w001

Only trusted local model assets/source modules should be loaded. E4 uses
weights_only=True with no silent unsafe deserialization fallback. The optional
fusion loader remains the user's existing implementation. A contract export
pins the provided asset pair; without a frozen report it does not independently
prove the pair's training provenance. Hardware/real-checkpoint acceptance must
be performed on the deployment machine; synthetic tests are not accuracy tests.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import re
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None
    nn = types.SimpleNamespace(Module=object)
    TORCH_IMPORT_ERROR = repr(exc)
else:
    TORCH_IMPORT_ERROR = None

VERSION = 'EEG-EMOTION-E4.1.0'
CONTRACT_SCHEMA = 'eav.eeg_emotion.e4.deployment.v1'
DEFAULT_CHECKPOINT = 'best_multiscale_dilated_tcn_eegnet_validation_selected.pt'
DEFAULT_NORMALIZATION = 'train_channel_normalization.json'
DEFAULT_FROZEN_REPORT = 'multiscale_dilated_tcn_eegnet_FINAL_FROZEN.json'
CONTRACT_NAME = 'eeg_emotion_e4_contract.json'
EMOTIONS = ['Neutral', 'Sadness', 'Anger', 'Happiness', 'Calmness']
CHANNELS = ['FP1','FP2','F7','F3','FZ','F4','F8','FC5','FC1','FC2','FC6','T7','C3','CZ','C4','T8',
            'CP5','CP1','CP2','CP6','P7','P3','PZ','P4','P8','PO9','O1','OZ','O2','PO10']
EAV_CHANNELS = CHANNELS
N_CHANNELS = 30
NUM_CLASSES = 5
SOURCE_FS = FS = 500
TARGET_FS = 200
WINDOW_SECONDS = 5.0
SOURCE_SAMPLES = 2500
TARGET_SAMPLES = 1000
DEFAULT_F1, DEFAULT_D, DEFAULT_F2 = 8, 2, 16
DEFAULT_TEMPORAL_KERNELS = (16, 32, 64, 100, 160)
DEFAULT_TEMPORAL_FILTERS = (1, 2, 2, 2, 1)
DEFAULT_SEPARABLE_KERNEL, DEFAULT_DROPOUT = 32, .50
DEFAULT_TCN_KERNEL, DEFAULT_TCN_DILATIONS = 5, (1, 2, 4)
DEFAULT_TCN_EXPANSION, DEFAULT_TCN_RESIDUAL_SCALE_INIT = 4, .10
UNIT_SCALES = {'V': 1., 'mV': 1e-3, 'uV': 1e-6}

class InputContractError(ValueError):
    """Wrong model/metadata/unit contract; do not infer or silently repair."""

class PayloadUnavailable(ValueError):
    """No current usable input; a fixed placeholder is appropriate."""

class E4InferenceError(RuntimeError):
    """Computation error, NOT a positive hardware-failure diagnosis."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise InputContractError(message)


def require_torch() -> None:
    if torch is None:
        raise RuntimeError('PyTorch import unavailable: '+str(TORCH_IMPORT_ERROR)+
                           '. Use the existing repaired environment; no automatic install.')


def finite(value: Any, name: str) -> float:
    require(not isinstance(value, (bool, np.bool_)), name+': not a boolean')
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise InputContractError(name+': need a numeric scalar') from exc
    require(math.isfinite(x), name+': NaN/Inf or missing value')
    return x


def exact_int(value: Any, name: str) -> int:
    x = finite(value, name)
    require(x == math.floor(x), name+': expected integer')
    return int(x)


def strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise InputContractError(name+': explicit bool or integer 0/1 required')


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(',', ':'))


def object_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode('utf-8')).hexdigest()


def read_json(path: str | Path) -> dict:
    a = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(isinstance(a, dict), f'Expected JSON object: {path}')
    return a


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(x: np.ndarray) -> str:
    """Same canonical float64 waveform hash as EEG Quality V1, in E4 order."""
    a = np.ascontiguousarray(x, dtype='<f8')
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def atomic_json(path: str | Path, obj: Any, *, protected: Sequence[Path] = ()) -> None:
    p = Path(path).expanduser().resolve()
    require(p not in {Path(q).resolve() for q in protected}, 'Output would overwrite an input/model asset.')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(json_safe(obj), ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    require(z.ndim == 2 and z.shape[1] == 5 and np.isfinite(z).all(), 'Need finite [N,5] logits.')
    z = z-z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=1, keepdims=True)


def normalize_probs(values: Any, name: str = 'eeg_probs') -> list[float]:
    p = np.asarray(values, dtype=np.float64)
    require(p.shape == (5,) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            and abs(float(p.sum())-1) <= 1e-3, name+': need five probabilities, not logits.')
    return (p/p.sum()).tolist()


# The following two classes are copied verbatim from the supplied E4 source.
# Their actual forward/state_dict, NOT the high-level prose, defines compatibility.
class DilatedResidualTemporalBlock(nn.Module):
    """Residual temporal refinement at one fixed dilation.

    E4-fast applies this block AFTER the EEGNet spatial depthwise convolution
    and its 4x temporal pooling. Input/output shape is identical:
        [B, C_scale, 1, 250]
    This preserves scale identity while avoiding TCN computation over all
    30 electrodes x 1000 samples.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int,
        residual_scale_init: float,
    ) -> None:
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be >=1")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd and >=3")
        if dilation < 1:
            raise ValueError("dilation must be >=1")
        if expansion < 1:
            raise ValueError("expansion must be >=1")

        hidden = int(channels * expansion)
        pad = int(dilation * (kernel_size - 1) // 2)

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.hidden_channels = hidden

        self.conv1 = nn.Conv2d(
            channels,
            hidden,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, pad),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, pad),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act_out = nn.ELU()

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if y.shape != residual.shape:
            raise RuntimeError(
                f"Residual TCN shape mismatch: y={tuple(y.shape)} "
                f"vs residual={tuple(residual.shape)}"
            )

        return self.act_out(residual + self.residual_scale * y)

class MultiScaleDilatedTCNEEGNet(nn.Module):
    """Stage 1A-E4-fast: E2 Multi-Scale EEGNet + efficient per-scale TCN.

    Input:
        [B, 1, 30, 1000]

    E2 temporal stems remain unchanged:
        kernels = [16, 32, 64, 100, 160]
        filters = [ 1,  2,  2,   2,   1]
        total F1 = 8

    Efficient E4 ordering:
        1) five E2 temporal stems at full resolution
        2) concatenate to F1=8
        3) original EEGNet spatial depthwise convolution across 30 electrodes
        4) original 4x temporal average pooling -> [B, 16, 1, 250]
        5) split channels back into five scale groups [2,4,4,4,2]
        6) per-scale residual TCN, dilations [1,2,4]
        7) concatenate back to 16 channels
        8) original E2 separable block and classifier

    The scientific intervention remains deeper temporal dependency modeling
    within each scale, but the TCN no longer runs on the expensive
    [electrodes=30, time=1000] maps.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_samples: int = TARGET_SAMPLES,
        n_classes: int = NUM_CLASSES,
        f1: int = DEFAULT_F1,
        d: int = DEFAULT_D,
        f2: int = DEFAULT_F2,
        temporal_kernels: Sequence[int] = DEFAULT_TEMPORAL_KERNELS,
        temporal_filters: Sequence[int] = DEFAULT_TEMPORAL_FILTERS,
        separable_kernel: int = DEFAULT_SEPARABLE_KERNEL,
        dropout: float = DEFAULT_DROPOUT,
        tcn_kernel: int = DEFAULT_TCN_KERNEL,
        tcn_dilations: Sequence[int] = DEFAULT_TCN_DILATIONS,
        tcn_expansion: int = DEFAULT_TCN_EXPANSION,
        tcn_residual_scale_init: float = DEFAULT_TCN_RESIDUAL_SCALE_INIT,
    ) -> None:
        super().__init__()

        temporal_kernels = tuple(int(k) for k in temporal_kernels)
        temporal_filters = tuple(int(v) for v in temporal_filters)
        tcn_dilations = tuple(int(v) for v in tcn_dilations)

        if len(temporal_kernels) != len(temporal_filters):
            raise ValueError(
                "temporal_kernels and temporal_filters must have equal length."
            )
        if len(temporal_kernels) < 2:
            raise ValueError("E4 requires at least two temporal scales.")
        if any(k <= 0 for k in temporal_kernels):
            raise ValueError("All temporal kernels must be positive.")
        if any(v <= 0 for v in temporal_filters):
            raise ValueError("All temporal filter counts must be positive.")
        if sum(temporal_filters) != f1:
            raise ValueError(
                f"sum(temporal_filters)={sum(temporal_filters)} must equal F1={f1}."
            )
        if f2 != f1 * d:
            raise ValueError("Formal Stage 1A-E4 expects F2 = F1 * D.")
        if tcn_kernel < 3 or tcn_kernel % 2 == 0:
            raise ValueError("TCN kernel must be odd and >=3.")
        if not tcn_dilations or any(v <= 0 for v in tcn_dilations):
            raise ValueError("TCN dilations must be positive.")
        if tcn_expansion < 1:
            raise ValueError("TCN expansion must be >=1.")
        if not 0.0 < tcn_residual_scale_init <= 1.0:
            raise ValueError("TCN residual scale init must be in (0,1].")

        self.temporal_kernels = temporal_kernels
        self.temporal_filters = temporal_filters
        self.f1_total = int(f1)
        self.depth_multiplier = int(d)
        self.spatial_scale_channels = tuple(int(v * d) for v in temporal_filters)
        self.tcn_kernel = int(tcn_kernel)
        self.tcn_dilations = tcn_dilations
        self.tcn_expansion = int(tcn_expansion)
        self.tcn_residual_scale_init = float(tcn_residual_scale_init)

        # EXACT E2 five-scale temporal stems.
        self.temporal_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    1,
                    n_filters,
                    kernel_size=(1, kernel),
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(n_filters),
            )
            for kernel, n_filters in zip(
                self.temporal_kernels,
                self.temporal_filters,
            )
        ])

        # EXACT E2 spatial block. Critically, this runs BEFORE TCN refinement.
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * d,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
            ),
            nn.Dropout(dropout),
        )

        # E4-fast TCNs operate on post-spatial/post-pooling scale groups.
        # E2 [1,2,2,2,1] temporal filters become [2,4,4,4,2] channels after D=2.
        self.tcn_refiners = nn.ModuleList([
            nn.Sequential(*[
                DilatedResidualTemporalBlock(
                    channels=scale_channels,
                    kernel_size=self.tcn_kernel,
                    dilation=dilation,
                    expansion=self.tcn_expansion,
                    residual_scale_init=self.tcn_residual_scale_init,
                )
                for dilation in self.tcn_dilations
            ])
            for scale_channels in self.spatial_scale_channels
        ])

        # EXACT E2 separable block.
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, separable_kernel),
                padding="same",
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(
                f1 * d,
                f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
            ),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(2, 1, n_channels, n_samples)
            z = self._features(dummy)
            flat_dim = int(np.prod(z.shape[1:]))

        self.feature_dim = flat_dim
        self.classifier = nn.Linear(flat_dim, n_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, DilatedResidualTemporalBlock):
                with torch.no_grad():
                    module.residual_scale.fill_(self.tcn_residual_scale_init)

    def forward_temporal_stems(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [stem(x) for stem in self.temporal_stems]

    def forward_spatial_multiscale(self, x: torch.Tensor) -> torch.Tensor:
        stem_maps = self.forward_temporal_stems(x)
        z = torch.cat(stem_maps, dim=1)
        if z.shape[1] != self.f1_total:
            raise RuntimeError(
                f"Temporal concat channels={z.shape[1]}, expected F1={self.f1_total}."
            )
        return self.spatial(z)

    def forward_spatial_scale_groups(self, x: torch.Tensor) -> List[torch.Tensor]:
        z = self.forward_spatial_multiscale(x)
        groups = list(torch.split(z, self.spatial_scale_channels, dim=1))
        if len(groups) != len(self.temporal_filters):
            raise RuntimeError("Unexpected number of post-spatial scale groups.")
        return groups

    def forward_temporal_scales(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return post-spatial TCN-refined scale groups."""
        groups = self.forward_spatial_scale_groups(x)
        return [
            refiner(scale_group)
            for scale_group, refiner in zip(groups, self.tcn_refiners)
        ]

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        refined_groups = self.forward_temporal_scales(x)
        z = torch.cat(refined_groups, dim=1)
        expected = self.f1_total * self.depth_multiplier
        if z.shape[1] != expected:
            raise RuntimeError(
                f"Refined concat channels={z.shape[1]}, expected={expected}."
            )
        return self.separable(z)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self._features(x), start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

SOURCE_TRAINING_SHA256 = '5a9125fe7672a13efb5888aaeb8e410305c3621eda850f5f05b917493d324afa'
MODEL_DEFINITION_SHA256 = '88510da171d8dbb0008dd3aa1cbb2e22de77e6f9c8d6b092b768792ec83fdd4d'

# =============================================================================
# Frozen checkpoint and paired Train normalization
# =============================================================================


def resolve_assets(*, assets_dir: str | Path | None = None,
                   e4_run: str | Path | None = None, checkpoint: str | Path | None = None,
                   normalization: str | Path | None = None) -> tuple[Path, Path]:
    require(not (assets_dir and e4_run), 'Use --assets-dir OR --e4-run.')
    require((checkpoint is None) == (normalization is None),
            'Explicit files require BOTH --checkpoint and --normalization; never guess the pair.')
    if checkpoint is not None:
        require(not (assets_dir or e4_run), 'Use an explicit file pair OR one asset directory.')
        cp, norm = Path(checkpoint).expanduser().resolve(), Path(normalization).expanduser().resolve()
    else:
        explicit = assets_dir or e4_run
        if explicit is not None:
            root = Path(explicit).expanduser().resolve()
        else:
            roots = list(dict.fromkeys([Path(__file__).resolve().parent, Path.cwd().resolve()]))
            found = [r for r in roots if (r/DEFAULT_CHECKPOINT).is_file() and (r/DEFAULT_NORMALIZATION).is_file()]
            require(len(found) == 1, 'Cannot select an unambiguous E4 asset pair. Use --e4-run for the '
                    'original frozen E4 directory, or put both files beside the script and use --assets-dir . '
                    'No recursive/latest-run discovery is performed.')
            root = found[0]
        cp, norm = root/DEFAULT_CHECKPOINT, root/DEFAULT_NORMALIZATION
    require(cp.is_file(), f'E4 checkpoint not found: {cp}')
    require(norm.is_file(), f'Paired Train normalization not found: {norm}')
    require(cp != norm, 'Checkpoint and normalization must be different files.')
    return cp, norm


def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    obj = read_json(path)
    rows = obj.get('channels')
    require(isinstance(rows, list) and len(rows) == 30, 'Normalization needs all 30 channel records.')
    require(isinstance(obj.get('policy'), str) and 'TRAIN' in obj['policy'].upper(),
            'Normalization must declare its original TRAIN-only policy.')
    count = exact_int(obj.get('sample_count_per_channel'), 'sample_count_per_channel')
    require(count > 0, 'Normalization sample count must be positive.')
    means, stds = [], []
    for i, (row, expected) in enumerate(zip(rows, CHANNELS)):
        require(isinstance(row, Mapping), 'Invalid channel normalization record.')
        require(exact_int(row.get('channel_index_0based'), 'channel index') == i and
                row.get('channel') == expected, 'Normalization channel/index order does not match E4; do not relabel it.')
        means.append(finite(row.get('train_mean'), expected+' mean'))
        s = finite(row.get('train_std'), expected+' std')
        require(s >= 1e-8, expected+': zero/negative/near-zero Train std.')
        stds.append(s)
    with np.errstate(over='ignore'):
        mean, std = np.asarray(means, np.float32)[:, None], np.asarray(stds, np.float32)[:, None]
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(),
            'Normalization cannot be represented in the original float32 arithmetic.')
    mean.setflags(write=False); std.setflags(write=False)
    return mean, std, obj


def architecture_kwargs(checkpoint: Mapping[str, Any]) -> dict:
    require(checkpoint.get('stage') == 'Stage 1A-E4', 'Checkpoint stage must be Stage 1A-E4, not E2/E3/E9.')
    require(checkpoint.get('model') == 'Multi-Scale Dilated-TCN EEGNet', 'Wrong E4 network type.')
    require(checkpoint.get('input_shape') == [1, 30, 1000] and checkpoint.get('sampling_rate_hz') == 200,
            'Checkpoint geometry must be [1,30,1000] at 200 Hz.')
    require(checkpoint.get('emotion_mapping') == {e: i for i, e in enumerate(EMOTIONS)},
            'Checkpoint five-class order differs from frozen fusion.')
    a = checkpoint.get('architecture')
    require(isinstance(a, Mapping), 'Architecture metadata is required; it is not inferred from names.')
    for k, v in {'attention': False, 'dilated_tcn': True, 'channel_attention': False}.items():
        require(a.get(k) is v, f'Unsupported E4 architecture flag {k}.')
    mapping = {'f1':'F1_total', 'd':'D', 'f2':'F2', 'separable_kernel':'separable_kernel',
               'tcn_kernel':'tcn_kernel_samples', 'tcn_expansion':'tcn_expansion'}
    kw = {k: exact_int(a.get(src), src) for k, src in mapping.items()}
    require(all(1 <= v <= 2048 for v in kw.values()), 'Invalid or excessive architecture dimensions.')
    for dest, src in [('temporal_kernels','temporal_kernels_samples'),
                      ('temporal_filters','temporal_filters_per_scale'), ('tcn_dilations','tcn_dilations')]:
        vals = a.get(src)
        require(isinstance(vals, (list, tuple)) and 1 <= len(vals) <= 32, 'Missing/invalid '+src)
        kw[dest] = tuple(exact_int(v, src) for v in vals)
        require(all(1 <= v <= 4096 for v in kw[dest]), 'Invalid or excessive '+src)
    kw['dropout'] = finite(a.get('dropout'), 'dropout')
    kw['tcn_residual_scale_init'] = finite(a.get('tcn_residual_scale_init'), 'residual scale init')
    require(0 <= kw['dropout'] < 1 and 0 < kw['tcn_residual_scale_init'] <= 1, 'Bad dropout/residual initialization.')
    require(kw['f1'] == sum(kw['temporal_filters']) and kw['f2'] == kw['f1']*kw['d'], 'Inconsistent E4 channel dimensions.')
    require(len(kw['temporal_filters']) == len(kw['temporal_kernels']), 'Scale count mismatch.')
    secs = a.get('temporal_kernel_seconds')
    if secs is not None:
        require(len(secs) == len(kw['temporal_kernels']) and
                np.allclose(secs, np.asarray(kw['temporal_kernels'])/200, rtol=0, atol=1e-12),
                'Temporal kernel samples/seconds mismatch.')
    return kw


def load_checkpoint(path: Path) -> tuple[dict, str]:
    require_torch()
    require(path.stat().st_size <= 512*1024*1024, 'Unexpectedly large E4 checkpoint (>512 MiB).')
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    try:
        obj = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    except Exception as exc:
        raise InputContractError('Restricted E4 checkpoint load failed. No unsafe pickle fallback is attempted; '
                                 'use the original tensor/state_dict E4 checkpoint. '+repr(exc)) from exc
    require(isinstance(obj, dict), 'E4 checkpoint must be a dictionary.')
    return obj, sha


def verify_state(model: Any, state: Any) -> dict:
    require(isinstance(state, Mapping) and bool(state), 'Missing model_state_dict.')
    expected = model.state_dict()
    missing, extra = sorted(set(expected)-set(state)), sorted(set(state)-set(expected))
    require(not missing and not extra, f'Strict E4 state mismatch. Missing={missing[:12]}, unexpected={extra[:12]}. '
            'No prefix stripping or partial loading is performed.')
    for k, ref in expected.items():
        v = state[k]
        require(isinstance(v, torch.Tensor) and v.layout == torch.strided, 'Non-dense-tensor state: '+k)
        require(v.shape == ref.shape and v.dtype == ref.dtype,
                f'Wrong state shape/dtype for {k}: {tuple(v.shape)}/{v.dtype}, expected {tuple(ref.shape)}/{ref.dtype}.')
        if v.is_floating_point():
            require(bool(torch.isfinite(v).all()), 'Nonfinite checkpoint value: '+k)
        if k.endswith('running_var'):
            require(bool((v >= 0).all()), 'Negative BatchNorm variance: '+k)
    model.load_state_dict(state, strict=True)
    return {'status':'PASS', 'state_tensors':len(state), 'strict_load':True,
            'all_finite':True, 'learned_parameters':sum(v.numel() for v in model.parameters())}


def verify_pair_report(report_path: Path | None, checkpoint_sha: str, norm: Mapping[str, Any]) -> dict:
    if report_path is None:
        return {'status':'CALLER_PAIRED_FILES_ONLY', 'independent_pair_provenance_verified':False,
                'note':'Shape/order checks cannot establish which experiment produced this pair. Keep both from the SAME E4 run.'}
    r = read_json(report_path)
    require(r.get('stage') == 'Stage 1A-E4' and r.get('model_frozen') is True, 'Not a final frozen E4 report.')
    require(r.get('provenance', {}).get('frozen_checkpoint_sha256') == checkpoint_sha,
            'E4 checkpoint SHA differs from the supplied frozen report.')
    prep = r.get('preprocessing', {})
    require(prep.get('source_fs_hz') == 500 and prep.get('target_fs_hz') == 200 and
            prep.get('source_window_shape') == [30,2500] and prep.get('extra_filtering') is None,
            'Frozen report preprocessing differs from this runtime.')
    recorded = prep.get('normalization', {})
    require(recorded.get('channels') == norm.get('channels') and
            recorded.get('sample_count_per_channel') == norm.get('sample_count_per_channel'),
            'Normalization differs from the frozen E4 report; do not mix runs.')
    return {'status':'MATCHES_SUPPLIED_FROZEN_REPORT', 'report':str(report_path),
            'sha256':digest_file(report_path), 'checkpoint_hash_and_normalization_checked':True,
            'independent_acquisition_unit_verified':False}


def choose_device(name: str) -> Any:
    require_torch()
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    require(name == 'cpu' or bool(re.fullmatch(r'cuda(?::\d+)?', name)), 'Device must be cpu, cuda, cuda:N or auto.')
    d = torch.device(name)
    if d.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA explicitly requested but unavailable; use --device cpu intentionally.')
        if d.index is not None:
            require(d.index < torch.cuda.device_count(), 'CUDA device index does not exist.')
    return d


def named_scale(unit: str | None, scale: float | None, label: str) -> float | None:
    require(not (unit is not None and scale is not None), label+': provide a named unit OR scalar, not both.')
    if unit is not None:
        require(unit in UNIT_SCALES, label+': supported named units are V/mV/uV.')
        return UNIT_SCALES[unit]
    if scale is not None:
        s = finite(scale, label)
        require(s > 0, label+': volts per unit must be positive.')
        return s
    return None


def ordered_source(eeg: Any, *, channel_names: Sequence[str] | None, channel_axis: int,
                   assume_e4_order: bool) -> tuple[np.ndarray, dict]:
    require(type(channel_axis) is int and channel_axis in (0, 1), 'channel_axis must be 0/1.')
    if eeg is None:
        raise PayloadUnavailable('NO_EEG_SAMPLES')
    try:
        a = np.asarray(eeg)
    except (TypeError, ValueError) as exc:
        raise PayloadUnavailable('INVALID_EEG_ARRAY') from exc
    if not a.size:
        raise PayloadUnavailable('NO_EEG_SAMPLES')
    if a.ndim != 2 or a.shape[channel_axis] != 30:
        raise PayloadUnavailable('E4_INPUT_GEOMETRY_MISMATCH')
    if a.dtype.kind not in 'iuf':
        raise PayloadUnavailable('NON_REAL_NUMERIC_EEG')
    a = a if channel_axis == 0 else a.T
    if a.shape[1] != 2500:
        raise PayloadUnavailable('INCOMPLETE_WINDOW' if a.shape[1] < 2500 else 'OVERSIZED_WINDOW')
    if channel_names is None:
        require(strict_bool(assume_e4_order, 'assume_e4_order'),
                'Provide real channel_names or explicitly set assume_e4_order=True.')
        idx = list(range(30)); decl = 'CALLER_DECLARED_E4_ORDER'
    else:
        require(not isinstance(channel_names, (str, bytes)), 'channel_names must be a list.')
        names = [str(c).strip().upper() for c in channel_names]
        require(len(names) == 30 and len(set(names)) == 30 and set(names) == set(CHANNELS),
                'Channel labels must identify every E4 electrode exactly once; no guessing/aliases.')
        idx = [names.index(c) for c in CHANNELS]; decl = 'EXPLICIT_CHANNEL_LABELS'
    a = np.array(a[idx], copy=True, order='C')  # Preserve float32/float64 source dtype as in E4.
    if not np.isfinite(a).all():
        raise PayloadUnavailable('NONFINITE_EEG_SAMPLES')
    if np.all(np.max(a,axis=1) == np.min(a,axis=1)):
        raise PayloadUnavailable('ALL_CHANNELS_FLAT')
    return a, {'channel_order':CHANNELS.copy(), 'source_dtype':str(a.dtype), 'source_shape':[30,2500],
               'channel_axis_received':channel_axis, 'channel_order_declaration':decl,
               'channels_reordered':idx != list(range(30)), 'source_waveform_sha256':array_sha(a)}


def source_to_training_scale(x: np.ndarray, *, input_unit: str,
                             input_scale_to_volts: float | None,
                             training_scale_to_volts: float | None) -> tuple[np.ndarray, dict]:
    if input_unit == 'training_native':
        require(input_scale_to_volts is None, 'training_native does not take input_scale_to_volts.')
        require(x.dtype.kind == 'f' and x.dtype.itemsize in (4,8),
                'training_native expects float32/float64 in the original training numeric scale. '
                'Convert raw ADC counts using documented calibration first.')
        y, gain, physical = x.copy(), 1.0, training_scale_to_volts
        status = 'CALLER_DECLARED_TRAINING_NUMERIC_SCALE_NOT_PHYSICAL_UNIT_VERIFICATION'
    else:
        if input_unit == 'custom':
            physical = named_scale(None, input_scale_to_volts, 'input_scale_to_volts')
            require(physical is not None, 'custom unit requires input_scale_to_volts.')
        else:
            require(input_scale_to_volts is None, 'A named input unit cannot also specify custom scaling.')
            physical = named_scale(input_unit, None, 'input_unit')
        require(training_scale_to_volts is not None,
                'Physical input requires a separately confirmed --training-eeg-unit or --training-scale-to-volts. '
                'The original training script does not establish that physical unit.')
        gain = physical/training_scale_to_volts
        with np.errstate(over='ignore', invalid='ignore'):
            y = np.asarray(x, dtype=np.float64)*gain
        status = 'CALLER_SUPPLIED_PHYSICAL_TO_TRAINING_SCALE'
    require(np.isfinite(y).all(), 'Input scale conversion overflowed.')
    return y, {'input_unit':input_unit, 'input_volts_per_unit':physical,
               'training_volts_per_unit':training_scale_to_volts, 'input_to_training_gain':gain,
               'scale_policy':status, 'physical_units_independently_verified':False,
               'training_scale_waveform_sha256':array_sha(y)}


def preprocess_training_window(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, dict]:
    """Faithful to extract_resampled_window + EEGMemmapDataset.__getitem__."""
    from scipy import signal
    require(x.shape == (30,2500) and x.dtype.kind == 'f' and np.isfinite(x).all(), 'Need finite float source window.')
    # Explicitly retain the original function's defaults, not a different filter.
    y = signal.resample_poly(x, up=2, down=5, axis=-1)
    if y.shape[-1] > 1000:
        y = y[...,:1000]
    elif y.shape[-1] < 1000:
        y = np.pad(y, ((0,0),(0,1000-y.shape[-1])), mode='edge')
    y = np.asarray(y, dtype=np.float32)
    require(y.shape == (30,1000) and np.isfinite(y).all(), 'Invalid resampled EEG.')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z = (np.array(y, dtype=np.float32, copy=True)-mean)/std
    require(np.isfinite(z).all(), 'Frozen normalization produced NaN/Inf; no clipping/imputation applied.')
    return np.ascontiguousarray(z[None,None,:,:]), {
        'source_fs_hz':500, 'model_fs_hz':200, 'resampled_shape':[30,1000], 'model_input_shape':[1,1,30,1000],
        'resample':'scipy.signal.resample_poly(up=2,down=5,axis=-1); original defaults',
        'normalization':'frozen TRAIN channel mean/std, float32', 'resampled_float32_sha256':array_sha(y),
        'normalized_input_sha256':array_sha(z), 'extra_filtering':None, 'source_modified':False}


def freshness(live: bool, newest: float | None, clock: Callable[[],float], max_age: float) -> tuple[str | None, float | None]:
    if not live:
        return None, None
    if newest is None:
        return 'MISSING_SAMPLE_TIMESTAMP', None
    try:
        age = finite(clock(), 'clock')-finite(newest, 'newest_sample_monotonic')
    except InputContractError:
        return 'INVALID_SAMPLE_TIMESTAMP', None
    if age < -1e-6:
        return 'FUTURE_OR_WRONG_CLOCK_TIMESTAMP', age
    age = max(0., age)
    return ('STALE_EEG_INPUT' if age > max_age else None), age


def checked_timestamps(values: Any, newest: float | None, tolerance: float) -> float | None:
    if values is None:
        return newest
    try:
        t = np.asarray(values, dtype=np.float64)
    except (TypeError,ValueError) as exc:
        raise PayloadUnavailable('INVALID_SAMPLE_TIMESTAMPS') from exc
    if t.shape != (2500,) or not np.isfinite(t).all():
        raise PayloadUnavailable('INVALID_SAMPLE_TIMESTAMPS')
    d = np.diff(t)
    if np.any(d <= 0) or np.any(np.abs(d-1/500)>tolerance):
        raise PayloadUnavailable('NONCONTIGUOUS_SAMPLE_TIMESTAMPS')
    if newest is not None and abs(finite(newest,'newest timestamp')-float(t[-1])) > 1e-6:
        raise PayloadUnavailable('NEWEST_TIMESTAMP_MISMATCH')
    return float(t[-1])
# =============================================================================
# Public emotion runtime
# =============================================================================

class EEGEmotionE4:
    """One loaded frozen E4 per stream; reusable file/array inference.

    Inference error default='raise'. Missing input returns a labelled placeholder.
    'training_native' preserves original EAV values; it never guesses microvolts.
    No returned confidence or availability is a calibrated sensor-quality score.
    """
    def __init__(self, *, assets_dir: str | Path | None = None,
                 e4_run: str | Path | None = None, checkpoint: str | Path | None = None,
                 normalization: str | Path | None = None, device: str = 'cuda',
                 frozen_report: str | Path | None = None, contract: str | Path | None = None,
                 training_eeg_unit: str | None = None, training_scale_to_volts: float | None = None,
                 unit_evidence: str = 'USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED',
                 on_inference_error: str = 'raise', stale_after_sec: float = 1.0,
                 timestamp_tolerance_sec: float = .0005,
                 clock: Callable[[],float] = time.monotonic):
        self.checkpoint_path, self.normalization_path = resolve_assets(
            assets_dir=assets_dir, e4_run=e4_run, checkpoint=checkpoint, normalization=normalization)
        require(on_inference_error in ('raise','unavailable'), 'on_inference_error must be raise/unavailable.')
        self.on_inference_error = on_inference_error
        self.stale_after_sec = finite(stale_after_sec, 'stale_after_sec')
        self.tolerance = finite(timestamp_tolerance_sec, 'timestamp_tolerance_sec')
        require(self.stale_after_sec > 0 and 0 < self.tolerance < 1/500, 'Invalid freshness/timing policy.')
        self.training_scale = named_scale(training_eeg_unit, training_scale_to_volts, 'training voltage scale')
        require(isinstance(unit_evidence,str) and bool(unit_evidence.strip()), 'Unit evidence must be a nonempty description.')
        self.unit_evidence = unit_evidence
        self.clock = clock; self._lock = threading.RLock(); self._generation = 0
        self._instance_id = uuid.uuid4().hex
        self.mean, self.std, norm = load_normalization(self.normalization_path)
        norm_sha = digest_file(self.normalization_path)
        ckpt, ckpt_sha = load_checkpoint(self.checkpoint_path)
        kw = architecture_kwargs(ckpt)
        require_torch()
        with torch.random.fork_rng(devices=[]):
            model = MultiScaleDilatedTCNEEGNet(**kw)
        state_report = verify_state(model, ckpt.get('model_state_dict'))
        paired_report = Path(frozen_report).expanduser().resolve() if frozen_report else None
        if paired_report is None and self.checkpoint_path.parent == self.normalization_path.parent:
            local = self.checkpoint_path.parent/DEFAULT_FROZEN_REPORT
            if local.is_file():
                paired_report = local
        self.pair_report = verify_pair_report(paired_report, ckpt_sha, norm)
        self.contract_path = Path(contract).expanduser().resolve() if contract else None
        if self.contract_path is None:
            local = self.checkpoint_path.parent/CONTRACT_NAME
            if local.is_file() and self.checkpoint_path.parent == self.normalization_path.parent:
                self.contract_path = local
        if self.contract_path is not None:
            c = read_json(self.contract_path)
            require(c.get('schema') == CONTRACT_SCHEMA and c.get('checkpoint_sha256') == ckpt_sha and
                    c.get('normalization_sha256') == norm_sha and c.get('model_definition_sha256') == MODEL_DEFINITION_SHA256
                    and c.get('class_order') == EMOTIONS and c.get('channel_order') == CHANNELS,
                    'E4 deployment contract / asset identity mismatch.')
            require(c.get('contract_id') == object_hash({k:v for k,v in c.items() if k != 'contract_id'}),
                    'Deployment contract checksum mismatch.')
            recorded_scale = c.get('training_volts_per_unit')
            if recorded_scale is not None:
                if self.training_scale is None:
                    self.training_scale = named_scale(None, recorded_scale, 'contract training scale')
                    self.unit_evidence = c.get('unit_evidence', unit_evidence)
                else:
                    require(self.training_scale == recorded_scale, 'Training scale conflicts with pinned deployment contract.')
        self.device = choose_device(device)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.identity = dict(module_version=VERSION, checkpoint=str(self.checkpoint_path), checkpoint_sha256=ckpt_sha,
            normalization=str(self.normalization_path), normalization_sha256=norm_sha,
            class_order=EMOTIONS.copy(), channel_order=CHANNELS.copy(), architecture=copy.deepcopy(ckpt['architecture']),
            actual_tcn_placement='after spatial depthwise convolution and 4x temporal pooling',
            model_definition_sha256=MODEL_DEFINITION_SHA256, supplied_training_script_sha256=SOURCE_TRAINING_SHA256,
            feature_dim=int(model.feature_dim), strict_state_load=state_report, pair_provenance=self.pair_report,
            training_volts_per_unit=self.training_scale, unit_evidence=self.unit_evidence,
            model_device=str(self.device), precision='FP32; no autocast; original float64 numpy softmax',
            selection_epoch=ckpt.get('epoch'), selection_metric=ckpt.get('selection_metric'),
            numpy=np.__version__, scipy=package_version('scipy'), torch=str(torch.__version__),
            stale_after_sec=self.stale_after_sec, timestamp_tolerance_sec=self.tolerance,
            quality_model_included=False, hardware_driver_included=False, new_model_fit=False,
            source_unit_independently_verified=False, on_inference_error=on_inference_error,
            imports_training_script=False, automatic_downloads=False, automatic_package_installs=False)
        self._norm_original = norm
        self._frozen_report_path = paired_report
        del ckpt

    @property
    def protected_paths(self) -> list[Path]:
        return [p for p in [self.checkpoint_path, self.normalization_path,
                            self.contract_path, self._frozen_report_path] if p is not None]

    def check_assets(self) -> dict:
        return {'status':'PASS', 'identity':copy.deepcopy(self.identity),
                'checkpoint_read_and_strict_loaded':True, 'network_constructor_shape_probe':True,
                'trained_model_forward_on_input':False, 'input_waveform_read':False,
                'normalization_channels':30, 'weights_frozen':all(not p.requires_grad for p in self.model.parameters()),
                'test_manifest_read':False, 'hardware_tested':False}

    def export_contract(self, path: str | Path) -> dict:
        p = Path(path).expanduser().resolve()
        require(not p.exists(), 'Use a new contract path; existing assets/contracts are not overwritten.')
        c = dict(schema=CONTRACT_SCHEMA, module_version=VERSION,
                 checkpoint_sha256=self.identity['checkpoint_sha256'],
                 normalization_sha256=self.identity['normalization_sha256'],
                 model_definition_sha256=MODEL_DEFINITION_SHA256,
                 class_order=EMOTIONS.copy(), channel_order=CHANNELS.copy(),
                 geometry={'source_rate':500, 'source_shape':[30,2500], 'model_shape':[1,30,1000], 'model_rate':200},
                 training_volts_per_unit=self.training_scale, unit_evidence=self.unit_evidence,
                 pair_provenance=self.pair_report,
                 note='Pins the provided asset pair; does not independently establish physical units or training provenance.')
        c['contract_id'] = object_hash(c)
        atomic_json(p, c, protected=self.protected_paths)
        return c

    def _base_result(self, wid: str, live: bool, newest: float | None) -> dict:
        return dict(module='EEG Emotion E4', module_version=VERSION, window_id=wid, window_seconds=5.,
                    class_order=EMOTIONS.copy(), status='UNAVAILABLE', eeg_available=False,
                    classifier_ok=False, payload_available=False, eeg_probs=[.2]*5,
                    pred_label_id=None, pred_emotion='NO_EEG_EVIDENCE', confidence=None,
                    logits=None, reason=None, error=None, probabilities_are_placeholder=True,
                    q_eeg_is_provided=False, eeg_weight=None, live=live,
                    newest_sample_monotonic=newest, newest_sample_age_seconds=None,
                    source_waveform_sha256=None, input_metadata=None, preprocessing=None,
                    runtime_instance_id=self._instance_id, generation=self._generation,
                    checkpoint_sha256=self.identity['checkpoint_sha256'],
                    normalization_sha256=self.identity['normalization_sha256'],
                    model_inference_performed=False, hardware_fault_confirmed=False)

    def _finish(self, r: dict, start: float) -> dict:
        r['elapsed_seconds'] = time.perf_counter()-start
        return json_safe(r)

    def predict_array(self, eeg: Any, *, sample_rate: float = 500,
                      channel_names: Sequence[str] | None = None, channel_axis: int = 0,
                      assume_e4_order: bool = False, input_unit: str = 'training_native',
                      input_scale_to_volts: float | None = None, window_id: str | None = None,
                      live: bool = False, newest_sample_monotonic: float | None = None,
                      sample_timestamps_monotonic: Any = None, contiguous: bool = True,
                      backend_error: bool = False) -> dict:
        """Predict one buffered window, no quality claim and no carried-over prediction."""
        with self._lock:
            self._generation += 1
            start = time.perf_counter()
            live = strict_bool(live,'live'); contiguous = strict_bool(contiguous,'contiguous')
            backend_error = strict_bool(backend_error,'backend_error')
            require(finite(sample_rate,'sample_rate') == 500, 'E4 raw entry expects 500 Hz, not a 200-Hz standardized tensor.')
            wid = window_id if window_id is not None else 'e4:'+uuid.uuid4().hex
            require(isinstance(wid,str) and bool(wid.strip()), 'window_id must be a nonempty string.')
            r = self._base_result(wid,live,newest_sample_monotonic)
            if backend_error or not contiguous:
                r['reason'] = 'CAPTURE_BACKEND_ERROR' if backend_error else 'NONCONTIGUOUS_EEG_WINDOW'
                return self._finish(r,start)
            try:
                newest = checked_timestamps(sample_timestamps_monotonic,newest_sample_monotonic,self.tolerance)
                r['newest_sample_monotonic'] = newest
                reason,age = freshness(live,newest,self.clock,self.stale_after_sec)
                r['newest_sample_age_seconds'] = age
                if reason:
                    raise PayloadUnavailable(reason)
                x,meta = ordered_source(eeg,channel_names=channel_names,channel_axis=channel_axis,
                                        assume_e4_order=assume_e4_order)
            except PayloadUnavailable as exc:
                r['reason'] = str(exc)
                return self._finish(r,start)
            r.update(payload_available=True, source_waveform_sha256=meta['source_waveform_sha256'],input_metadata=meta)
            # Wrong unit/contract is an explicit configuration error, not missing input.
            native,scale_info = source_to_training_scale(x,input_unit=input_unit,
                    input_scale_to_volts=input_scale_to_volts,training_scale_to_volts=self.training_scale)
            r['input_metadata'].update(scale_info, unit_evidence=self.unit_evidence)
            inp,prep = preprocess_training_window(native,self.mean,self.std)
            r['preprocessing'] = prep
            try:
                self.model.eval()
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
                    logits = self.model(torch.from_numpy(inp).to(self.device))
                    require(logits.shape == (1,5) and bool(torch.isfinite(logits).all()), 'E4 produced invalid logits.')
                    raw = logits.detach().float().cpu().numpy()
                p = softmax_np(raw)[0]
            except Exception as exc:
                if self.on_inference_error == 'raise':
                    raise E4InferenceError(f'E4 inference failed: {type(exc).__name__}: {exc}') from exc
                r.update(status='ERROR',reason='E4_INFERENCE_ERROR_GUARD',error=f'{type(exc).__name__}: {exc}',
                         guard_policy='explicit on_inference_error=unavailable; NOT hardware diagnosis')
                return self._finish(r,start)
            r['model_inference_performed'] = True
            reason,age = freshness(live,newest,self.clock,self.stale_after_sec)
            r['newest_sample_age_seconds'] = age
            if reason:
                r['reason'] = reason+'_AFTER_INFERENCE'
                return self._finish(r,start)  # Do not publish stale logits or probability evidence.
            pred = int(np.argmax(p))
            r.update(status='OK',eeg_available=True,classifier_ok=True,eeg_probs=p.tolist(),
                     pred_label_id=pred,pred_emotion=EMOTIONS[pred],confidence=float(p[pred]),
                     logits=raw[0].astype(float).tolist(),probabilities_are_placeholder=False,
                     reason='CURRENT_E4_WINDOW_SCORED')
            return self._finish(r,start)

    def current_prediction(self, result: Mapping[str,Any]) -> dict:
        """Recheck origin, generation and age immediately before consumption."""
        with self._lock:
            require(result.get('runtime_instance_id') == self._instance_id and
                    result.get('checkpoint_sha256') == self.identity['checkpoint_sha256'] and
                    result.get('normalization_sha256') == self.identity['normalization_sha256'],
                    'Prediction belongs to a different model/runtime.')
            r = copy.deepcopy(dict(result))
            reason = None
            if r.get('live'):
                if r.get('generation') != self._generation:
                    reason = 'SUPERSEDED_LIVE_E4_PREDICTION'
                else:
                    reason,_ = freshness(True,r.get('newest_sample_monotonic'),self.clock,self.stale_after_sec)
            if reason:
                r.update(status='UNAVAILABLE',eeg_available=False,classifier_ok=False,eeg_probs=[.2]*5,
                         pred_label_id=None,pred_emotion='NO_EEG_EVIDENCE',confidence=None,logits=None,
                         probabilities_are_placeholder=True,reason=reason+'_AT_CONSUMPTION')
            if r['classifier_ok']:
                r['eeg_probs'] = normalize_probs(r['eeg_probs'])
            return json_safe(r)

    def predict_file(self, path: str | Path, *, start_sample: int | None = None,
                     mat_variable: str = 'seg', trial_index: int | None = None,
                     window_index: int | None = None, channel_axis: int = 0,
                     channel_names: Sequence[str] | None = None,
                     sample_rate: float = 500, **kwargs: Any) -> dict:
        x,info = load_eeg_file(path,start_sample=start_sample,mat_variable=mat_variable,
                              trial_index=trial_index,window_index=window_index,channel_axis=channel_axis)
        if info.get('sample_rate') is not None:
            require(float(info['sample_rate']) == float(sample_rate), 'File/supplied sample rates disagree.')
        if info.get('channel_names') is not None:
            require(channel_names is None or list(channel_names) == info['channel_names'], 'File/supplied channel labels disagree.')
            channel_names = info['channel_names']
        r = self.predict_array(x,sample_rate=sample_rate,channel_names=channel_names,channel_axis=0,**kwargs)
        r['file_source'] = info
        return r

    predict = predict_file

    def predict_many(self, windows: Sequence[Mapping[str,Any]]) -> list[dict]:
        """Each packet contains 'samples' and predict_array keywords; no implicit batch slicing."""
        out = []
        for packet in windows:
            p = dict(packet)
            require('samples' in p, 'Each packet needs samples.')
            x = p.pop('samples')
            out.append(self.predict_array(x,**p))
        return out

    def predict_trial_array(self, eeg: Any, *, trial_id: str,
                            channel_axis: int = 0, **kwargs: Any) -> dict:
        """Explicit OFFLINE 20s emotion-only mean-softmax. No quality/fusion aggregation."""
        require(not kwargs.get('live',False) and not any(k in kwargs for k in
                ('sample_timestamps_monotonic','newest_sample_monotonic','window_id')),
                'Trial API is offline only; use aligned 5s packets for live quality/fusion.')
        require(type(channel_axis) is int and channel_axis in (0,1), 'channel_axis must be 0/1.')
        a = np.asarray(eeg)
        require(a.ndim == 2, 'Trial array must be 2-D.')
        a = a if channel_axis == 0 else a.T
        require(a.shape == (30,10000), 'Trial API requires exactly 30x10000 at 500 Hz.')
        require(isinstance(trial_id,str) and bool(trial_id.strip()), 'Explicit trial_id required.')
        with self._lock:
            rows = [self.predict_array(a[:,i*2500:(i+1)*2500],window_id=f'{trial_id}:w{i+1:02d}',
                                       channel_axis=0,**kwargs) for i in range(4)]
        ok = all(r['classifier_ok'] for r in rows)
        p = np.mean([r['eeg_probs'] for r in rows],axis=0) if ok else np.full(5,.2)
        p = p/p.sum(); pred = int(np.argmax(p))
        return dict(trial_id=trial_id,window_seconds=20.,class_order=EMOTIONS.copy(),
                    eeg_available=ok,classifier_ok=ok,eeg_probs=p.tolist(),
                    pred_emotion=EMOTIONS[pred] if ok else 'NO_EEG_EVIDENCE',
                    confidence=float(p[pred]) if ok else None, windows=rows,
                    aggregation='mean of four window softmax vectors; no missing-window dropping',
                    quality_aggregation_implemented=False,online_fusion_supported=False)

    def process_with_quality_array(self, eeg: Any, *, quality_runtime: Any,
                                   context: Mapping[str,Any] | None = None, bridge: Any = None,
                                   quality_scale_to_volts: float | None = None,
                                   **kwargs: Any) -> dict:
        """Original source -> current E4 + current EEGQualityV1 -> optional frozen AF4-C.

        context contains same-window Audio/Video predictions and explicit q/masks.
        Any pre-existing eeg_probs in context are replaced by THIS model's result.
        No second acquisition, no hidden calibration import/fit, no stale EEG reuse.
        """
        require(callable(getattr(quality_runtime,'assess_array',None)) and
                callable(getattr(quality_runtime,'make_fusion_input',None)), 'Expected EEGQualityV1 runtime.')
        if context is not None:
            require(isinstance(context,Mapping) and context.get('window_seconds') == 5.0 and
                    context.get('class_order') == EMOTIONS, 'Context must declare 5s and the frozen class order.')
            require(isinstance(context.get('window_id'),str) and bool(context['window_id']), 'Context needs window_id.')
            require('window_id' not in kwargs or kwargs['window_id'] == context['window_id'], 'Window IDs conflict.')
            kwargs['window_id'] = context['window_id']
        require(bridge is None or context is not None, 'Fusion bridge requires current Audio/Video context.')
        # Copy once to prevent a acquisition callback from changing one branch's source.
        snapshot = None if eeg is None else np.array(eeg,copy=True)
        frozen_timestamps = kwargs.get('sample_timestamps_monotonic')
        if frozen_timestamps is not None:
            kwargs['sample_timestamps_monotonic'] = np.array(frozen_timestamps,copy=True)
        with self._lock:
            emotion = self.predict_array(snapshot,**kwargs)
            unit = kwargs.get('input_unit','training_native')
            custom = kwargs.get('input_scale_to_volts')
            if unit == 'training_native':
                scale = self.training_scale
                if scale is None:
                    scale = quality_scale_to_volts if quality_scale_to_volts is not None else getattr(quality_runtime,'scale',None)
                elif quality_scale_to_volts is not None:
                    require(scale == finite(quality_scale_to_volts,'quality_scale_to_volts'), 'Training/quality scale conflict.')
                qdefault = getattr(quality_runtime,'scale',None)
                if qdefault is not None and scale is not None:
                    require(qdefault == scale, 'Native E4 and configured quality voltage scales disagree.')
            else:
                scale = named_scale(None,custom,'custom input scale') if unit == 'custom' else named_scale(unit,None,'input unit')
                require(quality_scale_to_volts is None or finite(quality_scale_to_volts,'quality scale') == scale,
                        'Explicit physical input and quality scale disagree.')
            # Missing input does not require an invented physical scale; valid scoring does.
            if emotion['payload_available']:
                require(scale is not None, 'Combined valid native input needs an explicit known quality voltage scale. '
                        'Configure EEGQualityV1(eeg_unit=...) or training_eeg_unit; do not infer uV from amplitude.')
            qkeys = ('sample_rate','channel_names','channel_axis','assume_e4_order','live',
                     'newest_sample_monotonic','sample_timestamps_monotonic','contiguous','backend_error')
            qargs = {k:v for k,v in kwargs.items() if k in qkeys}
            qargs['window_id'] = emotion['window_id']
            if scale is not None:
                qargs['scale_to_volts'] = scale
            assessment = quality_runtime.assess_array(snapshot,**qargs)
            if assessment.get('input_metadata') is not None and emotion['source_waveform_sha256'] is not None:
                require(assessment['input_metadata']['source_waveform_sha256'] == emotion['source_waveform_sha256'],
                        'E4 and quality source waveform hashes differ.')
            emotion = self.current_prediction(emotion)  # Scoring may have consumed freshness budget.
            payload = quality_runtime.make_fusion_input(assessment,emotion['eeg_probs'],
                prediction_window_id=emotion['window_id'],prediction_window_seconds=5.,
                classifier_ok=emotion['classifier_ok'],prediction_source_sha256=emotion['source_waveform_sha256'])
            report = dict(emotion=emotion,quality=assessment,fusion_eeg_input=payload,
                          fusion_input=None,fusion=None,eeg_weight=None,
                          joint_source_alignment='same snapshot and original canonical float64 source hash',
                          hardware_tested=False)
            if context is not None:
                ctx = copy.deepcopy(dict(context))
                require(isinstance(ctx.get('available'),Mapping) and isinstance(ctx.get('quality'),Mapping),
                        'Context must provide Audio/Video masks and qualities explicitly.')
                ctx['eeg_probs'] = emotion['eeg_probs']
                ctx['available'] = dict(ctx['available'],eeg=emotion['classifier_ok'])
                ctx['eeg_source_waveform_sha256'] = emotion['source_waveform_sha256']
                sample = make_full_fusion_sample(payload,ctx)
                report['fusion_input'] = sample
                if bridge is not None:
                    # Quality bridge verifies its own assessment generation and freshness again.
                    emotion = self.current_prediction(emotion)
                    ctx['eeg_probs'] = emotion['eeg_probs']; ctx['available']['eeg'] = emotion['classifier_ok']
                    fused = bridge.predict(quality_runtime,assessment,ctx)
                    report.update(emotion=emotion,fusion=fused,eeg_weight=fused['eeg_weight'],fusion_input=fused['fusion_input'])
                    report['fusion_eeg_input'] = dict(eeg_probs=fused['fusion_input']['eeg_probs'],
                                                     q_eeg=fused['q_eeg'],eeg_available=fused['eeg_available'])
            return json_safe(report)

    def preflight(self) -> dict:
        """Loaded trained E4 forward on SYNTHETIC signal, plus missing/recovery rules."""
        t = np.arange(2500,dtype=np.float64)/500
        x = self.mean.astype(float)+self.std.astype(float)*(
            np.sin(2*np.pi*10*t)[None,:]+.2*np.cos(2*np.pi*23*t)[None,:])
        original = x.copy(); before = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        a = self.predict_array(x,channel_names=CHANNELS,window_id='synthetic_e4_preflight')
        b = self.predict_array(x,channel_names=CHANNELS,window_id='synthetic_e4_repeat')
        missing = self.predict_array(None,window_id='synthetic_missing')
        back = self.predict_array(x,channel_names=CHANNELS,window_id='synthetic_restored')
        checks = dict(finite_probabilities=a['classifier_ok'] and np.isfinite(a['eeg_probs']).all(),
                      class_sum=abs(sum(a['eeg_probs'])-1)<1e-12,
                      repeat_within_fp_tolerance=np.allclose(a['eeg_probs'],b['eeg_probs'],rtol=1e-6,atol=1e-7),
                      no_parameter_or_BN_mutation=all(torch.equal(v,self.model.state_dict()[k].detach().cpu()) for k,v in before.items()),
                      source_unchanged=np.array_equal(x,original),
                      missing_placeholder=missing['eeg_probs']==[.2]*5 and not missing['classifier_ok'] and missing['confidence'] is None,
                      restored=np.allclose(a['eeg_probs'],back['eeg_probs'],rtol=1e-6,atol=1e-7),
                      model_frozen=all(not p.requires_grad for p in self.model.parameters()))
        require(all(checks.values()), 'E4 preflight failed: '+canonical_json(checks))
        return dict(status='PASS',checks=checks,n_checks=len(checks),identity=self.identity,
                    real_loaded_E4_checkpoint_forward=True,synthetic_input_only=True,
                    actual_EAV_waveform_used=False,emotion_accuracy_measured=False,
                    repeat_max_abs_difference=float(np.max(np.abs(np.asarray(a['eeg_probs'])-np.asarray(b['eeg_probs'])))),
                    repeat_tolerance={'rtol':1e-6,'atol':1e-7},
                    real_quality_detector_run=False,real_fusion_weights_run=False,hardware_tested=False)


def make_full_fusion_sample(eeg_payload: Mapping[str,Any], context: Mapping[str,Any]) -> dict:
    masks, qs, ps = {'eeg':bool(eeg_payload['eeg_available'])}, {'eeg':float(eeg_payload['q_eeg'])}, {'eeg':eeg_payload['eeg_probs']}
    for m in ('audio','video'):
        masks[m] = strict_bool(context['available'].get(m),m+' availability')
        if masks[m]:
            ps[m] = normalize_probs(context.get(m+'_probs'),m+'_probs')
            qs[m] = finite(context['quality'].get(m),m+' quality')
            require(0 <= qs[m] <= 1,'Quality must be [0,1].')
        else:
            ps[m] = [.2]*5; qs[m] = 0.
    return {**{m+'_probs':ps[m] for m in ('eeg','audio','video')},
            'quality':qs,'available':{m:int(v) for m,v in masks.items()}}

# File readers retained from the supplied EEG Quality V1 (no package dependency).


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


# =============================================================================
# CLI -- explicitly selected local inputs only
# =============================================================================

def import_quality_runtime(path: str | Path | None = None) -> Any:
    p = Path(path).expanduser().resolve() if path else Path(__file__).resolve().with_name('eeg_quality_v1_deployment.py')
    require(p.is_file(), f'Quality module not found: {p}. It is optional for emotion-only inference.')
    key = '_e4_quality_runtime_'+hashlib.sha256((str(p)+digest_file(p)).encode()).hexdigest()[:20]
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key,p)
    require(spec is not None and spec.loader is not None, 'Cannot import local quality module.')
    module = importlib.util.module_from_spec(spec); sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
        require(module.CHANNELS == CHANNELS and module.EMOTIONS == EMOTIONS and module.FS == 500 and
                hasattr(module,'EEGQualityV1') and hasattr(module,'AF4CEEGBridge'), 'Quality runtime interface mismatch.')
    except BaseException:
        sys.modules.pop(key,None)
        raise
    return module


def run_self_test() -> dict:
    """No external assets/data needed. Synthetic logic and original network shape."""
    from scipy.signal import resample_poly
    checks = {}
    def rejected(fn: Callable) -> bool:
        try:
            fn()
        except (InputContractError, PayloadUnavailable):
            return True
        return False
    rng = np.random.default_rng(20260919)
    x = rng.normal(0,15,(30,2500)).astype(np.float64); original = x.copy()
    opts = dict(channel_names=CHANNELS,channel_axis=0,assume_e4_order=False)
    a,meta = ordered_source(x,**opts)
    checks['source_not_mutated'] = np.array_equal(x,original) and not np.shares_memory(x,a)
    perm = rng.permutation(30)
    b,_ = ordered_source(x[perm],channel_names=[CHANNELS[i] for i in perm],channel_axis=0,assume_e4_order=False)
    checks['channel_reordering'] = np.array_equal(a,b)
    checks['unknown_channel_order_rejected'] = rejected(lambda: ordered_source(x,channel_names=None,channel_axis=0,assume_e4_order=False))
    checks['short_window_not_padded'] = rejected(lambda: ordered_source(x[:,:1000],**opts))
    checks['extra_channels_not_deleted'] = rejected(lambda: ordered_source(np.vstack([x,x[:1]]),**opts))
    checks['all_flat_rejected'] = rejected(lambda: ordered_source(np.ones_like(x),**opts))
    xn = x.copy(); xn[0,0] = np.nan
    checks['nan_rejected'] = rejected(lambda: ordered_source(xn,**opts))
    mean = np.arange(30,dtype=np.float32)[:,None]; std = np.linspace(1,10,30,dtype=np.float32)[:,None]
    z,prep = preprocess_training_window(x,mean,std)
    expected = (resample_poly(x,2,5,axis=-1).astype(np.float32)-mean)/std
    checks['original_preprocessing_exact'] = np.array_equal(z[0,0],expected)
    checks['output_tensor_shape'] = z.shape == (1,1,30,1000)
    checks['native_scale_preserved'] = np.array_equal(source_to_training_scale(x,input_unit='training_native',input_scale_to_volts=None,training_scale_to_volts=None)[0],x)
    checks['physical_scale_requires_training_unit'] = rejected(lambda: source_to_training_scale(x,input_unit='V',input_scale_to_volts=None,training_scale_to_volts=None))
    converted,_ = source_to_training_scale(x*1e-6,input_unit='V',input_scale_to_volts=None,training_scale_to_volts=1e-6)
    checks['volts_to_training_microvolts'] = np.allclose(converted,x,rtol=1e-14,atol=1e-14)
    checks['native_adc_not_guessed'] = rejected(lambda: source_to_training_scale(x.astype(np.int16),input_unit='training_native',input_scale_to_volts=None,training_scale_to_volts=None))
    p = softmax_np(np.array([[2.,-3.,4.,0.,1.]]))
    checks['five_class_probabilities'] = p.shape == (1,5) and abs(p.sum()-1) < 1e-12
    checks['logits_not_accepted_as_probabilities'] = rejected(lambda: normalize_probs([2,-3,4,0,1]))
    checks['stale_timestamp_rejected'] = freshness(True,10.,lambda:12.,1.)[0] == 'STALE_EEG_INPUT'
    checks['future_timestamp_rejected'] = freshness(True,13.,lambda:12.,1.)[0] == 'FUTURE_OR_WRONG_CLOCK_TIMESTAMP'
    checks['timestamp_required_for_live'] = freshness(True,None,lambda:12.,1.)[0] == 'MISSING_SAMPLE_TIMESTAMP'
    stamps = np.arange(2500)/500
    checks['timestamps_accepted'] = checked_timestamps(stamps,None,.0005) == stamps[-1]
    stamps[1000:] += .05
    checks['timestamp_gap_rejected'] = rejected(lambda: checked_timestamps(stamps,None,.0005))
    require_torch()
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            m = MultiScaleDilatedTCNEEGNet().eval()
        with torch.inference_mode():
            out = m(torch.from_numpy(z))
            groups = m.forward_spatial_scale_groups(torch.from_numpy(z))
        checks['untrained_source_network_shape'] = tuple(out.shape) == (1,5) and bool(torch.isfinite(out).all())
        checks['post_spatial_TCN_scale_groups'] = [tuple(g.shape[1:]) for g in groups] == [(2,1,250),(4,1,250),(4,1,250),(4,1,250),(2,1,250)]
    finally:
        torch.set_num_threads(prior_threads)
    require(all(checks.values()), 'Self-test failed: '+canonical_json(checks))
    return {'status':'PASS','n_checks':len(checks),'checks':checks,'synthetic_signal_only':True,
            'real_checkpoint_loaded':False,'real_PyPREP_executed':False,'fusion_executed':False,'hardware_tested':False}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Frozen E4 deployment: raw 5s EEG -> five probabilities; optional EEG Quality V1 / AF4-C.')
    p.add_argument('--version',action='version',version=VERSION)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--self-test',action='store_true')
    modes.add_argument('--check-assets',action='store_true')
    modes.add_argument('--preflight',action='store_true')
    dirs = p.add_mutually_exclusive_group()
    dirs.add_argument('--assets-dir',help='Directory holding the paired E4 checkpoint and Train normalization.')
    dirs.add_argument('--e4-run',help='Original frozen E4 run, explicitly selected; no latest-run guessing.')
    p.add_argument('--checkpoint'); p.add_argument('--normalization')
    p.add_argument('--frozen-report',help='Optional original FINAL_FROZEN JSON for checkpoint/norm pairing verification.')
    p.add_argument('--contract',help='Optional previously exported identity contract.')
    p.add_argument('--export-contract',help='Create a new pinned contract file after successful asset checks.')
    p.add_argument('--device',default='cuda',help='cuda, cuda:N, cpu or auto; default cuda, no silent fallback.')
    p.add_argument('--on-inference-error',choices=['raise','unavailable'],default='raise')
    p.add_argument('--stale-after-sec',type=float,default=1.0)
    p.add_argument('--input-unit',choices=['training_native','V','mV','uV','custom'],default='training_native')
    p.add_argument('--input-scale-to-volts',type=float)
    t = p.add_mutually_exclusive_group()
    t.add_argument('--training-eeg-unit',choices=['V','mV','uV'])
    t.add_argument('--training-scale-to-volts',type=float)
    p.add_argument('--unit-evidence',default='USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED')
    inp = p.add_mutually_exclusive_group()
    inp.add_argument('--eeg',help='Explicit .npy/.npz/classic .mat input; no data discovery.')
    inp.add_argument('--val-manifest',help='Explicit original Speaking val_window_manifest.csv, one --window-key only.')
    p.add_argument('--window-key'); p.add_argument('--window-id')
    p.add_argument('--sample-rate',type=float,default=500)
    p.add_argument('--channel-axis',type=int,choices=[0,1],default=0)
    p.add_argument('--channels-json',help='JSON list of actual electrode labels.')
    p.add_argument('--assume-e4-channel-order',action='store_true')
    p.add_argument('--start-sample',type=int)
    p.add_argument('--mat-variable',default='seg')
    p.add_argument('--trial-index',type=int); p.add_argument('--window-index',type=int)
    p.add_argument('--with-quality',action='store_true')
    p.add_argument('--quality-script'); p.add_argument('--quality-calibration')
    p.add_argument('--quality-candidate',choices=['pyprep_physical','pyprep_only'],default='pyprep_physical')
    q = p.add_mutually_exclusive_group()
    q.add_argument('--quality-eeg-unit',choices=['V','mV','uV'],help='Physical unit of source samples for quality (native-scale input).')
    q.add_argument('--quality-scale-to-volts',type=float)
    p.add_argument('--on-quality-error',choices=['raise','unavailable'],default='raise')
    p.add_argument('--fusion-context',help='Same 5s Audio/Video probs, qualities, masks and class_order/window_id.')
    p.add_argument('--fusion-script'); p.add_argument('--f4-checkpoint'); p.add_argument('--af4b-checkpoint')
    p.add_argument('--fusion-device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output',help='UTF-8 JSON result; cannot overwrite inputs or assets.')
    p.add_argument('--fusion-input-output',help='Optional prediction-only JSON for the original fusion CLI (offline replay).')
    return p.parse_args()


def print_result(r: Mapping[str,Any]) -> None:
    if 'emotion' not in r and 'eeg_probs' not in r:
        print(json.dumps(json_safe(r),ensure_ascii=False,indent=2,allow_nan=False))
        return
    e = r.get('emotion',r)
    print(f"Window                    : {e['window_id']}")
    print(f"E4 status                 : {e['status']}")
    print(f"Classifier available      : {e['classifier_ok']}")
    print(f"Prediction                : {e['pred_emotion']}")
    print(f"Confidence                : {e['confidence']}")
    print(f"EEG probabilities         : {e['eeg_probs']}")
    print(f"Reason                    : {e['reason']}")
    if 'fusion_eeg_input' in r:
        print('Fusion EEG input          : '+json.dumps(r['fusion_eeg_input']))
        print(f"Actual EEG branch weight  : {r['eeg_weight']}")
        if r.get('fusion'):
            print(f"Final                     : {r['fusion']['final']}")
    print('No training or benchmark Test evaluation was performed.')


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = run_self_test(); print_result(result)
        if args.output:
            atomic_json(args.output,result,protected=[Path(__file__)])
        return 0
    bridge_fields = [args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint]
    require(not any(bridge_fields) or all(bridge_fields), 'Fusion execution needs script plus BOTH frozen checkpoints.')
    require(not (any(bridge_fields) or args.fusion_context or args.fusion_input_output) or args.with_quality,
            'Fusion paths/context require --with-quality; quality cannot be invented by the emotion head.')
    require(not any(bridge_fields) or args.fusion_context, 'Actual fusion requires --fusion-context.')
    require(not args.val_manifest or args.window_key, 'VAL replay needs one explicit --window-key.')
    if not (args.check_assets or args.preflight):
        require(args.eeg or args.val_manifest, 'Choose --eeg, --val-manifest, --check-assets or --preflight.')
    head = EEGEmotionE4(assets_dir=args.assets_dir,e4_run=args.e4_run,
        checkpoint=args.checkpoint,normalization=args.normalization,
        device='cpu' if args.check_assets else args.device,frozen_report=args.frozen_report,
        contract=args.contract,training_eeg_unit=args.training_eeg_unit,
        training_scale_to_volts=args.training_scale_to_volts,unit_evidence=args.unit_evidence,
        on_inference_error=args.on_inference_error,stale_after_sec=args.stale_after_sec)
    protected = head.protected_paths+[Path(__file__)]
    for value in [args.eeg,args.val_manifest,args.quality_calibration,args.quality_script,args.fusion_script,
                  args.f4_checkpoint,args.af4b_checkpoint,args.fusion_context,args.channels_json]:
        if value:
            protected.append(Path(value).expanduser().resolve())
    outputs = [Path(v).expanduser().resolve() for v in [args.output,args.export_contract,args.fusion_input_output] if v]
    require(len(outputs) == len(set(outputs)), 'Output destinations must be different files.')
    require(not any(p in set(protected) for p in outputs), 'Output would overwrite source/model input.')
    if args.export_contract:
        head.export_contract(args.export_contract)
    print('='*106); print('EEG EMOTION E4 -- FROZEN STANDALONE DEPLOYMENT'); print('='*106)
    print('Version                   :',VERSION)
    print('Checkpoint                :',head.checkpoint_path)
    print('Normalization             :',head.normalization_path)
    print('Model device              :',head.device)
    print('Architecture              : original E4-fast; TCN after spatial/pool(4)')
    print('Input scale               :',args.input_unit)
    if args.check_assets:
        result = head.check_assets()
        if args.with_quality:
            qm = import_quality_runtime(args.quality_script)
            quality = qm.EEGQualityV1(calibration=args.quality_calibration,
                                      assets_dir=args.assets_dir,candidate=args.quality_candidate)
            result['quality_asset_check'] = quality.check_assets()
            result['status'] = 'PASS' if result['quality_asset_check']['status'] == 'PASS' else 'FAIL'
    elif args.preflight and not (args.eeg or args.val_manifest):
        require(not args.with_quality, 'For combined real scoring use --eeg or --val-manifest with --with-quality. '
                'Run the quality module own preflight separately; no physical source unit is guessed for synthetic E4 data.')
        result = head.preflight()
    else:
        names = None
        if args.channels_json:
            names = json.loads(Path(args.channels_json).read_text(encoding='utf-8-sig'))
            require(isinstance(names,list),'channels-json must contain a list.')
        if args.val_manifest:
            x,info = load_eav_val_window(args.val_manifest,args.window_key)
            require(names is None or names == info['channel_names'],'VAL/supplied channel names disagree.')
            names = info['channel_names']
        else:
            x,info = load_eeg_file(args.eeg,start_sample=args.start_sample,mat_variable=args.mat_variable,
                                   trial_index=args.trial_index,window_index=args.window_index,channel_axis=args.channel_axis)
            if info.get('channel_names') is not None:
                require(names is None or names == info['channel_names'],'File/supplied channel names disagree.')
                names = info['channel_names']
        protected.append(Path(info['path']))
        require(not any(p in set(protected) for p in outputs),'Output cannot overwrite the selected source MAT.')
        require(info.get('sample_rate') is None or float(info['sample_rate']) == args.sample_rate,'File/sample-rate mismatch.')
        wid = args.window_id or args.window_key or (Path(info['path']).stem+':'+str(info.get('start_sample',0)))
        call = dict(sample_rate=args.sample_rate,channel_names=names,channel_axis=0,
                    assume_e4_order=args.assume_e4_channel_order,input_unit=args.input_unit,
                    input_scale_to_volts=args.input_scale_to_volts,window_id=wid)
        if args.with_quality:
            qm = import_quality_runtime(args.quality_script)
            qs = named_scale(args.quality_eeg_unit,args.quality_scale_to_volts,'quality source scale')
            quality = qm.EEGQualityV1(calibration=args.quality_calibration,assets_dir=args.assets_dir,
                       candidate=args.quality_candidate,scale_to_volts=qs,on_quality_error=args.on_quality_error,
                       stale_after_sec=args.stale_after_sec,unit_evidence=args.unit_evidence)
            protected.append(quality.calibration_path)
            require(not any(p in set(protected) for p in outputs),'Output cannot overwrite calibration JSON.')
            context = read_json(args.fusion_context) if args.fusion_context else None
            if context is not None and args.window_id is None and args.window_key is None:
                call['window_id'] = context.get('window_id')
            bridge = qm.AF4CEEGBridge.from_paths(args.fusion_script,args.f4_checkpoint,args.af4b_checkpoint,
                                                device=args.fusion_device) if all(bridge_fields) else None
            result = head.process_with_quality_array(x,quality_runtime=quality,context=context,bridge=bridge,
                                                      quality_scale_to_volts=qs,**call)
            result['emotion']['file_source'] = info
        else:
            result = head.predict_array(x,**call); result['file_source'] = info
        if args.preflight:
            result['preflight_scope'] = 'one explicitly supplied real window, not an accuracy benchmark'
    print_result(result)
    if args.output:
        atomic_json(args.output,{'result':result,'model_identity':head.identity},protected=protected)
        print('Output                    :',Path(args.output).resolve())
    if args.fusion_input_output:
        require(result.get('fusion_input') is not None,'Fusion input export requires --fusion-context.')
        atomic_json(args.fusion_input_output,result['fusion_input'],protected=protected)
    # Normal unavailability is a valid input state. Computation failure is NOT.
    e = result.get('emotion',result)
    return 2 if e.get('status') in ('ERROR','FAIL') or result.get('quality',{}).get('status') == 'ERROR' else 0


if __name__ == '__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('E4 inference interrupted; no training/models modified.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'E4 DEPLOYMENT ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2)
