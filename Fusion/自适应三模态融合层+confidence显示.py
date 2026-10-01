#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
EAV FINAL FUSION — Standalone AF4-C + Confidence
=================================================

This is the frozen, standalone final fusion implementation.

It integrates the inference/runtime functionality previously split across:
    1) F4 frozen MLP fusion
    2) AF4-B quality-aware robust adaptive fusion
    3) AF4-C quality-triggered routing
    4) Per-modality and final confidence display

It does NOT dynamically import the old F4 / AF4-B Python scripts.

Frozen system
-------------
Emotion classes:
    Neutral, Sadness, Anger, Happiness, Calmness

Healthy route:
    if all modalities are available AND
       q_EEG >= 0.80 AND q_Audio >= 0.80 AND q_Video >= 0.80
    -> use frozen F4

Degraded route:
    if any modality is unavailable OR any q < 0.80
    -> use frozen AF4-B

All modalities unavailable:
    -> NO_DECISION
    -> final probability = [0.2,0.2,0.2,0.2,0.2]

Confidence
----------
For every modality and final output:

    confidence = max(class probabilities)

Important:
    confidence != quality

Confidence is classifier certainty.
Quality is sensor/modality health used by the router.

External files still required
-----------------------------
A Python script cannot replace trained neural-network weights.

This standalone script therefore still needs only the two frozen checkpoints:

F4:
    stagef4_oof_decision_fusion_.../
        mlp_stacker/
            best_validation_selected.pt

AF4-B:
    af4b_quality_aware_robust_adaptive_f4_.../
        af4b_qbeta_200/
            best_validation_robustness.pt

For formal benchmark evaluation it additionally reads F1 Val/Test probability
arrays. Runtime prediction does NOT require F1/F3B data.

Modes
-----
1) Preflight only
    python .\final_fusion_af4c_confidence_standalone.py --preflight-only

2) Frozen formal evaluation
    python .\final_fusion_af4c_confidence_standalone.py --mode evaluate

3) Single/batch runtime prediction from JSON
    python .\final_fusion_af4c_confidence_standalone.py `
        --mode predict `
        --input-json .\runtime_input.json

Runtime JSON format
-------------------
Single sample:
{
  "eeg_probs":   [0.1,0.2,0.3,0.2,0.2],
  "audio_probs": [0.1,0.1,0.6,0.1,0.1],
  "video_probs": [0.1,0.1,0.2,0.5,0.1],
  "quality": {
    "eeg": 0.92,
    "audio": 0.45,
    "video": 0.88
  },
  "available": {
    "eeg": 1,
    "audio": 1,
    "video": 1
  }
}

Batch:
{
  "samples": [
    { ... },
    { ... }
  ]
}

Frozen reference result
-----------------------
F4 clean Test Accuracy       : 0.7683
AF4-B clean Test Accuracy    : 0.7650
AF4-C clean Test Accuracy    : 0.7683
F4 stress mean Macro-F1      : 0.5846
AF4-C stress mean Macro-F1   : 0.5983
AF4-C missing mean Macro-F1  : 0.5818
All-missing NO_DECISION      : 1.0000

The script verifies exact F4 clean Test reproduction when run in evaluate mode.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


# =============================================================================
# Frozen constants
# =============================================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMOTIONS = [
    "Neutral",
    "Sadness",
    "Anger",
    "Happiness",
    "Calmness",
]

MODALITIES = [
    "eeg",
    "audio",
    "video",
]

NUM_CLASSES = 5
NUM_MODALITIES = 3
INPUT_DIM = 15
F4_HIDDEN_DEFAULT = 32

# Frozen AF4-C quality threshold.
ROUTER_TAU = 0.80

# Frozen AF4-B selected candidate.
EXPECTED_AF4B_CANDIDATE = "af4b_qbeta_200"

DEFAULT_SEED = 20260915

EXPECTED_F4_TEST_ACCURACY = 461.0 / 600.0

EXPECTED_REFERENCE = {
    "f4_clean_test_accuracy": 0.7683333333333333,
    "af4b_clean_test_accuracy": 0.7650,
    "af4c_clean_test_accuracy": 0.7683333333333333,
    "f4_stress_mean_macro_f1": 0.5846,
    "af4c_stress_mean_macro_f1": 0.5983,
    "af4c_missing_mean_macro_f1": 0.5818,
    "all_missing_no_decision_rate": 1.0,
}


# =============================================================================
# Generic utilities
# =============================================================================

def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def safe_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def ensure_probs(
    name: str,
    probs: np.ndarray,
) -> None:
    probs = np.asarray(
        probs
    )

    if (
        probs.ndim != 2
        or probs.shape[1] != NUM_CLASSES
    ):
        raise RuntimeError(
            f"{name}: expected [N,5], got {probs.shape}"
        )

    if not np.isfinite(
        probs
    ).all():
        raise RuntimeError(
            f"{name}: NaN/Inf detected."
        )

    if np.min(
        probs
    ) < -1e-6:
        raise RuntimeError(
            f"{name}: negative probability detected."
        )

    if not np.allclose(
        probs.sum(
            axis=1
        ),
        1.0,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            f"{name}: probability rows do not sum to 1."
        )


def normalize_probability_vector(
    values: Any,
    name: str,
) -> np.ndarray:
    p = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(
        -1
    )

    if p.shape != (
        NUM_CLASSES,
    ):
        raise ValueError(
            f"{name} must have exactly 5 probabilities."
        )

    if not np.isfinite(
        p
    ).all():
        raise ValueError(
            f"{name} contains NaN/Inf."
        )

    if np.min(
        p
    ) < 0:
        raise ValueError(
            f"{name} contains negative values."
        )

    total = float(
        p.sum()
    )

    if total <= 0:
        raise ValueError(
            f"{name} sums to zero."
        )

    p = (
        p
        / total
    ).astype(
        np.float32
    )

    return p


def metric_dict(
    y: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(
        y,
        dtype=np.int64,
    )

    probs = np.asarray(
        probs,
        dtype=np.float64,
    )

    pred = probs.argmax(
        axis=1
    )

    top2 = np.argpartition(
        probs,
        -2,
        axis=1,
    )[:, -2:]

    true_p = np.clip(
        probs[
            np.arange(
                len(y)
            ),
            y,
        ],
        1e-12,
        1.0,
    )

    return {
        "n": int(
            len(y)
        ),
        "accuracy": float(
            accuracy_score(
                y,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "uar": float(
            recall_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "top2_accuracy": float(
            np.mean(
                [
                    int(
                        y[i]
                    )
                    in top2[i]
                    for i in range(
                        len(y)
                    )
                ]
            )
        ),
        "nll": float(
            -np.log(
                true_p
            ).mean()
        ),
    }


def normalize_subject(x: Any) -> str:
    match = re.search(
        r"(\d+)$",
        str(x).strip(),
    )

    if not match:
        raise ValueError(
            f"Cannot parse subject {x!r}"
        )

    return (
        f"subject{int(match.group(1)):02d}"
    )


# =============================================================================
# F1 frozen trial-probability loader
# =============================================================================

def load_trial_metadata(
    path: Path,
    expected_trials: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    required = [
        "subject",
        "pair_key",
        "label_id",
        "emotion",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path.name}: missing columns {missing}"
        )

    if len(
        df
    ) != expected_trials:
        raise RuntimeError(
            f"{path.name}: rows={len(df)}, expected={expected_trials}"
        )

    out = df.copy()

    out[
        "subject"
    ] = out[
        "subject"
    ].map(
        normalize_subject
    )

    out[
        "pair_key"
    ] = (
        out[
            "pair_key"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    out[
        "label_id"
    ] = pd.to_numeric(
        out[
            "label_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if out.duplicated(
        [
            "subject",
            "pair_key",
        ]
    ).any():
        raise RuntimeError(
            f"{path.name}: duplicate trial identity."
        )

    return out.reset_index(
        drop=True
    )


def aggregate_window_probs_to_trials(
    window_probs: np.ndarray,
    trial_window_rows: np.ndarray,
    trial_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    window_probs = np.asarray(
        window_probs,
        dtype=np.float32,
    )

    trial_window_rows = np.asarray(
        trial_window_rows,
        dtype=np.int64,
    )

    trial_labels = np.asarray(
        trial_labels,
        dtype=np.int64,
    )

    if (
        trial_window_rows.ndim != 2
        or trial_window_rows.shape[1] != 4
    ):
        raise RuntimeError(
            f"trial_window_rows shape={trial_window_rows.shape}"
        )

    if len(
        trial_window_rows
    ) != len(
        trial_labels
    ):
        raise RuntimeError(
            "trial rows/labels length mismatch."
        )

    trial_probs = window_probs[
        trial_window_rows
    ].mean(
        axis=1
    )

    trial_probs = (
        trial_probs
        / trial_probs.sum(
            axis=1,
            keepdims=True,
        )
    )

    ensure_probs(
        "trial_probs",
        trial_probs,
    )

    return (
        trial_probs.astype(
            np.float32
        ),
        trial_labels.astype(
            np.int64
        ),
    )


def load_f1_split_trial_data(
    f1_dir: Path,
    split: str,
) -> Dict[str, Any]:
    if split not in (
        "val",
        "test",
    ):
        raise ValueError(
            split
        )

    expected = {
        "val": {
            "trials": 600,
            "windows": 2400,
            "subjects": 6,
        },
        "test": {
            "trials": 600,
            "windows": 2400,
            "subjects": 6,
        },
    }[
        split
    ]

    root = (
        f1_dir
        / "arrays"
        / split
    )

    required = {
        "eeg": root / "eeg_probs.npy",
        "audio": root / "audio_probs.npy",
        "video": root / "video_probs.npy",
        "trial_rows": root / "trial_window_rows.npy",
        "trial_labels": root / "trial_labels.npy",
    }

    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    window_probs = {}

    for modality in MODALITIES:
        p = np.load(
            required[
                modality
            ],
            mmap_mode="r",
        )

        if p.shape != (
            expected[
                "windows"
            ],
            NUM_CLASSES,
        ):
            raise RuntimeError(
                f"F1 {split}/{modality} probs shape={p.shape}"
            )

        ensure_probs(
            f"F1 {split}/{modality}",
            np.asarray(
                p
            ),
        )

        window_probs[
            modality
        ] = p

    trial_rows = np.load(
        required[
            "trial_rows"
        ]
    )

    trial_labels = np.load(
        required[
            "trial_labels"
        ]
    )

    probs = {}
    labels_ref = None

    for modality in MODALITIES:
        trial_probs, y = (
            aggregate_window_probs_to_trials(
                np.asarray(
                    window_probs[
                        modality
                    ]
                ),
                trial_rows,
                trial_labels,
            )
        )

        probs[
            modality
        ] = trial_probs

        if labels_ref is None:
            labels_ref = y
        elif not np.array_equal(
            labels_ref,
            y,
        ):
            raise RuntimeError(
                f"{split}: labels inconsistent."
            )

    meta = load_trial_metadata(
        f1_dir
        / f"{split}_trial_manifest.csv",
        expected[
            "trials"
        ],
    )

    if not np.array_equal(
        meta[
            "label_id"
        ].to_numpy(
            np.int64
        ),
        labels_ref,
    ):
        raise RuntimeError(
            f"{split}: trial manifest labels mismatch."
        )

    if meta[
        "subject"
    ].nunique() != expected[
        "subjects"
    ]:
        raise RuntimeError(
            f"{split}: subject count mismatch."
        )

    x15 = np.concatenate(
        [
            probs[
                "eeg"
            ],
            probs[
                "audio"
            ],
            probs[
                "video"
            ],
        ],
        axis=1,
    ).astype(
        np.float32
    )

    return {
        "split": split,
        "meta": meta,
        "labels": labels_ref,
        "probs": probs,
        "x15": x15,
    }


# =============================================================================
# Source discovery
# =============================================================================

def latest_pass_f1(
    project_root: Path,
) -> Path:
    root = (
        project_root
        / "EAV_dataset"
        / "models"
        / "fusion"
    )

    candidates = []

    for p in root.glob(
        "stagef1_unified_frozen_features_*"
    ):
        summary = (
            p
            / "stagef1_summary.json"
        )

        if not summary.is_file():
            continue

        try:
            obj = read_json(
                summary
            )
        except Exception:
            continue

        if str(
            obj.get(
                "status",
                "",
            )
        ).upper() == "PASS":
            candidates.append(
                (
                    p.stat().st_mtime,
                    p,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "No PASS F1 directory found."
        )

    return max(
        candidates,
        key=lambda item: item[0],
    )[1].resolve()


def latest_frozen_f4(
    project_root: Path,
) -> Path:
    root = (
        project_root
        / "EAV_dataset"
        / "models"
        / "fusion"
    )

    candidates = []

    for p in root.glob(
        "stagef4_oof_decision_fusion_*"
    ):
        selection = (
            p
            / "validation_model_selection.json"
        )

        checkpoint = (
            p
            / "mlp_stacker"
            / "best_validation_selected.pt"
        )

        if not (
            selection.is_file()
            and checkpoint.is_file()
        ):
            continue

        try:
            obj = read_json(
                selection
            )
        except Exception:
            continue

        if str(
            obj.get(
                "status",
                "",
            )
        ).upper() != "FROZEN_BEFORE_TEST":
            continue

        if str(
            obj.get(
                "selected_candidate",
                "",
            )
        ) != "mlp_stacker":
            continue

        candidates.append(
            (
                p.stat().st_mtime,
                p,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No frozen F4 mlp_stacker result found."
        )

    return max(
        candidates,
        key=lambda item: item[0],
    )[1].resolve()


def latest_frozen_af4b(
    project_root: Path,
) -> Path:
    root = (
        project_root
        / "EAV_dataset"
        / "models"
        / "fusion"
    )

    candidates = []

    for p in root.glob(
        "af4b_quality_aware_robust_adaptive_f4_*"
    ):
        selection = (
            p
            / "validation_model_selection.json"
        )

        summary = (
            p
            / "af4b_summary.json"
        )

        if not (
            selection.is_file()
            and summary.is_file()
        ):
            continue

        try:
            sel = read_json(
                selection
            )
            summ = read_json(
                summary
            )
        except Exception:
            continue

        if str(
            sel.get(
                "status",
                "",
            )
        ).upper() != "FROZEN_BEFORE_TEST":
            continue

        if str(
            summ.get(
                "status",
                "",
            )
        ).upper() != "FROZEN":
            continue

        candidate = str(
            sel.get(
                "best_af4b_candidate",
                "",
            )
        )

        checkpoint = (
            p
            / candidate
            / "best_validation_robustness.pt"
        )

        if not checkpoint.is_file():
            continue

        candidates.append(
            (
                p.stat().st_mtime,
                p,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No frozen AF4-B result found."
        )

    return max(
        candidates,
        key=lambda item: item[0],
    )[1].resolve()


def resolve_model_sources(
    project_root: Path,
    f4_dir_arg: Optional[str],
    af4b_dir_arg: Optional[str],
) -> Tuple[
    Path,
    Path,
    Path,
    Path,
]:
    f4_dir = (
        Path(
            f4_dir_arg
        ).expanduser().resolve()
        if f4_dir_arg
        else latest_frozen_f4(
            project_root
        )
    )

    af4b_dir = (
        Path(
            af4b_dir_arg
        ).expanduser().resolve()
        if af4b_dir_arg
        else latest_frozen_af4b(
            project_root
        )
    )

    f4_checkpoint = (
        f4_dir
        / "mlp_stacker"
        / "best_validation_selected.pt"
    )

    selection = read_json(
        af4b_dir
        / "validation_model_selection.json"
    )

    af4b_candidate = str(
        selection.get(
            "best_af4b_candidate",
            "",
        )
    )

    if af4b_candidate != EXPECTED_AF4B_CANDIDATE:
        raise RuntimeError(
            "Frozen final system expects "
            f"{EXPECTED_AF4B_CANDIDATE}, got {af4b_candidate!r}."
        )

    af4b_checkpoint = (
        af4b_dir
        / af4b_candidate
        / "best_validation_robustness.pt"
    )

    for path in (
        f4_checkpoint,
        af4b_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    return (
        f4_dir,
        af4b_dir,
        f4_checkpoint,
        af4b_checkpoint,
    )


# =============================================================================
# Frozen F4
# =============================================================================

class FrozenF4Core(
    nn.Module
):
    def __init__(
        self,
        hidden: int,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(
            INPUT_DIM
        )

        self.fc1 = nn.Linear(
            INPUT_DIM,
            hidden,
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden,
            NUM_CLASSES,
        )

        for parameter in self.parameters():
            parameter.requires_grad_(
                False
            )

    def forward(
        self,
        x15: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.act(
            self.fc1(
                self.norm(
                    x15
                )
            )
        )

        logits = self.fc2(
            hidden
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        return {
            "hidden": hidden,
            "logits": logits,
            "probs": probs,
        }


def load_frozen_f4_core(
    checkpoint: Path,
    device: torch.device,
) -> Tuple[
    FrozenF4Core,
    Dict[str, Any],
]:
    ckpt = safe_load(
        checkpoint
    )

    if str(
        ckpt.get(
            "stage",
            "",
        )
    ).upper() != "F4":
        raise RuntimeError(
            "Checkpoint is not an F4 checkpoint."
        )

    if str(
        ckpt.get(
            "candidate",
            "",
        )
    ) != "mlp_stacker":
        raise RuntimeError(
            "Final system requires F4 mlp_stacker."
        )

    hidden = int(
        ckpt.get(
            "mlp_hidden",
            F4_HIDDEN_DEFAULT,
        )
    )

    model = FrozenF4Core(
        hidden=hidden
    )

    state = ckpt[
        "model_state_dict"
    ]

    mapping = {
        "norm.weight": "net.0.weight",
        "norm.bias": "net.0.bias",
        "fc1.weight": "net.1.weight",
        "fc1.bias": "net.1.bias",
        "fc2.weight": "net.4.weight",
        "fc2.bias": "net.4.bias",
    }

    with torch.no_grad():
        for destination, source in mapping.items():
            if source not in state:
                raise RuntimeError(
                    f"F4 checkpoint missing {source}"
                )

            module_name, parameter_name = (
                destination.split(
                    "."
                )
            )

            parameter = getattr(
                getattr(
                    model,
                    module_name,
                ),
                parameter_name,
            )

            parameter.copy_(
                state[
                    source
                ]
            )

    model = model.to(
        device
    ).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    return (
        model,
        ckpt,
    )


@torch.inference_mode()
def infer_f4_core(
    model: FrozenF4Core,
    x15: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    outputs = []

    for start in range(
        0,
        len(x15),
        batch_size,
    ):
        xb = torch.from_numpy(
            np.asarray(
                x15[
                    start:
                    start
                    + batch_size
                ],
                dtype=np.float32,
            )
        ).to(
            device
        )

        outputs.append(
            model(
                xb
            )[
                "probs"
            ]
            .float()
            .cpu()
            .numpy()
        )

    probs = np.concatenate(
        outputs,
        axis=0,
    ).astype(
        np.float32
    )

    ensure_probs(
        "F4 output",
        probs,
    )

    return probs


# =============================================================================
# Frozen AF4-B
# =============================================================================

class AdaptiveF4B(
    nn.Module
):
    def __init__(
        self,
        frozen_f4: FrozenF4Core,
        hidden: int,
        quality_beta: float,
        residual_bias_init: float = -3.0,
    ) -> None:
        super().__init__()

        self.f4 = frozen_f4
        self.hidden = int(
            hidden
        )
        self.quality_beta = float(
            quality_beta
        )

        context_dim = (
            hidden
            + NUM_MODALITIES
            + NUM_MODALITIES
        )

        self.modality_gate = nn.Linear(
            context_dim,
            NUM_MODALITIES,
        )

        self.class_gate = nn.Linear(
            context_dim,
            NUM_MODALITIES
            * NUM_CLASSES,
        )

        self.residual_gate = nn.Linear(
            context_dim,
            1,
        )

        nn.init.zeros_(
            self.modality_gate.weight
        )
        nn.init.zeros_(
            self.modality_gate.bias
        )

        nn.init.zeros_(
            self.class_gate.weight
        )
        nn.init.zeros_(
            self.class_gate.bias
        )

        nn.init.zeros_(
            self.residual_gate.weight
        )
        nn.init.constant_(
            self.residual_gate.bias,
            residual_bias_init,
        )

        for parameter in self.f4.parameters():
            parameter.requires_grad_(
                False
            )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(
            mode
        )

        self.f4.eval()

        return self

    def forward(
        self,
        x15: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if (
            x15.ndim != 2
            or x15.shape[1] != INPUT_DIM
        ):
            raise RuntimeError(
                f"x15 must be [B,15], got {tuple(x15.shape)}"
            )

        if quality.shape != (
            len(x15),
            NUM_MODALITIES,
        ):
            raise RuntimeError(
                f"quality shape={tuple(quality.shape)}"
            )

        if mask.shape != (
            len(x15),
            NUM_MODALITIES,
        ):
            raise RuntimeError(
                f"mask shape={tuple(mask.shape)}"
            )

        quality = torch.clamp(
            quality,
            0.0,
            1.0,
        )

        mask = (
            mask
            > 0.5
        ).to(
            x15.dtype
        )

        quality = (
            quality
            * mask
        )

        modality_probs = x15.view(
            -1,
            NUM_MODALITIES,
            NUM_CLASSES,
        )

        with torch.no_grad():
            f4_out = self.f4(
                x15
            )

            hidden = f4_out[
                "hidden"
            ]

            p_f4 = f4_out[
                "probs"
            ]

        context = torch.cat(
            [
                hidden,
                quality,
                mask,
            ],
            dim=1,
        )

        eps = 1e-6

        raw_modality_logits = self.modality_gate(
            context
        )

        quality_prior = (
            self.quality_beta
            * torch.log(
                quality.clamp_min(
                    eps
                )
            )
        )

        modality_logits = (
            raw_modality_logits
            + quality_prior
        )

        available = (
            mask
            > 0.5
        )

        available_count = available.sum(
            dim=1
        )

        all_missing = (
            available_count
            == 0
        )

        safe_modality_logits = (
            modality_logits.masked_fill(
                ~available,
                -1e4,
            )
        )

        if all_missing.any():
            safe_modality_logits = (
                safe_modality_logits.clone()
            )

            safe_modality_logits[
                all_missing
            ] = 0.0

        alpha = torch.softmax(
            safe_modality_logits,
            dim=1,
        )

        alpha = (
            alpha
            * mask
        )

        alpha_sum = alpha.sum(
            dim=1,
            keepdim=True,
        )

        alpha = torch.where(
            alpha_sum
            > 0,
            alpha
            / alpha_sum.clamp_min(
                eps
            ),
            torch.full_like(
                alpha,
                1.0
                / NUM_MODALITIES,
            ),
        )

        class_logits = self.class_gate(
            context
        ).view(
            -1,
            NUM_MODALITIES,
            NUM_CLASSES,
        )

        class_reliability = torch.sigmoid(
            class_logits
        )

        effective_logits = (
            torch.log(
                alpha[
                    :,
                    :,
                    None,
                ].clamp_min(
                    eps
                )
            )
            + torch.log(
                class_reliability.clamp_min(
                    eps
                )
            )
        )

        class_available = available[
            :,
            :,
            None,
        ].expand(
            -1,
            -1,
            NUM_CLASSES,
        )

        effective_logits = (
            effective_logits.masked_fill(
                ~class_available,
                -1e4,
            )
        )

        if all_missing.any():
            effective_logits = (
                effective_logits.clone()
            )

            effective_logits[
                all_missing
            ] = 0.0

        effective_weights = torch.softmax(
            effective_logits,
            dim=1,
        )

        effective_weights = (
            effective_weights
            * class_available.to(
                effective_weights.dtype
            )
        )

        weight_sum = effective_weights.sum(
            dim=1,
            keepdim=True,
        )

        effective_weights = torch.where(
            weight_sum
            > 0,
            effective_weights
            / weight_sum.clamp_min(
                eps
            ),
            torch.full_like(
                effective_weights,
                1.0
                / NUM_MODALITIES,
            ),
        )

        adaptive_score = (
            effective_weights
            * modality_probs
        ).sum(
            dim=1
        )

        adaptive_probs = (
            adaptive_score
            / adaptive_score.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(
                eps
            )
        )

        gamma = torch.sigmoid(
            self.residual_gate(
                context
            ).squeeze(
                -1
            )
        )

        log_final = (
            (
                1.0
                - gamma[
                    :,
                    None,
                ]
            )
            * torch.log(
                p_f4.clamp_min(
                    eps
                )
            )
            + gamma[
                :,
                None,
            ]
            * torch.log(
                adaptive_probs.clamp_min(
                    eps
                )
            )
        )

        final_probs = torch.softmax(
            log_final,
            dim=1,
        )

        no_decision = all_missing.to(
            final_probs.dtype
        )

        if all_missing.any():
            uniform = torch.full(
                (
                    int(
                        all_missing.sum().item()
                    ),
                    NUM_CLASSES,
                ),
                1.0
                / NUM_CLASSES,
                device=final_probs.device,
                dtype=final_probs.dtype,
            )

            final_probs = (
                final_probs.clone()
            )

            adaptive_probs = (
                adaptive_probs.clone()
            )

            gamma = gamma.clone()

            final_probs[
                all_missing
            ] = uniform

            adaptive_probs[
                all_missing
            ] = uniform

            gamma[
                all_missing
            ] = 1.0

        return {
            "f4_probs": p_f4,
            "adaptive_probs": adaptive_probs,
            "final_probs": final_probs,
            "alpha": alpha,
            "class_reliability": class_reliability,
            "effective_weights": effective_weights,
            "gamma": gamma,
            "no_decision": no_decision,
            "available_count": available_count.to(
                final_probs.dtype
            ),
        }


def load_af4b_checkpoint(
    checkpoint: Path,
    f4_checkpoint: Path,
    device: torch.device,
) -> AdaptiveF4B:
    ckpt = safe_load(
        checkpoint
    )

    if str(
        ckpt.get(
            "stage",
            "",
        )
    ).upper() != "AF4-B":
        raise RuntimeError(
            "Checkpoint is not AF4-B."
        )

    if str(
        ckpt.get(
            "candidate",
            "",
        )
    ) != EXPECTED_AF4B_CANDIDATE:
        raise RuntimeError(
            "Final system requires "
            f"{EXPECTED_AF4B_CANDIDATE}."
        )

    core, _ = load_frozen_f4_core(
        f4_checkpoint,
        device,
    )

    model = AdaptiveF4B(
        frozen_f4=core,
        hidden=int(
            ckpt[
                "f4_hidden"
            ]
        ),
        quality_beta=float(
            ckpt[
                "quality_beta"
            ]
        ),
        residual_bias_init=float(
            ckpt[
                "residual_bias_init"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    return model


@torch.inference_mode()
def infer_af4b(
    model: AdaptiveF4B,
    x15: np.ndarray,
    quality: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model.eval()

    keys = [
        "f4_probs",
        "adaptive_probs",
        "final_probs",
        "alpha",
        "class_reliability",
        "effective_weights",
        "gamma",
        "no_decision",
        "available_count",
    ]

    parts = {
        key: []
        for key in keys
    }

    for start in range(
        0,
        len(x15),
        batch_size,
    ):
        xb = torch.from_numpy(
            np.asarray(
                x15[
                    start:
                    start
                    + batch_size
                ],
                dtype=np.float32,
            )
        ).to(
            device
        )

        qb = torch.from_numpy(
            np.asarray(
                quality[
                    start:
                    start
                    + batch_size
                ],
                dtype=np.float32,
            )
        ).to(
            device
        )

        mb = torch.from_numpy(
            np.asarray(
                mask[
                    start:
                    start
                    + batch_size
                ],
                dtype=np.float32,
            )
        ).to(
            device
        )

        out = model(
            xb,
            qb,
            mb,
        )

        for key in keys:
            parts[
                key
            ].append(
                out[
                    key
                ]
                .float()
                .cpu()
                .numpy()
            )

    result = {
        key: np.concatenate(
            values,
            axis=0,
        ).astype(
            np.float32
        )
        for key, values
        in parts.items()
    }

    ensure_probs(
        "AF4-B final",
        result[
            "final_probs"
        ],
    )

    return result


# =============================================================================
# Confidence
# =============================================================================

def modality_predictions_and_confidence(
    x15: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    x15 = np.asarray(
        x15,
        dtype=np.float32,
    )

    if (
        x15.ndim != 2
        or x15.shape[1] != INPUT_DIM
    ):
        raise RuntimeError(
            f"Expected [N,15], got {x15.shape}"
        )

    probs3 = x15.reshape(
        -1,
        NUM_MODALITIES,
        NUM_CLASSES,
    )

    result = {}

    for m, modality in enumerate(
        MODALITIES
    ):
        probs = probs3[
            :,
            m,
            :,
        ]

        pred = probs.argmax(
            axis=1
        )

        result[
            modality
        ] = {
            "probs": probs.astype(
                np.float32
            ),
            "pred_label_id": pred.astype(
                np.int64
            ),
            "pred_emotion": np.asarray(
                [
                    EMOTIONS[
                        int(i)
                    ]
                    for i in pred
                ],
                dtype=object,
            ),
            "confidence": probs.max(
                axis=1
            ).astype(
                np.float32
            ),
        }

    return result


def final_prediction_and_confidence(
    probs: np.ndarray,
) -> Dict[str, np.ndarray]:
    ensure_probs(
        "final confidence input",
        probs,
    )

    probs = np.asarray(
        probs,
        dtype=np.float32,
    )

    pred = probs.argmax(
        axis=1
    )

    return {
        "pred_label_id": pred.astype(
            np.int64
        ),
        "pred_emotion": np.asarray(
            [
                EMOTIONS[
                    int(i)
                ]
                for i in pred
            ],
            dtype=object,
        ),
        "confidence": probs.max(
            axis=1
        ).astype(
            np.float32
        ),
    }


# =============================================================================
# AF4-C router
# =============================================================================

def compute_degraded_mask(
    quality: np.ndarray,
    mask: np.ndarray,
    tau: float = ROUTER_TAU,
) -> np.ndarray:
    quality = np.asarray(
        quality,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    if quality.shape != mask.shape:
        raise RuntimeError(
            "quality/mask shape mismatch."
        )

    if (
        quality.ndim != 2
        or quality.shape[1] != NUM_MODALITIES
    ):
        raise RuntimeError(
            f"Expected quality/mask [N,3], got {quality.shape}"
        )

    any_unavailable = (
        mask
        < 0.5
    ).any(
        axis=1
    )

    low_quality = (
        quality.min(
            axis=1
        )
        < float(
            tau
        )
    )

    return (
        any_unavailable
        | low_quality
    )


def route_predictions(
    f4_probs: np.ndarray,
    af4b_output: Dict[str, np.ndarray],
    quality: np.ndarray,
    mask: np.ndarray,
    tau: float = ROUTER_TAU,
) -> Dict[str, np.ndarray]:
    ensure_probs(
        "router F4",
        f4_probs,
    )

    ensure_probs(
        "router AF4-B",
        af4b_output[
            "final_probs"
        ],
    )

    degraded = compute_degraded_mask(
        quality,
        mask,
        tau,
    )

    final_probs = np.asarray(
        f4_probs,
        dtype=np.float32,
    ).copy()

    final_probs[
        degraded
    ] = af4b_output[
        "final_probs"
    ][
        degraded
    ]

    no_decision = np.zeros(
        len(
            final_probs
        ),
        dtype=np.float32,
    )

    no_decision[
        degraded
    ] = af4b_output[
        "no_decision"
    ][
        degraded
    ]

    ensure_probs(
        "router final",
        final_probs,
    )

    return {
        "final_probs": final_probs,
        "use_af4b": degraded.astype(
            np.float32
        ),
        "use_f4": (
            ~degraded
        ).astype(
            np.float32
        ),
        "no_decision": no_decision,
    }


# =============================================================================
# Final system class
# =============================================================================

class FinalAF4CSystem:
    """
    Standalone frozen F4-C inference wrapper.

    Input:
        EEG 5-class probability
        Audio 5-class probability
        Video 5-class probability
        quality [q_eeg, q_audio, q_video]
        availability mask [m_eeg, m_audio, m_video]

    Output:
        per-modality prediction/confidence
        route/system_state
        F4 output
        AF4-B output
        final prediction/confidence
        AF4-B alpha/gamma when applicable
        NO_DECISION flag
    """

    def __init__(
        self,
        f4_checkpoint: Path,
        af4b_checkpoint: Path,
        device: torch.device,
        batch_size: int = 1024,
        tau: float = ROUTER_TAU,
    ) -> None:
        if not math.isclose(
            float(tau),
            ROUTER_TAU,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "F4-C is frozen at tau=0.80. "
                "Do not change tau in the final system."
            )

        self.device = device
        self.batch_size = int(
            batch_size
        )
        self.tau = float(
            tau
        )

        self.f4_checkpoint = Path(
            f4_checkpoint
        ).resolve()

        self.af4b_checkpoint = Path(
            af4b_checkpoint
        ).resolve()

        self.f4, self.f4_ckpt = (
            load_frozen_f4_core(
                self.f4_checkpoint,
                self.device,
            )
        )

        self.af4b = (
            load_af4b_checkpoint(
                self.af4b_checkpoint,
                self.f4_checkpoint,
                self.device,
            )
        )

    def predict_batch(
        self,
        x15: np.ndarray,
        quality: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        x15 = np.asarray(
            x15,
            dtype=np.float32,
        )

        quality = np.asarray(
            quality,
            dtype=np.float32,
        )

        mask = np.asarray(
            mask,
            dtype=np.float32,
        )

        if (
            x15.ndim != 2
            or x15.shape[1] != INPUT_DIM
        ):
            raise RuntimeError(
                f"x15 must be [N,15], got {x15.shape}"
            )

        if quality.shape != (
            len(x15),
            NUM_MODALITIES,
        ):
            raise RuntimeError(
                f"quality must be [N,3], got {quality.shape}"
            )

        if mask.shape != (
            len(x15),
            NUM_MODALITIES,
        ):
            raise RuntimeError(
                f"mask must be [N,3], got {mask.shape}"
            )

        quality = np.clip(
            quality,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        mask = (
            mask
            > 0.5
        ).astype(
            np.float32
        )

        f4_probs = infer_f4_core(
            self.f4,
            x15,
            self.device,
            self.batch_size,
        )

        af4b_output = infer_af4b(
            self.af4b,
            x15,
            quality,
            mask,
            self.device,
            self.batch_size,
        )

        routed = route_predictions(
            f4_probs,
            af4b_output,
            quality,
            mask,
            self.tau,
        )

        modality_info = (
            modality_predictions_and_confidence(
                x15
            )
        )

        f4_info = (
            final_prediction_and_confidence(
                f4_probs
            )
        )

        af4b_info = (
            final_prediction_and_confidence(
                af4b_output[
                    "final_probs"
                ]
            )
        )

        final_info = (
            final_prediction_and_confidence(
                routed[
                    "final_probs"
                ]
            )
        )

        route = np.where(
            routed[
                "use_af4b"
            ]
            > 0.5,
            "AF4-B",
            "F4",
        )

        all_missing = (
            mask.sum(
                axis=1
            )
            == 0
        )

        system_state = np.where(
            all_missing,
            "NO_DECISION",
            np.where(
                routed[
                    "use_af4b"
                ]
                > 0.5,
                "DEGRADED",
                "HEALTHY",
            ),
        )

        return {
            "x15": x15,
            "quality": quality,
            "mask": mask,
            "modality": modality_info,
            "f4_probs": f4_probs,
            "f4_info": f4_info,
            "af4b_output": af4b_output,
            "af4b_info": af4b_info,
            "router_output": routed,
            "final_info": final_info,
            "route": route,
            "system_state": system_state,
        }

    def predict_one(
        self,
        eeg_probs: Any,
        audio_probs: Any,
        video_probs: Any,
        q_eeg: float,
        q_audio: float,
        q_video: float,
        eeg_available: bool = True,
        audio_available: bool = True,
        video_available: bool = True,
    ) -> Dict[str, Any]:
        eeg = normalize_probability_vector(
            eeg_probs,
            "eeg_probs",
        )

        audio = normalize_probability_vector(
            audio_probs,
            "audio_probs",
        )

        video = normalize_probability_vector(
            video_probs,
            "video_probs",
        )

        x15 = np.concatenate(
            [
                eeg,
                audio,
                video,
            ],
            axis=0,
        )[
            None,
            :
        ].astype(
            np.float32
        )

        quality = np.asarray(
            [
                [
                    q_eeg,
                    q_audio,
                    q_video,
                ]
            ],
            dtype=np.float32,
        )

        mask = np.asarray(
            [
                [
                    int(
                        eeg_available
                    ),
                    int(
                        audio_available
                    ),
                    int(
                        video_available
                    ),
                ]
            ],
            dtype=np.float32,
        )

        result = self.predict_batch(
            x15,
            quality,
            mask,
        )

        return result_to_serializable_sample(
            result,
            0,
        )


# =============================================================================
# Runtime JSON
# =============================================================================

def sample_from_json(
    obj: Dict[str, Any],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    eeg = normalize_probability_vector(
        obj[
            "eeg_probs"
        ],
        "eeg_probs",
    )

    audio = normalize_probability_vector(
        obj[
            "audio_probs"
        ],
        "audio_probs",
    )

    video = normalize_probability_vector(
        obj[
            "video_probs"
        ],
        "video_probs",
    )

    quality_obj = obj.get(
        "quality",
        {}
    )

    available_obj = obj.get(
        "available",
        {}
    )

    quality = np.asarray(
        [
            float(
                quality_obj.get(
                    "eeg",
                    1.0,
                )
            ),
            float(
                quality_obj.get(
                    "audio",
                    1.0,
                )
            ),
            float(
                quality_obj.get(
                    "video",
                    1.0,
                )
            ),
        ],
        dtype=np.float32,
    )

    mask = np.asarray(
        [
            float(
                available_obj.get(
                    "eeg",
                    1
                )
            ),
            float(
                available_obj.get(
                    "audio",
                    1
                )
            ),
            float(
                available_obj.get(
                    "video",
                    1
                )
            ),
        ],
        dtype=np.float32,
    )

    x15 = np.concatenate(
        [
            eeg,
            audio,
            video,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    return (
        x15,
        quality,
        mask,
    )


def load_runtime_json(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    obj = read_json(
        path
    )

    if "samples" in obj:
        samples = obj[
            "samples"
        ]

        if not isinstance(
            samples,
            list,
        ) or not samples:
            raise ValueError(
                "'samples' must be a non-empty list."
            )
    else:
        samples = [
            obj
        ]

    x_list = []
    q_list = []
    m_list = []

    for sample in samples:
        x15, quality, mask = sample_from_json(
            sample
        )

        x_list.append(
            x15
        )

        q_list.append(
            quality
        )

        m_list.append(
            mask
        )

    return (
        np.stack(
            x_list,
            axis=0,
        ).astype(
            np.float32
        ),
        np.stack(
            q_list,
            axis=0,
        ).astype(
            np.float32
        ),
        np.stack(
            m_list,
            axis=0,
        ).astype(
            np.float32
        ),
    )


def result_to_serializable_sample(
    result: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    no_decision = bool(
        result[
            "router_output"
        ][
            "no_decision"
        ][
            index
        ]
        > 0.5
    )

    output = {
        "system_state": str(
            result[
                "system_state"
            ][
                index
            ]
        ),
        "route": str(
            result[
                "route"
            ][
                index
            ]
        ),
        "router_tau": float(
            ROUTER_TAU
        ),
        "no_decision": no_decision,
        "modalities": {},
        "f4": {
            "emotion": str(
                result[
                    "f4_info"
                ][
                    "pred_emotion"
                ][
                    index
                ]
            ),
            "confidence": float(
                result[
                    "f4_info"
                ][
                    "confidence"
                ][
                    index
                ]
            ),
            "probabilities": (
                result[
                    "f4_probs"
                ][
                    index
                ]
                .astype(
                    float
                )
                .tolist()
            ),
        },
        "af4b": {
            "emotion": str(
                result[
                    "af4b_info"
                ][
                    "pred_emotion"
                ][
                    index
                ]
            ),
            "confidence": float(
                result[
                    "af4b_info"
                ][
                    "confidence"
                ][
                    index
                ]
            ),
            "gamma": float(
                result[
                    "af4b_output"
                ][
                    "gamma"
                ][
                    index
                ]
            ),
            "alpha": {
                modality: float(
                    result[
                        "af4b_output"
                    ][
                        "alpha"
                    ][
                        index,
                        m
                    ]
                )
                for m, modality
                in enumerate(
                    MODALITIES
                )
            },
            "probabilities": (
                result[
                    "af4b_output"
                ][
                    "final_probs"
                ][
                    index
                ]
                .astype(
                    float
                )
                .tolist()
            ),
        },
    }

    for m, modality in enumerate(
        MODALITIES
    ):
        info = result[
            "modality"
        ][
            modality
        ]

        output[
            "modalities"
        ][
            modality
        ] = {
            "emotion": str(
                info[
                    "pred_emotion"
                ][
                    index
                ]
            ),
            "confidence": float(
                info[
                    "confidence"
                ][
                    index
                ]
            ),
            "quality": float(
                result[
                    "quality"
                ][
                    index,
                    m
                ]
            ),
            "available": bool(
                result[
                    "mask"
                ][
                    index,
                    m
                ]
                > 0.5
            ),
            "probabilities": (
                info[
                    "probs"
                ][
                    index
                ]
                .astype(
                    float
                )
                .tolist()
            ),
        }

    if no_decision:
        output[
            "final"
        ] = {
            "emotion": "NO_DECISION",
            "confidence": None,
            "probabilities": (
                result[
                    "router_output"
                ][
                    "final_probs"
                ][
                    index
                ]
                .astype(
                    float
                )
                .tolist()
            ),
        }
    else:
        output[
            "final"
        ] = {
            "emotion": str(
                result[
                    "final_info"
                ][
                    "pred_emotion"
                ][
                    index
                ]
            ),
            "confidence": float(
                result[
                    "final_info"
                ][
                    "confidence"
                ][
                    index
                ]
            ),
            "probabilities": (
                result[
                    "router_output"
                ][
                    "final_probs"
                ][
                    index
                ]
                .astype(
                    float
                )
                .tolist()
            ),
        }

    return output


# =============================================================================
# Synthetic robustness suite — retained only for frozen benchmark reproduction
# =============================================================================

def scenario_seed(
    base_seed: int,
    scenario_name: str,
) -> int:
    crc = zlib.crc32(
        scenario_name.encode(
            "utf-8"
        )
    )

    return int(
        (
            base_seed
            + crc
        )
        % (
            2**32
            - 1
        )
    )


def normalize_rows(
    x: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = np.clip(
        x,
        1e-12,
        None,
    )

    x = (
        x
        / x.sum(
            axis=-1,
            keepdims=True,
        )
    )

    return x.astype(
        np.float32
    )


def uniform_probability_vector() -> np.ndarray:
    return np.full(
        NUM_CLASSES,
        1.0
        / NUM_CLASSES,
        dtype=np.float32,
    )


def random_simplex(
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.dirichlet(
        np.ones(
            NUM_CLASSES,
            dtype=np.float64,
        )
    ).astype(
        np.float32
    )


def blend_with_random_probability(
    p: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    random_p = random_simplex(
        rng
    )

    out = (
        (
            1.0
            - float(
                epsilon
            )
        )
        * p
        + float(
            epsilon
        )
        * random_p
    )

    return normalize_rows(
        out[
            None,
            :
        ]
    )[0]


def overconfident_wrong_probability(
    true_label: int,
    rng: np.random.Generator,
    confidence: float = 0.90,
) -> np.ndarray:
    wrong_labels = [
        c
        for c in range(
            NUM_CLASSES
        )
        if c != int(
            true_label
        )
    ]

    target = int(
        rng.choice(
            wrong_labels
        )
    )

    remainder = (
        1.0
        - float(
            confidence
        )
    )

    p = np.full(
        NUM_CLASSES,
        remainder
        / (
            NUM_CLASSES
            - 1
        ),
        dtype=np.float32,
    )

    p[
        target
    ] = float(
        confidence
    )

    return p


def build_fixed_scenario(
    x15: np.ndarray,
    labels: np.ndarray,
    scenario: str,
    seed: int,
) -> Dict[str, Any]:
    probs = np.asarray(
        x15,
        dtype=np.float32,
    ).reshape(
        -1,
        NUM_MODALITIES,
        NUM_CLASSES,
    ).copy()

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    n = len(
        labels
    )

    quality = np.ones(
        (
            n,
            NUM_MODALITIES,
        ),
        dtype=np.float32,
    )

    mask = np.ones(
        (
            n,
            NUM_MODALITIES,
        ),
        dtype=np.float32,
    )

    rng = np.random.default_rng(
        scenario_seed(
            seed,
            scenario,
        )
    )

    if scenario == "clean":
        pass

    elif scenario == "all_missing":
        probs[
            :,
            :,
            :
        ] = (
            1.0
            / NUM_CLASSES
        )

        quality[
            :,
            :
        ] = 0.0

        mask[
            :,
            :
        ] = 0.0

    elif scenario.endswith(
        "_mild_noise_025"
    ):
        modality = scenario.split(
            "_"
        )[0]

        m = MODALITIES.index(
            modality
        )

        epsilon = 0.25

        for i in range(
            n
        ):
            probs[
                i,
                m
            ] = blend_with_random_probability(
                probs[
                    i,
                    m
                ],
                epsilon,
                rng,
            )

        quality[
            :,
            m
        ] = 0.75

    elif scenario.endswith(
        "_severe_noise_060"
    ):
        modality = scenario.split(
            "_"
        )[0]

        m = MODALITIES.index(
            modality
        )

        epsilon = 0.60

        for i in range(
            n
        ):
            probs[
                i,
                m
            ] = blend_with_random_probability(
                probs[
                    i,
                    m
                ],
                epsilon,
                rng,
            )

        quality[
            :,
            m
        ] = 0.40

    elif (
        scenario.endswith(
            "_missing"
        )
        and scenario.count(
            "_"
        ) == 1
    ):
        modality = scenario.split(
            "_"
        )[0]

        m = MODALITIES.index(
            modality
        )

        probs[
            :,
            m,
            :
        ] = uniform_probability_vector()

        quality[
            :,
            m
        ] = 0.0

        mask[
            :,
            m
        ] = 0.0

    elif scenario.endswith(
        "_overconf_wrong"
    ):
        modality = scenario.split(
            "_"
        )[0]

        m = MODALITIES.index(
            modality
        )

        for i in range(
            n
        ):
            probs[
                i,
                m
            ] = overconfident_wrong_probability(
                labels[
                    i
                ],
                rng,
                confidence=0.90,
            )

        quality[
            :,
            m
        ] = 0.15

    elif scenario.endswith(
        "_desync"
    ):
        modality = scenario.split(
            "_"
        )[0]

        m = MODALITIES.index(
            modality
        )

        if n > 1:
            order = np.arange(
                n
            )

            shift = int(
                rng.integers(
                    1,
                    n,
                )
            )

            source = np.roll(
                order,
                shift,
            )

            probs[
                :,
                m,
                :
            ] = probs[
                source,
                m,
                :
            ]

        quality[
            :,
            m
        ] = 0.35

    elif scenario in (
        "eeg_audio_missing",
        "eeg_video_missing",
        "audio_video_missing",
    ):
        pair = (
            scenario
            .replace(
                "_missing",
                "",
            )
            .split(
                "_"
            )
        )

        for modality in pair:
            m = MODALITIES.index(
                modality
            )

            probs[
                :,
                m,
                :
            ] = uniform_probability_vector()

            quality[
                :,
                m
            ] = 0.0

            mask[
                :,
                m
            ] = 0.0

    else:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    return {
        "scenario": scenario,
        "x15": probs.reshape(
            -1,
            INPUT_DIM,
        ).astype(
            np.float32
        ),
        "quality": quality,
        "mask": mask,
        "labels": labels,
    }


def scored_stress_scenarios() -> List[str]:
    scenarios = [
        "clean",
    ]

    for modality in MODALITIES:
        scenarios.append(
            f"{modality}_mild_noise_025"
        )

    for modality in MODALITIES:
        scenarios.append(
            f"{modality}_severe_noise_060"
        )

    for modality in MODALITIES:
        scenarios.append(
            f"{modality}_missing"
        )

    for modality in MODALITIES:
        scenarios.append(
            f"{modality}_overconf_wrong"
        )

    for modality in MODALITIES:
        scenarios.append(
            f"{modality}_desync"
        )

    scenarios.extend(
        [
            "eeg_audio_missing",
            "eeg_video_missing",
            "audio_video_missing",
        ]
    )

    return scenarios


def build_stress_suite(
    x15: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    return {
        scenario: build_fixed_scenario(
            x15,
            labels,
            scenario,
            seed,
        )
        for scenario
        in scored_stress_scenarios()
    }


def aggregate_suite_rows(
    rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    by_name = {
        row[
            "scenario"
        ]: row
        for row in rows
    }

    clean = by_name[
        "clean"
    ]

    stress = [
        row
        for row in rows
        if row[
            "scenario"
        ] != "clean"
    ]

    macro = np.asarray(
        [
            row[
                "macro_f1"
            ]
            for row in stress
        ],
        dtype=np.float64,
    )

    accuracy = np.asarray(
        [
            row[
                "accuracy"
            ]
            for row in stress
        ],
        dtype=np.float64,
    )

    nll = np.asarray(
        [
            row[
                "nll"
            ]
            for row in stress
        ],
        dtype=np.float64,
    )

    missing = [
        row
        for row in stress
        if "missing" in row[
            "scenario"
        ]
    ]

    return {
        "clean_accuracy": float(
            clean[
                "accuracy"
            ]
        ),
        "clean_macro_f1": float(
            clean[
                "macro_f1"
            ]
        ),
        "clean_nll": float(
            clean[
                "nll"
            ]
        ),
        "stress_mean_accuracy": float(
            accuracy.mean()
        ),
        "stress_mean_macro_f1": float(
            macro.mean()
        ),
        "stress_worst_macro_f1": float(
            macro.min()
        ),
        "stress_mean_nll": float(
            nll.mean()
        ),
        "missing_mean_macro_f1": float(
            np.mean(
                [
                    row[
                        "macro_f1"
                    ]
                    for row in missing
                ]
            )
        ),
    }


def evaluate_suite(
    system: FinalAF4CSystem,
    suite: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rows = []
    outputs = {}

    for scenario, data in suite.items():
        result = system.predict_batch(
            data[
                "x15"
            ],
            data[
                "quality"
            ],
            data[
                "mask"
            ],
        )

        final_probs = result[
            "router_output"
        ][
            "final_probs"
        ]

        metrics = metric_dict(
            data[
                "labels"
            ],
            final_probs,
        )

        modality_info = result[
            "modality"
        ]

        final_info = result[
            "final_info"
        ]

        row = {
            "scenario": scenario,
            **metrics,
            "route_to_af4b_rate": float(
                result[
                    "router_output"
                ][
                    "use_af4b"
                ].mean()
            ),
            "route_to_f4_rate": float(
                result[
                    "router_output"
                ][
                    "use_f4"
                ].mean()
            ),
            "no_decision_rate": float(
                result[
                    "router_output"
                ][
                    "no_decision"
                ].mean()
            ),
            "mean_eeg_confidence": float(
                modality_info[
                    "eeg"
                ][
                    "confidence"
                ].mean()
            ),
            "mean_audio_confidence": float(
                modality_info[
                    "audio"
                ][
                    "confidence"
                ].mean()
            ),
            "mean_video_confidence": float(
                modality_info[
                    "video"
                ][
                    "confidence"
                ].mean()
            ),
            "mean_final_confidence": float(
                final_info[
                    "confidence"
                ].mean()
            ),
            "mean_quality_eeg": float(
                data[
                    "quality"
                ][
                    :,
                    0
                ].mean()
            ),
            "mean_quality_audio": float(
                data[
                    "quality"
                ][
                    :,
                    1
                ].mean()
            ),
            "mean_quality_video": float(
                data[
                    "quality"
                ][
                    :,
                    2
                ].mean()
            ),
        }

        rows.append(
            row
        )

        outputs[
            scenario
        ] = result

    aggregate = aggregate_suite_rows(
        rows
    )

    stress_rows = [
        row
        for row in rows
        if row[
            "scenario"
        ] != "clean"
    ]

    aggregate[
        "stress_mean_route_to_af4b_rate"
    ] = float(
        np.mean(
            [
                row[
                    "route_to_af4b_rate"
                ]
                for row in stress_rows
            ]
        )
    )

    aggregate[
        "clean_route_to_af4b_rate"
    ] = float(
        by_scenario(
            rows
        )[
            "clean"
        ][
            "route_to_af4b_rate"
        ]
    )

    return {
        "rows": rows,
        "aggregate": aggregate,
        "outputs": outputs,
    }


def by_scenario(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        row[
            "scenario"
        ]: row
        for row in rows
    }


# =============================================================================
# Evaluation outputs
# =============================================================================

def detailed_prediction_frame(
    test: Dict[str, Any],
    suite: Dict[str, Dict[str, Any]],
    evaluation: Dict[str, Any],
) -> pd.DataFrame:
    frames = []

    for scenario in scored_stress_scenarios():
        data = suite[
            scenario
        ]

        result = evaluation[
            "outputs"
        ][
            scenario
        ]

        final_probs = result[
            "router_output"
        ][
            "final_probs"
        ]

        final_info = result[
            "final_info"
        ]

        frame = test[
            "meta"
        ][
            [
                "subject",
                "pair_key",
                "label_id",
                "emotion",
            ]
        ].copy()

        frame.insert(
            0,
            "scenario",
            scenario,
        )

        frame.insert(
            0,
            "router_tau",
            ROUTER_TAU,
        )

        frame[
            "system_state"
        ] = result[
            "system_state"
        ]

        frame[
            "route"
        ] = result[
            "route"
        ]

        frame[
            "no_decision"
        ] = result[
            "router_output"
        ][
            "no_decision"
        ]

        frame[
            "pred_label_id"
        ] = final_info[
            "pred_label_id"
        ]

        frame[
            "pred_emotion"
        ] = np.where(
            frame[
                "no_decision"
            ].to_numpy()
            > 0.5,
            "NO_DECISION",
            final_info[
                "pred_emotion"
            ],
        )

        frame[
            "final_confidence"
        ] = np.where(
            frame[
                "no_decision"
            ].to_numpy()
            > 0.5,
            np.nan,
            final_info[
                "confidence"
            ],
        )

        for m, modality in enumerate(
            MODALITIES
        ):
            info = result[
                "modality"
            ][
                modality
            ]

            frame[
                f"{modality}_pred_emotion"
            ] = info[
                "pred_emotion"
            ]

            frame[
                f"{modality}_confidence"
            ] = info[
                "confidence"
            ]

            frame[
                f"{modality}_quality"
            ] = result[
                "quality"
            ][
                :,
                m
            ]

            frame[
                f"{modality}_available"
            ] = result[
                "mask"
            ][
                :,
                m
            ]

            frame[
                f"af4b_alpha_{modality}"
            ] = result[
                "af4b_output"
            ][
                "alpha"
            ][
                :,
                m
            ]

        frame[
            "af4b_gamma"
        ] = result[
            "af4b_output"
        ][
            "gamma"
        ]

        for c, emotion in enumerate(
            EMOTIONS
        ):
            frame[
                f"prob_{c}_{emotion}"
            ] = final_probs[
                :,
                c
            ]

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone frozen final EAV AF4-C fusion system"
        )
    )

    parser.add_argument(
        "--project-root",
        default=str(
            DEFAULT_PROJECT_ROOT
        ),
    )

    parser.add_argument(
        "--mode",
        choices=(
            "evaluate",
            "predict",
        ),
        default="evaluate",
    )

    parser.add_argument(
        "--device",
        choices=(
            "cuda",
            "cpu",
        ),
        default="cuda",
    )

    parser.add_argument(
        "--f1-dir",
        default=None,
        help=(
            "Only needed for evaluate mode. "
            "If omitted, latest PASS F1 is auto-discovered."
        ),
    )

    parser.add_argument(
        "--f4-dir",
        default=None,
    )

    parser.add_argument(
        "--af4b-dir",
        default=None,
    )

    parser.add_argument(
        "--input-json",
        default=None,
        help=(
            "Required for predict mode."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Preflight
# =============================================================================

def run_preflight(
    f4_checkpoint: Path,
    af4b_checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    system = FinalAF4CSystem(
        f4_checkpoint=f4_checkpoint,
        af4b_checkpoint=af4b_checkpoint,
        device=device,
        batch_size=batch_size,
        tau=ROUTER_TAU,
    )

    p = np.asarray(
        [
            0.10,
            0.20,
            0.30,
            0.20,
            0.20,
        ],
        dtype=np.float32,
    )

    x15 = np.concatenate(
        [
            p,
            p,
            p,
        ],
        axis=0,
    )[
        None,
        :
    ]

    healthy = system.predict_batch(
        x15,
        np.asarray(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                ]
            ],
            dtype=np.float32,
        ),
        np.ones(
            (
                1,
                3,
            ),
            dtype=np.float32,
        ),
    )

    if str(
        healthy[
            "route"
        ][
            0
        ]
    ) != "F4":
        raise RuntimeError(
            "Healthy -> F4 preflight failed."
        )

    degraded = system.predict_batch(
        x15,
        np.asarray(
            [
                [
                    1.0,
                    0.50,
                    1.0,
                ]
            ],
            dtype=np.float32,
        ),
        np.ones(
            (
                1,
                3,
            ),
            dtype=np.float32,
        ),
    )

    if str(
        degraded[
            "route"
        ][
            0
        ]
    ) != "AF4-B":
        raise RuntimeError(
            "Degraded -> AF4-B preflight failed."
        )

    all_missing = system.predict_batch(
        np.full(
            (
                1,
                INPUT_DIM,
            ),
            1.0
            / NUM_CLASSES,
            dtype=np.float32,
        ),
        np.zeros(
            (
                1,
                3,
            ),
            dtype=np.float32,
        ),
        np.zeros(
            (
                1,
                3,
            ),
            dtype=np.float32,
        ),
    )

    if not bool(
        all_missing[
            "router_output"
        ][
            "no_decision"
        ][
            0
        ]
        > 0.5
    ):
        raise RuntimeError(
            "All-missing NO_DECISION preflight failed."
        )

    if not np.allclose(
        all_missing[
            "router_output"
        ][
            "final_probs"
        ][
            0
        ],
        1.0
        / NUM_CLASSES,
        atol=1e-6,
    ):
        raise RuntimeError(
            "All-missing uniform output preflight failed."
        )

    print(
        "Standalone model load       : PASS"
    )

    print(
        "Healthy -> F4               : PASS"
    )

    print(
        "Degraded -> AF4-B           : PASS"
    )

    print(
        "Confidence extraction       : PASS"
    )

    print(
        "All-missing NO_DECISION     : PASS"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    project_root = Path(
        args.project_root
    ).expanduser().resolve()

    device = choose_device(
        args.device
    )

    (
        f4_dir,
        af4b_dir,
        f4_checkpoint,
        af4b_checkpoint,
    ) = resolve_model_sources(
        project_root,
        args.f4_dir,
        args.af4b_dir,
    )

    print("=" * 128)
    print(
        "EAV FINAL FUSION — STANDALONE AF4-C + CONFIDENCE"
    )
    print("=" * 128)

    print(
        f"Device                    : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU                       : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"F4 checkpoint             : {f4_checkpoint}"
    )

    print(
        f"AF4-B checkpoint          : {af4b_checkpoint}"
    )

    print(
        f"Frozen router tau         : {ROUTER_TAU:.2f}"
    )

    print(
        "Healthy action            : F4"
    )

    print(
        "Degraded action           : AF4-B"
    )

    print(
        "All missing               : NO_DECISION"
    )

    print(
        "Confidence                 : max(class probability)"
    )

    print()

    run_preflight(
        f4_checkpoint,
        af4b_checkpoint,
        device,
        args.batch_size,
    )

    if args.preflight_only:
        print()

        print(
            "FINAL STANDALONE PREFLIGHT : PASS"
        )

        return 0

    system = FinalAF4CSystem(
        f4_checkpoint=f4_checkpoint,
        af4b_checkpoint=af4b_checkpoint,
        device=device,
        batch_size=args.batch_size,
        tau=ROUTER_TAU,
    )

    # ------------------------------------------------------------------
    # Runtime prediction mode.
    # ------------------------------------------------------------------
    if args.mode == "predict":
        if not args.input_json:
            raise ValueError(
                "--input-json is required in predict mode."
            )

        input_path = Path(
            args.input_json
        ).expanduser().resolve()

        (
            x15,
            quality,
            mask,
        ) = load_runtime_json(
            input_path
        )

        result = system.predict_batch(
            x15,
            quality,
            mask,
        )

        serializable = [
            result_to_serializable_sample(
                result,
                i,
            )
            for i in range(
                len(x15)
            )
        ]

        output_obj = (
            serializable[0]
            if len(
                serializable
            )
            == 1
            else {
                "samples": serializable
            }
        )

        if args.output:
            output_path = Path(
                args.output
            ).expanduser().resolve()
        else:
            output_path = (
                input_path.parent
                / (
                    input_path.stem
                    + "_af4c_output.json"
                )
            )

        write_json(
            output_path,
            output_obj,
        )

        print()

        for i, sample in enumerate(
            serializable,
            start=1,
        ):
            print(
                f"Sample {i}"
            )

            print(
                f"  State                   : "
                f"{sample['system_state']}"
            )

            print(
                f"  Route                   : "
                f"{sample['route']}"
            )

            for modality in MODALITIES:
                m = sample[
                    "modalities"
                ][
                    modality
                ]

                print(
                    f"  {modality.upper():5s}                   : "
                    f"{m['emotion']} | "
                    f"Confidence={m['confidence']:.4f} | "
                    f"Quality={m['quality']:.4f} | "
                    f"Available={m['available']}"
                )

            final = sample[
                "final"
            ]

            if sample[
                "no_decision"
            ]:
                print(
                    "  Final                   : NO_DECISION"
                )
            else:
                print(
                    f"  Final                   : "
                    f"{final['emotion']} | "
                    f"Confidence="
                    f"{final['confidence']:.4f}"
                )

            print()

        print(
            f"Output                    : {output_path}"
        )

        return 0

    # ------------------------------------------------------------------
    # Frozen formal evaluation mode.
    # ------------------------------------------------------------------
    f1_dir = (
        Path(
            args.f1_dir
        ).expanduser().resolve()
        if args.f1_dir
        else latest_pass_f1(
            project_root
        )
    )

    output_dir = (
        Path(
            args.output
        ).expanduser().resolve()
        if args.output
        else (
            project_root
            / "EAV_dataset"
            / "models"
            / "fusion"
            / (
                "final_af4c_standalone_evaluation_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print(
        "FORMAL FROZEN EVALUATION"
    )

    print(
        f"F1 source                 : {f1_dir}"
    )

    print(
        f"Output                    : {output_dir}"
    )

    # Validation is read for audit/reporting only.
    # No threshold/model selection occurs in this final frozen script.
    val = load_f1_split_trial_data(
        f1_dir,
        "val",
    )

    val_suite = build_stress_suite(
        val[
            "x15"
        ],
        val[
            "labels"
        ],
        seed=args.seed
        + 100_000,
    )

    val_eval = evaluate_suite(
        system,
        val_suite,
    )

    test = load_f1_split_trial_data(
        f1_dir,
        "test",
    )

    test_suite = build_stress_suite(
        test[
            "x15"
        ],
        test[
            "labels"
        ],
        seed=args.seed
        + 200_000,
    )

    test_eval = evaluate_suite(
        system,
        test_suite,
    )

    # Exact frozen clean-F4 gate.
    clean_result = test_eval[
        "outputs"
    ][
        "clean"
    ]

    f4_clean_metrics = metric_dict(
        test[
            "labels"
        ],
        clean_result[
            "f4_probs"
        ],
    )

    af4b_clean_metrics = metric_dict(
        test[
            "labels"
        ],
        clean_result[
            "af4b_output"
        ][
            "final_probs"
        ],
    )

    af4c_clean_metrics = metric_dict(
        test[
            "labels"
        ],
        clean_result[
            "router_output"
        ][
            "final_probs"
        ],
    )

    if abs(
        f4_clean_metrics[
            "accuracy"
        ]
        - EXPECTED_F4_TEST_ACCURACY
    ) > 1e-9:
        raise RuntimeError(
            "Frozen F4 clean Test reproduction failed: "
            f"{f4_clean_metrics['accuracy']:.12f}"
        )

    if not np.allclose(
        clean_result[
            "router_output"
        ][
            "final_probs"
        ],
        clean_result[
            "f4_probs"
        ],
        atol=1e-7,
        rtol=1e-7,
    ):
        raise RuntimeError(
            "Frozen AF4-C clean probabilities do not equal F4."
        )

    # All missing safety.
    all_missing = build_fixed_scenario(
        test[
            "x15"
        ],
        test[
            "labels"
        ],
        "all_missing",
        seed=args.seed
        + 300_000,
    )

    all_missing_result = system.predict_batch(
        all_missing[
            "x15"
        ],
        all_missing[
            "quality"
        ],
        all_missing[
            "mask"
        ],
    )

    all_missing_no_decision = float(
        all_missing_result[
            "router_output"
        ][
            "no_decision"
        ].mean()
    )

    pd.DataFrame(
        val_eval[
            "rows"
        ]
    ).to_csv(
        output_dir
        / "validation_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        test_eval[
            "rows"
        ]
    ).to_csv(
        output_dir
        / "test_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detailed_prediction_frame(
        test,
        test_suite,
        test_eval,
    ).to_csv(
        output_dir
        / "test_routed_predictions_with_confidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean_row = by_scenario(
        test_eval[
            "rows"
        ]
    )[
        "clean"
    ]

    summary = {
        "stage": "FINAL-AF4-C-STANDALONE",
        "status": "FROZEN",
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "router_tau": ROUTER_TAU,
        "healthy_action": "F4",
        "degraded_action": "AF4-B",
        "all_missing_action": "NO_DECISION",
        "confidence_definition": (
            "max class probability"
        ),
        "quality_and_confidence_are_distinct": True,
        "f4_dir": str(
            f4_dir
        ),
        "af4b_dir": str(
            af4b_dir
        ),
        "f4_checkpoint": str(
            f4_checkpoint
        ),
        "af4b_checkpoint": str(
            af4b_checkpoint
        ),
        "f1_source": str(
            f1_dir
        ),
        "test_used_for_selection": False,
        "validation_used_for_new_selection": False,
        "test_f4_clean": (
            f4_clean_metrics
        ),
        "test_af4b_clean": (
            af4b_clean_metrics
        ),
        "test_af4c_clean": (
            af4c_clean_metrics
        ),
        "test_af4c_aggregate": (
            test_eval[
                "aggregate"
            ]
        ),
        "all_missing_no_decision_rate": (
            all_missing_no_decision
        ),
        "clean_mean_confidence": {
            "eeg": (
                clean_row[
                    "mean_eeg_confidence"
                ]
            ),
            "audio": (
                clean_row[
                    "mean_audio_confidence"
                ]
            ),
            "video": (
                clean_row[
                    "mean_video_confidence"
                ]
            ),
            "final": (
                clean_row[
                    "mean_final_confidence"
                ]
            ),
        },
        "reference_from_frozen_experiment": (
            EXPECTED_REFERENCE
        ),
        "output_dir": str(
            output_dir
        ),
    }

    write_json(
        output_dir
        / "final_af4c_summary.json",
        summary,
    )

    print()

    print("=" * 128)
    print(
        "FINAL STANDALONE AF4-C"
    )
    print("=" * 128)

    print(
        "STATUS                    : FROZEN"
    )

    print(
        "OLD F4 SCRIPT REQUIRED    : NO"
    )

    print(
        "OLD AF4-B SCRIPT REQUIRED : NO"
    )

    print(
        "F4 CHECKPOINT REQUIRED    : YES"
    )

    print(
        "AF4-B CHECKPOINT REQUIRED : YES"
    )

    print(
        f"ROUTER TAU                : {ROUTER_TAU:.2f}"
    )

    print(
        f"F4 CLEAN TEST ACC         : "
        f"{f4_clean_metrics['accuracy']:.4f}"
    )

    print(
        f"AF4-B CLEAN TEST ACC      : "
        f"{af4b_clean_metrics['accuracy']:.4f}"
    )

    print(
        f"AF4-C CLEAN TEST ACC      : "
        f"{af4c_clean_metrics['accuracy']:.4f}"
    )

    print(
        f"AF4-C STRESS MEAN F1     : "
        f"{test_eval['aggregate']['stress_mean_macro_f1']:.4f}"
    )

    print(
        f"AF4-C MISSING MEAN F1    : "
        f"{test_eval['aggregate']['missing_mean_macro_f1']:.4f}"
    )

    print(
        f"ALL-MISSING NO-DECISION  : "
        f"{all_missing_no_decision:.4f}"
    )

    print(
        f"CLEAN ROUTE TO AF4-B     : "
        f"{test_eval['aggregate']['clean_route_to_af4b_rate']:.4f}"
    )

    print(
        f"STRESS ROUTE TO AF4-B    : "
        f"{test_eval['aggregate']['stress_mean_route_to_af4b_rate']:.4f}"
    )

    print(
        f"CLEAN EEG CONFIDENCE      : "
        f"{clean_row['mean_eeg_confidence']:.4f}"
    )

    print(
        f"CLEAN AUDIO CONFIDENCE    : "
        f"{clean_row['mean_audio_confidence']:.4f}"
    )

    print(
        f"CLEAN VIDEO CONFIDENCE    : "
        f"{clean_row['mean_video_confidence']:.4f}"
    )

    print(
        f"CLEAN FINAL CONFIDENCE    : "
        f"{clean_row['mean_final_confidence']:.4f}"
    )

    print(
        f"OUTPUT                    : {output_dir}"
    )

    print("=" * 128)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
