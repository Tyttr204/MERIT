#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
VIDEO Emotion V2B — Final Standalone Deployment
================================================

Purpose
-------
Standalone deployment inference for the frozen EAV VIDEO emotion head.

Input:
    one real video clip, normally one 5-second analysis window

Output:
    video_probs       : 5-class probabilities
    pred_label_id     : 0..4
    pred_emotion      : Neutral / Sadness / Anger / Happiness / Calmness
    confidence        : max(video_probs)
    video_available   : whether at least one usable face was detected

Frozen class order:
    0 Neutral
    1 Sadness
    2 Anger
    3 Happiness
    4 Calmness

Deployment assets
-----------------
By default this script looks beside itself for:

    DFEW-set1-model.pth
    best_v2b_frozen_dferclip_head_validation_selected.pt
    face_detection_yunet_2023mar.onnx

Important:
    - NO DFER-CLIP_official repository is required at runtime.
    - NO OpenAI CLIP checkpoint is downloaded at runtime.
    - The CLIP ViT-B/32 visual architecture is reconstructed locally.
    - The DFER temporal Transformer architecture is reconstructed locally.
    - DFEW-set1-model.pth directly supplies:
          module.image_encoder.*
          module.temporal_net.*
    - The EAV 5-class checkpoint supplies the trained EmotionHead.
    - Preprocessing follows the frozen V2B convention:
          16 face-focused RGB frames
          224 x 224
          float / 255
          NO CLIP mean/std normalization

Recommended runtime:
    python .\video_emotion_v2b_deployment.py `
      --video "C:\path\to\five_second_window.mp4"

Save JSON:
    python .\video_emotion_v2b_deployment.py `
      --video "C:\path\to\five_second_window.mp4" `
      --json-output ".\video_emotion_result.json"

Python API:
    from video_emotion_v2b_deployment import VideoEmotionV2B

    model = VideoEmotionV2B()
    result = model.predict(r"C:\path\to\window.mp4")

    video_probs = result["video_probs"]
    confidence = result["confidence"]
    video_available = result["video_available"]

AF4-C deployment contract:
    If video_available is False, this module returns a numerical placeholder
    [0.2,0.2,0.2,0.2,0.2]. The fusion layer MUST use video_available=False
    (mask_video=0), so the placeholder is never treated as classifier evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Frozen deployment constants
# =============================================================================

MODULE_VERSION = "VIDEO-EMOTION-V2B-DEPLOYMENT-V1.0-FROZEN"

EMOTIONS = [
    "Neutral",
    "Sadness",
    "Anger",
    "Happiness",
    "Calmness",
]

NUM_CLASSES = 5
FEATURE_DIM = 512

FRAMES_PER_WINDOW = 16
CROP_SIZE = 224

FACE_SCORE_THRESHOLD = 0.75
FACE_NMS_THRESHOLD = 0.30
FACE_TOP_K = 5000
FACE_DETECT_MAX_SIDE = 720
FACE_SCALE = 1.45

DEFAULT_DFEW_NAME = "DFEW-set1-model.pth"
DEFAULT_HEAD_NAME = "best_v2b_frozen_dferclip_head_validation_selected.pt"
DEFAULT_YUNET_NAME = "face_detection_yunet_2023mar.onnx"

# Frozen official DFER-CLIP visual encoder: OpenAI CLIP ViT-B/32.
CLIP_INPUT_RESOLUTION = 224
CLIP_PATCH_SIZE = 32
CLIP_VISION_WIDTH = 768
CLIP_VISION_LAYERS = 12
CLIP_VISION_HEADS = 12
CLIP_OUTPUT_DIM = 512

# Frozen official DFER temporal encoder.
TEMPORAL_NUM_PATCHES = 16
TEMPORAL_INPUT_DIM = 512
TEMPORAL_DEPTH = 1
TEMPORAL_HEADS = 8
TEMPORAL_MLP_DIM = 1024
TEMPORAL_DIM_HEAD = 64


# =============================================================================
# Generic utilities
# =============================================================================

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(obj),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_torch_load(path: Path) -> Any:
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


def resolve_asset(
    explicit: Optional[str],
    default_name: str,
) -> Path:
    """
    Deployment-first resolution:
      1) explicit path
      2) file beside this script
      3) current working directory
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    here = Path(__file__).resolve().parent

    candidates = [
        here / default_name,
        Path.cwd().resolve() / default_name,
    ]

    for p in candidates:
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        f"Cannot locate deployment asset {default_name!r}. "
        f"Place it beside this script or pass an explicit path."
    )


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")

    if name == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be 'cuda' or 'cpu'")


def uniform_probs() -> List[float]:
    return [1.0 / NUM_CLASSES] * NUM_CLASSES


# =============================================================================
# Self-contained OpenAI CLIP ViT-B/32 visual architecture
# =============================================================================

class CLIPLayerNorm(nn.LayerNorm):
    """OpenAI CLIP LayerNorm behavior for fp16-safe inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            n_head,
        )

        self.ln_1 = CLIPLayerNorm(
            d_model
        )

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c_fc",
                        nn.Linear(
                            d_model,
                            d_model * 4,
                        ),
                    ),
                    (
                        "gelu",
                        QuickGELU(),
                    ),
                    (
                        "c_proj",
                        nn.Linear(
                            d_model * 4,
                            d_model,
                        ),
                    ),
                ]
            )
        )

        self.ln_2 = CLIPLayerNorm(
            d_model
        )

        self.attn_mask = attn_mask

    def attention(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.attn_mask

        if mask is not None:
            mask = mask.to(
                dtype=x.dtype,
                device=x.device,
            )

        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )[0]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


class CLIPTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(
            *[
                CLIPResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """
    OpenAI CLIP VisionTransformer-compatible module.

    State-dict names intentionally match CLIP's `visual` module:
        conv1.weight
        class_embedding
        positional_embedding
        ln_pre.*
        transformer.resblocks.*
        ln_post.*
        proj
    """

    def __init__(
        self,
        input_resolution: int = CLIP_INPUT_RESOLUTION,
        patch_size: int = CLIP_PATCH_SIZE,
        width: int = CLIP_VISION_WIDTH,
        layers: int = CLIP_VISION_LAYERS,
        heads: int = CLIP_VISION_HEADS,
        output_dim: int = CLIP_OUTPUT_DIM,
    ) -> None:
        super().__init__()

        self.input_resolution = int(
            input_resolution
        )

        self.output_dim = int(
            output_dim
        )

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5

        self.class_embedding = nn.Parameter(
            scale * torch.randn(width)
        )

        grid_size = (
            input_resolution // patch_size
        )

        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                grid_size**2 + 1,
                width,
            )
        )

        self.ln_pre = CLIPLayerNorm(
            width
        )

        self.transformer = CLIPTransformer(
            width=width,
            layers=layers,
            heads=heads,
        )

        self.ln_post = CLIPLayerNorm(
            width
        )

        self.proj = nn.Parameter(
            scale
            * torch.randn(
                width,
                output_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # [B, 3, 224, 224] -> [B, width, 7, 7]
        x = self.conv1(x)

        # [B, width, grid, grid] -> [B, grid^2, width]
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        ).permute(
            0,
            2,
            1,
        )

        cls = (
            self.class_embedding
            .to(x.dtype)
            + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        )

        x = torch.cat(
            [
                cls,
                x,
            ],
            dim=1,
        )

        x = x + self.positional_embedding.to(
            x.dtype
        )

        x = self.ln_pre(x)

        # NLD -> LND
        x = x.permute(
            1,
            0,
            2,
        )

        x = self.transformer(x)

        # LND -> NLD
        x = x.permute(
            1,
            0,
            2,
        )

        x = self.ln_post(
            x[:, 0, :]
        )

        x = x @ self.proj

        return x


# =============================================================================
# Self-contained official DFER temporal Transformer
# =============================================================================

class DFERGELU(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    math.sqrt(
                        2 / math.pi
                    )
                    * (
                        x
                        + 0.044715
                        * torch.pow(
                            x,
                            3,
                        )
                    )
                )
            )
        )


class Residual(nn.Module):
    def __init__(
        self,
        fn: nn.Module,
    ) -> None:
        super().__init__()
        self.fn = fn

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.fn(
            x,
            **kwargs,
        ) + x


class PreNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        fn: nn.Module,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.fn = fn

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.fn(
            self.norm(x),
            **kwargs,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim,
            ),
            DFERGELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class DFERAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        inner_dim = dim_head * heads

        project_out = not (
            heads == 1
            and dim_head == dim
        )

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False,
        )

        self.to_out = (
            nn.Sequential(
                nn.Linear(
                    inner_dim,
                    dim,
                ),
                nn.Dropout(
                    dropout
                ),
            )
            if project_out
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Exact math of the official einops implementation,
        # implemented here without a runtime einops dependency.
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(
            x
        ).chunk(
            3,
            dim=-1,
        )

        q, k, v = [
            t.reshape(
                b,
                n,
                h,
                -1,
            ).permute(
                0,
                2,
                1,
                3,
            )
            for t in qkv
        ]

        dots = (
            q
            @ k.transpose(
                -2,
                -1,
            )
        ) * self.scale

        attn = dots.softmax(
            dim=-1
        )

        out = attn @ v

        out = out.permute(
            0,
            2,
            1,
            3,
        ).contiguous().reshape(
            b,
            n,
            -1,
        )

        return self.to_out(out)


class DFERTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            []
        )

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                DFERAttention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim,
                                FeedForward(
                                    dim,
                                    mlp_dim,
                                    dropout=dropout,
                                ),
                            )
                        ),
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


class TemporalTransformerCls(nn.Module):
    """
    Exact deployment reconstruction of official
    Temporal_Transformer_Cls(num_patches=16,input_dim=512,depth=1,
                             heads=8,mlp_dim=1024,dim_head=64).
    """

    def __init__(
        self,
        num_patches: int = TEMPORAL_NUM_PATCHES,
        input_dim: int = TEMPORAL_INPUT_DIM,
        depth: int = TEMPORAL_DEPTH,
        heads: int = TEMPORAL_HEADS,
        mlp_dim: int = TEMPORAL_MLP_DIM,
        dim_head: int = TEMPORAL_DIM_HEAD,
    ) -> None:
        super().__init__()

        dropout = 0.0

        self.num_patches = int(
            num_patches
        )

        self.input_dim = int(
            input_dim
        )

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                input_dim,
            )
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                num_patches + 1,
                input_dim,
            )
        )

        self.temporal_transformer = DFERTransformer(
            input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(
            b,
            -1,
            -1,
        )

        x = torch.cat(
            (
                cls_tokens,
                x,
            ),
            dim=1,
        )

        x = (
            x
            + self.pos_embedding[
                :,
                : n + 1,
            ]
        )

        x = self.temporal_transformer(
            x
        )

        return x[:, 0]


# =============================================================================
# Frozen DFER visual + temporal encoder
# =============================================================================

class FrozenDFEREncoder(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        temporal_net: nn.Module,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.temporal_net = temporal_net

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            x.ndim != 5
            or x.shape[1:] != (
                FRAMES_PER_WINDOW,
                3,
                CROP_SIZE,
                CROP_SIZE,
            )
        ):
            raise RuntimeError(
                "Expected input [B,16,3,224,224], "
                f"got {tuple(x.shape)}."
            )

        b, t, c, h, w = x.shape

        frames = x.contiguous().view(
            b * t,
            c,
            h,
            w,
        )

        frame_feat = self.image_encoder(
            frames
        )

        frame_feat = frame_feat.contiguous().view(
            b,
            t,
            -1,
        )

        z = self.temporal_net(
            frame_feat
        )

        return F.normalize(
            z,
            dim=-1,
            eps=1e-12,
        )


def extract_dfer_state(
    checkpoint: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = safe_torch_load(
        checkpoint
    )

    state = (
        ckpt["state_dict"]
        if (
            isinstance(
                ckpt,
                dict,
            )
            and "state_dict" in ckpt
        )
        else ckpt
    )

    if not isinstance(
        state,
        dict,
    ):
        raise RuntimeError(
            f"Unexpected DFEW checkpoint object: {type(state)}"
        )

    image_state: Dict[
        str,
        torch.Tensor,
    ] = {}

    temporal_state: Dict[
        str,
        torch.Tensor,
    ] = {}

    for key, value in state.items():
        key = str(key)

        if key.startswith(
            "module.image_encoder."
        ):
            image_state[
                key[
                    len(
                        "module.image_encoder."
                    ):
                ]
            ] = value

        elif key.startswith(
            "module.temporal_net."
        ):
            temporal_state[
                key[
                    len(
                        "module.temporal_net."
                    ):
                ]
            ] = value

    if not image_state:
        raise RuntimeError(
            "DFEW checkpoint contains no module.image_encoder.* weights."
        )

    if not temporal_state:
        raise RuntimeError(
            "DFEW checkpoint contains no module.temporal_net.* weights."
        )

    info = {
        "checkpoint": str(
            checkpoint
        ),
        "checkpoint_sha256": sha256_file(
            checkpoint
        ),
        "image_keys_found": len(
            image_state
        ),
        "temporal_keys_found": len(
            temporal_state
        ),
    }

    return (
        image_state,
        temporal_state,
        info,
    )


def load_direct_dfer_encoder(
    checkpoint: Path,
    device: torch.device,
) -> Tuple[FrozenDFEREncoder, Dict[str, Any]]:
    image_encoder = VisionTransformer()

    temporal_net = TemporalTransformerCls()

    (
        image_state,
        temporal_state,
        info,
    ) = extract_dfer_state(
        checkpoint
    )

    image_missing, image_unexpected = (
        image_encoder.load_state_dict(
            image_state,
            strict=False,
        )
    )

    temporal_missing, temporal_unexpected = (
        temporal_net.load_state_dict(
            temporal_state,
            strict=False,
        )
    )

    if (
        image_missing
        or image_unexpected
    ):
        raise RuntimeError(
            "Direct DFEW image_encoder state mismatch.\n"
            f"Missing: {list(image_missing)[:20]}\n"
            f"Unexpected: {list(image_unexpected)[:20]}"
        )

    if (
        temporal_missing
        or temporal_unexpected
    ):
        raise RuntimeError(
            "Direct DFEW temporal_net state mismatch.\n"
            f"Missing: {list(temporal_missing)[:20]}\n"
            f"Unexpected: {list(temporal_unexpected)[:20]}"
        )

    encoder = FrozenDFEREncoder(
        image_encoder=image_encoder,
        temporal_net=temporal_net,
    ).to(
        device
    ).eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(
            False
        )

    info.update(
        {
            "architecture": (
                "CLIP ViT-B/32 visual + "
                "1-layer DFER Temporal Transformer CLS"
            ),
            "official_repo_required": False,
            "openai_clip_checkpoint_required": False,
            "preprocessing": (
                "16 RGB face frames, 224x224, float/255, "
                "NO CLIP mean/std normalization"
            ),
        }
    )

    return encoder, info


# =============================================================================
# Frozen EAV 5-class head
# =============================================================================

class EmotionHead(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                FEATURE_DIM
            ),
            nn.Linear(
                FEATURE_DIM,
                hidden,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                NUM_CLASSES,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


def load_eav_head(
    checkpoint: Path,
    device: torch.device,
) -> Tuple[EmotionHead, Dict[str, Any]]:
    ckpt = safe_torch_load(
        checkpoint
    )

    if not isinstance(
        ckpt,
        dict,
    ):
        raise RuntimeError(
            "EAV head checkpoint is not a dictionary."
        )

    if str(
        ckpt.get(
            "stage",
            "",
        )
    ).upper() != "V2B":
        raise RuntimeError(
            "EAV head checkpoint stage is not V2B."
        )

    classes = list(
        ckpt.get(
            "classes",
            [],
        )
    )

    if classes != EMOTIONS:
        raise RuntimeError(
            "EAV head class order mismatch. "
            f"Expected {EMOTIONS}, got {classes}."
        )

    feature_dim = int(
        ckpt.get(
            "feature_dim",
            -1,
        )
    )

    if feature_dim != FEATURE_DIM:
        raise RuntimeError(
            "EAV head feature_dim mismatch: "
            f"{feature_dim}."
        )

    hidden = int(
        ckpt.get(
            "hidden_dim",
            256,
        )
    )

    dropout = float(
        ckpt.get(
            "dropout",
            0.35,
        )
    )

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            "EAV head checkpoint has no model_state_dict."
        )

    head = EmotionHead(
        hidden=hidden,
        dropout=dropout,
    )

    head.load_state_dict(
        ckpt[
            "model_state_dict"
        ],
        strict=True,
    )

    head = head.to(
        device
    ).eval()

    for parameter in head.parameters():
        parameter.requires_grad_(
            False
        )

    info = {
        "checkpoint": str(
            checkpoint
        ),
        "checkpoint_sha256": sha256_file(
            checkpoint
        ),
        "hidden_dim": hidden,
        "dropout": dropout,
        "classes": classes,
        "epoch": ckpt.get(
            "epoch"
        ),
        "selection_metric": ckpt.get(
            "selection_metric"
        ),
        "selection_score": ckpt.get(
            "selection_score"
        ),
    }

    return head, info


# =============================================================================
# YuNet face-focused preprocessing
# =============================================================================

class YuNetDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = FACE_SCORE_THRESHOLD,
        nms_threshold: float = FACE_NMS_THRESHOLD,
        top_k: int = FACE_TOP_K,
        detect_max_side: int = FACE_DETECT_MAX_SIDE,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                model_path
            )

        if (
            os.name == "nt"
            and not str(
                model_path
            ).isascii()
        ):
            raise RuntimeError(
                "YuNet runtime model path must be ASCII-only on Windows. "
                f"Current path: {model_path}"
            )

        self.model_path = model_path

        self.detector = cv2.FaceDetectorYN.create(
            str(
                model_path
            ),
            "",
            (
                320,
                320,
            ),
            float(
                score_threshold
            ),
            float(
                nms_threshold
            ),
            int(
                top_k
            ),
        )

        self.detect_max_side = int(
            detect_max_side
        )

    @staticmethod
    def iou(
        a: Optional[np.ndarray],
        b: np.ndarray,
    ) -> float:
        if a is None:
            return 0.0

        ax, ay, aw, ah = [
            float(x)
            for x in a
        ]

        bx, by, bw, bh = [
            float(x)
            for x in b
        ]

        x1 = max(
            ax,
            bx,
        )

        y1 = max(
            ay,
            by,
        )

        x2 = min(
            ax + aw,
            bx + bw,
        )

        y2 = min(
            ay + ah,
            by + bh,
        )

        inter = (
            max(
                0.0,
                x2 - x1,
            )
            * max(
                0.0,
                y2 - y1,
            )
        )

        union = (
            aw * ah
            + bw * bh
            - inter
        )

        return (
            inter / union
            if union > 0
            else 0.0
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        prev_box: Optional[np.ndarray],
    ) -> Tuple[
        Optional[np.ndarray],
        float,
    ]:
        h0, w0 = frame_bgr.shape[
            :2
        ]

        scale = min(
            1.0,
            self.detect_max_side
            / float(
                max(
                    h0,
                    w0,
                )
            ),
        )

        if scale < 1.0:
            wd = max(
                1,
                int(
                    round(
                        w0 * scale
                    )
                ),
            )

            hd = max(
                1,
                int(
                    round(
                        h0 * scale
                    )
                ),
            )

            det_img = cv2.resize(
                frame_bgr,
                (
                    wd,
                    hd,
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:
            det_img = frame_bgr
            hd, wd = h0, w0

        self.detector.setInputSize(
            (
                wd,
                hd,
            )
        )

        _, faces = self.detector.detect(
            det_img
        )

        if (
            faces is None
            or len(
                faces
            )
            == 0
        ):
            return (
                None,
                float(
                    "nan"
                ),
            )

        prev_scaled = (
            prev_box * scale
            if prev_box is not None
            else None
        )

        img_area = float(
            wd * hd
        )

        cx_img = wd / 2.0
        cy_img = hd / 2.0

        best_box: Optional[
            np.ndarray
        ] = None

        best_conf = float(
            "nan"
        )

        best_rank = -1e30

        for face in faces:
            x, y, w, h = [
                float(z)
                for z in face[
                    :4
                ]
            ]

            conf = float(
                face[-1]
            )

            box = np.asarray(
                [
                    x,
                    y,
                    w,
                    h,
                ],
                dtype=np.float32,
            )

            area_term = min(
                1.0,
                (
                    w * h
                )
                / max(
                    img_area * 0.15,
                    1.0,
                ),
            )

            cx = x + w / 2.0
            cy = y + h / 2.0

            dist = math.sqrt(
                (
                    (
                        cx - cx_img
                    )
                    / max(
                        wd,
                        1,
                    )
                )
                ** 2
                + (
                    (
                        cy - cy_img
                    )
                    / max(
                        hd,
                        1,
                    )
                )
                ** 2
            )

            center_term = max(
                0.0,
                1.0 - 2.0 * dist,
            )

            track_term = self.iou(
                prev_scaled,
                box,
            )

            rank = (
                2.0 * conf
                + 0.75 * area_term
                + 0.35 * center_term
                + 1.25 * track_term
            )

            if rank > best_rank:
                best_rank = rank
                best_box = box
                best_conf = conf

        if best_box is None:
            return (
                None,
                float(
                    "nan"
                ),
            )

        return (
            (
                best_box / scale
            ).astype(
                np.float32
            ),
            best_conf,
        )


def uniform_sample_indices(
    frame_count: int,
    n_frames: int = FRAMES_PER_WINDOW,
) -> List[int]:
    if frame_count <= 0:
        raise RuntimeError(
            f"Invalid frame_count={frame_count}."
        )

    edges = np.linspace(
        0,
        frame_count,
        n_frames + 1,
        endpoint=True,
    )

    indices: List[
        int
    ] = []

    for i in range(
        n_frames
    ):
        center = int(
            math.floor(
                (
                    edges[i]
                    + edges[i + 1]
                    - 1.0
                )
                / 2.0
            )
        )

        indices.append(
            max(
                0,
                min(
                    center,
                    frame_count - 1,
                ),
            )
        )

    return indices


def decode_uniform_frames(
    video_path: Path,
) -> Tuple[
    List[np.ndarray],
    float,
    int,
]:
    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    try:
        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = int(
            round(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
        )

        if (
            not math.isfinite(
                fps
            )
            or fps <= 1e-6
        ):
            fps = 30.0

        if frame_count <= 0:
            raise RuntimeError(
                "Video frame count is invalid."
            )

        targets = uniform_sample_indices(
            frame_count,
            FRAMES_PER_WINDOW,
        )

        target_set = set(
            targets
        )

        frame_store: Dict[
            int,
            np.ndarray,
        ] = {}

        max_target = max(
            targets
        )

        fi = 0

        while fi <= max_target:
            ok, frame = cap.read()

            if not ok:
                break

            if fi in target_set:
                frame_store[
                    fi
                ] = frame.copy()

            fi += 1

        frames: List[
            np.ndarray
        ] = []

        for target in targets:
            if target not in frame_store:
                raise RuntimeError(
                    f"Could not decode sampled frame {target}."
                )

            frames.append(
                frame_store[
                    target
                ]
            )

        return (
            frames,
            fps,
            frame_count,
        )

    finally:
        cap.release()


def fill_missing_boxes(
    boxes: List[
        Optional[
            np.ndarray
        ]
    ],
) -> Tuple[
    np.ndarray,
    List[str],
]:
    n = len(
        boxes
    )

    valid = [
        i
        for i, box
        in enumerate(
            boxes
        )
        if box is not None
    ]

    if not valid:
        raise RuntimeError(
            "YuNet failed on all 16 sampled frames."
        )

    out = np.zeros(
        (
            n,
            4,
        ),
        dtype=np.float32,
    )

    source = [
        "detected"
    ] * n

    for i in valid:
        out[i] = boxes[i]  # type: ignore[index]

    for i in range(n):
        if boxes[i] is not None:
            continue

        left = max(
            [
                j
                for j in valid
                if j < i
            ],
            default=None,
        )

        right = min(
            [
                j
                for j in valid
                if j > i
            ],
            default=None,
        )

        if (
            left is not None
            and right is not None
        ):
            alpha = (
                i - left
            ) / float(
                right - left
            )

            out[i] = (
                (
                    1.0 - alpha
                )
                * out[left]
                + alpha
                * out[right]
            )

            source[i] = (
                "interpolated"
            )

        elif left is not None:
            out[i] = out[
                left
            ]

            source[i] = (
                "forward_fill"
            )

        elif right is not None:
            out[i] = out[
                right
            ]

            source[i] = (
                "backward_fill"
            )

        else:
            raise RuntimeError(
                "Missing-box filling failure."
            )

    return (
        out,
        source,
    )


def rolling_median_1d(
    x: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    y = np.empty_like(
        x
    )

    for i in range(
        len(
            x
        )
    ):
        lo = max(
            0,
            i - radius,
        )

        hi = min(
            len(
                x
            ),
            i + radius + 1,
        )

        y[i] = np.median(
            x[
                lo:hi
            ]
        )

    return y


def smooth_boxes(
    boxes: np.ndarray,
    median_radius: int = 2,
    ema_alpha: float = 0.65,
) -> np.ndarray:
    x, y, w, h = [
        boxes[
            :,
            i,
        ]
        for i in range(
            4
        )
    ]

    cx = x + w / 2.0
    cy = y + h / 2.0
    size = np.maximum(
        w,
        h,
    )

    cx = rolling_median_1d(
        cx,
        median_radius,
    )

    cy = rolling_median_1d(
        cy,
        median_radius,
    )

    size = rolling_median_1d(
        size,
        median_radius,
    )

    for arr in (
        cx,
        cy,
        size,
    ):
        for i in range(
            1,
            len(
                arr
            ),
        ):
            arr[i] = (
                ema_alpha
                * arr[i]
                + (
                    1.0
                    - ema_alpha
                )
                * arr[
                    i - 1
                ]
            )

    out = np.zeros_like(
        boxes,
        dtype=np.float32,
    )

    out[
        :,
        2,
    ] = size

    out[
        :,
        3,
    ] = size

    out[
        :,
        0,
    ] = cx - size / 2.0

    out[
        :,
        1,
    ] = cy - size / 2.0

    return out


def square_face_crop(
    frame: np.ndarray,
    box: np.ndarray,
    face_scale: float = FACE_SCALE,
    out_size: int = CROP_SIZE,
) -> np.ndarray:
    h, w = frame.shape[
        :2
    ]

    x, y, bw, bh = [
        float(z)
        for z in box
    ]

    cx = x + bw / 2.0
    cy = y + bh / 2.0

    side = max(
        24.0,
        max(
            bw,
            bh,
        )
        * float(
            face_scale
        ),
    )

    x1 = int(
        math.floor(
            cx - side / 2.0
        )
    )

    y1 = int(
        math.floor(
            cy - side / 2.0
        )
    )

    x2 = int(
        math.ceil(
            cx + side / 2.0
        )
    )

    y2 = int(
        math.ceil(
            cy + side / 2.0
        )
    )

    pad_l = max(
        0,
        -x1,
    )

    pad_t = max(
        0,
        -y1,
    )

    pad_r = max(
        0,
        x2 - w,
    )

    pad_b = max(
        0,
        y2 - h,
    )

    if any(
        v > 0
        for v in (
            pad_l,
            pad_t,
            pad_r,
            pad_b,
        )
    ):
        padded = cv2.copyMakeBorder(
            frame,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            borderType=cv2.BORDER_REFLECT_101,
        )

        x1 += pad_l
        x2 += pad_l
        y1 += pad_t
        y2 += pad_t

    else:
        padded = frame

    crop = padded[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        raise RuntimeError(
            "Empty face crop."
        )

    return cv2.resize(
        crop,
        (
            out_size,
            out_size,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


def preprocess_video_clip(
    video_path: Path,
    detector: YuNetDetector,
) -> Dict[str, Any]:
    (
        frames,
        fps,
        frame_count,
    ) = decode_uniform_frames(
        video_path
    )

    raw_boxes: List[
        Optional[
            np.ndarray
        ]
    ] = []

    confidences: List[
        float
    ] = []

    prev: Optional[
        np.ndarray
    ] = None

    for frame in frames:
        box, conf = detector.detect(
            frame,
            prev,
        )

        raw_boxes.append(
            box
        )

        confidences.append(
            conf
        )

        if box is not None:
            prev = box

    detected_count = sum(
        box is not None
        for box in raw_boxes
    )

    if detected_count == 0:
        return {
            "video_available": False,
            "clip_tensor": None,
            "fps": float(
                fps
            ),
            "frame_count": int(
                frame_count
            ),
            "duration_sec": float(
                frame_count
                / max(
                    fps,
                    1e-6,
                )
            ),
            "sampled_frames": FRAMES_PER_WINDOW,
            "face_detected_frames": 0,
            "face_fallback_frames": FRAMES_PER_WINDOW,
            "face_raw_detection_rate": 0.0,
            "face_mean_detection_score": None,
        }

    filled, source = fill_missing_boxes(
        raw_boxes
    )

    smoothed = smooth_boxes(
        filled
    )

    crops: List[
        np.ndarray
    ] = []

    detected_scores: List[
        float
    ] = []

    fallback_count = 0

    for i, frame in enumerate(
        frames
    ):
        crop = square_face_crop(
            frame,
            smoothed[i],
            face_scale=FACE_SCALE,
            out_size=CROP_SIZE,
        )

        rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        crops.append(
            rgb
        )

        if source[i] == "detected":
            if math.isfinite(
                confidences[i]
            ):
                detected_scores.append(
                    float(
                        confidences[i]
                    )
                )
        else:
            fallback_count += 1

    arr = np.stack(
        crops,
        axis=0,
    )

    x = (
        torch.from_numpy(
            arr
        )
        .permute(
            0,
            3,
            1,
            2,
        )
        .contiguous()
        .float()
        .div_(
            255.0
        )
    )

    if x.shape != (
        FRAMES_PER_WINDOW,
        3,
        CROP_SIZE,
        CROP_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected preprocessed shape: {tuple(x.shape)}"
        )

    return {
        "video_available": True,
        "clip_tensor": x,
        "fps": float(
            fps
        ),
        "frame_count": int(
            frame_count
        ),
        "duration_sec": float(
            frame_count
            / max(
                fps,
                1e-6,
            )
        ),
        "sampled_frames": FRAMES_PER_WINDOW,
        "face_detected_frames": int(
            detected_count
        ),
        "face_fallback_frames": int(
            fallback_count
        ),
        "face_raw_detection_rate": float(
            detected_count
            / FRAMES_PER_WINDOW
        ),
        "face_mean_detection_score": (
            float(
                np.mean(
                    detected_scores
                )
            )
            if detected_scores
            else None
        ),
    }


# =============================================================================
# Public deployment API
# =============================================================================

class VideoEmotionV2B:
    """
    Initialize once and reuse for many 5-second video windows.
    """

    def __init__(
        self,
        dfew_checkpoint: Optional[str] = None,
        head_checkpoint: Optional[str] = None,
        yunet_model: Optional[str] = None,
        device: str = "cuda",
        amp_dtype: str = "fp16",
        strict: bool = False,
    ) -> None:
        self.device = choose_device(
            device
        )

        if amp_dtype not in (
            "fp16",
            "bf16",
        ):
            raise ValueError(
                "amp_dtype must be 'fp16' or 'bf16'."
            )

        self.amp_dtype = amp_dtype
        self.strict = bool(
            strict
        )

        self.dfew_checkpoint = resolve_asset(
            dfew_checkpoint,
            DEFAULT_DFEW_NAME,
        )

        self.head_checkpoint = resolve_asset(
            head_checkpoint,
            DEFAULT_HEAD_NAME,
        )

        self.yunet_model = resolve_asset(
            yunet_model,
            DEFAULT_YUNET_NAME,
        )

        self.detector = YuNetDetector(
            self.yunet_model
        )

        (
            self.encoder,
            self.encoder_info,
        ) = load_direct_dfer_encoder(
            self.dfew_checkpoint,
            self.device,
        )

        (
            self.head,
            self.head_info,
        ) = load_eav_head(
            self.head_checkpoint,
            self.device,
        )

        self.model_identity = {
            "module_version": MODULE_VERSION,
            "classes": EMOTIONS,
            "dfew_checkpoint": str(
                self.dfew_checkpoint
            ),
            "dfew_checkpoint_sha256": self.encoder_info[
                "checkpoint_sha256"
            ],
            "eav_head_checkpoint": str(
                self.head_checkpoint
            ),
            "eav_head_sha256": self.head_info[
                "checkpoint_sha256"
            ],
            "yunet_model": str(
                self.yunet_model
            ),
            "yunet_sha256": sha256_file(
                self.yunet_model
            ),
            "official_dfer_repo_required": False,
            "openai_clip_checkpoint_required": False,
            "device": str(
                self.device
            ),
            "gpu": (
                torch.cuda.get_device_name(
                    0
                )
                if self.device.type == "cuda"
                else None
            ),
            "amp_dtype": self.amp_dtype,
            "preprocessing": (
                "16 face-focused RGB frames; 224x224; "
                "float/255; no CLIP mean/std normalization"
            ),
        }

        self._preflight()

    @torch.inference_mode()
    def _preflight(
        self,
    ) -> None:
        # Architecture / weight contract check.
        dummy = torch.zeros(
            (
                1,
                FRAMES_PER_WINDOW,
                3,
                CROP_SIZE,
                CROP_SIZE,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        amp_t = (
            torch.bfloat16
            if self.amp_dtype == "bf16"
            else torch.float16
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=amp_t,
            enabled=(
                self.device.type
                == "cuda"
            ),
        ):
            z = self.encoder(
                dummy
            )

            logits = self.head(
                z
            )

        if z.shape != (
            1,
            FEATURE_DIM,
        ):
            raise RuntimeError(
                f"Encoder preflight shape={tuple(z.shape)}"
            )

        if logits.shape != (
            1,
            NUM_CLASSES,
        ):
            raise RuntimeError(
                f"Head preflight shape={tuple(logits.shape)}"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "Preflight produced NaN/Inf."
            )

    def _unavailable_result(
        self,
        video_path: Path,
        reason: str,
        preprocessing: Optional[
            Dict[str, Any]
        ] = None,
        error: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        p = uniform_probs()

        return {
            "module": "VIDEO Emotion V2B",
            "module_version": MODULE_VERSION,
            "video_path": str(
                video_path
            ),
            "video_available": False,
            "video_probs": p,
            "pred_label_id": None,
            "pred_emotion": "NO_VIDEO_EVIDENCE",
            "confidence": None,
            "reason": reason,
            "error": error,
            "preprocessing": preprocessing,
            "elapsed_sec": elapsed_sec,
            "af4c_video_payload": {
                "video_probs": p,
                "video_available": 0,
            },
        }

    @torch.inference_mode()
    def predict(
        self,
        video_path: str | Path,
    ) -> Dict[str, Any]:
        t0 = time.time()

        path = Path(
            video_path
        ).expanduser().resolve()

        if not path.is_file():
            if self.strict:
                raise FileNotFoundError(
                    path
                )

            return self._unavailable_result(
                video_path=path,
                reason="VIDEO_FILE_MISSING",
                error=f"File not found: {path}",
                elapsed_sec=(
                    time.time()
                    - t0
                ),
            )

        try:
            prep = preprocess_video_clip(
                path,
                self.detector,
            )
        except Exception as exc:
            if self.strict:
                raise

            return self._unavailable_result(
                video_path=path,
                reason="VIDEO_PREPROCESSING_ERROR",
                error=repr(
                    exc
                ),
                elapsed_sec=(
                    time.time()
                    - t0
                ),
            )

        if not bool(
            prep[
                "video_available"
            ]
        ):
            prep_public = {
                key: value
                for key, value
                in prep.items()
                if key != "clip_tensor"
            }

            return self._unavailable_result(
                video_path=path,
                reason="FACE_NOT_OBSERVABLE",
                preprocessing=prep_public,
                elapsed_sec=(
                    time.time()
                    - t0
                ),
            )

        x = prep[
            "clip_tensor"
        ].unsqueeze(
            0
        ).to(
            self.device
        )

        amp_t = (
            torch.bfloat16
            if self.amp_dtype == "bf16"
            else torch.float16
        )

        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_t,
                enabled=(
                    self.device.type
                    == "cuda"
                ),
            ):
                z = self.encoder(
                    x
                )

                logits = self.head(
                    z
                )

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )

        except Exception as exc:
            if self.strict:
                raise

            prep_public = {
                key: value
                for key, value
                in prep.items()
                if key != "clip_tensor"
            }

            return self._unavailable_result(
                video_path=path,
                reason="V2B_INFERENCE_ERROR",
                preprocessing=prep_public,
                error=repr(
                    exc
                ),
                elapsed_sec=(
                    time.time()
                    - t0
                ),
            )

        p = (
            probs[
                0
            ]
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        p = np.clip(
            p,
            0.0,
            None,
        )

        total = float(
            p.sum()
        )

        if (
            not math.isfinite(
                total
            )
            or total <= 0
        ):
            raise RuntimeError(
                "Invalid V2B probability sum."
            )

        p = p / total

        pred = int(
            np.argmax(
                p
            )
        )

        confidence = float(
            p[
                pred
            ]
        )

        prep_public = {
            key: value
            for key, value
            in prep.items()
            if key != "clip_tensor"
        }

        result = {
            "module": "VIDEO Emotion V2B",
            "module_version": MODULE_VERSION,
            "video_path": str(
                path
            ),
            "video_available": True,
            "video_probs": p.tolist(),
            "pred_label_id": pred,
            "pred_emotion": EMOTIONS[
                pred
            ],
            "confidence": confidence,
            "reason": "FACE_OBSERVABLE_AND_V2B_SCORED",
            "error": None,
            "preprocessing": prep_public,
            "feature_norm": float(
                z[
                    0
                ]
                .float()
                .norm()
                .item()
            ),
            "elapsed_sec": float(
                time.time()
                - t0
            ),
            "af4c_video_payload": {
                "video_probs": p.tolist(),
                "video_available": 1,
            },
        }

        return result

    def predict_many(
        self,
        video_paths: Iterable[
            str | Path
        ],
    ) -> List[
        Dict[
            str,
            Any,
        ]
    ]:
        return [
            self.predict(
                path
            )
            for path in video_paths
        ]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone frozen V2B VIDEO emotion inference. "
            "Directly loads DFEW-set1-model.pth without the "
            "DFER-CLIP_official runtime repository."
        )
    )

    p.add_argument(
        "--video",
        required=True,
        help=(
            "Input real video clip; normally one 5-second analysis window."
        ),
    )

    p.add_argument(
        "--dfew-checkpoint",
        default=None,
        help=(
            "DFEW-set1-model.pth. Default: beside this script."
        ),
    )

    p.add_argument(
        "--head-checkpoint",
        default=None,
        help=(
            "best_v2b_frozen_dferclip_head_validation_selected.pt. "
            "Default: beside this script."
        ),
    )

    p.add_argument(
        "--yunet-model",
        default=None,
        help=(
            "face_detection_yunet_2023mar.onnx. "
            "Default: beside this script."
        ),
    )

    p.add_argument(
        "--device",
        choices=(
            "cuda",
            "cpu",
        ),
        default="cuda",
    )

    p.add_argument(
        "--amp-dtype",
        choices=(
            "fp16",
            "bf16",
        ),
        default="fp16",
    )

    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Raise runtime exceptions instead of returning "
            "a fail-safe unavailable result."
        ),
    )

    p.add_argument(
        "--json-output",
        default=None,
    )

    p.add_argument(
        "--show-model-identity",
        action="store_true",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(
        "=" * 118
    )

    print(
        "VIDEO EMOTION V2B — FINAL STANDALONE DEPLOYMENT"
    )

    print(
        "=" * 118
    )

    system = VideoEmotionV2B(
        dfew_checkpoint=args.dfew_checkpoint,
        head_checkpoint=args.head_checkpoint,
        yunet_model=args.yunet_model,
        device=args.device,
        amp_dtype=args.amp_dtype,
        strict=args.strict,
    )

    print(
        f"Device                    : {system.device}"
    )

    if system.device.type == "cuda":
        print(
            f"GPU                       : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Official DFER repo        : NOT REQUIRED"
    )

    print(
        f"DFEW checkpoint           : {system.dfew_checkpoint}"
    )

    print(
        f"EAV 5-class head          : {system.head_checkpoint}"
    )

    print(
        f"YuNet                     : {system.yunet_model}"
    )

    print(
        "Preprocessing             : 16 x RGB 224x224 /255, no CLIP normalization"
    )

    print()

    if args.show_model_identity:
        print(
            json.dumps(
                _json_safe(
                    system.model_identity
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

        print()

    result = system.predict(
        args.video
    )

    print(
        f"Video                     : {result['video_path']}"
    )

    print(
        f"Available                 : {result['video_available']}"
    )

    if result[
        "video_available"
    ]:
        print(
            f"Prediction                : "
            f"{result['pred_emotion']} "
            f"(class {result['pred_label_id']})"
        )

        print(
            f"Confidence                : "
            f"{result['confidence']:.4f}"
        )

        print(
            "Video probs               : "
            + np.array2string(
                np.asarray(
                    result[
                        "video_probs"
                    ]
                ),
                precision=5,
                separator=", ",
            )
        )

    else:
        print(
            "Prediction                : NO_VIDEO_EVIDENCE"
        )

        print(
            f"Reason                    : {result['reason']}"
        )

    prep = result.get(
        "preprocessing"
    )

    if prep is not None:
        print(
            f"Face detection            : "
            f"{prep['face_detected_frames']}/"
            f"{prep['sampled_frames']} "
            f"({prep['face_raw_detection_rate']:.4f})"
        )

        print(
            f"Fallback frames           : "
            f"{prep['face_fallback_frames']}"
        )

        print(
            f"Video duration            : "
            f"{prep['duration_sec']:.3f} s"
        )

    print(
        "AF4-C video payload       : "
        + json.dumps(
            result[
                "af4c_video_payload"
            ],
            ensure_ascii=False,
        )
    )

    print(
        f"Elapsed                   : "
        f"{float(result.get('elapsed_sec') or 0.0):.3f} s"
    )

    if result.get(
        "error"
    ):
        print(
            f"Error                     : "
            f"{result['error']}"
        )

    if args.json_output:
        output = Path(
            args.json_output
        ).expanduser().resolve()

        write_json(
            output,
            {
                "result": result,
                "model_identity": system.model_identity,
            },
        )

        print(
            f"JSON output               : {output}"
        )

    print(
        "=" * 118
    )

    # Face-unavailable is a valid deployment state, not a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
