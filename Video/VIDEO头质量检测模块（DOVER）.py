#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
VIDEO Quality V1 — Frozen Deployment Module
===========================================

Purpose
-------
A single deployment-oriented module for the VIDEO quality branch of the
EAV multimodal emotion-recognition system.

It exposes exactly the two signals required by frozen AF4-C:

    video_available : bool
    q_video         : float in [0, 1]

Scientific / engineering contract
---------------------------------
1) Availability / observability:
       YuNet detects whether a usable face is observable in a 5-s video clip.
       The frozen V1 rule is the same rule used in VQ2-B:
           >= 1 detected face among 16 uniformly sampled frames
               -> video_available = True
           0 / 16 detected
               -> video_available = False

2) Physical video quality:
       If video_available is True:
           official DOVER Technical raw score
               -> frozen VQ2-A isotonic calibrator
               -> q_video in [0,1]

       If video_available is False:
           q_video = 0
           DOVER is skipped by default because the video modality will not be
           used by AF4-C anyway.

3) Separation of concepts:
       - face observability controls availability
       - DOVER controls physical quality
       - V2B classifier confidence is NOT mixed into q_video

4) Frozen AF4-C threshold:
       tau = 0.80

       available and q_video >= 0.80:
           quality_state = HEALTHY

       available and q_video < 0.80:
           quality_state = DEGRADED

       unavailable:
           quality_state = UNAVAILABLE

This module DOES NOT:
    - run V2B emotion classification
    - train or update any model
    - alter F4 / AF4-B / AF4-C
    - use classifier confidence as quality

Recommended runtime usage
-------------------------
from video_quality_v1_deployment import VideoQualityV1

vq = VideoQualityV1()
result = vq.assess(r"C:\path\to\five_second_window.mp4")

q_video = result["q_video"]
video_available = result["video_available"]

# Direct AF4-C-compatible payload:
result["af4c_payload"]
# {
#   "quality": {"video": ...},
#   "available": {"video": 0 or 1}
# }

Command line
------------
python .\video_quality_v1_deployment.py `
  --video "C:\path\to\window.mp4"

Save JSON:
python .\video_quality_v1_deployment.py `
  --video "C:\path\to\window.mp4" `
  --json-output ".\video_quality_result.json"

Deployment note
---------------
The module is calibrated/validated on 5-second EAV windows. In a live robot,
the camera stream should normally be buffered into the same 5-second analysis
window before calling assess().
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import torch
import yaml


# =============================================================================
# Frozen V1 constants
# =============================================================================

MODULE_VERSION = "VIDEO-QUALITY-V1.0-FROZEN"

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen AF4-C quality threshold.
ROUTER_TAU = 0.80

# VQ2-B / frozen V2A-compatible face observability.
FACE_SAMPLE_FRAMES = 16
FACE_SCORE_THRESHOLD = 0.75
FACE_NMS_THRESHOLD = 0.30
FACE_TOP_K = 5000
FACE_DETECT_MAX_SIDE = 720
MIN_FACE_DETECTED_FRAMES = 1

# Deterministic DOVER view sampling.
DOVER_SEED = 20260917
DOVER_DOPT_KEY = "val-l1080p"

# Exact DOVER raw-score semantics used by VQ1-C / VQ2-A.
# Official DOVER single-video inference uses result[0] for Technical and
# result[1] for Aesthetic.
DOVER_TECH_MEAN = 0.1107
DOVER_TECH_STD = 0.07355
DOVER_AESTH_MEAN = -0.08285
DOVER_AESTH_STD = 0.03774
DOVER_TECH_FUSION_WEIGHT = 0.6104
DOVER_AESTH_FUSION_WEIGHT = 0.3896

DEFAULT_DOVER_REPO_REL = Path(
    "EAV_dataset/models/video_quality/DOVER_official"
)
DEFAULT_DOVER_CHECKPOINT_REL = (
    DEFAULT_DOVER_REPO_REL / "pretrained_weights/DOVER.pth"
)
DEFAULT_DOVER_CONFIG_REL = DEFAULT_DOVER_REPO_REL / "dover.yml"

DEFAULT_VQ2A_RUN_NAME = "vq2a_dover_qvideo_calibration_20260917_173627"
DEFAULT_CALIBRATOR_REL = Path(
    "EAV_dataset/models/video_quality"
) / DEFAULT_VQ2A_RUN_NAME / "video_quality_calibrator.joblib"

DEFAULT_YUNET_ASCII = (
    Path.home()
    / ".cache"
    / "eav_yunet_runtime"
    / "face_detection_yunet_2023mar.onnx"
)


# =============================================================================
# Utilities
# =============================================================================

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def sigmoid(x: float) -> float:
    # Stable enough for the DOVER score range here.
    return float(1.0 / (1.0 + np.exp(-float(x))))


def resolve_project_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "EAV_dataset").exists():
        return cwd

    here = Path(__file__).resolve().parent
    if (here / "EAV_dataset").exists():
        return here

    return DEFAULT_PROJECT_ROOT.resolve()


def resolve_under_project(
    project_root: Path,
    explicit: Optional[str],
    default_relative: Path,
    expect_file: bool = True,
) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = project_root / p
        p = p.resolve()
    else:
        p = (project_root / default_relative).resolve()

    exists = p.is_file() if expect_file else p.is_dir()
    if not exists:
        kind = "file" if expect_file else "directory"
        raise FileNotFoundError(f"Required {kind} not found: {p}")

    return p


def load_calibrator(path: Path):
    bundle = joblib.load(path)

    if isinstance(bundle, dict):
        if "model" not in bundle:
            raise RuntimeError(
                f"Calibrator bundle has no 'model' key: {path}"
            )
        model = bundle["model"]
        metadata = {
            k: v
            for k, v in bundle.items()
            if k != "model"
        }
    else:
        model = bundle
        metadata = {}

    if not hasattr(model, "predict"):
        raise RuntimeError(
            "Loaded VQ2-A calibrator does not implement predict()."
        )

    # Runtime contract smoke test.
    probe = np.asarray(
        model.predict(np.asarray([0.10], dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)

    if probe.size != 1 or not np.isfinite(probe[0]):
        raise RuntimeError("VQ2-A calibrator preflight failed.")

    return model, metadata


# =============================================================================
# Face observability
# =============================================================================

class YuNetFaceObservability:
    """
    Frozen V1 face-observability detector.

    This reproduces the detector-selection logic used in the successful V2A /
    VQ2-B pipeline, but does not need to generate face crops.
    """

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = FACE_SCORE_THRESHOLD,
        nms_threshold: float = FACE_NMS_THRESHOLD,
        top_k: int = FACE_TOP_K,
        detect_max_side: int = FACE_DETECT_MAX_SIDE,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        # OpenCV FaceDetectorYN on Windows was previously validated using an
        # ASCII-only model path.
        if os.name == "nt" and not str(model_path).isascii():
            raise RuntimeError(
                "YuNet model path must be ASCII-only on Windows. "
                f"Current path: {model_path}"
            )

        self.model_path = model_path
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.detect_max_side = int(detect_max_side)

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

    @staticmethod
    def _iou(
        a: Optional[np.ndarray],
        b: np.ndarray,
    ) -> float:
        if a is None:
            return 0.0

        ax, ay, aw, ah = [float(x) for x in a]
        bx, by, bw, bh = [float(x) for x in b]

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = aw * ah + bw * bh - inter

        return inter / union if union > 0 else 0.0

    def _detect_one(
        self,
        frame_bgr: np.ndarray,
        prev_box: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float]:
        h0, w0 = frame_bgr.shape[:2]

        scale = min(
            1.0,
            self.detect_max_side / float(max(h0, w0)),
        )

        if scale < 1.0:
            wd = max(1, int(round(w0 * scale)))
            hd = max(1, int(round(h0 * scale)))
            det_img = cv2.resize(
                frame_bgr,
                (wd, hd),
                interpolation=cv2.INTER_AREA,
            )
        else:
            det_img = frame_bgr
            hd, wd = h0, w0

        self.detector.setInputSize((wd, hd))
        _, faces = self.detector.detect(det_img)

        if faces is None or len(faces) == 0:
            return None, float("nan")

        prev_scaled = (
            prev_box * scale
            if prev_box is not None
            else None
        )

        img_area = float(wd * hd)
        cx_img = wd / 2.0
        cy_img = hd / 2.0

        best_box: Optional[np.ndarray] = None
        best_conf = float("nan")
        best_rank = -1e30

        for f in faces:
            x, y, w, h = [float(z) for z in f[:4]]
            conf = float(f[-1])
            box = np.asarray([x, y, w, h], dtype=np.float32)

            area_term = min(
                1.0,
                (w * h) / max(img_area * 0.15, 1.0),
            )

            cx = x + w / 2.0
            cy = y + h / 2.0

            dist = math.sqrt(
                ((cx - cx_img) / max(wd, 1)) ** 2
                + ((cy - cy_img) / max(hd, 1)) ** 2
            )

            center_term = max(0.0, 1.0 - 2.0 * dist)
            track_term = self._iou(prev_scaled, box)

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
            return None, float("nan")

        return (
            (best_box / scale).astype(np.float32),
            best_conf,
        )

    @staticmethod
    def _uniform_indices(
        frame_count: int,
        n: int = FACE_SAMPLE_FRAMES,
    ) -> List[int]:
        if frame_count <= 0:
            raise RuntimeError(
                f"Invalid video frame_count={frame_count}"
            )

        # Same "bin-center" style as V2A/VQ2-B over the full input clip.
        edges = np.linspace(
            0,
            frame_count,
            n + 1,
            endpoint=True,
        )

        out: List[int] = []

        for i in range(n):
            center = int(
                math.floor(
                    (edges[i] + edges[i + 1] - 1.0) / 2.0
                )
            )
            out.append(
                max(
                    0,
                    min(center, frame_count - 1),
                )
            )

        return out

    def assess(
        self,
        video_path: Path,
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(
                round(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )
            )

            if not math.isfinite(fps) or fps <= 1e-6:
                fps = 30.0

            if frame_count <= 0:
                raise RuntimeError(
                    f"Video has invalid frame count: {video_path}"
                )

            duration_sec = frame_count / fps

            targets = self._uniform_indices(
                frame_count,
                FACE_SAMPLE_FRAMES,
            )

            target_set = set(targets)
            frames: Dict[int, np.ndarray] = {}

            fi = 0
            max_target = max(targets)

            while fi <= max_target:
                ok, frame = cap.read()
                if not ok:
                    break

                if fi in target_set:
                    frames[fi] = frame.copy()

                fi += 1

            sampled_frames: List[np.ndarray] = []

            for index in targets:
                if index not in frames:
                    raise RuntimeError(
                        f"Could not decode sampled frame {index} "
                        f"from {video_path}"
                    )
                sampled_frames.append(frames[index])

        finally:
            cap.release()

        prev: Optional[np.ndarray] = None
        detected = 0
        scores: List[float] = []

        for frame in sampled_frames:
            box, conf = self._detect_one(
                frame,
                prev,
            )

            if box is not None:
                detected += 1
                prev = box

                if math.isfinite(conf):
                    scores.append(float(conf))

        available = detected >= MIN_FACE_DETECTED_FRAMES

        return {
            "video_available": bool(available),
            "sampled_frames": FACE_SAMPLE_FRAMES,
            "detected_frames": int(detected),
            "raw_detection_rate": float(
                detected / FACE_SAMPLE_FRAMES
            ),
            "mean_detection_score": (
                float(np.mean(scores))
                if scores
                else None
            ),
            "fps": float(fps),
            "frame_count": int(frame_count),
            "duration_sec": float(duration_sec),
            "availability_rule": (
                f"detected_frames >= {MIN_FACE_DETECTED_FRAMES} "
                f"of {FACE_SAMPLE_FRAMES}"
            ),
        }


# =============================================================================
# DOVER Technical scorer
# =============================================================================

class DOVERTechnicalScorer:
    """
    Frozen official DOVER inference.

    Uses the same score semantics as VQ1-C:
        results[0] -> technical_raw
        results[1] -> aesthetic_raw

    qVideo calibration uses ONLY technical_raw.
    """

    def __init__(
        self,
        repo: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        dopt_key: str = DOVER_DOPT_KEY,
        seed: int = DOVER_SEED,
    ) -> None:
        self.repo = repo
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dopt_key = str(dopt_key)
        self.seed = int(seed)

        if not (repo / "dover").is_dir():
            raise FileNotFoundError(
                f"Invalid DOVER repo: {repo}"
            )

        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            datasets_mod = importlib.import_module(
                "dover.datasets"
            )
            models_mod = importlib.import_module(
                "dover.models"
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot import official DOVER. "
                "Activate the .venv-video environment and ensure "
                "the DOVER official dependencies are installed."
            ) from exc

        self.UnifiedFrameSampler = (
            datasets_mod.UnifiedFrameSampler
        )
        self.spatial_temporal_view_decomposition = (
            datasets_mod.spatial_temporal_view_decomposition
        )
        DOVER = models_mod.DOVER

        opt = yaml.safe_load(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        if self.dopt_key not in opt["data"]:
            raise RuntimeError(
                f"DOVER config has no data key '{self.dopt_key}'."
            )

        self.dopt = opt["data"][self.dopt_key]["args"]

        self.model = DOVER(
            **opt["model"]["args"]
        ).to(
            self.device
        )

        state = self._load_checkpoint(
            checkpoint_path
        )

        # Official DOVER script loads this state directly.
        self.model.load_state_dict(
            state,
            strict=True,
        )
        self.model.eval()

        self.temporal_samplers: Dict[str, Any] = {}

        for stype, sopt in self.dopt[
            "sample_types"
        ].items():
            if "t_frag" not in sopt:
                self.temporal_samplers[
                    stype
                ] = self.UnifiedFrameSampler(
                    sopt["clip_len"],
                    sopt["num_clips"],
                    sopt["frame_interval"],
                )
            else:
                self.temporal_samplers[
                    stype
                ] = self.UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        self.mean = torch.FloatTensor(
            [123.675, 116.28, 103.53]
        )
        self.std = torch.FloatTensor(
            [58.395, 57.12, 57.375]
        )

    @staticmethod
    def _load_checkpoint(
        path: Path,
    ) -> Dict[str, Any]:
        try:
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(
                path,
                map_location="cpu",
            )

        # Official DOVER.pth is normally already a state_dict.
        # Keep a compatibility fallback without changing normal behavior.
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and not any(
                str(k).startswith(
                    ("technical", "aesthetic", "backbone")
                )
                for k in state.keys()
            )
        ):
            candidate = state["state_dict"]
            if isinstance(candidate, dict):
                state = candidate

        if not isinstance(state, dict):
            raise RuntimeError(
                f"Unexpected DOVER checkpoint object: {type(state)}"
            )

        return state

    def _set_deterministic_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @torch.inference_mode()
    def score(
        self,
        video_path: Path,
    ) -> Dict[str, float]:
        self._set_deterministic_seed()

        views, _ = (
            self.spatial_temporal_view_decomposition(
                str(video_path),
                self.dopt["sample_types"],
                self.temporal_samplers,
            )
        )

        for key, value in list(views.items()):
            num_clips = (
                self.dopt["sample_types"][key]
                .get("num_clips", 1)
            )

            views[key] = (
                (
                    (
                        value.permute(1, 2, 3, 0)
                        - self.mean
                    )
                    / self.std
                )
                .permute(3, 0, 1, 2)
                .reshape(
                    value.shape[0],
                    num_clips,
                    -1,
                    *value.shape[2:],
                )
                .transpose(0, 1)
                .to(self.device)
            )

        outputs = self.model(views)

        results = [
            float(r.mean().item())
            for r in outputs
        ]

        if len(results) < 2:
            raise RuntimeError(
                f"DOVER returned {len(results)} outputs; expected >=2."
            )

        technical_raw = float(results[0])
        aesthetic_raw = float(results[1])

        if not (
            math.isfinite(technical_raw)
            and math.isfinite(aesthetic_raw)
        ):
            raise RuntimeError(
                "DOVER produced non-finite raw scores."
            )

        technical_z = (
            technical_raw - DOVER_TECH_MEAN
        ) / DOVER_TECH_STD

        aesthetic_z = (
            aesthetic_raw - DOVER_AESTH_MEAN
        ) / DOVER_AESTH_STD

        technical_01 = sigmoid(
            technical_z
        )
        aesthetic_01 = sigmoid(
            aesthetic_z
        )

        overall_logit = (
            technical_z
            * DOVER_TECH_FUSION_WEIGHT
            + aesthetic_z
            * DOVER_AESTH_FUSION_WEIGHT
        )

        overall_01 = sigmoid(
            overall_logit
        )

        return {
            "technical_raw": technical_raw,
            "aesthetic_raw": aesthetic_raw,
            "technical_z": float(technical_z),
            "aesthetic_z": float(aesthetic_z),
            "technical_01": technical_01,
            "aesthetic_01": aesthetic_01,
            "overall_01": overall_01,
        }


# =============================================================================
# Public deployment API
# =============================================================================

@dataclass
class VideoQualityV1Paths:
    project_root: Path
    dover_repo: Path
    dover_config: Path
    dover_checkpoint: Path
    calibrator: Path
    yunet: Path


class VideoQualityV1:
    """
    Public VIDEO Quality V1 deployment class.

    Initialize once, reuse for many 5-second video windows.
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        dover_repo: Optional[str] = None,
        dover_config: Optional[str] = None,
        dover_checkpoint: Optional[str] = None,
        calibrator: Optional[str] = None,
        yunet_model: Optional[str] = None,
        device: str = "cuda",
        router_tau: float = ROUTER_TAU,
        score_dover_if_unavailable: bool = False,
        strict: bool = False,
    ) -> None:
        self.project_root = resolve_project_root(
            project_root
        )

        repo = resolve_under_project(
            self.project_root,
            dover_repo,
            DEFAULT_DOVER_REPO_REL,
            expect_file=False,
        )

        config_path = resolve_under_project(
            self.project_root,
            dover_config,
            DEFAULT_DOVER_CONFIG_REL,
            expect_file=True,
        )

        checkpoint_path = resolve_under_project(
            self.project_root,
            dover_checkpoint,
            DEFAULT_DOVER_CHECKPOINT_REL,
            expect_file=True,
        )

        calibrator_path = resolve_under_project(
            self.project_root,
            calibrator,
            DEFAULT_CALIBRATOR_REL,
            expect_file=True,
        )

        if yunet_model:
            yp = Path(yunet_model).expanduser()
            if not yp.is_absolute():
                yp = self.project_root / yp
            yunet_path = yp.resolve()
        else:
            yunet_path = DEFAULT_YUNET_ASCII.resolve()

        if not yunet_path.is_file():
            raise FileNotFoundError(
                "YuNet runtime model not found: "
                f"{yunet_path}"
            )

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but torch.cuda.is_available() is False."
                )
            self.device = torch.device("cuda")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(
                "device must be 'cuda' or 'cpu'"
            )

        self.router_tau = float(router_tau)

        if not 0.0 <= self.router_tau <= 1.0:
            raise ValueError(
                "router_tau must be in [0,1]."
            )

        self.score_dover_if_unavailable = bool(
            score_dover_if_unavailable
        )
        self.strict = bool(strict)

        self.paths = VideoQualityV1Paths(
            project_root=self.project_root,
            dover_repo=repo,
            dover_config=config_path,
            dover_checkpoint=checkpoint_path,
            calibrator=calibrator_path,
            yunet=yunet_path,
        )

        self.face = YuNetFaceObservability(
            yunet_path
        )

        self.calibrator, self.calibrator_metadata = (
            load_calibrator(
                calibrator_path
            )
        )

        self.dover = DOVERTechnicalScorer(
            repo=repo,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        self.model_identity = {
            "module_version": MODULE_VERSION,
            "router_tau": self.router_tau,
            "dover_repo": str(repo),
            "dover_config": str(config_path),
            "dover_checkpoint": str(
                checkpoint_path
            ),
            "dover_checkpoint_sha256": sha256_file(
                checkpoint_path
            ),
            "vq2a_calibrator": str(
                calibrator_path
            ),
            "vq2a_calibrator_sha256": sha256_file(
                calibrator_path
            ),
            "yunet_model": str(yunet_path),
            "yunet_sha256": sha256_file(
                yunet_path
            ),
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if self.device.type == "cuda"
                else None
            ),
        }

    def _calibrate_q(
        self,
        technical_raw: float,
    ) -> float:
        pred = np.asarray(
            self.calibrator.predict(
                np.asarray(
                    [technical_raw],
                    dtype=np.float64,
                )
            ),
            dtype=np.float64,
        ).reshape(-1)

        if pred.size != 1 or not np.isfinite(
            pred[0]
        ):
            raise RuntimeError(
                "VQ2-A calibrator returned an invalid value."
            )

        return float(
            np.clip(
                pred[0],
                0.0,
                1.0,
            )
        )

    def _unavailable_result(
        self,
        video_path: Path,
        face_result: Optional[Dict[str, Any]],
        reason: str,
        error: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        result = {
            "module": "VIDEO Quality V1",
            "module_version": MODULE_VERSION,
            "video_path": str(video_path),
            "video_available": False,
            "q_video": 0.0,
            "quality_state": "UNAVAILABLE",
            "router_tau": self.router_tau,
            "physical_quality": {
                "dover_used": False,
                "technical_raw": None,
                "q_video_physical": None,
            },
            "face_observability": face_result,
            "reason": reason,
            "error": error,
            "af4c_payload": {
                "quality": {
                    "video": 0.0,
                },
                "available": {
                    "video": 0,
                },
            },
            "elapsed_sec": elapsed_sec,
        }
        return result

    def assess(
        self,
        video_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Assess one video clip.

        Returns an AF4-C-ready dictionary.
        """
        t0 = time.time()

        path = Path(video_path).expanduser().resolve()

        if not path.is_file():
            if self.strict:
                raise FileNotFoundError(path)

            return self._unavailable_result(
                video_path=path,
                face_result=None,
                reason="VIDEO_FILE_MISSING",
                error=f"File not found: {path}",
                elapsed_sec=time.time() - t0,
            )

        # ------------------------------------------------------------------
        # 1) Availability / face observability
        # ------------------------------------------------------------------
        try:
            face_result = self.face.assess(
                path
            )
        except Exception as exc:
            if self.strict:
                raise

            return self._unavailable_result(
                video_path=path,
                face_result=None,
                reason="VIDEO_OR_FACE_OBSERVABILITY_ERROR",
                error=repr(exc),
                elapsed_sec=time.time() - t0,
            )

        available = bool(
            face_result[
                "video_available"
            ]
        )

        # Default deployment behavior:
        # no face -> immediately unavailable, no expensive DOVER call.
        if (
            not available
            and not self.score_dover_if_unavailable
        ):
            return self._unavailable_result(
                video_path=path,
                face_result=face_result,
                reason="FACE_NOT_OBSERVABLE",
                elapsed_sec=time.time() - t0,
            )

        # ------------------------------------------------------------------
        # 2) Physical quality / DOVER
        # ------------------------------------------------------------------
        try:
            dover_result = self.dover.score(
                path
            )

            q_physical = self._calibrate_q(
                dover_result[
                    "technical_raw"
                ]
            )

        except Exception as exc:
            if self.strict:
                raise

            # If DOVER itself fails, the quality signal is not trustworthy.
            # Fail safe: mark VIDEO unavailable rather than silently using q=1.
            result = self._unavailable_result(
                video_path=path,
                face_result=face_result,
                reason="DOVER_OR_CALIBRATOR_ERROR",
                error=repr(exc),
                elapsed_sec=time.time() - t0,
            )

            result["physical_quality"][
                "dover_used"
            ] = True

            return result

        # If face was unavailable but caller explicitly requested DOVER for
        # diagnostics, preserve the hard availability contract.
        if not available:
            result = self._unavailable_result(
                video_path=path,
                face_result=face_result,
                reason="FACE_NOT_OBSERVABLE",
                elapsed_sec=time.time() - t0,
            )

            result["physical_quality"] = {
                "dover_used": True,
                **dover_result,
                "q_video_physical": q_physical,
                "note": (
                    "DOVER was evaluated for diagnostics only. "
                    "Final q_video remains 0 because face observability "
                    "declared the VIDEO modality unavailable."
                ),
            }

            return result

        # ------------------------------------------------------------------
        # 3) Available: qVideo comes from frozen VQ2-A calibrator
        # ------------------------------------------------------------------
        q_video = float(q_physical)

        quality_state = (
            "HEALTHY"
            if q_video >= self.router_tau
            else "DEGRADED"
        )

        result = {
            "module": "VIDEO Quality V1",
            "module_version": MODULE_VERSION,
            "video_path": str(path),
            "video_available": True,
            "q_video": q_video,
            "quality_state": quality_state,
            "router_tau": self.router_tau,
            "physical_quality": {
                "dover_used": True,
                **dover_result,
                "q_video_physical": q_video,
                "calibrator": "VQ2-A frozen isotonic",
            },
            "face_observability": face_result,
            "reason": (
                "FACE_OBSERVABLE_AND_PHYSICAL_QUALITY_SCORED"
            ),
            "error": None,
            "af4c_payload": {
                "quality": {
                    "video": q_video,
                },
                "available": {
                    "video": 1,
                },
            },
            "elapsed_sec": float(
                time.time() - t0
            ),
        }

        return result

    def assess_many(
        self,
        video_paths: Iterable[str | Path],
    ) -> List[Dict[str, Any]]:
        return [
            self.assess(path)
            for path in video_paths
        ]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "VIDEO Quality V1: YuNet availability + DOVER Technical "
            "+ frozen VQ2-A qVideo calibration."
        )
    )

    p.add_argument(
        "--video",
        required=True,
        help="Input video clip, normally one 5-s analysis window.",
    )

    p.add_argument(
        "--project-root",
        default=None,
    )

    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )

    p.add_argument(
        "--dover-repo",
        default=None,
    )

    p.add_argument(
        "--dover-config",
        default=None,
    )

    p.add_argument(
        "--dover-checkpoint",
        default=None,
    )

    p.add_argument(
        "--calibrator",
        default=None,
    )

    p.add_argument(
        "--yunet-model",
        default=None,
    )

    p.add_argument(
        "--router-tau",
        type=float,
        default=ROUTER_TAU,
    )

    p.add_argument(
        "--score-dover-if-unavailable",
        action="store_true",
        help=(
            "Diagnostic only: still score DOVER when no face is observable. "
            "Final video_available remains 0 and q_video remains 0."
        ),
    )

    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Raise exceptions instead of fail-safe unavailable outputs."
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

    print("=" * 118)
    print("VIDEO QUALITY V1 — FROZEN DEPLOYMENT MODULE")
    print("=" * 118)

    system = VideoQualityV1(
        project_root=args.project_root,
        dover_repo=args.dover_repo,
        dover_config=args.dover_config,
        dover_checkpoint=args.dover_checkpoint,
        calibrator=args.calibrator,
        yunet_model=args.yunet_model,
        device=args.device,
        router_tau=args.router_tau,
        score_dover_if_unavailable=(
            args.score_dover_if_unavailable
        ),
        strict=args.strict,
    )

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

    result = system.assess(
        args.video
    )

    print(f"Video                     : {result['video_path']}")
    print(f"Available                 : {result['video_available']}")
    print(f"q_video                   : {result['q_video']:.4f}")
    print(f"Quality state             : {result['quality_state']}")
    print(f"Reason                    : {result['reason']}")

    face = result.get(
        "face_observability"
    )

    if face is not None:
        print(
            f"Face detection            : "
            f"{face['detected_frames']}/{face['sampled_frames']} "
            f"({face['raw_detection_rate']:.4f})"
        )

    physical = result.get(
        "physical_quality",
        {}
    )

    if physical.get(
        "technical_raw"
    ) is not None:
        print(
            f"DOVER Technical raw       : "
            f"{physical['technical_raw']:.8f}"
        )

        print(
            f"DOVER Technical [0,1]     : "
            f"{physical['technical_01']:.4f}"
        )

    print(
        "AF4-C payload             : "
        + json.dumps(
            result["af4c_payload"],
            ensure_ascii=False,
        )
    )

    print(
        f"Elapsed                   : "
        f"{float(result.get('elapsed_sec') or 0.0):.3f} s"
    )

    if result.get("error"):
        print(
            f"Error                     : "
            f"{result['error']}"
        )

    if args.json_output:
        out = Path(
            args.json_output
        ).expanduser().resolve()

        write_json(
            out,
            {
                "result": result,
                "model_identity": system.model_identity,
            },
        )

        print(
            f"JSON output               : {out}"
        )

    print("=" * 118)

    # Deployment-friendly exit policy:
    # unavailable is a valid system state, not a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
