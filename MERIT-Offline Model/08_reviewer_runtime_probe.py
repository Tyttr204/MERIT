#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARIT reviewer runtime probe.

Run this script WITH the Python interpreter that will execute main.py.

It performs import/readiness checks only:
- no model training
- no checkpoint forward
- no parameter changes
- no network access
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

VERSION = "MARIT-RUNTIME-PROBE.1.1"

REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("soundfile", "soundfile"),
    ("onnxruntime", "onnxruntime"),
    ("mne", "mne"),
    ("pyprep", "pyprep"),
    ("opencv-python", "cv2"),
    ("sounddevice", "sounddevice"),
    ("joblib", "joblib"),
    ("PyYAML", "yaml"),
    ("funasr", "funasr"),
    ("decord", "decord"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
]

DEPLOYMENT_FILES = [
    ("Realtime AV probe", Path("MERIT-Realtime Model") / "01_probe_av_devices.py"),
    ("EEG emotion", Path("EEG") / "EEG头部署模块.py"),
    ("EEG quality", Path("EEG") / "EEG质量检测模块（PyPREP）.py"),
    ("Audio emotion", Path("Audio") / "Audio头部署模块.py"),
    ("Audio quality", Path("Audio") / "Audio头质量检测模块（DNSMOS）.py"),
    ("Video emotion", Path("Video") / "VIDEO头部署脚本.py"),
    ("Video quality", Path("Video") / "VIDEO头质量检测模块（DOVER）.py"),
    ("Fusion", Path("Fusion") / "融合层部署模块.py"),
]


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def import_package(dist_name: str, module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "ok": True,
            "distribution": dist_name,
            "module": module_name,
            "version": json_safe(getattr(module, "__version__", None)),
        }
    except BaseException as exc:
        return {
            "ok": False,
            "distribution": dist_name,
            "module": module_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def import_file(path: Path, index: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "ok": False,
            "file": str(path),
            "error": "FILE_NOT_FOUND",
        }

    name = f"_marit_runtime_probe_{index}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("No import loader returned.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        return {"ok": True, "file": path.name}
    except BaseException as exc:
        sys.modules.pop(name, None)
        return {
            "ok": False,
            "file": path.name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_dover(root: Path) -> dict[str, Any]:
    dover_root = root / "Video" / "DOVER_official"
    if not dover_root.is_dir():
        return {
            "ok": False,
            "path": str(dover_root),
            "error": "DOVER_official directory not found",
        }

    inserted = False
    try:
        if str(dover_root) not in sys.path:
            sys.path.insert(0, str(dover_root))
            inserted = True
        importlib.import_module("dover.datasets")
        importlib.import_module("dover.models")
        return {"ok": True, "path": str(dover_root)}
    except BaseException as exc:
        return {
            "ok": False,
            "path": str(dover_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if inserted:
            try:
                sys.path.remove(str(dover_root))
            except ValueError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()

    root = args.deployment_root.expanduser().resolve()

    package_results = [
        import_package(dist, module)
        for dist, module in REQUIRED_IMPORTS
    ]

    deployment_results = [
        import_file(root / rel, i)
        for i, (_label, rel) in enumerate(DEPLOYMENT_FILES)
    ]

    dover = probe_dover(root)

    torch_info: dict[str, Any] = {}
    try:
        import torch
        torch_info = {
            "version": getattr(torch, "__version__", None),
            "cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
    except BaseException as exc:
        torch_info = {
            "error": f"{type(exc).__name__}: {exc}",
            "cuda_available": False,
        }

    missing_packages = [
        item["distribution"]
        for item in package_results
        if not item["ok"]
    ]
    module_errors = [
        item for item in deployment_results
        if not item["ok"]
    ]

    executables = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }

    ready = (
        root.is_dir()
        and not missing_packages
        and not module_errors
        and bool(dover.get("ok"))
        and all(executables.values())
    )

    result = {
        "schema": "marit.reviewer.runtime_probe.v1",
        "version": VERSION,
        "ready": ready,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "deployment_root": str(root),
        "packages": package_results,
        "missing_packages": missing_packages,
        "deployment_modules": deployment_results,
        "deployment_module_errors": module_errors,
        "dover": dover,
        "executables": executables,
        "torch": torch_info,
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
