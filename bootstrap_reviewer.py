#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Bootstrap the repository-local MERIT reviewer Python environment.

Validated baseline:
- Python 3.11 (validated on 3.11.9)
- PyTorch 2.7.1 / torchvision 0.22.1 / torchaudio 2.7.1
- CUDA 12.8 profile on Windows/Linux with NVIDIA GPU
- CPU profile is available as a portability fallback

This script:
1) locates the repository from this script's own directory;
2) validates the repository by required marker files/directories, NOT by folder name;
3) creates .venv-reviewer inside the repository;
4) installs the official PyTorch wheel profile (CUDA 12.8 or CPU);
5) installs requirements-reviewer.txt;
6) runs pip check;
7) checks tkinter, ffmpeg and ffprobe;
8) runs MERIT-Offline Model/08_reviewer_runtime_probe.py;
9) prints the exact GUI launch command.

It does NOT download MERIT model assets. Release-asset bootstrap is a separate step.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Sequence

VERSION = "MERIT-REVIEWER-BOOTSTRAP.1.2"

TORCH = "2.7.1"
TORCHVISION = "0.22.1"
TORCHAUDIO = "2.7.1"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

REPO_MARKER_FILES = (
    Path("main.py"),
    Path("system_config.json"),
    Path("requirements-reviewer.txt"),
    Path("MERIT-Offline Model") / "03_reviewer_demo.py",
    Path("MERIT-Offline Model") / "08_reviewer_runtime_probe.py",
)

REPO_MARKER_DIRS = (
    Path("EEG"),
    Path("Audio"),
    Path("Video"),
    Path("Fusion"),
    Path("Quality"),
    Path("MERIT-Offline Model"),
    Path("MERIT-Realtime Model"),
)


class BootstrapError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BootstrapError(message)


def validate_repository_root(root: Path) -> None:
    missing: list[str] = []

    for rel in REPO_MARKER_FILES:
        if not (root / rel).is_file():
            missing.append(rel.as_posix())

    for rel in REPO_MARKER_DIRS:
        if not (root / rel).is_dir():
            missing.append(rel.as_posix() + "/")

    require(
        not missing,
        (
            "This does not look like a complete MERIT repository checkout. "
            "Missing repository marker(s): "
            + ", ".join(missing)
        ),
    )

    require(
        "最终部署" not in root.parts,
        "Refusing to bootstrap inside the private 最终部署 tree.",
    )


def run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    print()
    print(">", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=None,
        stderr=None,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise BootstrapError(f"Command failed with exit code {proc.returncode}: {cmd[0]}")
    return proc


def venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def detect_profile(requested: str) -> str:
    if requested in ("cuda", "cpu"):
        return requested

    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def check_base_python() -> None:
    require(
        sys.version_info[:2] == (3, 11),
        (
            "Python 3.11 is required for the validated reviewer environment; "
            f"current interpreter is {sys.version.split()[0]}."
        ),
    )


def ensure_ffmpeg() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if ffmpeg and ffprobe:
        print(f"ffmpeg : {ffmpeg}")
        print(f"ffprobe: {ffprobe}")
        return

    system = platform.system()
    if system == "Windows":
        hint = (
            "Install FFmpeg and ensure ffmpeg.exe and ffprobe.exe are on PATH. "
            "On Windows, winget is one possible installation method."
        )
    elif system == "Darwin":
        hint = (
            "Install FFmpeg and ensure ffmpeg and ffprobe are on PATH. "
            "For example: brew install ffmpeg."
        )
    else:
        hint = (
            "Install FFmpeg and ensure ffmpeg and ffprobe are on PATH. "
            "For example on Debian/Ubuntu: sudo apt install ffmpeg."
        )

    raise BootstrapError(hint)


def torch_install(root: Path, py: Path, profile: str) -> None:
    index = CUDA_INDEX if profile == "cuda" else CPU_INDEX
    run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            f"torch=={TORCH}",
            f"torchvision=={TORCHVISION}",
            f"torchaudio=={TORCHAUDIO}",
            "--index-url",
            index,
        ],
        cwd=root,
    )


def torch_probe(root: Path, py: Path, profile: str) -> None:
    code = (
        "import json,torch,torchvision,torchaudio;"
        "print(json.dumps({"
        "'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,"
        "'torchaudio':torchaudio.__version__,"
        "'cuda_build':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'gpu':(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
        "}))"
    )

    proc = subprocess.run(
        [str(py), "-c", code],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    require(proc.returncode == 0, f"PyTorch probe failed:\n{proc.stderr}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    print("PyTorch probe:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if profile == "cuda":
        require(
            payload.get("cuda_available") is True,
            (
                "CUDA profile was installed but torch.cuda.is_available() is False. "
                "Check the NVIDIA driver, or recreate the environment with --device cpu."
            ),
        )


def tkinter_probe(root: Path, py: Path) -> None:
    proc = subprocess.run(
        [str(py), "-c", "import tkinter; print(tkinter.TkVersion)"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    require(
        proc.returncode == 0,
        (
            "tkinter is unavailable in this Python installation. "
            "Install a Python 3.11 build that includes Tk/Tcl.\n"
            + proc.stderr
        ),
    )


def runtime_probe(root: Path, py: Path) -> None:
    probe = root / "MERIT-Offline Model" / "08_reviewer_runtime_probe.py"
    require(probe.is_file(), f"Missing runtime probe: {probe}")

    proc = subprocess.run(
        [
            str(py),
            "-X",
            "utf8",
            str(probe),
            "--deployment-root",
            str(root),
        ],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
        check=False,
    )

    lines = [x for x in proc.stdout.splitlines() if x.strip()]
    require(lines, f"Runtime probe produced no JSON:\n{proc.stderr}")

    report = json.loads(lines[-1])

    if report.get("ready") is not True:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise BootstrapError("Reviewer runtime probe did not pass.")

    print("Reviewer runtime probe: PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="PyTorch wheel profile. Auto selects CUDA when nvidia-smi is visible.",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild .venv-reviewer if it already exists.",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install packages; only validate an existing .venv-reviewer.",
    )
    p.add_argument("--version", action="version", version=VERSION)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # The checkout directory can be named MERIT, MERIT_CLEAN_TEST,
    # anonymous-review-artifact, or anything else. A reviewer should not be
    # forced to clone into one exact folder name.
    root = Path(__file__).resolve().parent
    validate_repository_root(root)

    print("=" * 86)
    print("MERIT REVIEWER ENVIRONMENT BOOTSTRAP")
    print("=" * 86)
    print(f"Version    : {VERSION}")
    print(f"Repository : {root}")
    print(f"Python     : {sys.version.split()[0]}")

    check_base_python()

    requirements = root / "requirements-reviewer.txt"
    require(requirements.is_file(), "requirements-reviewer.txt is missing.")

    ensure_ffmpeg()

    env_dir = root / ".venv-reviewer"

    if args.recreate and env_dir.exists():
        print(f"Removing existing reviewer environment: {env_dir}")
        shutil.rmtree(env_dir)

    if not env_dir.exists():
        print(f"Creating repository-local environment: {env_dir.name}")
        venv.EnvBuilder(
            with_pip=True,
            clear=False,
            symlinks=False,
        ).create(env_dir)

    py = venv_python(env_dir)
    require(py.is_file(), f"Virtual-environment Python missing: {py}")

    profile = detect_profile(args.device)
    print(f"Reviewer device profile: {profile.upper()}")

    if not args.skip_install:
        run(
            [str(py), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=root,
        )
        torch_install(root, py, profile)
        run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements),
            ],
            cwd=root,
        )

    run([str(py), "-m", "pip", "check"], cwd=root)
    tkinter_probe(root, py)
    torch_probe(root, py, profile)
    runtime_probe(root, py)

    rel_py = (
        r".\.venv-reviewer\Scripts\python.exe"
        if os.name == "nt"
        else "./.venv-reviewer/bin/python"
    )

    print()
    print("=" * 86)
    print("MERIT REVIEWER ENVIRONMENT READY")
    print("=" * 86)
    print("Environment : .venv-reviewer")
    print(f"Profile     : {profile.upper()}")
    print("Runtime     : PASS")
    print()
    print("Launch MARIT - Offline Mode with:")

    if os.name == "nt":
        print(
            f'  & "{rel_py}" ".\\MERIT-Offline Model\\03_reviewer_demo.py" '
            f'--python "{rel_py}" --device auto --fusion-device cpu'
        )
    else:
        print(
            f"  {rel_py} ./MERIT-Offline Model/03_reviewer_demo.py "
            f"--python {rel_py} --device auto --fusion-device cpu"
        )

    print("=" * 86)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except BootstrapError as exc:
        print(f"\nBOOTSTRAP FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
