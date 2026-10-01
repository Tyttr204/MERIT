#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
EAV ReviewerDemo Stage 03 — reviewer-friendly GUI with temporal trajectory
==========================================================================

This GUI lives entirely inside ReviewerDemo and orchestrates the already frozen
deployment pipeline. It never reimplements or modifies the EEG / Audio / Video
heads, Quality V1, F4, AF4-B, thresholds, checkpoints, or robot-control policy.

Two reviewer workflows are exposed:

1) Quick 4-window review
       Stage 01 -> Stage 02 -> frozen main.py
   Preserves the original four non-overlapping 5-s windows.

2) Temporal emotion trajectory
       Stage 01 -> Stage 04 -> Stage 05 -> frozen main.py
   Uses the SAME 5-s frozen model input window with a 1-s stride, yielding
   16 time-indexed predictions over a 20-s EAV trial:
       0-5, 1-6, ..., 15-20 s
   The GUI displays:
       - categorical final-emotion trajectory
       - five final fusion probability curves over time
       - q_EEG / q_Audio / q_Video curves over time
       - F4 / AF4-B route for each time step
       - a detailed 16-row table
   Stage 05 automatically saves:
       emotion_trajectory.jsonl
       emotion_trajectory.csv
       emotion_trajectory_summary.json

Important interpretation:
- Reference emotion is audit-only and is not supplied to model inference.
- Confidence is model confidence, not psychological truth.
- q values are quality scores, not total modality contribution weights.
- The temporal trajectory is a 5-s sliding-window trajectory, not per-video-frame
  emotion ground truth.
- No smoothing, majority vote, HMM, persistence rule, or new 20-s classifier is
  applied by this GUI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import tkinter.font as tkfont
    from tkinter.scrolledtext import ScrolledText
except Exception as exc:
    raise SystemExit(
        "ReviewerDemo Stage 03 requires Python tkinter/ttk. "
        f"Could not import tkinter: {type(exc).__name__}: {exc}"
    )

VERSION = "EAV-REVIEWER-DEMO-GUI.1.2.1"

EMOTIONS = ("Neutral", "Sadness", "Anger", "Happiness", "Calmness")
MODALITIES = ("eeg", "audio", "video")
TEST_SUBJECTS = {
    "subject03", "subject05", "subject20",
    "subject31", "subject35", "subject39",
}

SUBJECT_RE = re.compile(r"^subject0*(\d+)$", re.IGNORECASE)
MEDIA_BASE = (
    r"(?P<instance>\d+)_Trial_(?P<trial>\d+)_"
    r"(?P<task>Speaking|Listening)_"
    r"(?P<emotion>Neutral|Sadness|Anger|Happiness|Calmness)"
)
AUDIO_RE = re.compile(rf"^{MEDIA_BASE}(?:_aud)?\.wav$", re.IGNORECASE)
VIDEO_RE = re.compile(rf"^{MEDIA_BASE}\.mp4$", re.IGNORECASE)

EMOTION_COLORS = {
    "Neutral": "#6b7280",
    "Sadness": "#2563eb",
    "Anger": "#dc2626",
    "Happiness": "#d97706",
    "Calmness": "#059669",
}
QUALITY_COLORS = {
    "EEG": "#7c3aed",
    "Audio": "#0891b2",
    "Video": "#16a34a",
}


class ReviewerDemoError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ReviewerDemoError(message)


def normalize_subject(value: str) -> str:
    m = SUBJECT_RE.fullmatch(str(value).strip())
    require(m is not None, f"Invalid EAV subject name: {value!r}")
    n = int(m.group(1))
    require(1 <= n <= 42, f"EAV subject number outside 1..42: {n}")
    return f"subject{n:02d}"


def subject_number(value: str) -> int:
    return int(normalize_subject(value).replace("subject", ""))


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReviewerDemoError(f"Could not read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise ReviewerDemoError(
                    f"Could not parse JSONL {path}:{line_no}: {exc}"
                ) from exc
            require(isinstance(value, dict), f"JSONL row {line_no} is not an object")
            rows.append(value)
    return rows


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def reveal_path(path: Path) -> None:
    path = path.resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


@dataclass(frozen=True)
class TrialChoice:
    subject_id: str
    subject_folder: Path
    instance: int
    emotion: str
    audio_trial_number: int
    video_trial_number: int
    audio_path: Path
    video_path: Path

    @property
    def pair_key(self) -> str:
        return f"{self.subject_id}_instance{self.instance:03d}"

    @property
    def display(self) -> str:
        return (
            f"{self.instance:03d}  |  {self.emotion:<9}  |  "
            f"Audio {self.audio_trial_number:03d}  |  Video {self.video_trial_number:03d}"
        )


def parse_media(path: Path, pattern: re.Pattern[str]) -> tuple[int, int, str, str] | None:
    match = pattern.fullmatch(path.name)
    if not match:
        return None
    emotion = next(
        x for x in EMOTIONS
        if x.casefold() == match.group("emotion").casefold()
    )
    return (
        int(match.group("instance")),
        int(match.group("trial")),
        match.group("task").casefold(),
        emotion,
    )


def scan_subject_trials(subject_dir: Path) -> list[TrialChoice]:
    subject_id = normalize_subject(subject_dir.name)
    audio_dir = subject_dir / "Audio"
    video_dir = subject_dir / "Video"
    eeg_dir = subject_dir / "EEG"

    require(audio_dir.is_dir(), f"Audio folder missing: {audio_dir}")
    require(video_dir.is_dir(), f"Video folder missing: {video_dir}")
    require(eeg_dir.is_dir(), f"EEG folder missing: {eeg_dir}")

    mats = [p for p in eeg_dir.glob("*.mat") if p.is_file()]
    require(len(mats) >= 2, f"Expected EEG signal + label MAT files in {eeg_dir}")

    audio: dict[tuple[int, str], tuple[int, Path]] = {}
    for path in audio_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        parsed = parse_media(path, AUDIO_RE)
        if parsed is None:
            continue
        instance, trial, task, emotion = parsed
        if task != "speaking":
            continue
        key = (instance, emotion)
        require(key not in audio, f"Duplicate Speaking Audio instance: {path}")
        audio[key] = (trial, path.resolve())

    video: dict[tuple[int, str], tuple[int, Path]] = {}
    for path in video_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        parsed = parse_media(path, VIDEO_RE)
        if parsed is None:
            continue
        instance, trial, task, emotion = parsed
        if task != "speaking":
            continue
        key = (instance, emotion)
        require(key not in video, f"Duplicate Speaking Video instance: {path}")
        video[key] = (trial, path.resolve())

    choices: list[TrialChoice] = []
    for key in sorted(set(audio) & set(video), key=lambda x: x[0]):
        instance, emotion = key
        at, ap = audio[key]
        vt, vp = video[key]
        choices.append(
            TrialChoice(
                subject_id=subject_id,
                subject_folder=subject_dir.resolve(),
                instance=instance,
                emotion=emotion,
                audio_trial_number=at,
                video_trial_number=vt,
                audio_path=ap,
                video_path=vp,
            )
        )

    require(choices, f"No complete Speaking Audio+Video pairs in {subject_dir}")
    return choices


def discover_testing_data(testing_data: Path) -> dict[str, list[TrialChoice]]:
    require(testing_data.is_dir(), f"testing Data folder not found: {testing_data}")
    result: dict[str, list[TrialChoice]] = {}
    errors: list[str] = []

    dirs = [
        p for p in testing_data.iterdir()
        if p.is_dir() and SUBJECT_RE.fullmatch(p.name)
    ]
    dirs.sort(key=lambda p: subject_number(p.name))
    require(dirs, f"No EAV subject folders found in: {testing_data}")

    for subject_dir in dirs:
        subject_id = normalize_subject(subject_dir.name)
        if subject_id in TEST_SUBJECTS:
            continue
        try:
            result[subject_id] = scan_subject_trials(subject_dir)
        except Exception as exc:
            errors.append(f"{subject_dir.name}: {type(exc).__name__}: {exc}")

    require(
        result,
        "No usable non-TEST EAV subject was found.\n" + "\n".join(errors),
    )
    return result


def candidate_python(project_root: Path, explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("EAV_REVIEWER_PYTHON")
    if env:
        candidates.append(Path(env).expanduser())

    if sys.platform.startswith("win"):
        candidates += [
            project_root / ".venv-reviewer" / "Scripts" / "python.exe",
            project_root / ".venv-video" / "Scripts" / "python.exe",
            project_root / ".venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates += [
            project_root / ".venv-reviewer" / "bin" / "python",
            project_root / ".venv-video" / "bin" / "python",
            project_root / ".venv" / "bin" / "python",
        ]
    candidates.append(Path(sys.executable))

    seen: set[str] = set()
    for path in candidates:
        try:
            p = path.resolve()
        except Exception:
            p = path
        key = str(p).casefold()
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p

    raise ReviewerDemoError("No usable Python interpreter found.")


def discover_release(config: Path, deployment_root: Path) -> Path | None:
    if config.is_file():
        try:
            cfg = read_json(config)
            q = cfg.get("quality_layer")
            if isinstance(q, Mapping):
                value = q.get("release")
                if isinstance(value, str) and value.strip():
                    p = Path(value).expanduser()
                    p = (p if p.is_absolute() else config.parent / p).resolve()
                    if p.is_file():
                        return p
        except Exception:
            pass

    known = (
        deployment_root
        / "system_checks"
        / "audio_quality_closeout_v2_1_20260927_145459_043"
        / "quality_layer_release_candidate.json"
    )
    if known.is_file():
        return known.resolve()

    checks = deployment_root / "system_checks"
    if checks.is_dir():
        candidates = sorted(checks.rglob("quality_layer_release_candidate.json"))
        if len(candidates) == 1:
            return candidates[0].resolve()
    return None


def environment_probe(python_exe: Path) -> dict[str, Any]:
    command = [
        str(python_exe),
        "-c",
        (
            "import json,sys;"
            "d={'python':sys.version.split()[0],'executable':sys.executable};"
            "\ntry:\n import torch;"
            "\n d.update(torch=getattr(torch,'__version__',None),"
            "cuda=getattr(torch.version,'cuda',None),"
            "cuda_available=bool(torch.cuda.is_available()))"
            "\nexcept Exception as e:\n d['torch_error']=type(e).__name__+': '+str(e)"
            "\nprint(json.dumps(d))"
        ),
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    require(proc.returncode == 0, f"Python environment probe failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def resolve_inference_device(
    requested: str,
    probe: Mapping[str, Any],
) -> str:
    # Resolve reviewer neural inference device without silently demanding CUDA.
    choice = str(requested).strip().lower()
    require(choice in ("auto", "cpu", "cuda"), f"Unknown inference device: {requested!r}")

    if "torch_error" in probe:
        raise ReviewerDemoError(
            "PyTorch probe failed in the selected Python environment: "
            + str(probe["torch_error"])
        )

    cuda_available = bool(probe.get("cuda_available"))

    if choice == "auto":
        return "cuda" if cuda_available else "cpu"

    if choice == "cuda":
        require(
            cuda_available,
            (
                "CUDA was explicitly selected, but torch.cuda.is_available() is False "
                "in the selected Python environment. Choose Auto or CPU, or select a "
                "CUDA-enabled Python environment."
            ),
        )
        return "cuda"

    return "cpu"


def runtime_dependency_probe(
    python_exe: Path,
    deployment_root: Path,
) -> dict[str, Any]:
    probe_script = Path(__file__).resolve().with_name("08_reviewer_runtime_probe.py")
    require(probe_script.is_file(), f"Runtime probe script missing: {probe_script}")

    proc = subprocess.run(
        [
            str(python_exe),
            "-X", "utf8",
            str(probe_script),
            "--deployment-root", str(deployment_root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=150,
        check=False,
    )

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    require(
        lines,
        (
            "Runtime probe produced no JSON output. "
            f"returncode={proc.returncode}; stderr={proc.stderr[-2000:]}"
        ),
    )
    try:
        result = json.loads(lines[-1])
    except Exception as exc:
        raise ReviewerDemoError(
            "Could not parse runtime probe JSON. "
            f"stdout={proc.stdout[-3000:]} stderr={proc.stderr[-3000:]}"
        ) from exc
    require(isinstance(result, dict), "Runtime probe result is not a JSON object.")
    return result


def require_runtime_ready(report: Mapping[str, Any]) -> None:
    if bool(report.get("ready")):
        return

    details: list[str] = []

    missing = report.get("missing_packages")
    if isinstance(missing, list) and missing:
        details.append("Missing/import-failing packages: " + ", ".join(map(str, missing)))

    module_errors = report.get("deployment_module_errors")
    if isinstance(module_errors, list) and module_errors:
        parts = []
        for item in module_errors:
            if isinstance(item, Mapping):
                parts.append(f"{item.get('file')}: {item.get('error')}")
        if parts:
            details.append("Deployment module import errors: " + "; ".join(parts))

    dover = report.get("dover")
    if isinstance(dover, Mapping) and not bool(dover.get("ok")):
        details.append("DOVER import error: " + str(dover.get("error")))

    executables = report.get("executables")
    if isinstance(executables, Mapping):
        missing_exe = [k for k, v in executables.items() if not v]
        if missing_exe:
            details.append("Missing executables: " + ", ".join(missing_exe))

    if not details:
        details.append("Runtime probe status is not ready.")

    raise ReviewerDemoError(
        "Selected Python environment is NOT runtime-ready.\n"
        + "\n".join(details)
        + "\nChoose a complete Python interpreter before running inference."
    )


class LineChart(tk.Canvas):
    """Small dependency-free line chart for probability/quality trajectories."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        title: str,
        y_label: str,
        height: int = 260,
    ) -> None:
        super().__init__(
            master,
            height=height,
            background="white",
            highlightthickness=1,
            highlightbackground="#d1d5db",
        )
        self.title = title
        self.y_label = y_label
        self.x_values: list[float] = []
        self.series: list[tuple[str, list[float | None], str]] = []
        self.y_min = 0.0
        self.y_max = 1.0
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(
        self,
        x_values: Sequence[float],
        series: Sequence[tuple[str, Sequence[float | None], str]],
        *,
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> None:
        self.x_values = [float(x) for x in x_values]
        self.series = [
            (name, [None if v is None else float(v) for v in values], color)
            for name, values, color in series
        ]
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.redraw()

    def clear_chart(self) -> None:
        self.x_values = []
        self.series = []
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 500)
        height = max(self.winfo_height(), 220)

        self.create_text(
            12, 10,
            text=self.title,
            anchor="nw",
            font=("Segoe UI", 11, "bold"),
            fill="#111827",
        )

        if not self.x_values or not self.series:
            self.create_text(
                width / 2,
                height / 2,
                text="No trajectory result yet",
                fill="#6b7280",
                font=("Segoe UI", 12),
            )
            return

        left, right, top, bottom = 58, 18, 38, 42
        x0, x1 = left, width - right
        y0, y1 = top, height - bottom

        # Grid and y-axis
        for i in range(5):
            frac = i / 4
            y = y1 - frac * (y1 - y0)
            value = self.y_min + frac * (self.y_max - self.y_min)
            self.create_line(x0, y, x1, y, fill="#e5e7eb")
            self.create_text(x0 - 8, y, text=f"{value:.2f}", anchor="e",
                             fill="#4b5563", font=("Segoe UI", 9))
        self.create_line(x0, y0, x0, y1, fill="#374151")
        self.create_line(x0, y1, x1, y1, fill="#374151")

        self.create_text(
            14, (y0 + y1) / 2,
            text=self.y_label,
            angle=90,
            fill="#4b5563",
            font=("Segoe UI", 9),
        )
        self.create_text(
            (x0 + x1) / 2,
            height - 10,
            text="Trial-relative center time (s)",
            fill="#4b5563",
            font=("Segoe UI", 9),
        )

        xmin, xmax = min(self.x_values), max(self.x_values)
        span = xmax - xmin if xmax > xmin else 1.0

        def px(x: float) -> float:
            return x0 + (x - xmin) / span * (x1 - x0)

        def py(y: float) -> float:
            value = min(self.y_max, max(self.y_min, y))
            return y1 - (value - self.y_min) / (self.y_max - self.y_min) * (y1 - y0)

        # x ticks: preserve first/last and roughly six labels
        count = len(self.x_values)
        tick_indices = sorted(
            set(
                [0, count - 1]
                + [round(i * (count - 1) / 5) for i in range(1, 5)]
            )
        )
        for idx in tick_indices:
            x = px(self.x_values[idx])
            self.create_line(x, y1, x, y1 + 4, fill="#374151")
            self.create_text(
                x, y1 + 7,
                text=f"{self.x_values[idx]:.1f}",
                anchor="n",
                fill="#4b5563",
                font=("Segoe UI", 9),
            )

        # Lines
        for name, values, color in self.series:
            points: list[float] = []
            for x, value in zip(self.x_values, values):
                if value is None or not math.isfinite(value):
                    if len(points) >= 4:
                        self.create_line(*points, fill=color, width=2, smooth=False)
                    points = []
                    continue
                points.extend([px(x), py(value)])
            if len(points) >= 4:
                self.create_line(*points, fill=color, width=2, smooth=False)

            for x, value in zip(self.x_values, values):
                if value is None or not math.isfinite(value):
                    continue
                cx, cy = px(x), py(value)
                self.create_oval(cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5,
                                 fill=color, outline=color)

        # Legend
        legend_x = x1 - 8
        for name, _values, color in reversed(self.series):
            tw = max(58, len(name) * 7 + 24)
            legend_x -= tw
            self.create_line(legend_x, 23, legend_x + 14, 23, fill=color, width=3)
            self.create_text(
                legend_x + 18, 23,
                text=name,
                anchor="w",
                fill="#374151",
                font=("Segoe UI", 9),
            )


class EmotionTimeline(tk.Canvas):
    """Categorical emotion trajectory; AF4-B points receive a dark outline."""

    def __init__(self, master: tk.Misc, *, height: int = 210) -> None:
        super().__init__(
            master,
            height=height,
            background="white",
            highlightthickness=1,
            highlightbackground="#d1d5db",
        )
        self.rows: list[dict[str, Any]] = []
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(x) for x in rows]
        self.redraw()

    def clear_chart(self) -> None:
        self.rows = []
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 500)
        height = max(self.winfo_height(), 190)
        self.create_text(
            12, 10,
            text="Final classification trajectory",
            anchor="nw",
            font=("Segoe UI", 11, "bold"),
            fill="#111827",
        )

        if not self.rows:
            self.create_text(width / 2, height / 2, text="No trajectory result yet",
                             fill="#6b7280", font=("Segoe UI", 12))
            return

        left, right, top, bottom = 82, 18, 38, 36
        x0, x1 = left, width - right
        y0, y1 = top, height - bottom

        centers = [float(x["center_seconds"]) for x in self.rows]
        xmin, xmax = min(centers), max(centers)
        span = xmax - xmin if xmax > xmin else 1.0

        def px(x: float) -> float:
            return x0 + (x - xmin) / span * (x1 - x0)

        row_gap = (y1 - y0) / (len(EMOTIONS) - 1)
        y_by_emotion = {
            emotion: y0 + i * row_gap
            for i, emotion in enumerate(EMOTIONS)
        }

        for emotion in EMOTIONS:
            y = y_by_emotion[emotion]
            self.create_line(x0, y, x1, y, fill="#eef2f7")
            self.create_text(
                x0 - 9, y,
                text=emotion,
                anchor="e",
                fill=EMOTION_COLORS[emotion],
                font=("Segoe UI", 9, "bold"),
            )

        pts: list[float] = []
        for item in self.rows:
            emotion = str(item.get("final_emotion"))
            if emotion not in y_by_emotion:
                continue
            pts.extend([px(float(item["center_seconds"])), y_by_emotion[emotion]])
        if len(pts) >= 4:
            self.create_line(*pts, fill="#9ca3af", width=1.5)

        for item in self.rows:
            emotion = str(item.get("final_emotion"))
            if emotion not in y_by_emotion:
                continue
            x = px(float(item["center_seconds"]))
            y = y_by_emotion[emotion]
            route = str(item.get("route") or "")
            outline = "#111827" if route == "AF4-B" else EMOTION_COLORS[emotion]
            width_o = 2 if route == "AF4-B" else 1
            self.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill=EMOTION_COLORS[emotion],
                outline=outline,
                width=width_o,
            )

        count = len(centers)
        tick_indices = sorted(
            set([0, count - 1] + [round(i * (count - 1) / 5) for i in range(1, 5)])
        )
        self.create_line(x0, y1 + 10, x1, y1 + 10, fill="#374151")
        for idx in tick_indices:
            x = px(centers[idx])
            self.create_line(x, y1 + 10, x, y1 + 14, fill="#374151")
            self.create_text(x, y1 + 17, text=f"{centers[idx]:.1f}",
                             anchor="n", fill="#4b5563", font=("Segoe UI", 9))

        self.create_text(
            x1, 21,
            text="Dark outline = AF4-B",
            anchor="e",
            fill="#4b5563",
            font=("Segoe UI", 9),
        )


class ReviewerDemoApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args

        self.script_dir = Path(__file__).resolve().parent
        self.deployment_root = self.script_dir.parent
        self.project_root = self.deployment_root.parent

        self.stage01 = self.script_dir / "01_prepare_eav_trial.py"
        self.stage02 = self.script_dir / "02_run_eav_reviewer_inference.py"
        self.stage04 = self.script_dir / "04_prepare_temporal_trajectory.py"
        self.stage05 = self.script_dir / "05_run_temporal_trajectory.py"
        self.main_py = self.deployment_root / "main.py"
        self.config = Path(args.config).expanduser().resolve()
        self.testing_data = Path(args.testing_data).expanduser().resolve()

        self.python_exe = candidate_python(self.deployment_root, args.python)
        self.default_release = (
            Path(args.release).expanduser().resolve()
            if args.release
            else discover_release(self.config, self.deployment_root)
        )

        self.inventory: dict[str, list[TrialChoice]] = {}
        self.selected_trial: TrialChoice | None = None

        self.running = False
        self.worker_thread: threading.Thread | None = None
        self.active_process: subprocess.Popen[str] | None = None
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.current_session_root: Path | None = None
        self.current_output_dir: Path | None = None

        self.last_baseline_json: Path | None = None
        self.last_baseline_csv: Path | None = None

        self.last_trajectory_jsonl: Path | None = None
        self.last_trajectory_csv: Path | None = None
        self.last_trajectory_summary: Path | None = None

        # Resolved from the selected Python environment. "auto" never silently
        # requires CUDA: CUDA is used only when torch reports it available.
        self.resolved_inference_device: str | None = None

        self._set_dpi_awareness()
        self._configure_window()
        self._configure_fonts()
        self._build_style()
        self._build_ui()
        self._bind_events()

        self.root.after(80, self._drain_ui_queue)
        self.root.after(120, self.refresh_inventory)
        self.root.after(180, self.check_setup)

    def _set_dpi_awareness(self) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _configure_window(self) -> None:
        self.root.title("MARIT - Offline Mode")
        self.root.geometry("1640x1020")
        self.root.minsize(1320, 840)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_fonts(self) -> None:
        """Increase GUI readability without changing any inference behavior."""
        # Update Tk named fonts so ordinary ttk/tk labels, entries, menus,
        # comboboxes, dialogs, etc. inherit a larger readable default.
        named = {
            "TkDefaultFont": ("Segoe UI", 11),
            "TkTextFont": ("Segoe UI", 11),
            "TkMenuFont": ("Segoe UI", 11),
            "TkHeadingFont": ("Segoe UI", 11, "bold"),
            "TkCaptionFont": ("Segoe UI", 11),
            "TkSmallCaptionFont": ("Segoe UI", 10),
            "TkIconFont": ("Segoe UI", 10),
            "TkTooltipFont": ("Segoe UI", 10),
            "TkFixedFont": ("Consolas", 10),
        }
        for name, spec in named.items():
            try:
                tkfont.nametofont(name).configure(
                    family=spec[0],
                    size=spec[1],
                    weight=spec[2] if len(spec) > 2 else "normal",
                )
            except Exception:
                pass

        # Slightly increase Tk's scaling on very dense displays, but keep it
        # conservative so layout dimensions and charts remain stable.
        try:
            current = float(self.root.tk.call("tk", "scaling"))
            self.root.tk.call("tk", "scaling", max(current, 1.15))
        except Exception:
            pass

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista" if sys.platform.startswith("win") else "clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("BigAction.TButton", font=("Segoe UI", 12, "bold"), padding=(16, 11))
        style.configure("Treeview", rowheight=32, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(9, 6))
        style.configure("TLabel", font=("Segoe UI", 12))
        style.configure("TEntry", font=("Segoe UI", 12))
        style.configure("TCombobox", font=("Segoe UI", 12))
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=(12, 7))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="MARIT - Offline Mode",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Offline reviewer interface • frozen EEG / Audio / Video heads • Quality V1 • "
                "F4 / AF4-B • temporal trajectory saved automatically"
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 4))
        ttk.Label(
            header,
            text=(
                "Temporal mode = 5-s frozen input window with 1-s stride. "
                "Reference emotion is audit-only; no smoothing or new temporal classifier is added."
            ),
        ).pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Initialising…")
        status_row = ttk.Frame(header)
        status_row.pack(fill="x", pady=(0, 8))
        ttk.Label(status_row, text="Status:", style="Status.TLabel").pack(side="left")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(status_row, mode="determinate", maximum=100, value=0)
        self.progress.pack(side="right", fill="x", expand=True, padx=(24, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=3)

        # ----- Selection
        selection = ttk.Labelframe(left, text="1. Select reviewer trial")
        selection.pack(fill="x", pady=(0, 9))
        row = ttk.Frame(selection, padding=8)
        row.pack(fill="x")
        ttk.Label(row, text="Testing data").grid(row=0, column=0, sticky="w")
        self.testing_data_var = tk.StringVar(value=str(self.testing_data))
        ttk.Entry(row, textvariable=self.testing_data_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(row, text="Browse…", command=self.browse_testing_data).grid(
            row=1, column=1, padx=(0, 5)
        )
        ttk.Button(row, text="Refresh", command=self.refresh_inventory).grid(row=1, column=2)
        row.columnconfigure(0, weight=1)

        choice = ttk.Frame(selection, padding=(8, 0, 8, 8))
        choice.pack(fill="x")
        ttk.Label(choice, text="Subject").grid(row=0, column=0, sticky="w")
        ttk.Label(choice, text="Speaking trial").grid(row=0, column=1, sticky="w")
        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(
            choice, textvariable=self.subject_var, state="readonly", width=19
        )
        self.subject_combo.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.trial_var = tk.StringVar()
        self.trial_combo = ttk.Combobox(
            choice, textvariable=self.trial_var, state="readonly", width=50
        )
        self.trial_combo.grid(row=1, column=1, sticky="ew")
        choice.columnconfigure(1, weight=1)

        self.selection_info_var = tk.StringVar(value="No trial selected.")
        ttk.Label(
            selection,
            textvariable=self.selection_info_var,
            wraplength=470,
            justify="left",
            padding=(8, 0, 8, 8),
        ).pack(fill="x")

        # ----- Runtime setup
        setup = ttk.Labelframe(left, text="2. Runtime setup")
        setup.pack(fill="x", pady=(0, 9))
        s = ttk.Frame(setup, padding=8)
        s.pack(fill="x")

        self.python_var = tk.StringVar(value=str(self.python_exe))
        self.release_var = tk.StringVar(
            value=str(self.default_release) if self.default_release else ""
        )
        self.eeg_unit_var = tk.StringVar(value=self.args.eeg_unit)
        self.inference_device_var = tk.StringVar(value=self.args.device)
        self.fusion_device_var = tk.StringVar(value=self.args.fusion_device)
        self.stride_var = tk.StringVar(value=f"{self.args.trajectory_stride:g}")

        ttk.Label(s, text="Python").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Entry(s, textvariable=self.python_var).grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=(0, 5)
        )
        ttk.Button(s, text="Browse…", command=self.browse_python).grid(row=1, column=3)

        ttk.Label(s, text="Quality release").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(7, 0)
        )
        ttk.Entry(s, textvariable=self.release_var).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=(0, 5)
        )
        ttk.Button(s, text="Browse…", command=self.browse_release).grid(row=3, column=3)

        ttk.Label(s, text="EEG unit").grid(row=4, column=0, sticky="w", pady=(7, 0))
        ttk.Label(s, text="Neural device").grid(row=4, column=1, sticky="w", pady=(7, 0))
        ttk.Label(s, text="Fusion device").grid(row=4, column=2, sticky="w", pady=(7, 0))
        ttk.Label(s, text="Trajectory stride").grid(row=4, column=3, sticky="w", pady=(7, 0))

        ttk.Combobox(
            s, textvariable=self.eeg_unit_var,
            values=("uV", "mV", "V"), state="readonly", width=9
        ).grid(row=5, column=0, sticky="w")
        ttk.Combobox(
            s, textvariable=self.inference_device_var,
            values=("auto", "cuda", "cpu"), state="readonly", width=9
        ).grid(row=5, column=1, sticky="w")
        ttk.Combobox(
            s, textvariable=self.fusion_device_var,
            values=("cpu", "cuda"), state="readonly", width=9
        ).grid(row=5, column=2, sticky="w")
        ttk.Combobox(
            s, textvariable=self.stride_var,
            values=("1",), state="readonly", width=9
        ).grid(row=5, column=3, sticky="w")
        ttk.Label(s, text="seconds").grid(row=5, column=3, sticky="e")

        s.columnconfigure(0, weight=1)
        s.columnconfigure(1, weight=1)
        s.columnconfigure(2, weight=1)
        s.columnconfigure(3, weight=1)

        self.setup_status_var = tk.StringVar(value="Not checked")
        setup_bottom = ttk.Frame(setup, padding=(8, 0, 8, 8))
        setup_bottom.pack(fill="x")
        ttk.Label(
            setup_bottom,
            textvariable=self.setup_status_var,
            wraplength=380,
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(setup_bottom, text="Check setup", command=self.check_setup).pack(side="right")

        # ----- Run
        actions = ttk.Labelframe(left, text="3. Run reviewer workflow")
        actions.pack(fill="x", pady=(0, 9))
        a = ttk.Frame(actions, padding=8)
        a.pack(fill="x")

        self.prepare_button = ttk.Button(
            a, text="Prepare trial only", command=self.prepare_selected_trial
        )
        self.prepare_button.pack(fill="x", pady=(0, 5))

        self.baseline_button = ttk.Button(
            a,
            text="Run quick 4-window review",
            command=self.run_baseline_demo,
        )
        self.baseline_button.pack(fill="x", pady=(0, 5))

        self.trajectory_button = ttk.Button(
            a,
            text="Run emotion recognition",
            command=self.run_temporal_demo,
            style="BigAction.TButton",
        )
        self.trajectory_button.pack(fill="x")

        ttk.Label(
            actions,
            text=(
                "Temporal workflow: Stage 01 → Stage 04 → Stage 05 → frozen main.py. "
                "For a 20-s trial, 5-s windows with 1-s stride produce 16 saved trajectory points."
            ),
            wraplength=470,
            justify="left",
            padding=(8, 0, 8, 8),
        ).pack(fill="x")

        # ----- Outputs
        outputs = ttk.Labelframe(left, text="4. Saved outputs")
        outputs.pack(fill="x")
        o = ttk.Frame(outputs, padding=8)
        o.pack(fill="x")

        self.open_folder_button = ttk.Button(
            o, text="Open output folder", state="disabled", command=self.open_output_folder
        )
        self.open_folder_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        self.open_csv_button = ttk.Button(
            o, text="Open trajectory CSV", state="disabled", command=self.open_trajectory_csv
        )
        self.open_csv_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))

        self.open_summary_button = ttk.Button(
            o, text="Open summary JSON", state="disabled", command=self.open_trajectory_summary
        )
        self.open_summary_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))

        self.export_csv_button = ttk.Button(
            o, text="Save CSV copy…", state="disabled", command=self.export_trajectory_csv
        )
        self.export_csv_button.grid(row=2, column=0, sticky="ew", padx=(0, 3), pady=(5, 0))

        self.export_jsonl_button = ttk.Button(
            o, text="Save JSONL copy…", state="disabled", command=self.export_trajectory_jsonl
        )
        self.export_jsonl_button.grid(row=2, column=1, sticky="ew", padx=(3, 0), pady=(5, 0))

        o.columnconfigure(0, weight=1)
        o.columnconfigure(1, weight=1)

        # ----- Right summary
        summary = ttk.Labelframe(right, text="Current result summary")
        summary.pack(fill="x", pady=(0, 8))
        si = ttk.Frame(summary, padding=8)
        si.pack(fill="x")

        self.reference_var = tk.StringVar(value="-")
        self.result_subject_var = tk.StringVar(value="-")
        self.mode_var = tk.StringVar(value="-")
        self.release_fp_var = tk.StringVar(value="-")
        self.route_counts_var = tk.StringVar(value="-")
        self.emotion_counts_var = tk.StringVar(value="-")
        self.asset_check_var = tk.StringVar(value="-")

        labels = (
            ("Reference emotion", self.reference_var),
            ("Subject / instance", self.result_subject_var),
            ("Display mode", self.mode_var),
            ("Release fingerprint", self.release_fp_var),
            ("Route counts", self.route_counts_var),
            ("Emotion counts", self.emotion_counts_var),
            ("Frozen assets verified", self.asset_check_var),
        )
        for r, (name, var) in enumerate(labels):
            ttk.Label(si, text=name).grid(row=r, column=0, sticky="nw", padx=(0, 12), pady=1)
            ttk.Label(si, textvariable=var, wraplength=850, justify="left").grid(
                row=r, column=1, sticky="w", pady=1
            )
        si.columnconfigure(1, weight=1)

        # ----- Tabs
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        # Trajectory tab
        self.trajectory_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.trajectory_tab, text="Emotion trajectory")

        self.timeline_chart = EmotionTimeline(self.trajectory_tab, height=210)
        self.timeline_chart.pack(fill="x", padx=8, pady=(8, 5))

        self.probability_chart = LineChart(
            self.trajectory_tab,
            title="Final fusion probabilities over time",
            y_label="Probability",
            height=260,
        )
        self.probability_chart.pack(fill="both", expand=True, padx=8, pady=5)

        self.quality_chart = LineChart(
            self.trajectory_tab,
            title="Modality quality over time",
            y_label="Quality q",
            height=235,
        )
        self.quality_chart.pack(fill="both", expand=True, padx=8, pady=(5, 5))

        self.trajectory_note_var = tk.StringVar(
            value="No temporal trajectory yet."
        )
        ttk.Label(
            self.trajectory_tab,
            textvariable=self.trajectory_note_var,
            wraplength=980,
            justify="left",
            padding=(8, 2, 8, 8),
        ).pack(fill="x")

        # Time-step table tab
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text="Time-aligned Annotation")

        trajectory_columns = (
            "step", "span", "center", "route", "final", "conf",
            "eeg", "qe", "audio", "qa", "video", "qv",
        )
        self.trajectory_tree = ttk.Treeview(
            self.table_tab,
            columns=trajectory_columns,
            show="headings",
            height=20,
        )
        heading_map = {
            "step": "#",
            "span": "Window",
            "center": "Center",
            "route": "Route",
            "final": "Final emotion",
            "conf": "Confidence",
            "eeg": "EEG",
            "qe": "q_EEG",
            "audio": "Audio",
            "qa": "q_Audio",
            "video": "Video",
            "qv": "q_Video",
        }
        widths = {
            "step": 38, "span": 82, "center": 65, "route": 74,
            "final": 95, "conf": 78, "eeg": 92, "qe": 65,
            "audio": 92, "qa": 68, "video": 92, "qv": 68,
        }
        for col in trajectory_columns:
            self.trajectory_tree.heading(col, text=heading_map[col])
            self.trajectory_tree.column(col, width=widths[col], anchor="center")
        tree_scroll = ttk.Scrollbar(
            self.table_tab, orient="vertical", command=self.trajectory_tree.yview
        )
        self.trajectory_tree.configure(yscrollcommand=tree_scroll.set)
        self.trajectory_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        tree_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        # Baseline tab
        self.baseline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.baseline_tab, text="4-window review")

        baseline_columns = (
            "window", "route", "final", "conf",
            "eeg", "qe", "audio", "qa", "video", "qv",
        )
        self.baseline_tree = ttk.Treeview(
            self.baseline_tab,
            columns=baseline_columns,
            show="headings",
            height=10,
        )
        baseline_headings = {
            "window": "W", "route": "Route", "final": "Final",
            "conf": "Confidence", "eeg": "EEG", "qe": "q_EEG",
            "audio": "Audio", "qa": "q_Audio", "video": "Video", "qv": "q_Video",
        }
        for col in baseline_columns:
            self.baseline_tree.heading(col, text=baseline_headings[col])
            self.baseline_tree.column(col, width=82 if col != "window" else 40, anchor="center")
        self.baseline_tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.baseline_note_var = tk.StringVar(
            value="No quick 4-window review yet."
        )
        ttk.Label(
            self.baseline_tab,
            textvariable=self.baseline_note_var,
            wraplength=980,
            justify="left",
            padding=(8, 0, 8, 8),
        ).pack(fill="x")

        # Log tab
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Execution log")
        log_toolbar = ttk.Frame(self.log_tab, padding=(8, 6, 8, 0))
        log_toolbar.pack(fill="x")
        ttk.Button(
            log_toolbar, text="Clear log", command=lambda: self.log.delete("1.0", "end")
        ).pack(side="right")
        self.log = ScrolledText(
            self.log_tab,
            height=20,
            wrap="word",
            font=("Consolas", 10),
        )
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def _bind_events(self) -> None:
        self.subject_combo.bind("<<ComboboxSelected>>", self._on_subject_changed)
        self.trial_combo.bind("<<ComboboxSelected>>", self._on_trial_changed)

    # ------------------------ generic UI ------------------------

    def set_status(self, text: str, progress: int | None = None) -> None:
        self.status_var.set(text)
        if progress is not None:
            self.progress.configure(value=max(0, min(100, progress)))

    def append_log(self, text: str) -> None:
        self.log.insert("end", text)
        if not text.endswith("\n"):
            self.log.insert("end", "\n")
        self.log.see("end")

    def show_error(self, title: str, exc: BaseException | str) -> None:
        message = str(exc)
        self.append_log(f"[ERROR] {message}")
        messagebox.showerror(title, message, parent=self.root)

    def _set_running(self, running: bool) -> None:
        self.running = running
        state = "disabled" if running else "normal"
        self.prepare_button.configure(state=state)
        self.baseline_button.configure(state=state)
        self.trajectory_button.configure(state=state)
        self.subject_combo.configure(state="disabled" if running else "readonly")
        self.trial_combo.configure(state="disabled" if running else "readonly")

    # ------------------------ discovery ------------------------

    def browse_testing_data(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Select ReviewerDemo testing Data folder",
            initialdir=self.testing_data_var.get() or str(self.script_dir),
        )
        if selected:
            self.testing_data_var.set(selected)
            self.refresh_inventory()

    def refresh_inventory(self) -> None:
        if self.running:
            return
        try:
            path = Path(self.testing_data_var.get()).expanduser().resolve()
            inventory = discover_testing_data(path)
            self.testing_data = path
            self.inventory = inventory

            subjects = list(inventory)
            self.subject_combo.configure(values=subjects)
            if subjects:
                current = self.subject_var.get()
                self.subject_var.set(current if current in subjects else subjects[0])
                self._populate_trials_for_subject(self.subject_var.get())

            total = sum(len(v) for v in inventory.values())
            self.append_log(
                f"[DISCOVERY] {len(subjects)} reviewer subject(s), "
                f"{total} complete Speaking trial(s) under {path}"
            )
            excluded = [
                p.name for p in path.iterdir()
                if p.is_dir()
                and SUBJECT_RE.fullmatch(p.name)
                and normalize_subject(p.name) in TEST_SUBJECTS
            ]
            if excluded:
                self.append_log(
                    "[DISCOVERY] Frozen TEST subjects excluded: " + ", ".join(excluded)
                )
        except Exception as exc:
            self.inventory = {}
            self.subject_combo.configure(values=[])
            self.trial_combo.configure(values=[])
            self.selected_trial = None
            self.selection_info_var.set(str(exc))
            self.append_log(f"[DISCOVERY ERROR] {type(exc).__name__}: {exc}")

    def _populate_trials_for_subject(self, subject_id: str) -> None:
        trials = self.inventory.get(subject_id, [])
        displays = [x.display for x in trials]
        self.trial_combo.configure(values=displays)
        if displays:
            self.trial_var.set(displays[0])
            self.selected_trial = trials[0]
            self._show_selected_trial()
        else:
            self.trial_var.set("")
            self.selected_trial = None
            self.selection_info_var.set("No complete Speaking trials.")

    def _on_subject_changed(self, _event: Any = None) -> None:
        self._populate_trials_for_subject(self.subject_var.get())

    def _on_trial_changed(self, _event: Any = None) -> None:
        trials = self.inventory.get(self.subject_var.get(), [])
        display = self.trial_var.get()
        self.selected_trial = next((x for x in trials if x.display == display), None)
        self._show_selected_trial()

    def _show_selected_trial(self) -> None:
        t = self.selected_trial
        if t is None:
            self.selection_info_var.set("No trial selected.")
            return
        self.selection_info_var.set(
            f"{t.pair_key} • Reference={t.emotion} (audit only; not model input) • "
            "Stage 01 validates EEG labels and Audio/Video pairing before inference."
        )

    # ------------------------ setup ------------------------

    def browse_python(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select Python interpreter",
            initialdir=str(self.python_exe.parent),
            filetypes=[("Python executable", "python.exe"), ("All files", "*.*")],
        )
        if selected:
            self.python_var.set(selected)
            self.check_setup()

    def browse_release(self) -> None:
        initial = (
            str(self.default_release.parent)
            if self.default_release is not None
            else str(self.deployment_root)
        )
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select quality_layer_release_candidate.json",
            initialdir=initial,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.release_var.set(selected)

    def check_setup(self) -> None:
        if self.running:
            return
        try:
            python_exe = Path(self.python_var.get()).expanduser().resolve()
            checks = {
                "Stage 01": self.stage01.is_file(),
                "Stage 02": self.stage02.is_file(),
                "Stage 04": self.stage04.is_file(),
                "Stage 05": self.stage05.is_file(),
                "main.py": self.main_py.is_file(),
                "system_config.json": self.config.is_file(),
                "Python": python_exe.is_file(),
                "ffmpeg": shutil.which("ffmpeg") is not None,
                "ffprobe": shutil.which("ffprobe") is not None,
            }
            release_text = self.release_var.get().strip()
            if release_text:
                checks["Release"] = Path(release_text).expanduser().is_file()
            failed = [name for name, ok in checks.items() if not ok]
            require(not failed, "Missing setup item(s): " + ", ".join(failed))

            probe = environment_probe(python_exe)
            runtime_report = runtime_dependency_probe(
                python_exe,
                self.deployment_root,
            )
            require_runtime_ready(runtime_report)
            resolved_device = resolve_inference_device(
                self.inference_device_var.get(),
                probe,
            )
            self.resolved_inference_device = resolved_device
            cuda_text = (
                f"CUDA available={probe.get('cuda_available')}, "
                f"Torch={probe.get('torch')}, CUDA={probe.get('cuda')}"
                if "torch_error" not in probe
                else f"Torch probe failed: {probe['torch_error']}"
            )
            requested = self.inference_device_var.get().strip().lower()
            self.setup_status_var.set(
                f"PASS • Python {probe.get('python')} • "
                f"Neural={requested}→{resolved_device.upper()} • {cuda_text}"
            )
            self.append_log(
                f"[SETUP PASS] executable={probe.get('executable')} | "
                f"neural_device={requested}->{resolved_device} | {cuda_text}"
            )
            self.append_log(
                f"[RUNTIME READY] required packages/imports PASS | "
                f"DOVER={runtime_report.get('dover', {}).get('ok')}"
            )
            self.set_status(f"Ready • Neural device: {resolved_device.upper()}", 0)
        except Exception as exc:
            self.setup_status_var.set(f"FAILED • {exc}")
            self.append_log(f"[SETUP ERROR] {type(exc).__name__}: {exc}")

    def _validate_run_inputs(self) -> tuple[TrialChoice, Path, Path | None]:
        trial = self.selected_trial
        require(trial is not None, "Select a subject and trial first.")

        python_exe = Path(self.python_var.get()).expanduser().resolve()
        require(python_exe.is_file(), f"Python interpreter not found: {python_exe}")
        for p, name in (
            (self.stage01, "Stage 01"),
            (self.stage02, "Stage 02"),
            (self.stage04, "Stage 04"),
            (self.stage05, "Stage 05"),
            (self.main_py, "main.py"),
            (self.config, "system_config.json"),
        ):
            require(p.is_file(), f"{name} missing: {p}")

        release: Path | None = None
        release_text = self.release_var.get().strip()
        if release_text:
            release = Path(release_text).expanduser().resolve()
            require(release.is_file(), f"Quality release not found: {release}")

        probe = environment_probe(python_exe)
        runtime_report = runtime_dependency_probe(
            python_exe,
            self.deployment_root,
        )
        require_runtime_ready(runtime_report)
        self.resolved_inference_device = resolve_inference_device(
            self.inference_device_var.get(),
            probe,
        )
        self.append_log(
            f"[DEVICE] requested={self.inference_device_var.get()} "
            f"resolved={self.resolved_inference_device} "
            f"cuda_available={probe.get('cuda_available')}"
        )
        return trial, python_exe, release

    # ------------------------ commands ------------------------

    def _stage01_command(
        self,
        *,
        trial: TrialChoice,
        python_exe: Path,
        output_dir: Path,
    ) -> list[str]:
        return [
            str(python_exe), "-u", str(self.stage01),
            "--testing-data", str(self.testing_data),
            "--subject", trial.subject_id,
            "--instance", str(trial.instance),
            "--audio-policy", "a0_pcm16",
            "--output", str(output_dir),
        ]

    def _stage02_command(
        self,
        *,
        python_exe: Path,
        prepared: Path,
        output_dir: Path,
        release: Path | None,
    ) -> list[str]:
        cmd = [
            str(python_exe), "-u", str(self.stage02),
            "--prepared", str(prepared),
            "--python", str(python_exe),
            "--config", str(self.config),
            "--main", str(self.main_py),
            "--eeg-unit", self.eeg_unit_var.get(),
            "--device", str(self.resolved_inference_device or "cpu"),
            "--fusion-device", self.fusion_device_var.get(),
            "--output", str(output_dir),
        ]
        if release is not None:
            cmd += ["--release", str(release)]
        return cmd

    def _stage04_command(
        self,
        *,
        python_exe: Path,
        prepared: Path,
        output_dir: Path,
    ) -> list[str]:
        return [
            str(python_exe), "-u", str(self.stage04),
            "--prepared", str(prepared),
            "--stride-seconds", self.stride_var.get(),
            "--output", str(output_dir),
        ]

    def _stage05_command(
        self,
        *,
        python_exe: Path,
        temporal_prepared: Path,
        output_dir: Path,
        release: Path | None,
    ) -> list[str]:
        cmd = [
            str(python_exe), "-u", str(self.stage05),
            "--prepared", str(temporal_prepared),
            "--python", str(python_exe),
            "--config", str(self.config),
            "--main", str(self.main_py),
            "--eeg-unit", self.eeg_unit_var.get(),
            "--device", str(self.resolved_inference_device or "cpu"),
            "--fusion-device", self.fusion_device_var.get(),
            "--output", str(output_dir),
        ]
        if release is not None:
            cmd += ["--release", str(release)]
        return cmd

    # ------------------------ workflows ------------------------

    def prepare_selected_trial(self) -> None:
        if self.running:
            return
        try:
            trial, python_exe, _release = self._validate_run_inputs()
        except Exception as exc:
            self.show_error("Cannot prepare trial", exc)
            return

        session_root = (
            self.script_dir / "gui_sessions" / f"{trial.pair_key}_{now_stamp()}_prepare"
        ).resolve()
        session_root.mkdir(parents=True, exist_ok=False)
        prepared = session_root / "prepared"

        self.current_session_root = session_root
        self._set_running(True)
        self.set_status("Stage 01: validating raw EAV trial…", 15)
        self.notebook.select(self.log_tab)
        self.append_log("\n" + "=" * 100)
        self.append_log(f"[GUI SESSION] {session_root}")

        def worker() -> None:
            try:
                rc = self._run_streamed(
                    self._stage01_command(
                        trial=trial,
                        python_exe=python_exe,
                        output_dir=prepared,
                    ),
                    cwd=self.deployment_root,
                    label="STAGE 01",
                )
                require(rc == 0, f"Stage 01 exited with code {rc}")
                require((prepared / "reviewer_windows.jsonl").is_file(),
                        "Stage 01 did not produce reviewer_windows.jsonl")
                self.ui_queue.put(("prepare_success", {"prepared": prepared, "session_root": session_root}))
            except Exception as exc:
                self.ui_queue.put(("worker_error", ("Stage 01 failed", exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def run_baseline_demo(self) -> None:
        if self.running:
            return
        try:
            trial, python_exe, release = self._validate_run_inputs()
        except Exception as exc:
            self.show_error("Cannot run quick review", exc)
            return

        session_root = (
            self.script_dir / "gui_sessions" / f"{trial.pair_key}_{now_stamp()}_baseline"
        ).resolve()
        session_root.mkdir(parents=True, exist_ok=False)
        prepared = session_root / "prepared"
        baseline = session_root / "baseline_results"

        self.current_session_root = session_root
        self._set_running(True)
        self._clear_baseline()
        self.set_status("Stage 01: validating raw EAV trial…", 10)
        self.notebook.select(self.log_tab)
        self.append_log("\n" + "=" * 100)
        self.append_log(f"[QUICK REVIEW] {trial.pair_key}")

        def worker() -> None:
            try:
                rc1 = self._run_streamed(
                    self._stage01_command(
                        trial=trial, python_exe=python_exe, output_dir=prepared
                    ),
                    cwd=self.deployment_root,
                    label="STAGE 01",
                )
                require(rc1 == 0, f"Stage 01 exited with code {rc1}")
                self.ui_queue.put(("progress", ("Stage 02: frozen 4-window inference…", 45)))

                rc2 = self._run_streamed(
                    self._stage02_command(
                        python_exe=python_exe,
                        prepared=prepared,
                        output_dir=baseline,
                        release=release,
                    ),
                    cwd=self.deployment_root,
                    label="STAGE 02",
                )
                require(rc2 == 0, f"Stage 02 exited with code {rc2}")

                result_json = baseline / "reviewer_results.json"
                result_csv = baseline / "reviewer_results.csv"
                require(result_json.is_file(), f"Reviewer JSON missing: {result_json}")
                result = read_json(result_json)
                require(result.get("status") == "PASS", "Quick review status is not PASS")

                self.ui_queue.put((
                    "baseline_success",
                    {
                        "session_root": session_root,
                        "output_dir": baseline,
                        "json": result_json,
                        "csv": result_csv,
                        "result": result,
                    },
                ))
            except Exception as exc:
                self.ui_queue.put(("worker_error", ("Quick reviewer workflow failed", exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def run_temporal_demo(self) -> None:
        if self.running:
            return
        try:
            trial, python_exe, release = self._validate_run_inputs()
        except Exception as exc:
            self.show_error("Cannot run temporal trajectory", exc)
            return

        session_root = (
            self.script_dir / "gui_sessions" / f"{trial.pair_key}_{now_stamp()}_trajectory"
        ).resolve()
        session_root.mkdir(parents=True, exist_ok=False)

        prepared = session_root / "prepared"
        temporal_prepared = session_root / "temporal_prepared"
        trajectory_output = session_root / "trajectory_results"

        self.current_session_root = session_root
        self.current_output_dir = None
        self._set_running(True)
        self._clear_trajectory()
        self.set_status("Stage 01: validating raw EAV trial…", 8)
        self.notebook.select(self.log_tab)
        self.append_log("\n" + "=" * 100)
        self.append_log(f"[TEMPORAL REVIEW] {trial.pair_key}")
        self.append_log(
            "[BOUNDARY] GUI orchestrates Stage 01 → Stage 04 → Stage 05 only; "
            "all model/quality/fusion logic remains frozen outside ReviewerDemo."
        )

        def worker() -> None:
            try:
                # Stage 01
                rc1 = self._run_streamed(
                    self._stage01_command(
                        trial=trial, python_exe=python_exe, output_dir=prepared
                    ),
                    cwd=self.deployment_root,
                    label="STAGE 01",
                )
                require(rc1 == 0, f"Stage 01 exited with code {rc1}")
                require((prepared / "reviewer_windows.jsonl").is_file(),
                        "Stage 01 did not produce reviewer_windows.jsonl")
                self.ui_queue.put((
                    "progress",
                    ("Stage 04: creating overlapping 5-s / 1-s-stride windows…", 28),
                ))

                # Stage 04
                rc4 = self._run_streamed(
                    self._stage04_command(
                        python_exe=python_exe,
                        prepared=prepared,
                        output_dir=temporal_prepared,
                    ),
                    cwd=self.deployment_root,
                    label="STAGE 04",
                )
                require(rc4 == 0, f"Stage 04 exited with code {rc4}")
                require((temporal_prepared / "temporal_windows.jsonl").is_file(),
                        "Stage 04 did not produce temporal_windows.jsonl")
                self.ui_queue.put((
                    "progress",
                    ("Stage 05: frozen temporal inference (16 windows)…", 52),
                ))

                # Stage 05
                rc5 = self._run_streamed(
                    self._stage05_command(
                        python_exe=python_exe,
                        temporal_prepared=temporal_prepared,
                        output_dir=trajectory_output,
                        release=release,
                    ),
                    cwd=self.deployment_root,
                    label="STAGE 05",
                )
                require(rc5 == 0, f"Stage 05 exited with code {rc5}")

                jsonl_path = trajectory_output / "emotion_trajectory.jsonl"
                csv_path = trajectory_output / "emotion_trajectory.csv"
                summary_path = trajectory_output / "emotion_trajectory_summary.json"
                require(jsonl_path.is_file(), f"Trajectory JSONL missing: {jsonl_path}")
                require(csv_path.is_file(), f"Trajectory CSV missing: {csv_path}")
                require(summary_path.is_file(), f"Trajectory summary missing: {summary_path}")

                rows = read_jsonl(jsonl_path)
                summary = read_json(summary_path)
                require(summary.get("status") == "PASS", "Trajectory summary status is not PASS")
                require(len(rows) == 16, f"Expected 16 trajectory points, got {len(rows)}")

                self.ui_queue.put((
                    "trajectory_success",
                    {
                        "session_root": session_root,
                        "output_dir": trajectory_output,
                        "jsonl": jsonl_path,
                        "csv": csv_path,
                        "summary_path": summary_path,
                        "rows": rows,
                        "summary": summary,
                    },
                ))
            except Exception as exc:
                self.ui_queue.put(("worker_error", ("Temporal reviewer workflow failed", exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _run_streamed(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        label: str,
    ) -> int:
        self.ui_queue.put(("log", f"\n[{label}] COMMAND:\n  " + " ".join(command) + "\n"))
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )
        self.active_process = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.ui_queue.put(("log", line))
            return int(proc.wait())
        finally:
            self.active_process = None

    # ------------------------ queue ------------------------

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                event, payload = self.ui_queue.get_nowait()
                if event == "log":
                    self.append_log(str(payload))
                elif event == "progress":
                    text, value = payload
                    self.set_status(text, value)
                elif event == "prepare_success":
                    self.current_session_root = payload["session_root"]
                    self.current_output_dir = payload["prepared"]
                    self.set_status("PASS — Stage 01 trial preparation complete", 100)
                    self.append_log("[STAGE 01 PASS] Trial prepared.")
                    self._set_running(False)
                    self.open_folder_button.configure(state="normal")
                elif event == "baseline_success":
                    self.current_session_root = payload["session_root"]
                    self.current_output_dir = payload["output_dir"]
                    self.last_baseline_json = payload["json"]
                    self.last_baseline_csv = payload["csv"]
                    self._render_baseline(payload["result"])
                    self.set_status("PASS — quick 4-window review complete", 100)
                    self.append_log("[QUICK REVIEW PASS]")
                    self._set_running(False)
                    self.open_folder_button.configure(state="normal")
                    self.notebook.select(self.baseline_tab)
                elif event == "trajectory_success":
                    self.current_session_root = payload["session_root"]
                    self.current_output_dir = payload["output_dir"]
                    self.last_trajectory_jsonl = payload["jsonl"]
                    self.last_trajectory_csv = payload["csv"]
                    self.last_trajectory_summary = payload["summary_path"]
                    self._render_trajectory(payload["rows"], payload["summary"])
                    self.set_status("PASS — temporal emotion trajectory complete", 100)
                    self.append_log("[TEMPORAL TRAJECTORY PASS] 16 frozen-runtime time steps saved.")
                    self._set_running(False)
                    self._enable_trajectory_outputs()
                    self.notebook.select(self.trajectory_tab)
                elif event == "worker_error":
                    title, exc = payload
                    self.set_status("FAILED", 0)
                    self._set_running(False)
                    self.show_error(title, exc)
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self._drain_ui_queue)

    # ------------------------ rendering ------------------------

    def _clear_common_summary(self) -> None:
        self.reference_var.set("-")
        self.result_subject_var.set("-")
        self.mode_var.set("-")
        self.release_fp_var.set("-")
        self.route_counts_var.set("-")
        self.emotion_counts_var.set("-")
        self.asset_check_var.set("-")

    def _clear_baseline(self) -> None:
        for item in self.baseline_tree.get_children():
            self.baseline_tree.delete(item)
        self.baseline_note_var.set("Running quick 4-window review…")

    def _clear_trajectory(self) -> None:
        self._clear_common_summary()
        for item in self.trajectory_tree.get_children():
            self.trajectory_tree.delete(item)
        self.timeline_chart.clear_chart()
        self.probability_chart.clear_chart()
        self.quality_chart.clear_chart()
        self.trajectory_note_var.set(
            "Running temporal trajectory. No smoothing or independent aggregation is applied."
        )
        self.last_trajectory_jsonl = None
        self.last_trajectory_csv = None
        self.last_trajectory_summary = None
        for button in (
            self.open_csv_button,
            self.open_summary_button,
            self.export_csv_button,
            self.export_jsonl_button,
        ):
            button.configure(state="disabled")

    def _render_baseline(self, result: Mapping[str, Any]) -> None:
        for item in self.baseline_tree.get_children():
            self.baseline_tree.delete(item)

        reference = result.get("reference")
        ref_emotion = reference.get("emotion") if isinstance(reference, Mapping) else None
        self.reference_var.set(f"{ref_emotion or '-'} (audit only; not model input)")

        stage1 = result.get("stage1")
        stage1 = stage1 if isinstance(stage1, Mapping) else {}
        subject = stage1.get("subject", "-")
        instance = stage1.get("media_instance")
        pair_key = stage1.get("pair_key")
        self.result_subject_var.set(
            f"{subject} / {format_float(instance, 0)}"
            + (f" • {pair_key}" if pair_key else "")
        )
        self.mode_var.set("Quick review: 4 × non-overlapping 5-s windows")

        frozen = result.get("frozen_runtime")
        frozen = frozen if isinstance(frozen, Mapping) else {}
        self.release_fp_var.set(str(frozen.get("release_sha256") or "-"))
        run_summary = frozen.get("run_summary")
        run_summary = run_summary if isinstance(run_summary, Mapping) else {}

        display = result.get("display_summary")
        display = display if isinstance(display, Mapping) else {}
        self.route_counts_var.set(str(display.get("route_counts", run_summary.get("route_counts", "-"))))
        self.emotion_counts_var.set("See four decisions below")
        self.asset_check_var.set(
            "PASS" if run_summary.get("assets_verified_after_run") is True else "-"
        )

        windows = result.get("windows")
        require(isinstance(windows, list), "reviewer_results.json has no windows list")
        for w in windows:
            final = w.get("final")
            final = final if isinstance(final, Mapping) else {}
            modalities = w.get("modalities")
            modalities = modalities if isinstance(modalities, Mapping) else {}

            def modal(name: str) -> tuple[str, str]:
                x = modalities.get(name)
                x = x if isinstance(x, Mapping) else {}
                emotion = x.get("emotion") or ("UNAVAILABLE" if not x.get("available") else "-")
                return str(emotion), format_float(x.get("quality_for_fusion"))

            eeg_e, qe = modal("eeg")
            aud_e, qa = modal("audio")
            vid_e, qv = modal("video")
            self.baseline_tree.insert(
                "", "end",
                values=(
                    w.get("window_index_0based", "-"),
                    w.get("route") or "-",
                    final.get("emotion") or "NO_DECISION",
                    format_float(final.get("confidence")),
                    eeg_e, qe, aud_e, qa, vid_e, qv,
                ),
            )

        trial_records = result.get("authoritative_trial_results_from_main")
        trial_count = len(trial_records) if isinstance(trial_records, list) else 0
        self.baseline_note_var.set(
            f"{len(windows)} frozen 5-s windows • main.py trial records={trial_count} • "
            "Stage 03 performs no independent 20-s aggregation."
        )

    def _render_trajectory(
        self,
        rows: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> None:
        require(rows, "Trajectory rows are empty")
        for item in self.trajectory_tree.get_children():
            self.trajectory_tree.delete(item)

        reference = summary.get("reference_emotion")
        self.reference_var.set(f"{reference or '-'} (audit only; not model input)")
        self.result_subject_var.set(
            f"{summary.get('subject', '-')} • {summary.get('pair_key', '-')}"
        )
        policy = summary.get("temporal_policy")
        policy = policy if isinstance(policy, Mapping) else {}
        self.mode_var.set(
            f"Temporal: 5-s sliding window / {format_float(policy.get('stride_seconds'), 1)}-s stride / "
            f"{policy.get('window_count', len(rows))} points"
        )
        runtime = summary.get("runtime")
        runtime = runtime if isinstance(runtime, Mapping) else {}
        self.release_fp_var.set(str(runtime.get("release_sha256") or "-"))
        self.route_counts_var.set(str(summary.get("route_counts") or "-"))
        self.emotion_counts_var.set(str(summary.get("final_emotion_counts") or "-"))
        self.asset_check_var.set(
            "PASS" if runtime.get("assets_verified_after_run") is True else "-"
        )

        x_values: list[float] = []
        probability_series: dict[str, list[float | None]] = {
            e: [] for e in EMOTIONS
        }
        quality_series: dict[str, list[float | None]] = {
            "EEG": [], "Audio": [], "Video": []
        }

        for item in rows:
            center = float(item["center_seconds"])
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
            x_values.append(center)

            probs = item.get("final_probabilities")
            probs = probs if isinstance(probs, list) and len(probs) == 5 else [None] * 5
            for i, emotion in enumerate(EMOTIONS):
                probability_series[emotion].append(
                    None if probs[i] is None else float(probs[i])
                )

            modalities = item.get("modalities")
            modalities = modalities if isinstance(modalities, Mapping) else {}
            eeg = modalities.get("eeg")
            audio = modalities.get("audio")
            video = modalities.get("video")
            eeg = eeg if isinstance(eeg, Mapping) else {}
            audio = audio if isinstance(audio, Mapping) else {}
            video = video if isinstance(video, Mapping) else {}

            quality_series["EEG"].append(
                None if eeg.get("quality") is None else float(eeg["quality"])
            )
            quality_series["Audio"].append(
                None if audio.get("quality") is None else float(audio["quality"])
            )
            quality_series["Video"].append(
                None if video.get("quality") is None else float(video["quality"])
            )

            self.trajectory_tree.insert(
                "", "end",
                values=(
                    item.get("step", "-"),
                    f"{start:.1f}–{end:.1f}s",
                    f"{center:.1f}s",
                    item.get("route") or "-",
                    item.get("final_emotion") or "NO_DECISION",
                    format_float(item.get("final_confidence")),
                    eeg.get("emotion") or "-",
                    format_float(eeg.get("quality")),
                    audio.get("emotion") or "-",
                    format_float(audio.get("quality")),
                    video.get("emotion") or "-",
                    format_float(video.get("quality")),
                ),
            )

        self.timeline_chart.set_data(rows)
        self.probability_chart.set_data(
            x_values,
            [
                (emotion, probability_series[emotion], EMOTION_COLORS[emotion])
                for emotion in EMOTIONS
            ],
            y_min=0.0,
            y_max=1.0,
        )
        self.quality_chart.set_data(
            x_values,
            [
                (name, quality_series[name], QUALITY_COLORS[name])
                for name in ("EEG", "Audio", "Video")
            ],
            y_min=0.0,
            y_max=1.0,
        )

        self.trajectory_note_var.set(
            f"{len(rows)} saved time steps • centers "
            f"{x_values[0]:.1f}s → {x_values[-1]:.1f}s • "
            f"routes={summary.get('route_counts')} • "
            f"NO_DECISION={summary.get('no_decision_windows', 0)} • "
            "raw frozen-model trajectory; no smoothing, voting, HMM, or new 20-s classifier."
        )

    # ------------------------ outputs ------------------------

    def _enable_trajectory_outputs(self) -> None:
        if self.current_output_dir and self.current_output_dir.exists():
            self.open_folder_button.configure(state="normal")
        if self.last_trajectory_csv and self.last_trajectory_csv.is_file():
            self.open_csv_button.configure(state="normal")
            self.export_csv_button.configure(state="normal")
        if self.last_trajectory_summary and self.last_trajectory_summary.is_file():
            self.open_summary_button.configure(state="normal")
        if self.last_trajectory_jsonl and self.last_trajectory_jsonl.is_file():
            self.export_jsonl_button.configure(state="normal")

    def open_output_folder(self) -> None:
        target = self.current_output_dir or self.current_session_root
        if target:
            try:
                reveal_path(target)
            except Exception as exc:
                self.show_error("Could not open output folder", exc)

    def open_trajectory_csv(self) -> None:
        if self.last_trajectory_csv:
            try:
                reveal_path(self.last_trajectory_csv)
            except Exception as exc:
                self.show_error("Could not open trajectory CSV", exc)

    def open_trajectory_summary(self) -> None:
        if self.last_trajectory_summary:
            try:
                reveal_path(self.last_trajectory_summary)
            except Exception as exc:
                self.show_error("Could not open trajectory summary", exc)

    def _export_copy(self, source: Path | None, *, title: str, ext: str) -> None:
        if source is None or not source.is_file():
            self.show_error(title, "No trajectory result is available to export.")
            return
        target = filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            defaultextension=ext,
            initialfile=source.name,
            filetypes=[(ext.upper().lstrip("."), f"*{ext}"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            shutil.copy2(source, Path(target))
            messagebox.showinfo(title, f"Saved copy:\n{target}", parent=self.root)
        except Exception as exc:
            self.show_error(title, exc)

    def export_trajectory_csv(self) -> None:
        self._export_copy(
            self.last_trajectory_csv,
            title="Save trajectory CSV copy",
            ext=".csv",
        )

    def export_trajectory_jsonl(self) -> None:
        self._export_copy(
            self.last_trajectory_jsonl,
            title="Save trajectory JSONL copy",
            ext=".jsonl",
        )

    # ------------------------ close ------------------------

    def on_close(self) -> None:
        if self.running:
            messagebox.showwarning(
                "Inference is still running",
                (
                    "The frozen runtime is still processing data. Wait for completion "
                    "before closing so GPU/native subprocesses and saved results remain complete."
                ),
                parent=self.root,
            )
            return
        self.root.destroy()


def cli_self_test(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent
    project_root = deployment_root.parent

    python_exe = candidate_python(deployment_root, args.python)
    testing_data = Path(args.testing_data).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()

    checks: dict[str, Any] = {
        "stage01_exists": (script_dir / "01_prepare_eav_trial.py").is_file(),
        "stage02_exists": (script_dir / "02_run_eav_reviewer_inference.py").is_file(),
        "stage04_exists": (script_dir / "04_prepare_temporal_trajectory.py").is_file(),
        "stage05_exists": (script_dir / "05_run_temporal_trajectory.py").is_file(),
        "main_exists": (deployment_root / "main.py").is_file(),
        "config_exists": config.is_file(),
        "python_exists": python_exe.is_file(),
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
        "ffprobe_found": shutil.which("ffprobe") is not None,
    }

    try:
        inventory = discover_testing_data(testing_data)
        checks["inventory_subjects"] = list(inventory)
        checks["inventory_trials"] = sum(len(v) for v in inventory.values())
        checks["inventory_pass"] = True
    except Exception as exc:
        checks["inventory_pass"] = False
        checks["inventory_error"] = f"{type(exc).__name__}: {exc}"

    try:
        checks["environment"] = environment_probe(python_exe)
        checks["runtime_dependencies"] = runtime_dependency_probe(
            python_exe,
            deployment_root,
        )
        require_runtime_ready(checks["runtime_dependencies"])
        checks["resolved_inference_device"] = resolve_inference_device(
            args.device,
            checks["environment"],
        )
        checks["requested_inference_device"] = args.device
        checks["environment_pass"] = True
    except Exception as exc:
        checks["environment_pass"] = False
        checks["environment_error"] = f"{type(exc).__name__}: {exc}"

    release = (
        Path(args.release).expanduser().resolve()
        if args.release
        else discover_release(config, deployment_root)
    )
    checks["release_candidate"] = str(release) if release else None
    checks["release_file_exists"] = bool(release and release.is_file())

    required_bools = [
        v for k, v in checks.items()
        if isinstance(v, bool) and k != "release_file_exists"
    ]
    status = "PASS" if all(required_bools) else "FAIL"

    print(json.dumps(
        {
            "version": VERSION,
            "status": status,
            "script_dir": str(script_dir),
            "deployment_root": str(deployment_root),
            "testing_data": str(testing_data),
            "python": str(python_exe),
            "checks": checks,
            "trajectory_policy": {
                "window_seconds": 5.0,
                "stride_seconds": args.trajectory_stride,
                "expected_points_for_20s_trial": 16 if args.trajectory_stride == 1.0 else None,
            },
            "note": "Setup self-test only; no model/quality/fusion inference is executed.",
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if status == "PASS" else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    deployment_root = here.parent

    p = argparse.ArgumentParser(
        description="Reviewer-friendly GUI for frozen EAV multimodal emotion recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--testing-data",
        default=str(here / "testing Data"),
        help="Local ReviewerDemo EAV subject folders",
    )
    p.add_argument(
        "--python",
        default=None,
        help="Interpreter for Stage 01/02/04/05/main; project .venv-video is preferred",
    )
    p.add_argument("--config", default=str(deployment_root / "system_config.json"))
    p.add_argument("--release", default=None)
    p.add_argument("--eeg-unit", choices=("uV", "mV", "V"), default="uV")
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Neural inference device. Auto uses CUDA only when available in the selected Python.",
    )
    p.add_argument("--fusion-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--trajectory-stride",
        type=float,
        default=1.0,
        help="Reviewer temporal stride in seconds. GUI v1.1 exposes the validated 1-s setting.",
    )
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--version", action="version", version=VERSION)
    args = p.parse_args(argv)
    require(
        abs(args.trajectory_stride - 1.0) < 1e-12,
        "GUI v1.1 currently exposes the validated 1.0-s temporal stride only.",
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return cli_self_test(args)

    root = tk.Tk()
    ReviewerDemoApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewerDemoError as exc:
        print(f"REVIEWER DEMO ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
