#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReviewerDemo Stage 01
=====================
Auto-discover one EAV Speaking trial from MERIT-Offline Model/testing Data and prepare
four standard 5-second replay descriptors for the frozen final-deployment runtime.

This script is an INPUT ADAPTER only. It does NOT run emotion recognition,
quality estimation, fusion, or robot behaviour.

Default folder layout
---------------------
MERIT-Offline Model/
    01_prepare_eav_trial.py
    testing Data/
        subject1/
            EEG/
                subject1_eeg.mat
                subject1_eeg_label.mat
            Audio/
                *.wav
            Video/
                *.mp4
        subject2/
            ...
        subject4/
            ...

Default use
-----------
From 最终部署:

    python .\\MERIT-Offline Model\\01_prepare_eav_trial.py

The script will deterministically select:
    1) the first available subject folder, and
    2) the first complete Speaking trial for that subject.

Optional selection:

    python .\\MERIT-Offline Model\\01_prepare_eav_trial.py --subject subject1
    python .\\MERIT-Offline Model\\01_prepare_eav_trial.py --subject subject1 --instance 24
    python .\\MERIT-Offline Model\\01_prepare_eav_trial.py --subject subject1 --emotion Calmness
    python .\\MERIT-Offline Model\\01_prepare_eav_trial.py --list

Outputs
-------
By default:

MERIT-Offline Model/prepared/<subject>_instanceNNN/
    reviewer_windows.jsonl
    input_audit.json

The JSONL is compatible with the existing final-deployment main.py replay mode.

EAV binding used here
---------------------
The reviewed EAV protocol contains 200 EEG trials and alternating Listening /
Speaking media instances. For a media instance I in 1..200, the corresponding
zero-based EEG trial index is I-1. Speaking media instances are therefore even,
and their zero-based EEG indices are odd.

This automatic binding is NOT accepted blindly. When the subject's
`*_eeg_label.mat` is available, this script loads the official 200 x 10 one-hot EEG label matrix and checks
that the inferred EEG trial is a Speaking trial with the same emotion encoded
by the selected Audio/Video files.

Audio/Video pairing follows the frozen deployment contract:
    - same leading EAV instance number
    - same task = Speaking
    - same emotion

The secondary `Trial_XX` field is recorded for audit but is NOT required to be
identical across Audio and Video, matching the existing final-deployment logic.

Dependencies
------------
Python:
    scipy
    soundfile

System:
    ffprobe
    ffmpeg

No model package is loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "EAV-REVIEWER-DEMO-PREPARE.2.2"
SOURCE_SCHEMA = "eav.system.source_window.v1"

EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
EMOTION_TO_ID = {name.lower(): i for i, name in enumerate(EMOTIONS)}
MODALITIES = ("eeg", "audio", "video")

CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "CZ", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "PZ", "P4", "P8",
    "PO9", "O1", "OZ", "O2", "PO10",
]

EEG_SAMPLE_RATE = 500
EEG_TRIAL_SAMPLES = 10000
EEG_CHANNELS = 30
EEG_TRIAL_COUNT = 200
EEG_EXPECTED_AXIS_SIZES = {EEG_TRIAL_SAMPLES, EEG_CHANNELS, EEG_TRIAL_COUNT}
WINDOW_SECONDS = 5.0
WINDOW_COUNT = 4
EEG_WINDOW_SAMPLES = 2500

AUDIO_RUNTIME_RATE = 16000
AUDIO_RUNTIME_WINDOW_SAMPLES = 80000

VIDEO_EXPECTED_FPS = 30.0
VIDEO_REQUIRED_FRAMES = 600
VIDEO_WINDOW_FRAMES = 150

MEDIA_BASE_RE = (
    r"(?P<instance>\d+)_Trial_(?P<trial>\d+)_"
    r"(?P<task>Speaking|Listening)_"
    r"(?P<emotion>Neutral|Sadness|Anger|Happiness|Calmness)"
)
AUDIO_RE = re.compile(rf"^{MEDIA_BASE_RE}(?:_aud)?\.wav$", re.IGNORECASE)
VIDEO_RE = re.compile(rf"^{MEDIA_BASE_RE}\.mp4$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"^subject0*(\d+)$", re.IGNORECASE)


class PreparationError(RuntimeError):
    """Input/contract error for ReviewerDemo Stage 01."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise PreparationError(
            f"Refusing to overwrite existing file: {path}\n"
            "Use --overwrite only if replacement is intentional."
        )
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise PreparationError(
            f"Refusing to overwrite existing file: {path}\n"
            "Use --overwrite only if replacement is intentional."
        )
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path, *, with_hash: bool) -> dict[str, Any]:
    st = path.stat()
    out: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "modified_ns": int(st.st_mtime_ns),
    }
    if with_hash:
        out["sha256"] = sha256_file(path)
    return out


def normalize_subject(value: str) -> str:
    m = SUBJECT_RE.fullmatch(value.strip())
    require(m is not None, f"Invalid subject name: {value!r}")
    number = int(m.group(1))
    require(1 <= number <= 42, f"Subject number outside EAV range 1..42: {number}")
    return f"subject{number:02d}"


def subject_number_from_name(value: str) -> int:
    m = SUBJECT_RE.fullmatch(value.strip())
    require(m is not None, f"Invalid subject directory name: {value!r}")
    return int(m.group(1))


@dataclass(frozen=True)
class MediaIdentity:
    instance: int
    trial_number: int
    task: str
    emotion: str
    path: Path

    @property
    def label_id(self) -> int:
        return EMOTION_TO_ID[self.emotion.lower()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "trial_number": self.trial_number,
            "task": self.task,
            "emotion": self.emotion,
            "label_id": self.label_id,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class TrialPair:
    instance: int
    emotion: str
    audio: MediaIdentity
    video: MediaIdentity

    @property
    def eeg_trial_index_0based(self) -> int:
        return self.instance - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "emotion": self.emotion,
            "audio_trial_number": self.audio.trial_number,
            "video_trial_number": self.video.trial_number,
            "eeg_trial_index_0based": self.eeg_trial_index_0based,
            "audio_path": str(self.audio.path),
            "video_path": str(self.video.path),
        }


def parse_media_identity(path: Path, kind: str) -> MediaIdentity | None:
    matcher = AUDIO_RE if kind == "audio" else VIDEO_RE
    m = matcher.fullmatch(path.name)
    if m is None:
        return None
    emotion_raw = m.group("emotion")
    emotion = next(name for name in EMOTIONS if name.lower() == emotion_raw.lower())
    return MediaIdentity(
        instance=int(m.group("instance")),
        trial_number=int(m.group("trial")),
        task=m.group("task").capitalize(),
        emotion=emotion,
        path=path.resolve(),
    )


def locate_executable(value: str, role: str) -> str:
    found = shutil.which(value)
    if found:
        return str(Path(found).resolve())
    p = Path(value).expanduser()
    if p.is_file():
        return str(p.resolve())
    raise PreparationError(
        f"{role} executable not found: {value!r}. "
        f"Install {role} or pass its absolute path."
    )


def run_checked(cmd: Sequence[str], *, timeout: float = 120.0) -> str:
    proc = subprocess.run(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        raise PreparationError(
            "External command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"STDERR:\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


def parse_rate(value: str | None, field: str) -> float:
    require(value not in (None, "", "N/A", "0/0"), f"Missing/invalid {field}: {value!r}")
    assert value is not None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        require(den_f != 0, f"Invalid {field}: {value}")
        return float(num) / den_f
    return float(value)


def discover_subjects(testing_data: Path) -> list[Path]:
    require(testing_data.is_dir(), f"testing Data directory not found: {testing_data}")
    subjects = [
        d.resolve()
        for d in testing_data.iterdir()
        if d.is_dir() and SUBJECT_RE.fullmatch(d.name)
    ]
    subjects.sort(key=lambda p: subject_number_from_name(p.name))
    require(subjects, f"No subject folders found under: {testing_data}")
    return subjects


def select_subject(testing_data: Path, requested: str | None) -> Path:
    subjects = discover_subjects(testing_data)
    if requested is None:
        return subjects[0]

    target = normalize_subject(requested)
    matches = [p for p in subjects if normalize_subject(p.name) == target]
    require(
        len(matches) == 1,
        f"Requested subject {requested!r} not found uniquely under {testing_data}",
    )
    return matches[0]


def discover_eeg_files(subject_dir: Path) -> tuple[Path, Path]:
    eeg_dir = subject_dir / "EEG"
    require(eeg_dir.is_dir(), f"EEG directory missing: {eeg_dir}")

    mats = sorted(p.resolve() for p in eeg_dir.glob("*.mat") if p.is_file())
    signal = [p for p in mats if "label" not in p.name.lower()]
    labels = [p for p in mats if "label" in p.name.lower()]

    require(
        len(signal) == 1,
        f"Expected exactly one EEG signal MAT in {eeg_dir}; found: {[p.name for p in signal]}",
    )
    require(
        len(labels) == 1,
        f"Expected exactly one EEG label MAT in {eeg_dir}; found: {[p.name for p in labels]}",
    )
    return signal[0], labels[0]


def discover_media(subject_dir: Path) -> tuple[list[MediaIdentity], list[MediaIdentity]]:
    audio_dir = subject_dir / "Audio"
    video_dir = subject_dir / "Video"
    require(audio_dir.is_dir(), f"Audio directory missing: {audio_dir}")
    require(video_dir.is_dir(), f"Video directory missing: {video_dir}")

    audio: list[MediaIdentity] = []
    for p in sorted(audio_dir.glob("*.wav")):
        ident = parse_media_identity(p, "audio")
        if ident is not None and ident.task.lower() == "speaking":
            audio.append(ident)

    video: list[MediaIdentity] = []
    for p in sorted(video_dir.glob("*.mp4")):
        ident = parse_media_identity(p, "video")
        if ident is not None and ident.task.lower() == "speaking":
            video.append(ident)

    require(audio, f"No parseable Speaking WAV files found in {audio_dir}")
    require(video, f"No parseable Speaking MP4 files found in {video_dir}")
    return audio, video


def make_trial_pairs(
    audio: Sequence[MediaIdentity],
    video: Sequence[MediaIdentity],
) -> list[TrialPair]:
    amap: dict[int, list[MediaIdentity]] = {}
    vmap: dict[int, list[MediaIdentity]] = {}

    for x in audio:
        amap.setdefault(x.instance, []).append(x)
    for x in video:
        vmap.setdefault(x.instance, []).append(x)

    pairs: list[TrialPair] = []
    for instance in sorted(set(amap) & set(vmap)):
        aa = amap[instance]
        vv = vmap[instance]
        require(
            len(aa) == 1,
            f"Instance {instance:03d}: expected one Speaking Audio file, got {[x.path.name for x in aa]}",
        )
        require(
            len(vv) == 1,
            f"Instance {instance:03d}: expected one Speaking Video file, got {[x.path.name for x in vv]}",
        )
        a, v = aa[0], vv[0]

        require(a.task == "Speaking" and v.task == "Speaking", "Internal task-filter error")
        require(
            a.emotion == v.emotion,
            (
                f"Instance {instance:03d}: Audio/Video emotion mismatch: "
                f"{a.emotion} vs {v.emotion}"
            ),
        )
        require(
            1 <= instance <= EEG_TRIAL_COUNT,
            f"Media instance outside 1..{EEG_TRIAL_COUNT}: {instance}",
        )
        # EAV paired protocol: Speaking is every even 1-based media instance,
        # hence an odd zero-based EEG index = instance - 1.
        require(
            instance % 2 == 0,
            (
                f"Speaking instance {instance:03d} is not even. "
                "This contradicts the reviewed EAV paired protocol."
            ),
        )
        pairs.append(
            TrialPair(
                instance=instance,
                emotion=a.emotion,
                audio=a,
                video=v,
            )
        )

    require(pairs, "No complete Speaking Audio/Video trial pairs were found")
    return pairs


def select_trial(
    pairs: Sequence[TrialPair],
    *,
    instance: int | None,
    emotion: str | None,
) -> TrialPair:
    candidates = list(pairs)

    if emotion is not None:
        canonical = next(
            (x for x in EMOTIONS if x.lower() == emotion.lower()),
            None,
        )
        require(canonical is not None, f"Unsupported emotion: {emotion!r}")
        candidates = [p for p in candidates if p.emotion == canonical]

    if instance is not None:
        require(1 <= instance <= 200, "--instance must be in 1..200")
        candidates = [p for p in candidates if p.instance == instance]

    require(
        candidates,
        (
            "No complete Speaking trial matches the requested selection. "
            f"Available instances: {[p.instance for p in pairs[:20]]}"
            + (" ..." if len(pairs) > 20 else "")
        ),
    )
    return sorted(candidates, key=lambda p: p.instance)[0]


def inspect_eeg_mat(path: Path, requested_variable: str) -> dict[str, Any]:
    try:
        from scipy.io import whosmat
    except ImportError as exc:
        raise PreparationError("Python package 'scipy' is required.") from exc

    try:
        entries = whosmat(str(path))
    except Exception as exc:
        raise PreparationError(f"Could not inspect EEG MAT: {path}\n{exc}") from exc

    variables = {name: tuple(int(v) for v in shape) for name, shape, _ in entries}

    def valid(shape: tuple[int, ...]) -> bool:
        return len(shape) == 3 and set(shape) == EEG_EXPECTED_AXIS_SIZES

    if requested_variable != "auto":
        require(requested_variable in variables, f"EEG variable {requested_variable!r} not found")
        variable = requested_variable
        require(
            valid(variables[variable]),
            f"EEG variable {variable!r} has incompatible shape {variables[variable]}",
        )
    else:
        preferred = [
            name for name in ("seg", "seg1")
            if name in variables and valid(variables[name])
        ]
        if preferred:
            variable = "seg" if "seg" in preferred else preferred[0]
        else:
            compatible = [name for name, shape in variables.items() if valid(shape)]
            require(
                len(compatible) == 1,
                (
                    "Could not uniquely identify the EAV EEG tensor. "
                    f"Compatible variables: {compatible}; all variables: {variables}"
                ),
            )
            variable = compatible[0]

    shape = variables[variable]
    return {
        "variable": variable,
        "stored_shape": list(shape),
        "sample_axis": shape.index(EEG_TRIAL_SAMPLES),
        "channel_axis": shape.index(EEG_CHANNELS),
        "trial_axis": shape.index(EEG_TRIAL_COUNT),
        "canonical_shape": [EEG_TRIAL_SAMPLES, EEG_CHANNELS, EEG_TRIAL_COUNT],
        "sample_rate_hz": EEG_SAMPLE_RATE,
        "trial_duration_seconds": 20.0,
        "trial_count": EEG_TRIAL_COUNT,
    }


def load_and_validate_eeg_labels(
    label_path: Path,
    *,
    selected_trial_index: int,
    expected_emotion: str,
) -> dict[str, Any]:
    """
    Validate the EAV EEG label MAT.

    Official EAV label format:
        200 trials x 10 classes, one-hot encoded
    where the 10 columns represent:
        0 Listening/Neutral
        1 Speaking/Neutral
        2 Listening/Sadness
        3 Speaking/Sadness
        4 Listening/Anger
        5 Speaking/Anger
        6 Listening/Happiness
        7 Speaking/Happiness
        8 Listening/Calmness
        9 Speaking/Calmness

    A transposed 10 x 200 one-hot matrix is also accepted defensively.
    The older 200-element integer-code vector format is retained only as a
    compatibility fallback; it is not the expected EAV representation.
    """
    try:
        import numpy as np
        from scipy.io import loadmat
    except ImportError as exc:
        raise PreparationError("Python packages 'numpy' and 'scipy' are required.") from exc

    try:
        obj = loadmat(str(label_path))
    except Exception as exc:
        raise PreparationError(f"Could not read EEG label MAT: {label_path}\n{exc}") from exc

    require(
        0 <= selected_trial_index < EEG_TRIAL_COUNT,
        f"EEG trial index out of range: {selected_trial_index}",
    )

    # ------------------------------------------------------------------
    # 1) Expected EAV representation: 200 x 10 one-hot.
    # ------------------------------------------------------------------
    matrix_candidates: list[tuple[str, Any, str]] = []

    for name, value in obj.items():
        if name.startswith("__"):
            continue

        arr = np.asarray(value)
        if arr.dtype.kind not in "biuf" or arr.ndim != 2:
            continue

        if arr.shape == (EEG_TRIAL_COUNT, 10):
            matrix = arr.astype(float, copy=False)
            orientation = "200x10"
        elif arr.shape == (10, EEG_TRIAL_COUNT):
            matrix = arr.T.astype(float, copy=False)
            orientation = "10x200_transposed_to_200x10"
        else:
            continue

        if not np.isfinite(matrix).all():
            continue

        # Accept standard one-hot values represented as bool/int/float.
        rounded = np.rint(matrix)
        if not np.allclose(matrix, rounded, atol=1e-8, rtol=0):
            continue

        onehot = rounded.astype(int)
        if not np.isin(onehot, [0, 1]).all():
            continue
        if not np.all(onehot.sum(axis=1) == 1):
            continue

        matrix_candidates.append((name, onehot, orientation))

    if matrix_candidates:
        preferred = [x for x in matrix_candidates if "label" in x[0].lower()]
        if preferred:
            matrix_candidates = preferred

        require(
            len(matrix_candidates) == 1,
            (
                "Could not uniquely identify the EAV 200x10 one-hot EEG label matrix; "
                f"candidates: {[name for name, _, _ in matrix_candidates]}"
            ),
        )

        variable, onehot, orientation = matrix_candidates[0]
        row = onehot[selected_trial_index]
        code = int(np.argmax(row))
        encoding = "one_hot_200x10"

    else:
        # ------------------------------------------------------------------
        # 2) Compatibility fallback: a 200-element integer class-code vector.
        # ------------------------------------------------------------------
        vector_candidates: list[tuple[str, Any]] = []

        for name, value in obj.items():
            if name.startswith("__"):
                continue

            arr = np.asarray(value)
            if arr.size != EEG_TRIAL_COUNT or arr.dtype.kind not in "iuf":
                continue

            flat = arr.reshape(-1).astype(float)
            if not np.isfinite(flat).all():
                continue

            rounded = np.rint(flat)
            if not np.allclose(flat, rounded, atol=1e-8, rtol=0):
                continue

            iv = rounded.astype(int)
            if np.all((0 <= iv) & (iv <= 9)):
                vector_candidates.append((name, iv))
            elif np.all((1 <= iv) & (iv <= 10)):
                vector_candidates.append((name, iv - 1))

        preferred = [x for x in vector_candidates if "label" in x[0].lower()]
        if preferred:
            vector_candidates = preferred

        require(
            len(vector_candidates) == 1,
            (
                "Could not identify the EAV EEG labels. Expected a 200x10 one-hot "
                f"matrix (or 10x200 transpose) in {label_path}. "
                f"Available non-private MAT variables: "
                f"{[(k, list(np.asarray(v).shape), str(np.asarray(v).dtype)) for k, v in obj.items() if not k.startswith('__')]}"
            ),
        )

        variable, labels = vector_candidates[0]
        code = int(labels[selected_trial_index])
        orientation = "200_element_integer_vector"
        encoding = "integer_code_vector_compatibility"

    expected_emotion_id = EMOTION_TO_ID[expected_emotion.lower()]
    expected_code = expected_emotion_id * 2 + 1  # odd code => Speaking

    code_to_description = {
        0: ("Listening", "Neutral"),
        1: ("Speaking", "Neutral"),
        2: ("Listening", "Sadness"),
        3: ("Speaking", "Sadness"),
        4: ("Listening", "Anger"),
        5: ("Speaking", "Anger"),
        6: ("Listening", "Happiness"),
        7: ("Speaking", "Happiness"),
        8: ("Listening", "Calmness"),
        9: ("Speaking", "Calmness"),
    }

    require(0 <= code <= 9, f"Decoded EEG label code outside 0..9: {code}")
    task, emotion = code_to_description[code]

    require(
        code == expected_code,
        (
            "Automatic media-instance -> EEG-trial binding failed label validation.\n"
            f"Selected media emotion : {expected_emotion}\n"
            f"Inferred EEG index      : {selected_trial_index}\n"
            f"EEG label code          : {code} ({task}/{emotion})\n"
            f"Expected EEG code        : {expected_code} (Speaking/{expected_emotion})\n"
            "The script refuses to guess a different mapping."
        ),
    )

    return {
        "variable": variable,
        "source_encoding": encoding,
        "source_orientation": orientation,
        "class_order": [
            "Listening/Neutral",
            "Speaking/Neutral",
            "Listening/Sadness",
            "Speaking/Sadness",
            "Listening/Anger",
            "Speaking/Anger",
            "Listening/Happiness",
            "Speaking/Happiness",
            "Listening/Calmness",
            "Speaking/Calmness",
        ],
        "selected_trial_index_0based": selected_trial_index,
        "selected_code": code,
        "selected_task": task,
        "selected_emotion": emotion,
        "expected_code": expected_code,
        "mapping_validated": True,
    }


def inspect_audio(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PreparationError("Python package 'soundfile' is required.") from exc

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise PreparationError(f"Could not inspect Audio WAV: {path}\n{exc}") from exc

    require(info.samplerate > 0 and info.frames > 0 and info.channels >= 1, "Invalid Audio metadata")
    duration = float(info.frames) / float(info.samplerate)
    require(
        duration >= 20.0,
        f"Audio is shorter than 20 s; runtime does not invent padding: {duration:.3f} s",
    )
    return {
        "source_sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": duration,
        "format": str(info.format),
        "subtype": str(info.subtype),
        "runtime_policy": {
            "downmix_if_multichannel": True,
            "resample_target_hz": AUDIO_RUNTIME_RATE,
            "use_first_seconds": 20.0,
            "window_seconds": WINDOW_SECONDS,
            "samples_per_runtime_window": AUDIO_RUNTIME_WINDOW_SAMPLES,
            "window_count": WINDOW_COUNT,
        },
    }


def inspect_video(path: Path, ffprobe: str) -> dict[str, Any]:
    raw = run_checked(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of", "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreparationError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc

    streams = payload.get("streams")
    require(isinstance(streams, list) and len(streams) == 1, f"Expected one video stream: {path}")
    s = streams[0]

    r_fps = parse_rate(s.get("r_frame_rate"), "r_frame_rate")
    avg_fps = parse_rate(s.get("avg_frame_rate"), "avg_frame_rate")
    require(
        math.isclose(r_fps, VIDEO_EXPECTED_FPS, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(avg_fps, VIDEO_EXPECTED_FPS, rel_tol=0.0, abs_tol=1e-6),
        (
            "The frozen EAV trial replay path requires recorded 30-fps video; "
            f"observed r={r_fps}, avg={avg_fps}: {path}"
        ),
    )

    frame_text = s.get("nb_read_frames")
    if frame_text in (None, "", "N/A"):
        frame_text = s.get("nb_frames")
    frame_count = None if frame_text in (None, "", "N/A") else int(frame_text)

    duration = None
    if s.get("duration") not in (None, "", "N/A"):
        duration = float(s["duration"])
    if duration is None and frame_count is not None:
        duration = frame_count / avg_fps

    require(duration is not None, f"Could not determine Video duration: {path}")
    require(
        duration + 1.0 / VIDEO_EXPECTED_FPS >= 20.0,
        f"Video shorter than required 20 s span: {duration:.3f} s",
    )
    if frame_count is not None:
        require(
            frame_count >= VIDEO_REQUIRED_FRAMES,
            f"Video contains {frame_count} frames; at least {VIDEO_REQUIRED_FRAMES} are required",
        )

    return {
        "codec_name": s.get("codec_name"),
        "width": int(s["width"]) if s.get("width") is not None else None,
        "height": int(s["height"]) if s.get("height") is not None else None,
        "r_frame_rate_fps": r_fps,
        "avg_frame_rate_fps": avg_fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "runtime_windows": [
            {
                "window_index": w,
                "start_frame": w * VIDEO_WINDOW_FRAMES,
                "end_frame_exclusive": (w + 1) * VIDEO_WINDOW_FRAMES,
            }
            for w in range(WINDOW_COUNT)
        ],
    }


def make_window_id(subject: str, pair: TrialPair, w: int) -> str:
    return (
        f"{subject.upper()}_INSTANCE_{pair.instance:03d}_"
        f"{pair.emotion.upper()}_W{w}"
    )


def build_descriptors(
    *,
    subject: str,
    eeg_path: Path,
    eeg_variable: str,
    pair: TrialPair,
    audio_policy: str,
) -> list[dict[str, Any]]:
    eeg_trial_index = pair.eeg_trial_index_0based
    rows: list[dict[str, Any]] = []

    for w in range(WINDOW_COUNT):
        rows.append(
            {
                "schema": SOURCE_SCHEMA,
                "window_id": make_window_id(subject, pair, w),
                "window_seconds": WINDOW_SECONDS,
                "task_condition": "Speaking",
                "split": "replay",
                "identity": {
                    "subject": subject,
                    # Required by the frozen QualityController for an EAV replay
                    # without wall-clock timing. This is the same pair-key form
                    # used by the formal Stage0C/raw replay pipeline.
                    "pair_key": f"{subject}_instance{pair.instance:03d}",
                    "source_type": "EAV_REVIEWER_DEMO",
                    "media_instance": pair.instance,
                    "audio_trial_number": pair.audio.trial_number,
                    "video_trial_number": pair.video.trial_number,
                    "eeg_trial_index_0based": eeg_trial_index,
                    "window_idx_0based": w,
                },
                "task_evidence": "EAV_FILENAME_AND_EEG_LABEL_VALIDATED",
                # Ground-truth emotion is intentionally excluded from model input.
                "modalities": {
                    "eeg": {
                        "present": True,
                        "path": str(eeg_path),
                        "kind": "eav_mat",
                        "variable": eeg_variable,
                        "trial_index": eeg_trial_index,
                        "window_index": w,
                        "sample_rate": EEG_SAMPLE_RATE,
                        "input_unit": "training_native",
                        "channel_names": CHANNELS,
                    },
                    "audio": {
                        "present": True,
                        "path": str(pair.audio.path),
                        "kind": "audio_trial",
                        "window_index": w,
                        "audio_policy": audio_policy,
                    },
                    "video": {
                        "present": True,
                        "path": str(pair.video.path),
                        "kind": "video_trial",
                        "window_index": w,
                    },
                },
            }
        )
    return rows


def self_validate_descriptors(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == WINDOW_COUNT, f"Expected {WINDOW_COUNT} descriptors")
    seen: set[str] = set()
    for w, d in enumerate(rows):
        require(d.get("schema") == SOURCE_SCHEMA, "Descriptor schema mismatch")
        require(d.get("window_seconds") == 5.0, "Descriptor must be exactly 5 s")
        require(d.get("task_condition") == "Speaking", "Descriptor must be Speaking")
        require(d.get("split") == "replay", "Descriptor must use replay split")
        wid = d.get("window_id")
        require(isinstance(wid, str) and wid and wid not in seen, f"Duplicate/invalid window ID: {wid}")
        seen.add(wid)

        mods = d.get("modalities")
        require(isinstance(mods, Mapping) and set(mods) == set(MODALITIES), "Exactly 3 modality descriptors required")
        for m in MODALITIES:
            item = mods[m]
            require(item.get("present") is True, f"{m} unexpectedly absent")
            require(Path(item["path"]).is_file(), f"{m} source missing: {item['path']}")
            require(int(item["window_index"]) == w, f"{m} window index mismatch")


def print_inventory(testing_data: Path) -> None:
    subjects = discover_subjects(testing_data)
    print("=" * 100)
    print("REVIEWER DEMO TESTING DATA INVENTORY")
    print("=" * 100)
    print(f"Root: {testing_data}")
    print()

    for subject_dir in subjects:
        subject = normalize_subject(subject_dir.name)
        try:
            eeg, label = discover_eeg_files(subject_dir)
            audio, video = discover_media(subject_dir)
            pairs = make_trial_pairs(audio, video)
            by_emotion: dict[str, int] = {x: 0 for x in EMOTIONS}
            for pair in pairs:
                by_emotion[pair.emotion] += 1
            print(
                f"{subject:10s} | folder={subject_dir.name:8s} | "
                f"Speaking pairs={len(pairs):3d} | "
                + ", ".join(f"{k}={v}" for k, v in by_emotion.items())
            )
            print(f"             EEG={eeg.name}; labels={label.name}")
            preview = ", ".join(
                f"{p.instance:03d}:{p.emotion}"
                for p in pairs[:10]
            )
            print(f"             First trials: {preview}")
        except Exception as exc:
            print(f"{subject:10s} | ERROR: {exc}")
        print()

    print("=" * 100)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_testing = script_dir / "testing Data"

    p = argparse.ArgumentParser(
        description=(
            "ReviewerDemo Stage 01 v2: auto-discover a complete EAV Speaking trial "
            "from MERIT-Offline Model/testing Data and prepare 4 x 5-s replay descriptors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--testing-data",
        default=str(default_testing),
        help="Folder containing subject1/subject2/... test copies",
    )
    p.add_argument(
        "--subject",
        default=None,
        help="Subject to use, e.g. subject1. If omitted, the first available subject is selected.",
    )
    p.add_argument(
        "--instance",
        type=int,
        default=None,
        help="Leading EAV media instance number. If omitted, the first matching Speaking trial is selected.",
    )
    p.add_argument(
        "--emotion",
        choices=EMOTIONS,
        default=None,
        help="Optional emotion filter when automatically choosing a trial.",
    )
    p.add_argument(
        "--eeg-variable",
        default="auto",
        help="Normally auto; explicit seg/seg1 is supported for troubleshooting.",
    )
    p.add_argument(
        "--audio-policy",
        choices=("a0_pcm16", "float_window"),
        default="a0_pcm16",
        help="Existing final-deployment raw Audio replay policy.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory; default is MERIT-Offline Model/prepared/<subject>_instanceNNN",
    )
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--hash-inputs", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--list",
        action="store_true",
        help="List discovered subjects and complete Speaking trials, then exit.",
    )
    p.add_argument("--version", action="version", version=VERSION)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    testing_data = Path(args.testing_data).expanduser().resolve()

    if args.list:
        print_inventory(testing_data)
        return 0

    print("=" * 100)
    print("EAV REVIEWER DEMO — STAGE 01: AUTO-PREPARE RAW EAV TRIAL")
    print("=" * 100)
    print(f"Version      : {VERSION}")
    print(f"testing Data : {testing_data}")
    print()

    # ------------------------------------------------------------------
    # 1. Select subject and discover local raw files.
    # ------------------------------------------------------------------
    print("[1/7] Selecting subject and discovering raw EAV files...")
    subject_dir = select_subject(testing_data, args.subject)
    subject = normalize_subject(subject_dir.name)
    eeg_path, eeg_label_path = discover_eeg_files(subject_dir)
    audio_items, video_items = discover_media(subject_dir)
    pairs = make_trial_pairs(audio_items, video_items)

    print(f"      Subject folder : {subject_dir.name}")
    print(f"      Subject ID     : {subject}")
    print(f"      EEG signal     : {eeg_path.name}")
    print(f"      EEG labels     : {eeg_label_path.name}")
    print(f"      Speaking pairs : {len(pairs)}")

    # ------------------------------------------------------------------
    # 2. Select one complete Speaking trial.
    # ------------------------------------------------------------------
    print("[2/7] Selecting complete Speaking trial...")
    pair = select_trial(pairs, instance=args.instance, emotion=args.emotion)
    eeg_trial_index = pair.eeg_trial_index_0based
    print(f"      Media instance : {pair.instance:03d}")
    print(f"      Emotion        : {pair.emotion}")
    print(f"      Audio          : {pair.audio.path.name}")
    print(f"      Video          : {pair.video.path.name}")
    print(f"      Audio Trial_XX : {pair.audio.trial_number}")
    print(f"      Video Trial_XX : {pair.video.trial_number}")
    if pair.audio.trial_number != pair.video.trial_number:
        print("      NOTE            : secondary Trial_XX differs; this is allowed by the frozen binding contract.")
    print(f"      EEG trial index: {eeg_trial_index} (0-based; inferred as instance - 1)")

    # ------------------------------------------------------------------
    # 3. Inspect EEG tensor and validate automatic trial mapping with labels.
    # ------------------------------------------------------------------
    print("[3/7] Validating EEG tensor and automatic trial binding...")
    eeg_info = inspect_eeg_mat(eeg_path, args.eeg_variable)
    label_info = load_and_validate_eeg_labels(
        eeg_label_path,
        selected_trial_index=eeg_trial_index,
        expected_emotion=pair.emotion,
    )
    print(
        f"      PASS: variable={eeg_info['variable']}, "
        f"shape={tuple(eeg_info['stored_shape'])}"
    )
    print(
        f"      PASS: EEG label={label_info['selected_code']} "
        f"({label_info['selected_task']}/{label_info['selected_emotion']})"
    )

    # ------------------------------------------------------------------
    # 4. Inspect Audio.
    # ------------------------------------------------------------------
    print("[4/7] Inspecting Audio...")
    audio_info = inspect_audio(pair.audio.path)
    print(
        f"      PASS: sr={audio_info['source_sample_rate_hz']} Hz, "
        f"channels={audio_info['channels']}, "
        f"duration={audio_info['duration_seconds']:.3f} s"
    )

    # ------------------------------------------------------------------
    # 5. Inspect Video and external tools.
    # ------------------------------------------------------------------
    print("[5/7] Inspecting Video and FFmpeg tools...")
    ffprobe = locate_executable(args.ffprobe, "ffprobe")
    ffmpeg = locate_executable(args.ffmpeg, "ffmpeg")
    video_info = inspect_video(pair.video.path, ffprobe)
    print(
        f"      PASS: fps={video_info['avg_frame_rate_fps']:.3f}, "
        f"frames={video_info['frame_count']}, "
        f"duration={video_info['duration_seconds']:.3f} s"
    )

    # ------------------------------------------------------------------
    # 6. Generate standard replay descriptors.
    # ------------------------------------------------------------------
    print("[6/7] Generating 4 x 5-s final-runtime descriptors...")
    rows = build_descriptors(
        subject=subject,
        eeg_path=eeg_path,
        eeg_variable=eeg_info["variable"],
        pair=pair,
        audio_policy=args.audio_policy,
    )
    self_validate_descriptors(rows)

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = (
            script_dir
            / "prepared"
            / f"{subject}_instance{pair.instance:03d}"
        ).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "reviewer_windows.jsonl"
    audit_path = output_dir / "input_audit.json"
    write_jsonl(jsonl_path, rows, overwrite=args.overwrite)

    # ------------------------------------------------------------------
    # 7. Audit/provenance.
    # ------------------------------------------------------------------
    print("[7/7] Writing audit...")
    audit = {
        "schema": "eav.reviewer_demo.input_audit.v2",
        "version": VERSION,
        "status": "PASS",
        "created_utc": utc_now(),
        "purpose": (
            "Auto-discover one raw EAV Speaking trial from MERIT-Offline Model/testing Data "
            "and bind it to the unchanged final-deployment 5-s replay contract."
        ),
        "selection": {
            "testing_data": str(testing_data),
            "subject_folder": subject_dir.name,
            "subject": subject,
            "selection_mode": {
                "subject": "EXPLICIT" if args.subject else "FIRST_AVAILABLE_DETERMINISTIC",
                "instance": "EXPLICIT" if args.instance is not None else "AUTO_FIRST_MATCH",
                "emotion_filter": args.emotion,
            },
            "trial_pair": pair.to_dict(),
        },
        "eeg_binding": {
            "rule": "zero_based_eeg_trial_index = one_based_media_instance - 1",
            "protocol_note": "Speaking media instances are even; corresponding zero-based EEG indices are odd.",
            "selected_trial_index_0based": eeg_trial_index,
            "label_validation": label_info,
            "ground_truth_emotion_used_as_model_input": False,
        },
        "inputs": {
            "eeg": {
                **file_record(eeg_path, with_hash=args.hash_inputs),
                "inspection": eeg_info,
            },
            "eeg_labels": {
                **file_record(eeg_label_path, with_hash=args.hash_inputs),
                "inspection": label_info,
            },
            "audio": {
                **file_record(pair.audio.path, with_hash=args.hash_inputs),
                "media_identity": pair.audio.to_dict(),
                "inspection": audio_info,
            },
            "video": {
                **file_record(pair.video.path, with_hash=args.hash_inputs),
                "media_identity": pair.video.to_dict(),
                "inspection": video_info,
            },
        },
        "runtime_tools": {
            "ffprobe": ffprobe,
            "ffmpeg": ffmpeg,
        },
        "output_contract": {
            "descriptor_schema": SOURCE_SCHEMA,
            "task_condition": "Speaking",
            "split": "replay",
            "pair_key": f"{subject}_instance{pair.instance:03d}",
            "pair_key_semantics": (
                "EAV trial identity required by frozen QualityController to bind "
                "the four 5-s windows to one trial-relative replay span."
            ),
            "window_seconds": WINDOW_SECONDS,
            "window_count": WINDOW_COUNT,
            "jsonl": str(jsonl_path),
            "algorithm_changes": False,
            "model_inference_performed": False,
            "quality_inference_performed": False,
            "fusion_performed": False,
        },
        "stage_boundary": {
            "does": [
                "discover local ReviewerDemo testing Data",
                "select one complete Speaking Audio/Video trial",
                "infer EEG trial index from the reviewed EAV media-instance protocol",
                "validate inferred EEG trial against the subject EEG label MAT",
                "inspect EEG/Audio/Video source contracts",
                "generate four standard 5-s replay descriptors",
            ],
            "does_not": [
                "run EEG/Audio/Video emotion inference",
                "run PyPREP/DNSMOS/DOVER quality inference",
                "compute B_m, T_m, or q_m",
                "run adaptive fusion",
                "update Emotion State Manager",
                "perform robot actions",
            ],
        },
    }
    write_json(audit_path, audit, overwrite=args.overwrite)

    print()
    print("=" * 100)
    print("STAGE 01 COMPLETE")
    print("=" * 100)
    print("STATUS           : PASS")
    print(f"SUBJECT          : {subject}")
    print(f"INSTANCE         : {pair.instance:03d}")
    print(f"REFERENCE EMOTION: {pair.emotion} (audit only; NOT model input)")
    print(f"EEG TRIAL INDEX  : {eeg_trial_index} (auto + label validated)")
    print(f"WINDOWS          : {WINDOW_COUNT} x {WINDOW_SECONDS:.0f} s")
    print(f"DESCRIPTORS      : {jsonl_path}")
    print(f"AUDIT            : {audit_path}")
    print("MODEL INFERENCE  : NO")
    print("QUALITY INFERENCE: NO")
    print("FUSION           : NO")
    print("=" * 100)
    print()
    print("Next stage:")
    print("  Use reviewer_windows.jsonl with ReviewerDemo Stage 02 / the frozen main.py replay runtime.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as exc:
        print()
        print("=" * 100, file=sys.stderr)
        print("STAGE 01 FAILED", file=sys.stderr)
        print("=" * 100, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 100, file=sys.stderr)
        raise SystemExit(2)
