#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ReviewerDemo Stage 02
=====================
Run a Stage-01 prepared EAV trial through the SAME frozen final-deployment
`main.py --mode replay` runtime and build reviewer-facing outputs.

This wrapper does not reimplement emotion heads, quality scoring, routing, or
fusion. It only:
    1) resolves a Stage-01 reviewer_windows.jsonl,
    2) resolves the frozen quality-layer release candidate,
    3) launches the current final-deployment main.py,
    4) preserves its complete runtime output,
    5) creates compact JSON/CSV views for the future Reviewer GUI.

The current final-deployment main entry requires the frozen quality release
candidate explicitly unless `quality_layer.release` is already present in the
configuration. Stage 02 therefore resolves and passes `--release`.

For the currently frozen system, the expected release *semantic fingerprint*
(the JSON field `release_sha256`) is:
    516740caea3c8a672989d1e02a3179d5ef1ab9e9df394d6c27ddb42c42e4c098

Important: this is NOT the SHA-256 of the JSON file bytes. The release builder
computes it from canonical JSON content after excluding the `release_sha256`
field. Stage 02 reproduces that canonical fingerprint check exactly.

Default workflow
----------------
From the `最终部署` directory:

    python .\ReviewerDemo\02_run_eav_reviewer_inference.py --eeg-unit uV

If system_config.json already contains `"training_unit": "uV"`, simply:

    python .\ReviewerDemo\02_run_eav_reviewer_inference.py

Explicit prepared trial:

    python .\ReviewerDemo\02_run_eav_reviewer_inference.py ^
        --prepared ".\ReviewerDemo\prepared\subject01_instance002" ^
        --eeg-unit uV

Important
---------
- `latest_state.json` may intentionally be cleared by current main.py when the
  process exits. Historical predictions remain in window_results.jsonl and, when
  emitted by main.py, trial_results.jsonl.
- Stage 02 never averages four window predictions itself. If a trial-level
  result exists, it is copied from main.py's own `trial_results.jsonl`.
- No robot action is authorized.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "EAV-REVIEWER-DEMO-INFERENCE.1.4"

SOURCE_SCHEMA = "eav.system.source_window.v1"
CONFIG_SCHEMA = "eav.system.config.v1"

EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
MODALITIES = ("eeg", "audio", "video")
WINDOW_COUNT = 4
WINDOW_SECONDS = 5.0

TEST_SUBJECTS = {
    "subject03",
    "subject05",
    "subject20",
    "subject31",
    "subject35",
    "subject39",
}

SUBJECT_RE = re.compile(r"^subject0*(\d+)$", re.IGNORECASE)

# Frozen quality-layer semantic fingerprint reported by the final integrated run.
# This is the embedded/canonical release_sha256, NOT the raw file-byte SHA256.
EXPECTED_RELEASE_SHA256 = (
    "516740caea3c8a672989d1e02a3179d5ef1ab9e9df394d6c27ddb42c42e4c098"
)

# Frozen fusion runtime file used by current final deployment.
DEFAULT_FUSION_SCRIPT_RELATIVE = Path("Fusion") / "融合层部署模块.py"
DEFAULT_FUSION_ASSETS_RELATIVE = Path("Fusion")


class Stage2Error(RuntimeError):
    """ReviewerDemo Stage-02 contract/runtime error."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise Stage2Error(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_subject(value: str) -> str:
    m = SUBJECT_RE.fullmatch(str(value).strip())
    require(m is not None, f"Invalid subject identity: {value!r}")
    n = int(m.group(1))
    require(1 <= n <= 42, f"Subject number outside EAV range 1..42: {n}")
    return f"subject{n:02d}"


def expected_pair_key(subject: str, instance: int) -> str:
    """Canonical EAV pair identity used by the frozen replay/quality runtime."""
    return f"{normalize_subject(subject)}_instance{int(instance):03d}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    """Match evaluate_audio_quality_v2.py fingerprint() exactly."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_release_file(path: Path) -> dict[str, Any]:
    """Validate release semantic fingerprint, not raw file-byte hash."""
    release = read_json(path)
    require(
        release.get("schema") == "eav.quality_layer.release_candidate.v1",
        f"Wrong quality release schema: {release.get('schema')!r}",
    )
    embedded = release.get("release_sha256")
    require(
        isinstance(embedded, str) and re.fullmatch(r"[0-9a-fA-F]{64}", embedded),
        "Release is missing a valid release_sha256 field.",
    )
    payload = {k: v for k, v in release.items() if k != "release_sha256"}
    recomputed = canonical_fingerprint(payload)
    require(
        recomputed.lower() == embedded.lower(),
        (
            "Quality release canonical fingerprint does not match its embedded "
            "release_sha256. The file may have been modified.\n"
            f"Path       : {path}\n"
            f"Embedded   : {embedded}\n"
            f"Recomputed : {recomputed}"
        ),
    )
    require(
        embedded.lower() == EXPECTED_RELEASE_SHA256.lower(),
        (
            "Quality release is internally valid but is not the frozen release "
            "used by the final integrated system.\n"
            f"Path       : {path}\n"
            f"Expected   : {EXPECTED_RELEASE_SHA256}\n"
            f"Observed   : {embedded}"
        ),
    )
    require(
        release.get("variant") == "KEEP_V1",
        f"Expected frozen KEEP_V1 release, got {release.get('variant')!r}",
    )
    require(
        release.get("structural_search_closed") is True,
        "Release does not declare structural_search_closed=true.",
    )
    require(
        release.get("production_approved") is False,
        "Release unexpectedly declares production_approved=true.",
    )
    return {
        "release": release,
        "embedded_release_sha256": embedded.lower(),
        "canonical_recomputed_sha256": recomputed.lower(),
        "file_byte_sha256": sha256_file(path),
    }


def no_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise Stage2Error(f"Duplicate JSON key rejected: {key!r}")
        out[key] = value
    return out


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    try:
        obj = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=no_duplicate_object_pairs,
        )
    except Exception as exc:
        if isinstance(exc, Stage2Error):
            raise
        raise Stage2Error(f"Could not read JSON: {path}\n{exc}") from exc
    require(isinstance(obj, dict), f"Expected JSON object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            require(
                len(line) <= 4 * 1024 * 1024,
                f"Oversized JSONL line {line_no}: {path}",
            )
            try:
                obj = json.loads(
                    line,
                    object_pairs_hook=no_duplicate_object_pairs,
                )
            except Exception as exc:
                if isinstance(exc, Stage2Error):
                    raise
                raise Stage2Error(
                    f"Invalid JSONL at {path}:{line_no}\n{exc}"
                ) from exc
            require(
                isinstance(obj, dict),
                f"JSONL line {line_no} is not an object: {path}",
            )
            rows.append(obj)
    require(rows, f"No JSONL records found: {path}")
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric or null",
    )
    x = float(value)
    require(math.isfinite(x), f"{field} contains NaN/Inf")
    return x


def probability_vector(value: Any, *, field: str) -> list[float] | None:
    if value is None:
        return None
    require(
        isinstance(value, (list, tuple)) and len(value) == 5,
        f"{field} must contain 5 probabilities",
    )
    out = [finite_float(x, field=field) for x in value]
    require(all(x is not None for x in out), f"{field} contains null")
    return [float(x) for x in out]  # type: ignore[arg-type]


def argmax_emotion(probs: list[float] | None) -> tuple[str | None, float | None]:
    if not probs:
        return None, None
    idx = max(range(len(probs)), key=probs.__getitem__)
    return EMOTIONS[idx], float(probs[idx])


def discover_prepared(script_dir: Path) -> list[Path]:
    root = script_dir / "prepared"
    if not root.is_dir():
        return []
    items = [
        p.resolve()
        for p in root.rglob("reviewer_windows.jsonl")
        if p.is_file()
    ]
    items.sort(
        key=lambda p: (p.stat().st_mtime_ns, str(p).lower()),
        reverse=True,
    )
    return items


def resolve_manifest(
    *,
    script_dir: Path,
    prepared: str | None,
    manifest: str | None,
) -> tuple[Path, str]:
    require(
        not (prepared and manifest),
        "Choose either --prepared or --manifest, not both.",
    )

    if manifest:
        p = Path(manifest).expanduser().resolve()
        require(p.is_file(), f"--manifest not found: {p}")
        return p, "EXPLICIT_MANIFEST"

    if prepared:
        p = Path(prepared).expanduser().resolve()
        if p.is_file():
            require(
                p.name == "reviewer_windows.jsonl",
                f"Prepared file must be reviewer_windows.jsonl: {p}",
            )
            return p, "EXPLICIT_PREPARED_FILE"
        require(p.is_dir(), f"--prepared path not found: {p}")
        m = p / "reviewer_windows.jsonl"
        require(m.is_file(), f"Prepared manifest missing: {m}")
        return m.resolve(), "EXPLICIT_PREPARED_DIRECTORY"

    items = discover_prepared(script_dir)
    require(
        items,
        (
            f"No Stage-01 reviewer_windows.jsonl found under {script_dir / 'prepared'}.\n"
            "Run ReviewerDemo Stage 01 first."
        ),
    )
    return items[0], "AUTO_LATEST_PREPARED"


def validate_stage1_manifest(
    manifest: Path,
    *,
    allow_test: bool,
) -> dict[str, Any]:
    rows = read_jsonl(manifest)

    require(
        len(rows) == WINDOW_COUNT,
        (
            f"Expected one complete {WINDOW_COUNT}-window Stage-01 trial, "
            f"got {len(rows)} rows in {manifest}"
        ),
    )

    ids: set[str] = set()
    window_indices: list[int] = []
    subjects: set[str] = set()
    instances: set[int] = set()
    eeg_indices: set[int] = set()
    declared_pair_keys: set[str] = set()
    missing_pair_key_rows = 0

    for i, d in enumerate(rows):
        require(
            d.get("schema") == SOURCE_SCHEMA,
            f"Row {i}: wrong schema {d.get('schema')!r}",
        )
        require(
            d.get("window_seconds") == WINDOW_SECONDS,
            f"Row {i}: expected 5-s source window",
        )
        require(
            str(d.get("task_condition", "")).lower() == "speaking",
            f"Row {i}: ReviewerDemo is Speaking-only",
        )
        require(
            d.get("split") == "replay",
            f"Row {i}: Stage 01 should emit split='replay'",
        )

        wid = d.get("window_id")
        require(
            isinstance(wid, str) and wid and wid not in ids,
            f"Row {i}: invalid/duplicate window_id {wid!r}",
        )
        ids.add(wid)

        identity = d.get("identity")
        require(isinstance(identity, dict), f"Row {i}: missing identity")

        w = identity.get("window_idx_0based")
        require(
            isinstance(w, int) and not isinstance(w, bool) and 0 <= w < 4,
            f"Row {i}: invalid window_idx_0based {w!r}",
        )
        window_indices.append(w)

        subject = normalize_subject(str(identity.get("subject")))
        subjects.add(subject)

        inst = identity.get("media_instance")
        require(
            isinstance(inst, int) and not isinstance(inst, bool) and 1 <= inst <= 200,
            f"Row {i}: invalid media_instance {inst!r}",
        )
        instances.add(inst)

        pair_key = identity.get("pair_key")
        if pair_key is None:
            # Stage-01 V2.1 omitted this field. It is not an emotion label or
            # model feature; it is required replay source identity used by the
            # frozen QualityController to construct the trial-relative 5-s span.
            missing_pair_key_rows += 1
        else:
            require(
                isinstance(pair_key, str) and pair_key.strip() == pair_key and pair_key,
                f"Row {i}: invalid identity.pair_key {pair_key!r}",
            )
            declared_pair_keys.add(pair_key)

        eeg_idx = identity.get("eeg_trial_index_0based")
        require(
            isinstance(eeg_idx, int)
            and not isinstance(eeg_idx, bool)
            and 0 <= eeg_idx < 200,
            f"Row {i}: invalid eeg_trial_index_0based {eeg_idx!r}",
        )
        eeg_indices.add(eeg_idx)

        modalities = d.get("modalities")
        require(
            isinstance(modalities, dict)
            and set(modalities) == set(MODALITIES),
            f"Row {i}: exactly EEG/Audio/Video descriptors are required",
        )
        for m in MODALITIES:
            item = modalities[m]
            require(
                isinstance(item, dict) and item.get("present") is True,
                f"Row {i}: Stage-01 reviewer trial should contain {m}",
            )
            p = Path(str(item.get("path", ""))).expanduser().resolve()
            require(p.is_file(), f"Row {i}: missing {m} source: {p}")
            require(
                item.get("window_index") == w,
                f"Row {i}: {m} window_index mismatch",
            )

    require(
        sorted(window_indices) == [0, 1, 2, 3],
        f"Expected window indices [0,1,2,3], got {sorted(window_indices)}",
    )
    require(len(subjects) == 1, f"Cross-subject manifest rejected: {subjects}")
    require(len(instances) == 1, f"Cross-instance manifest rejected: {instances}")
    require(len(eeg_indices) == 1, f"Cross-EEG-trial manifest rejected: {eeg_indices}")

    subject = next(iter(subjects))
    instance = next(iter(instances))
    canonical_pair = expected_pair_key(subject, instance)

    require(
        not declared_pair_keys or declared_pair_keys == {canonical_pair},
        (
            "Stage-01 identity.pair_key disagrees with validated subject/media "
            f"instance. Expected {canonical_pair!r}, got {sorted(declared_pair_keys)!r}"
        ),
    )
    require(
        missing_pair_key_rows in (0, WINDOW_COUNT),
        (
            "Mixed Stage-01 pair_key state rejected: either all four windows "
            "must contain the same pair_key or all four must be legacy V2.1 rows."
        ),
    )

    if subject in TEST_SUBJECTS and not allow_test:
        raise Stage2Error(
            f"{subject} belongs to the frozen independent TEST cohort. "
            "ReviewerDemo refuses it by default. Use a non-TEST demo subject "
            "or pass --allow-test explicitly."
        )

    return {
        "rows": rows,
        "subject": subject,
        "instance": instance,
        "pair_key": canonical_pair,
        "pair_key_was_missing": missing_pair_key_rows == WINDOW_COUNT,
        "eeg_trial_index_0based": next(iter(eeg_indices)),
    }



def write_runtime_manifest(
    *,
    original_manifest: Path,
    manifest_info: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """
    Write an immutable Stage-02 runtime copy of the Stage-01 descriptor set.

    Stage-01 V2.1 omitted identity.pair_key. The frozen QualityController
    explicitly requires pair_key + window_idx_0based for an EAV replay that
    has no wall-clock timing/quality_window. We therefore add ONLY this
    deterministic source-identity field, derived from the already validated
    subject + media instance. No signal, probability, quality value, label,
    threshold, model input, or source path is changed.
    """
    rows = copy.deepcopy(list(manifest_info["rows"]))
    pair_key = str(manifest_info["pair_key"])
    modifications: list[dict[str, Any]] = []

    for row in rows:
        identity = row["identity"]
        before = identity.get("pair_key")
        if before is None:
            identity["pair_key"] = pair_key
            modifications.append(
                {
                    "window_id": row["window_id"],
                    "field": "identity.pair_key",
                    "before": None,
                    "after": pair_key,
                    "reason": "FROZEN_QUALITY_CONTROLLER_EAV_REPLAY_IDENTITY_CONTRACT",
                }
            )
        else:
            require(
                before == pair_key,
                f"Runtime pair_key mismatch for {row['window_id']}: {before!r}",
            )

    path = output_dir / "runtime_manifest.jsonl"
    require(not path.exists(), f"Runtime manifest already exists: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )

    audit = {
        "original_manifest": str(original_manifest),
        "runtime_manifest": str(path),
        "original_manifest_sha256": sha256_file(original_manifest),
        "runtime_manifest_sha256": sha256_file(path),
        "pair_key": pair_key,
        "legacy_stage01_pair_key_missing": bool(
            manifest_info.get("pair_key_was_missing")
        ),
        "modifications": modifications,
        "changed_fields": sorted({m["field"] for m in modifications}),
        "raw_source_paths_changed": False,
        "signals_changed": False,
        "reference_label_added": False,
        "model_or_quality_logic_changed": False,
    }
    write_json(output_dir / "runtime_manifest_audit.json", audit)
    return path, audit


def inspect_config(
    config: Path,
    *,
    eeg_unit_override: str | None,
) -> dict[str, Any]:
    cfg = read_json(config)

    require(
        cfg.get("schema") == CONFIG_SCHEMA,
        f"Unexpected config schema: {cfg.get('schema')!r}",
    )
    require(
        cfg.get("class_order") == EMOTIONS,
        "Deployment class order does not match the frozen five-class order.",
    )
    require(
        cfg.get("window_seconds") == 5.0,
        "Deployment config is not a 5-s runtime.",
    )

    eeg = cfg.get("eeg")
    require(isinstance(eeg, dict), "Config missing eeg block")

    effective_unit = eeg_unit_override or eeg.get("training_unit")
    require(
        effective_unit in {"V", "mV", "uV"},
        (
            "Raw EEG replay needs the documented training EEG unit. "
            f"Config currently has training_unit={eeg.get('training_unit')!r}. "
            "Set it in system_config.json or pass --eeg-unit V|mV|uV."
        ),
    )

    quality_layer = cfg.get("quality_layer")
    configured_release = None
    if isinstance(quality_layer, dict):
        configured_release = quality_layer.get("release")

    return {
        "raw_config": cfg,
        "training_unit_effective": effective_unit,
        "input_unit": eeg.get("input_unit"),
        "configured_release": configured_release,
    }


def resolve_maybe_relative(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def resolve_release(
    *,
    deployment_root: Path,
    config: Path,
    config_info: Mapping[str, Any],
    explicit_release: str | None,
) -> tuple[Path, str, dict[str, Any]]:
    """Resolve frozen release by canonical embedded release_sha256."""
    candidates: list[tuple[Path, str]] = []

    if explicit_release:
        candidates.append((Path(explicit_release).expanduser().resolve(), "EXPLICIT_CLI"))
    else:
        cfg_value = config_info.get("configured_release")
        if isinstance(cfg_value, str) and cfg_value.strip():
            candidates.append((resolve_maybe_relative(cfg_value, config.parent), "SYSTEM_CONFIG"))

        candidates.append((
            (deployment_root / "system_checks" / "audio_quality_closeout_v2_1_20260927_145459_043" / "quality_layer_release_candidate.json").resolve(),
            "KNOWN_FROZEN_SYSTEM_CHECK_PATH",
        ))

        root = deployment_root / "system_checks"
        if root.is_dir():
            for p in sorted(root.rglob("quality_layer_release_candidate.json")):
                rp = p.resolve()
                if all(rp != q for q, _ in candidates):
                    candidates.append((rp, "DISCOVERED_SYSTEM_CHECKS"))

    attempted: list[dict[str, Any]] = []
    matches: list[tuple[Path, str, dict[str, Any]]] = []

    for p, source in candidates:
        if not p.is_file():
            attempted.append({"path": str(p), "source": source, "status": "NOT_FOUND"})
            continue
        try:
            info = validate_release_file(p)
            matches.append((p, source, info))
            attempted.append({
                "path": str(p),
                "source": source,
                "status": "MATCH",
                "embedded_release_sha256": info["embedded_release_sha256"],
                "file_byte_sha256": info["file_byte_sha256"],
            })
        except Stage2Error as exc:
            attempted.append({
                "path": str(p),
                "source": source,
                "status": "REJECTED",
                "reason": str(exc),
                "file_byte_sha256": sha256_file(p),
            })
            if explicit_release:
                raise

    unique: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for p, source, info in matches:
        unique[str(p).casefold()] = (p, source, info)
    matches = list(unique.values())

    require(
        len(matches) == 1,
        (
            "Could not uniquely resolve the frozen quality-layer release by its "
            "canonical release fingerprint.\n"
            f"Expected embedded release_sha256: {EXPECTED_RELEASE_SHA256}\n"
            f"Valid matching files: {[str(x[0]) for x in matches]}\n"
            "Use --release with the known frozen file if needed.\n"
            f"Attempted candidates: {attempted}"
        ),
    )

    return matches[0]



def infer_stage1_audit(manifest: Path) -> Path | None:
    p = manifest.parent / "input_audit.json"
    return p if p.is_file() else None


def reference_from_audit(audit: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not audit:
        return None

    selection = audit.get("selection")
    if isinstance(selection, dict):
        pair = selection.get("trial_pair")
        if isinstance(pair, dict) and pair.get("emotion") in EMOTIONS:
            return {
                "emotion": pair["emotion"],
                "source": "Stage-01 media identity audit",
                "used_as_model_input": False,
            }

    source_binding = audit.get("source_binding")
    if isinstance(source_binding, dict):
        media = source_binding.get("media_identity")
        if isinstance(media, dict) and media.get("emotion") in EMOTIONS:
            return {
                "emotion": media["emotion"],
                "source": "Stage-01 media identity audit",
                "used_as_model_input": False,
            }

    return None


def choose_output_dir(
    *,
    script_dir: Path,
    output: str | None,
    subject: str,
    instance: int,
) -> Path:
    if output:
        p = Path(output).expanduser().resolve()
        require(not p.exists(), f"Output already exists: {p}")
        p.mkdir(parents=True)
        return p

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    p = (
        script_dir
        / "runs"
        / f"{subject}_instance{instance:03d}_{stamp}"
    ).resolve()
    require(not p.exists(), f"Generated output already exists: {p}")
    p.mkdir(parents=True)
    return p


def build_main_command(
    *,
    python_exe: Path,
    main_py: Path,
    config: Path,
    release: Path,
    manifest: Path,
    runtime_output: Path,
    fusion_script: Path,
    assets_dir: Path,
    fusion_device: str,
    device: str | None,
    eeg_unit: str | None,
    error_policy: str | None,
    allow_test: bool,
) -> list[str]:
    cmd = [
        str(python_exe),
        str(main_py),
        "--mode", "replay",
        "--config", str(config),
        "--release", str(release),
        "--manifest", str(manifest),
        "--fusion-script", str(fusion_script),
        "--assets-dir", str(assets_dir),
        "--fusion-device", fusion_device,
        "--output", str(runtime_output),
    ]

    if device:
        cmd += ["--device", device]
    if eeg_unit:
        cmd += ["--eeg-unit", eeg_unit]
    if error_policy:
        cmd += ["--error-policy", error_policy]
    if allow_test:
        cmd += ["--allow-test"]

    return cmd


def run_and_tee(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> int:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print()
    print("=" * 112)
    print("LAUNCHING FROZEN FINAL-DEPLOYMENT RUNTIME")
    print("=" * 112)
    print(f"Working directory: {cwd}")
    print("Command:")
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    print("=" * 112)
    print()

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return int(proc.wait())
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise


def load_runtime_outputs(runtime_dir: Path) -> dict[str, Any]:
    require(runtime_dir.is_dir(), f"Runtime output directory missing: {runtime_dir}")

    window_path = runtime_dir / "window_results.jsonl"
    summary_path = runtime_dir / "run_summary.json"
    latest_path = runtime_dir / "latest_state.json"
    trial_path = runtime_dir / "trial_results.jsonl"

    windows = read_jsonl(window_path)
    summary = read_json(summary_path)

    latest = read_json(latest_path) if latest_path.is_file() else None

    # main.py always creates trial_results.jsonl at shutdown, but it writes
    # trial rows only for windows that carry a reference_label. ReviewerDemo
    # deliberately omits the reference emotion from the runtime descriptor, so
    # an empty trial_results.jsonl is a valid and EXPECTED outcome here.
    #
    # Do NOT synthesize a trial prediction in Stage 02. Historical 5-s model
    # outputs remain authoritative in window_results.jsonl.
    trial_file_present = trial_path.is_file()
    trial_file_empty = trial_file_present and trial_path.stat().st_size == 0

    if trial_file_present and not trial_file_empty:
        trials = read_jsonl(trial_path)
    else:
        trials = []

    return {
        "windows": windows,
        "summary": summary,
        "latest_state": latest,
        "trial_results": trials,
        "trial_results_file_present": trial_file_present,
        "trial_results_file_empty": trial_file_empty,
        "trial_results_empty_is_expected_without_reference_label": True,
        "window_results_path": window_path,
        "run_summary_path": summary_path,
        "latest_state_path": latest_path if latest_path.is_file() else None,
        "trial_results_path": trial_path if trial_file_present else None,
    }


def extract_probs_from_report(
    modality: str,
    report: Mapping[str, Any],
) -> list[float] | None:
    # Current main v2 report.
    emotion = report.get("emotion")
    if isinstance(emotion, Mapping):
        for key in ("probabilities", f"{modality}_probs"):
            if key in emotion:
                try:
                    return probability_vector(
                        emotion.get(key),
                        field=f"{modality}.emotion.{key}",
                    )
                except Stage2Error:
                    pass

    # Some wrappers use modality-specific emotion blocks.
    block = report.get(f"{modality}_emotion")
    if isinstance(block, Mapping):
        for key in ("probabilities", f"{modality}_probs"):
            if key in block:
                try:
                    return probability_vector(
                        block.get(key),
                        field=f"{modality}.{key}",
                    )
                except Stage2Error:
                    pass

    # Legacy packet path.
    packet = report.get("packet")
    if isinstance(packet, Mapping) and "probabilities" in packet:
        try:
            return probability_vector(
                packet.get("probabilities"),
                field=f"{modality}.packet.probabilities",
            )
        except Stage2Error:
            pass

    # Flat report compatibility.
    for key in ("probabilities", f"{modality}_probs"):
        if key in report:
            try:
                return probability_vector(
                    report.get(key),
                    field=f"{modality}.{key}",
                )
            except Stage2Error:
                pass

    return None


def extract_window_index(row: Mapping[str, Any]) -> int | None:
    for block_name in ("identity", "source_window"):
        block = row.get(block_name)
        if isinstance(block, Mapping):
            if block_name == "source_window":
                ident = block.get("identity")
                if isinstance(ident, Mapping):
                    value = ident.get("window_idx_0based")
                else:
                    value = block.get("window_idx_0based")
            else:
                value = block.get("window_idx_0based")
            if isinstance(value, int) and not isinstance(value, bool):
                return value

    wid = str(row.get("window_id", ""))
    m = re.search(r"(?:_W|_w)([0-3])(?:\D|$)", wid)
    return int(m.group(1)) if m else None


def compact_window(row: Mapping[str, Any]) -> dict[str, Any]:
    fusion = row.get("fusion")
    require(
        isinstance(fusion, Mapping),
        f"Runtime window lacks fusion block: {row.get('window_id')}",
    )

    final = fusion.get("final")
    require(
        isinstance(final, Mapping),
        f"Runtime fusion lacks final block: {row.get('window_id')}",
    )

    final_probs = probability_vector(
        final.get("probabilities"),
        field=f"{row.get('window_id')}.fusion.final.probabilities",
    )
    final_emotion = final.get("emotion")
    if final_emotion is None and final_probs is not None:
        final_emotion, _ = argmax_emotion(final_probs)
    if final_emotion is None:
        final_emotion = "NO_DECISION"

    route = (
        fusion.get("active_branch")
        or fusion.get("executed_branch")
        or fusion.get("route")
    )
    candidate_route = fusion.get("candidate_route")

    availability = fusion.get("availability")
    if not isinstance(availability, Mapping):
        fi = fusion.get("fusion_input")
        availability = fi.get("available") if isinstance(fi, Mapping) else {}
    if not isinstance(availability, Mapping):
        availability = {}

    quality_for_fusion = fusion.get("quality_for_fusion")
    if not isinstance(quality_for_fusion, Mapping):
        fi = fusion.get("fusion_input")
        quality_for_fusion = fi.get("quality") if isinstance(fi, Mapping) else {}
    if not isinstance(quality_for_fusion, Mapping):
        quality_for_fusion = {}

    reports = row.get("reports")
    if not isinstance(reports, Mapping):
        reports = row.get("heads")
    if not isinstance(reports, Mapping):
        reports = {}

    old_modalities = fusion.get("modalities")
    if not isinstance(old_modalities, Mapping):
        old_modalities = {}

    modalities: dict[str, Any] = {}
    for m in MODALITIES:
        old = old_modalities.get(m)
        old = old if isinstance(old, Mapping) else {}

        report = reports.get(m)
        report = report if isinstance(report, Mapping) else {}

        probs = None
        if old.get("probabilities") is not None:
            try:
                probs = probability_vector(
                    old.get("probabilities"),
                    field=f"{m}.fusion.probabilities",
                )
            except Stage2Error:
                probs = None
        if probs is None:
            probs = extract_probs_from_report(m, report)

        predicted, confidence = argmax_emotion(probs)

        av_raw = availability.get(m)
        if isinstance(av_raw, bool):
            available = av_raw
        elif old.get("available") is not None:
            available = bool(old.get("available"))
        elif report.get("available") is not None:
            available = bool(report.get("available"))
        elif report.get("status") == "OK":
            available = True
        else:
            available = False

        q = quality_for_fusion.get(m)
        if q is None:
            q = old.get("quality")
        q = finite_float(q, field=f"{m}.quality") if q is not None else None

        if old.get("emotion") in EMOTIONS:
            predicted = str(old.get("emotion"))
        if old.get("confidence") is not None:
            confidence = finite_float(
                old.get("confidence"),
                field=f"{m}.confidence",
            )

        modalities[m] = {
            "available": available,
            "quality_for_fusion": q,
            "emotion": predicted if available else "UNAVAILABLE",
            "confidence": confidence if available else None,
            "probabilities": probs,
            "report_status": report.get("status"),
        }

    return {
        "window_index_0based": extract_window_index(row),
        "window_id": row.get("window_id"),
        "status": row.get("status"),
        "selected_variant": row.get("selected_variant"),
        "route": route,
        "candidate_route": candidate_route,
        "no_decision": final_emotion == "NO_DECISION",
        "final": {
            "emotion": final_emotion,
            "confidence": (
                finite_float(
                    final.get("confidence"),
                    field="fusion.final.confidence",
                )
                if final.get("confidence") is not None
                else argmax_emotion(final_probs)[1]
            ),
            "probabilities": final_probs,
        },
        "modalities": modalities,
        "fusion_status": fusion.get("status"),
        "fusion_executed": fusion.get("fusion_executed"),
        "fusion_runtime_identity": fusion.get("identity"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "raw_fusion_diagnostics": fusion.get("diagnostics"),
    }


def create_reviewer_results(
    *,
    manifest: Path,
    runtime_manifest: Path,
    runtime_manifest_audit: Mapping[str, Any],
    manifest_info: Mapping[str, Any],
    selection_mode: str,
    stage1_audit_path: Path | None,
    stage1_audit: Mapping[str, Any] | None,
    main_py: Path,
    config: Path,
    release: Path,
    release_source: str,
    release_info: Mapping[str, Any],
    fusion_script: Path,
    assets_dir: Path,
    command: Sequence[str],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    windows = [compact_window(x) for x in runtime["windows"]]
    windows.sort(
        key=lambda x: (
            99
            if x["window_index_0based"] is None
            else x["window_index_0based"]
        )
    )

    routes = Counter(str(x["route"]) for x in windows)
    final_counts = Counter(str(x["final"]["emotion"]) for x in windows)

    return {
        "schema": "eav.reviewer_demo.results.v1.1",
        "version": VERSION,
        "status": "PASS",
        "created_utc": utc_now(),
        "purpose": (
            "Compact reviewer view of the unchanged final-deployment replay run."
        ),
        "stage1": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "selection_mode": selection_mode,
            "input_audit": (
                str(stage1_audit_path)
                if stage1_audit_path is not None
                else None
            ),
            "subject": manifest_info["subject"],
            "media_instance": manifest_info["instance"],
            "pair_key": manifest_info["pair_key"],
            "eeg_trial_index_0based": manifest_info["eeg_trial_index_0based"],
            "runtime_manifest": str(runtime_manifest),
            "runtime_manifest_sha256": sha256_file(runtime_manifest),
            "runtime_manifest_audit": dict(runtime_manifest_audit),
        },
        "reference": reference_from_audit(stage1_audit),
        "frozen_runtime": {
            "main_py": str(main_py),
            "main_py_sha256": sha256_file(main_py),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "release": str(release),
            "release_sha256": release_info["embedded_release_sha256"],
            "release_canonical_recomputed_sha256": release_info["canonical_recomputed_sha256"],
            "release_file_byte_sha256": release_info["file_byte_sha256"],
            "release_resolution": release_source,
            "fusion_script": str(fusion_script),
            "fusion_script_sha256": sha256_file(fusion_script),
            "assets_dir": str(assets_dir),
            "command": list(command),
            "run_summary": runtime["summary"],
        },
        "windows": windows,
        "authoritative_trial_results_from_main": runtime["trial_results"],
        "authoritative_trial_results_note": (
            "Copied verbatim from main.py trial_results.jsonl when records are present. "
            "For ReviewerDemo, reference_label is intentionally omitted from runtime "
            "descriptors, so main.py may create an empty trial_results.jsonl. "
            "Stage 02 treats that as valid and performs no independent four-window averaging."
        ),
        "trial_results_file_present": runtime.get("trial_results_file_present"),
        "trial_results_file_empty": runtime.get("trial_results_file_empty"),
        "latest_state_after_process_exit": runtime["latest_state"],
        "latest_state_note": (
            "Current main.py may intentionally clear current evidence on process "
            "exit. Historical predictions are preserved in window_results.jsonl "
            "and trial_results.jsonl."
        ),
        "display_summary": {
            "window_count": len(windows),
            "route_counts": dict(routes),
            "window_final_emotion_counts": dict(final_counts),
            "no_decision_windows": sum(
                int(x["no_decision"]) for x in windows
            ),
        },
        "integrity": {
            "new_model_logic_added_by_stage2": False,
            "quality_recomputed_by_stage2": False,
            "fusion_recomputed_by_stage2": False,
            "new_trial_aggregation_added_by_stage2": False,
            "reference_label_passed_to_models": False,
            "release_canonical_fingerprint_verified": True,
            "robot_action_authorized": False,
        },
    }


def write_results_csv(
    path: Path,
    windows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "window_index_0based",
        "window_id",
        "status",
        "route",
        "candidate_route",
        "final_emotion",
        "final_confidence",
        "eeg_available",
        "eeg_quality",
        "eeg_emotion",
        "eeg_confidence",
        "audio_available",
        "audio_quality",
        "audio_emotion",
        "audio_confidence",
        "video_available",
        "video_quality",
        "video_emotion",
        "video_confidence",
        "elapsed_seconds",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for w in windows:
            row: dict[str, Any] = {
                "window_index_0based": w["window_index_0based"],
                "window_id": w["window_id"],
                "status": w["status"],
                "route": w["route"],
                "candidate_route": w["candidate_route"],
                "final_emotion": w["final"]["emotion"],
                "final_confidence": w["final"]["confidence"],
                "elapsed_seconds": w["elapsed_seconds"],
            }
            for m in MODALITIES:
                x = w["modalities"][m]
                row[f"{m}_available"] = x["available"]
                row[f"{m}_quality"] = x["quality_for_fusion"]
                row[f"{m}_emotion"] = x["emotion"]
                row[f"{m}_confidence"] = x["confidence"]
            writer.writerow(row)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return str(value)


def print_reviewer_summary(
    result: Mapping[str, Any],
    output_dir: Path,
) -> None:
    print()
    print("=" * 126)
    print("EAV REVIEWER DEMO — STAGE 02 RESULTS")
    print("=" * 126)

    reference = result.get("reference")
    if isinstance(reference, Mapping):
        print(
            f"Reference emotion : {reference.get('emotion')} "
            "(Stage-01 audit only; NOT model input)"
        )

    print(
        f"Subject / instance: "
        f"{result['stage1']['subject']} / "
        f"{result['stage1']['media_instance']:03d}"
    )
    print(
        f"Release fingerprint: "
        f"{result['frozen_runtime']['release_sha256']}"
    )
    print("-" * 126)
    print(
        f"{'W':<3} {'ROUTE':<10} {'FINAL':<12} {'CONF':>7} | "
        f"{'EEG':<12} {'qE':>7} | "
        f"{'AUDIO':<12} {'qA':>7} | "
        f"{'VIDEO':<12} {'qV':>7}"
    )
    print("-" * 126)

    for w in result["windows"]:
        e = w["modalities"]["eeg"]
        a = w["modalities"]["audio"]
        v = w["modalities"]["video"]
        print(
            f"{str(w['window_index_0based']):<3} "
            f"{str(w['route']):<10} "
            f"{str(w['final']['emotion']):<12} "
            f"{fmt(w['final']['confidence']):>7} | "
            f"{str(e['emotion']):<12} {fmt(e['quality_for_fusion']):>7} | "
            f"{str(a['emotion']):<12} {fmt(a['quality_for_fusion']):>7} | "
            f"{str(v['emotion']):<12} {fmt(v['quality_for_fusion']):>7}"
        )

    print("-" * 126)

    trial_results = result["authoritative_trial_results_from_main"]
    if trial_results:
        print(
            f"Authoritative main.py trial results: {len(trial_results)} record(s) "
            "(copied, not recomputed by Stage 02)"
        )
    else:
        print(
            "Authoritative main.py trial results: 0 records "
            "(expected because ReviewerDemo does not pass reference_label to runtime); "
            "Stage 02 does not invent a 20-s aggregate."
        )

    print(f"Route counts        : {result['display_summary']['route_counts']}")
    print(
        f"No-decision windows : "
        f"{result['display_summary']['no_decision_windows']}"
    )
    print(f"Reviewer JSON       : {output_dir / 'reviewer_results.json'}")
    print(f"Reviewer CSV        : {output_dir / 'reviewer_results.csv'}")
    print(f"Raw runtime outputs : {output_dir / 'runtime_run'}")
    print("=" * 126)



def collect_runtime_failure_diagnostics(runtime_dir: Path) -> dict[str, Any]:
    """Extract the original first-window failure instead of only reporting exit code 2."""
    out: dict[str, Any] = {}

    fatal = runtime_dir / "fatal_error.json"
    if fatal.is_file():
        try:
            out["fatal_error"] = read_json(fatal)
        except Exception as exc:
            out["fatal_error_read_error"] = repr(exc)

    window_file = runtime_dir / "window_results.jsonl"
    if window_file.is_file():
        try:
            rows = read_jsonl(window_file)
            out["window_records"] = len(rows)
            if rows:
                row = rows[-1]
                out["last_window_id"] = row.get("window_id")
                out["last_window_status"] = row.get("status")

                quality = row.get("quality_result")
                if isinstance(quality, Mapping):
                    out["quality_status"] = quality.get("status")
                    out["quality_errors"] = quality.get("errors")
                    out["candidate_route"] = quality.get("candidate_route")

                fusion = row.get("fusion")
                if isinstance(fusion, Mapping):
                    out["fusion_status"] = fusion.get("status")
                    out["fusion_errors"] = fusion.get("errors")
                    out["active_branch"] = fusion.get("active_branch")

                reports = row.get("reports")
                if isinstance(reports, Mapping):
                    out["producer_reports"] = {
                        m: {
                            "status": reports.get(m, {}).get("status")
                            if isinstance(reports.get(m), Mapping)
                            else None,
                            "reason": reports.get(m, {}).get("reason")
                            if isinstance(reports.get(m), Mapping)
                            else None,
                            "error": reports.get(m, {}).get("error")
                            if isinstance(reports.get(m), Mapping)
                            else None,
                            "algorithm_error": reports.get(m, {}).get("algorithm_error")
                            if isinstance(reports.get(m), Mapping)
                            else None,
                        }
                        for m in MODALITIES
                    }
        except Exception as exc:
            out["window_results_read_error"] = repr(exc)

    summary = runtime_dir / "run_summary.json"
    if summary.is_file():
        try:
            out["run_summary"] = read_json(summary)
        except Exception as exc:
            out["run_summary_read_error"] = repr(exc)

    return out


def print_runtime_failure_diagnostics(diag: Mapping[str, Any]) -> None:
    print()
    print("-" * 112, file=sys.stderr)
    print("AUTHORITATIVE RUNTIME FAILURE DIAGNOSTIC", file=sys.stderr)
    print("-" * 112, file=sys.stderr)
    if diag.get("last_window_id"):
        print(f"Window         : {diag.get('last_window_id')}", file=sys.stderr)
    if diag.get("quality_status"):
        print(f"Quality status : {diag.get('quality_status')}", file=sys.stderr)
    if diag.get("quality_errors"):
        print(
            "Quality errors : "
            + json.dumps(diag.get("quality_errors"), ensure_ascii=False),
            file=sys.stderr,
        )
    if diag.get("fusion_status"):
        print(f"Fusion status  : {diag.get('fusion_status')}", file=sys.stderr)
    if diag.get("fusion_errors"):
        print(
            "Fusion errors  : "
            + json.dumps(diag.get("fusion_errors"), ensure_ascii=False),
            file=sys.stderr,
        )
    reports = diag.get("producer_reports")
    if isinstance(reports, Mapping):
        for m in MODALITIES:
            print(f"{m.upper():<7}        : {reports.get(m)}", file=sys.stderr)
    fatal = diag.get("fatal_error")
    if isinstance(fatal, Mapping):
        print(f"Fatal error     : {fatal.get('error')}", file=sys.stderr)
    print("-" * 112, file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "ReviewerDemo Stage 02: run Stage-01 raw EAV replay input through "
            "the current frozen final-deployment main.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--prepared", default=None)
    p.add_argument("--manifest", default=None)

    p.add_argument(
        "--config",
        default=str(deployment_root / "system_config.json"),
    )
    p.add_argument(
        "--main",
        default=str(deployment_root / "main.py"),
    )
    p.add_argument(
        "--release",
        default=None,
        help=(
            "Frozen quality_layer_release_candidate.json. If omitted, Stage 02 "
            "uses config quality_layer.release or searches system_checks by the "
            "frozen SHA256 pin."
        ),
    )
    p.add_argument(
        "--fusion-script",
        default=str(deployment_root / DEFAULT_FUSION_SCRIPT_RELATIVE),
    )
    p.add_argument(
        "--assets-dir",
        default=str(deployment_root / DEFAULT_FUSION_ASSETS_RELATIVE),
    )
    p.add_argument(
        "--fusion-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch main.py",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.add_argument("--eeg-unit", choices=("V", "mV", "uV"), default=None)
    p.add_argument("--error-policy", choices=("raise", "exclude"), default=None)
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--list-prepared", action="store_true")
    p.add_argument("--version", action="version", version=VERSION)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    deployment_root = script_dir.parent

    if args.list_prepared:
        items = discover_prepared(script_dir)
        print("=" * 100)
        print("REVIEWER DEMO PREPARED TRIALS")
        print("=" * 100)
        if not items:
            print("No Stage-01 prepared trials found.")
        else:
            for i, p in enumerate(items, 1):
                print(f"{i:02d}. {p}")
        print("=" * 100)
        return 0

    main_py = Path(args.main).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    python_exe = Path(args.python).expanduser().resolve()
    fusion_script = Path(args.fusion_script).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()

    require(main_py.is_file(), f"main.py not found: {main_py}")
    require(config.is_file(), f"system_config.json not found: {config}")
    require(python_exe.is_file(), f"Python interpreter not found: {python_exe}")
    require(
        fusion_script.is_file(),
        f"Frozen fusion script not found: {fusion_script}",
    )
    require(
        assets_dir.is_dir(),
        f"Fusion assets directory not found: {assets_dir}",
    )

    manifest, selection_mode = resolve_manifest(
        script_dir=script_dir,
        prepared=args.prepared,
        manifest=args.manifest,
    )

    print("=" * 112)
    print("EAV REVIEWER DEMO — STAGE 02: FROZEN RUNTIME INFERENCE")
    print("=" * 112)
    print(f"Version           : {VERSION}")
    print(f"main.py           : {main_py}")
    print(f"system_config     : {config}")
    print(f"Stage-01 manifest : {manifest}")
    print(f"Selection mode    : {selection_mode}")
    print()

    print("[1/6] Validating Stage-01 prepared trial...")
    manifest_info = validate_stage1_manifest(
        manifest,
        allow_test=args.allow_test,
    )
    print(
        f"      PASS: {manifest_info['subject']} / "
        f"instance {manifest_info['instance']:03d} / "
        f"EEG index {manifest_info['eeg_trial_index_0based']} / 4 x 5 s"
    )
    print(
        f"      Canonical replay pair_key: {manifest_info['pair_key']} "
        f"({'compatibility insertion required' if manifest_info['pair_key_was_missing'] else 'already present'})"
    )

    print("[2/6] Validating deployment configuration...")
    config_info = inspect_config(
        config,
        eeg_unit_override=args.eeg_unit,
    )
    print(
        f"      PASS: effective EEG training unit="
        f"{config_info['training_unit_effective']}"
    )

    print("[3/6] Resolving frozen quality-layer release...")
    release, release_source, release_info = resolve_release(
        deployment_root=deployment_root,
        config=config,
        config_info=config_info,
        explicit_release=args.release,
    )
    print(f"      PASS: {release}")
    print(f"      Source                  : {release_source}")
    print(f"      Embedded release_sha256: {release_info['embedded_release_sha256']}")
    print(f"      Canonical recomputed    : {release_info['canonical_recomputed_sha256']}")
    print(f"      File-byte SHA256        : {release_info['file_byte_sha256']}")

    stage1_audit_path = infer_stage1_audit(manifest)
    stage1_audit = (
        read_json(stage1_audit_path)
        if stage1_audit_path is not None
        else None
    )
    reference = reference_from_audit(stage1_audit)
    if reference:
        print(
            f"      Stage-01 reference emotion: {reference['emotion']} "
            "(audit only; NOT model input)"
        )

    output_dir = choose_output_dir(
        script_dir=script_dir,
        output=args.output,
        subject=manifest_info["subject"],
        instance=manifest_info["instance"],
    )
    runtime_manifest, runtime_manifest_audit = write_runtime_manifest(
        original_manifest=manifest,
        manifest_info=manifest_info,
        output_dir=output_dir,
    )
    if runtime_manifest_audit["legacy_stage01_pair_key_missing"]:
        print(
            "      Compatibility fix: Stage-01 V2.1 omitted identity.pair_key; "
            f"runtime copy adds {manifest_info['pair_key']!r} to all four windows."
        )
    else:
        print(
            f"      Replay pair_key verified: {manifest_info['pair_key']}"
        )

    runtime_dir = output_dir / "runtime_run"
    console_log = output_dir / "main_console.log"

    print("[4/6] Preparing frozen main.py command...")
    cmd = build_main_command(
        python_exe=python_exe,
        main_py=main_py,
        config=config,
        release=release,
        manifest=runtime_manifest,
        runtime_output=runtime_dir,
        fusion_script=fusion_script,
        assets_dir=assets_dir,
        fusion_device=args.fusion_device,
        device=args.device,
        eeg_unit=args.eeg_unit,
        error_policy=args.error_policy,
        allow_test=args.allow_test,
    )

    print("[5/6] Running final-deployment main.py...")
    rc = run_and_tee(
        cmd,
        cwd=deployment_root,
        log_path=console_log,
    )

    if rc != 0:
        diagnostics = collect_runtime_failure_diagnostics(runtime_dir)
        print_runtime_failure_diagnostics(diagnostics)
        failure: dict[str, Any] = {
            "schema": "eav.reviewer_demo.results.failure.v1",
            "version": VERSION,
            "status": "FAILED_RUNTIME",
            "created_utc": utc_now(),
            "return_code": rc,
            "manifest": str(manifest),
            "main_py": str(main_py),
            "release": str(release),
            "release_sha256": release_info["embedded_release_sha256"],
            "release_file_byte_sha256": release_info["file_byte_sha256"],
            "runtime_output": str(runtime_dir),
            "console_log": str(console_log),
            "runtime_manifest": str(runtime_manifest),
            "runtime_manifest_audit": runtime_manifest_audit,
            "diagnostics": diagnostics,
            "note": (
                "Frozen main.py returned non-zero. Stage 02 did not substitute "
                "another release, fabricate data, or bypass the runtime."
            ),
        }
        if (runtime_dir / "run_summary.json").is_file():
            failure["run_summary"] = read_json(
                runtime_dir / "run_summary.json"
            )
        if (runtime_dir / "fatal_error.json").is_file():
            failure["fatal_error"] = read_json(
                runtime_dir / "fatal_error.json"
            )
        write_json(output_dir / "reviewer_results.json", failure)

        raise Stage2Error(
            f"Frozen main.py exited with code {rc}. "
            f"See {console_log}"
        )

    print("[6/6] Reading authoritative runtime results...")
    runtime = load_runtime_outputs(runtime_dir)
    summary = runtime["summary"]

    require(
        summary.get("status") == "PASS",
        f"main.py returned zero but run_summary status={summary.get('status')!r}",
    )

    n_processed = summary.get("n_processed")
    if n_processed is None:
        n_processed = summary.get("windows_completed")

    require(
        n_processed == WINDOW_COUNT,
        f"Runtime processed {n_processed!r} windows; expected {WINDOW_COUNT}",
    )
    require(
        len(runtime["windows"]) == WINDOW_COUNT,
        (
            f"window_results.jsonl contains {len(runtime['windows'])} records; "
            f"expected {WINDOW_COUNT}"
        ),
    )

    require(
        summary.get("training_performed") is False,
        "Runtime unexpectedly reports training_performed != false",
    )
    require(
        summary.get("robot_actions_performed") is False,
        "Runtime unexpectedly reports robot_actions_performed != false",
    )

    if summary.get("release_sha256") is not None:
        require(
            summary.get("release_sha256") == EXPECTED_RELEASE_SHA256,
            (
                "Runtime summary release SHA256 differs from the frozen release "
                f"pin: {summary.get('release_sha256')}"
            ),
        )

    result = create_reviewer_results(
        manifest=manifest,
        runtime_manifest=runtime_manifest,
        runtime_manifest_audit=runtime_manifest_audit,
        manifest_info=manifest_info,
        selection_mode=selection_mode,
        stage1_audit_path=stage1_audit_path,
        stage1_audit=stage1_audit,
        main_py=main_py,
        config=config,
        release=release,
        release_source=release_source,
        release_info=release_info,
        fusion_script=fusion_script,
        assets_dir=assets_dir,
        command=cmd,
        runtime=runtime,
    )

    results_json = output_dir / "reviewer_results.json"
    results_csv = output_dir / "reviewer_results.csv"

    write_json(results_json, result)
    write_results_csv(results_csv, result["windows"])

    print_reviewer_summary(result, output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSTAGE 02 INTERRUPTED", file=sys.stderr)
        raise SystemExit(130)
    except Stage2Error as exc:
        print()
        print("=" * 112, file=sys.stderr)
        print("STAGE 02 FAILED", file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 112, file=sys.stderr)
        raise SystemExit(2)
