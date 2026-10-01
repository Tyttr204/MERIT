#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""EAV quality-to-fusion adapter, candidate V1.1 (Python >= 3.10).

V1.1 compatibility fix: use deterministic structural AST fingerprints instead
of interpreter-dependent ast.dump() defaults. No scoring/mapping/network/weight
changes. --check-fusion-api diagnoses source mismatches without importing code.

Public API
----------
    adapter = QualityFusionAdapter(params_path="Quality/distribution_quality_params.json")
    result = adapter.predict_window(quality_result=qc_result,
                                    emotion_reports=reports, mode="replay")

This is a NEW candidate policy, not the original frozen AF4-C router.
    available: q_new = max(1e-6, exp(-0.5 * (B/T)**2))
    unavailable: q_new = 0; mask = 0; probabilities cleared to [0.2]*5
The mapping is an explicit engineering hypothesis, NOT fitted/calibrated to
emotion correctness and NOT known to match AF4-B's training quality scale.
B > T decides the branch in float64. The old q<0.80 router is NEVER called.
HEALTHY_FUSION -> original F4 core, ROBUST_FUSION -> original AF4-B core.
AF4-B contains its own frozen F4 residual: invoking AF4-B legitimately computes
that residual, but does not invoke the old router or the separate F4 branch.

Required siblings (unchanged): quality_controller.py, distribution_quality.py,
distribution_quality_params.json. Actual inference also needs the user's existing
fusion/fusion_af4c_deployment.py and the two reviewed .pt files, or an explicit
--system-config / --fusion-script / checkpoint pair. No checkpoint download,
unsafe torch.load fallback, random-weight fallback, sensor I/O, training,
parameter fitting, label usage, old q usage, or robot actuation occurs.

No-weight tests and real-model tests are SEPARATE:
    python Quality/quality_fusion_adapter.py --preflight-only
    python Quality/quality_fusion_adapter.py --preflight
    python Quality/quality_fusion_adapter.py --write-example example.json
    python Quality/quality_fusion_adapter.py --input-json example.json --output out.json
--prepare-only validates/joins/maps but deliberately does not load fusion models.
Examples contain SYNTHETIC probabilities, never actual classifier predictions.

Input forms: (1) native emotion packets, schema eav.quality_fusion.emotion_packet.v1;
(2) full native quality-controller reports with an explicit emotion class_order;
(3) full reviewed main.py reports. Only the last form consumes packet identity,
probabilities and availability; it NEVER reads packet.quality or old q fields.
No bare probability arrays may bypass the same-window/source-binding checks.

Replay has no current-clock freshness guarantee. Shadow is same-host diagnostic
inference ONLY, checks a trusted callable clock before/after inference, and
withholds expired outputs instead of silently changing availability/routing.
Caller owns synchronization, truthful source metadata and publication dedup.
Parameter/hash checks prove consistency, NOT authentic sensor provenance.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PureWindowsPath
import re
import struct
import sys
import threading
import time
from typing import Any, Callable

VERSION = "EAV-QUALITY-FUSION-ADAPTER.1.1"
INPUT_SCHEMA = "eav.quality_fusion_adapter.input.v1"
OUTPUT_SCHEMA = "eav.quality_fusion_adapter.result.v1"
BATCH_SCHEMA = "eav.quality_fusion_adapter.batch.v1"
EMOTION_SCHEMA = "eav.quality_fusion.emotion_packet.v1"
POLICY_SCHEMA = "eav.quality_fusion_adapter.policy.v1"
CONTROLLER_SCHEMA = "eav.quality_controller.result.v1"
NATIVE_REPORT_SCHEMA = "eav.quality_controller.report.v1"
MAIN_PACKET_SCHEMA = "eav.af4c.modality.packet.v1"
MODALITIES = ("eeg", "audio", "video")
EMOTIONS = ("Neutral", "Sadness", "Anger", "Happiness", "Calmness")
CONTROLLER_VERSION = "EAV-QUALITY-CONTROLLER.1.0"
CONTROLLER_SHA256 = "7feead21763a4f0b7e3c05ae1fda3d1d7bca200a97d67208ba23dcd6dac28e72"
FUSION_VERSION = "FUSION-AF4C-DEPLOYMENT.1.0"
FUSION_CORE_AST_FORMAT = "eav.fusion_core.structural_ast.v1"
# V1.0 used Python 3.13 ast.dump defaults: incompatible with Python 3.11.
# These pins describe the SAME reviewed core with a stable serializer.
LEGACY_FUSION_CORE_AST_SHA256 = 'ddbe867ff5c75449892605f1138b56826ecc3eb65ce4f573dfb44115abd6dfa5'
FUSION_CORE_AST_SHA256 = '456daaf543201c7411340c6cd1d4002b69dd2fbae3a6610de673bf1e09874fe8'
FUSION_CORE_NODE_SHA256 = {
    "AdaptiveF4B": "0ad3f6fa5216b36429068305c1a33f197fc6f6698b640ce5b829c8fac62297c7",
    "FrozenF4Core": "6ac45cf19ef17f7e0f17c062398a96c456bafffbc637929033c7216f0de0d269",
    "canonical_arrays": "768ee932d981dd674565b8f130a4e40e9013ea5361f95fb8992455e94de144cf",
    "choose_device": "f70f1d1db3fd9332c966b53dbc77d7ff940884c31d478758baca3be0e1c1344f",
    "ensure_probs": "fc57c1b6344fc73deacd774b2d8b59c9a5e5be0cea669c3d8352582f10c79d93",
    "finite_number": "e4f696efe6def6364d676a9864dd727f943bc56204968ebdc8e1a71fbd1e8959",
    "infer_af4b": "a1496a16796a05d8050e6a1a46b6c5a8d5387a84153b580af04e79bd9602987c",
    "infer_f4_core": "97b77a0b85caebd81ceb161068e0de801f41c7f415e9e04ebc736acef480c45c",
    "load_pair": "2461e61728570459daf83d3ea9edb5eb144120b2532250fd3cc4d5107b5829a1",
    "require": "8d902abd15278c338af436bbf99182979f065857810184c3b8fd990ff88ba216",
    "resolve_assets": "dbf46dd70d37ac133643652226a5f885a632fded9b45ef18eb5f5564e2655066",
    "restricted_checkpoint": "e0eb231c3f99f203d10388d1b3b169f463e728ee2a33194d47cddb8055cd49c1",
    "state_digest": "7cb5dad058d1859f1853038179c1577e6f45929b990a32964c776b46cd2a0d07",
    "strict_bool": "886bfa5bd4d6cd8e63f81b9b7e0f5b688afd1e8b2f379cdb2a816ab4461058c7",
    "validated_probability": "430cca99ead32ce7e58f5b7ab6a6b77e2ce873f568fa0d0e5d8cf53f467bf789"
}
FUSION_CORE_NAMES = ("FrozenF4Core", "AdaptiveF4B", "infer_f4_core", "infer_af4b", "load_pair",
    "restricted_checkpoint", "canonical_arrays", "state_digest", "choose_device", "resolve_assets",
    "ensure_probs", "validated_probability", "require", "finite_number", "strict_bool")
CHECKPOINT_HASHES = {
    "f4": "3baee8225f871591f1f28da39717553c8d7a2fcc879dcd6de9297d485629d3d1",
    "af4b": "de1056dd1a2e31778443a47cc8229911ce573f5be3d63886d10d22382dfb088b",
}
F4_NAME = "best_validation_selected.pt"
AF4B_NAME = "best_validation_robustness.pt"
PROBABILITY_ATOL = 1e-5
MAX_JSON_BYTES = 64 * 1024 * 1024
_IMPORT_LOCK = threading.RLock()


class AdapterContractError(ValueError):
    """Malformed or contradictory input; never an automatic missing modality."""


class AdapterAssetError(RuntimeError):
    """Missing/incompatible dependencies or frozen model assets."""


class AdapterInferenceError(RuntimeError):
    """Model failure; do not substitute a successful result."""


def need(ok: Any, message: str, kind: type[Exception] = AdapterContractError) -> None:
    if not ok:
        raise kind(message)


def obj(value: Any, name: str) -> Mapping:
    need(isinstance(value, Mapping), f"{name}: mapping required")
    return value


def text(value: Any, name: str) -> str:
    need(isinstance(value, str) and value.strip() == value and 0 < len(value) <= 2048,
         f"{name}: nonempty trimmed string required")
    return value


def real(value: Any, name: str) -> float:
    need(isinstance(value, Real) and not isinstance(value, bool), f"{name}: real scalar required, not string/bool")
    try:
        x = float(value)
    except (ValueError, OverflowError, TypeError) as exc:
        raise AdapterContractError(f"{name}: not representable as float64") from exc
    need(math.isfinite(x), f"{name}: NaN/Inf/overflow forbidden")
    return x


def boolean(value: Any, name: str) -> bool:
    # The existing main.py boundary admits integer 0/1, but no numeric strings.
    if type(value) is bool:
        return value
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(value)
    raise AdapterContractError(f"{name}: bool or integer 0/1 required")


def sha(value: Any, name: str) -> str:
    need(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
         f"{name}: SHA256 hex required")
    return value.lower()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        need(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise AdapterContractError(f"Nonstandard JSON constant: {value}")


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser()
    need(p.is_file(), f"JSON not found: {p}")
    need(p.stat().st_size <= MAX_JSON_BYTES, "JSON exceeds 64 MiB")
    raw = p.read_bytes()
    need(len(raw) <= MAX_JSON_BYTES, "JSON grew beyond 64 MiB")
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs, parse_constant=_bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AdapterContractError(f"Invalid JSON: {p}: {exc}") from exc


def write_json(path: str | Path, value: Any) -> None:
    # Serialize BEFORE creating the output, then exclusive-create. Never overwrite.
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)


def _canonical_ast_value(value: Any) -> Any:
    """Stable structural representation; never use ast.dump() display defaults.

    Preserve nodes, field names, operators, literal types, order, empty lists
    and None. Ignore location attributes, as the original checker did. The only
    schema normalization is an EMPTY type_params field added by Python 3.12;
    nonempty type parameters remain in the fingerprint and will be rejected.
    Parsed Python 3.10-3.13 trees of this reviewed core have the same remaining
    fields. Unknown/new fields are preserved, not silently accepted.
    """
    if isinstance(value, ast.AST):
        fields = []
        for name, item in sorted(ast.iter_fields(value), key=lambda pair: pair[0]):
            if (name == "type_params" and isinstance(item, list) and not item
                    and isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))):
                continue
            fields.append([name, _canonical_ast_value(item)])
        return ["node", type(value).__name__, fields]
    if isinstance(value, list):
        return ["list", [_canonical_ast_value(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_ast_value(item) for item in value]]
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    raise AdapterAssetError(f"Unsupported AST value: {type(value).__name__}")


def _fusion_core_structure(data: bytes, *, filename: str) -> dict:
    """Read-only, no import/exec, no Torch, no checkpoint or sensor access."""
    try:
        tree = ast.parse(data, filename=filename)
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise AdapterAssetError(f"Cannot parse fusion source {filename}: {exc}") from exc
    found = {name: [] for name in FUSION_CORE_NAMES}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in found:
            found[node.name].append(node)
    missing = sorted(name for name, seq in found.items() if not seq)
    duplicates = sorted(name for name, seq in found.items() if len(seq) > 1)
    structures = {name: _canonical_ast_value(seq[0]) for name, seq in found.items() if len(seq) == 1}
    hashes = {name: fingerprint(structure) for name, structure in structures.items()}
    mismatch = sorted(name for name in structures if hashes[name] != FUSION_CORE_NODE_SHA256[name])
    combined = fingerprint([[name, structures[name]] for name in sorted(structures)])
    ok = not missing and not duplicates and not mismatch and combined == FUSION_CORE_AST_SHA256
    return {"status": "PASS" if ok else "FUSION_CORE_MISMATCH",
            "adapter_version": VERSION, "python_version": sys.version.split()[0],
            "fusion_script": filename, "fusion_script_sha256": hashlib.sha256(data).hexdigest(),
            "serializer": FUSION_CORE_AST_FORMAT,
            "expected_core_sha256": FUSION_CORE_AST_SHA256,
            "actual_core_sha256": combined,
            "missing_symbols": missing, "duplicate_symbols": duplicates,
            "mismatched_symbols": mismatch,
            "actual_symbol_sha256": hashes,
            "expected_symbol_sha256": dict(FUSION_CORE_NODE_SHA256),
            "fusion_module_imported": False, "real_checkpoint_loaded": False,
            "fusion_executed": False,
            "scope": "Reviewed core syntax only; not full-module authentication or model preflight"}


def check_fusion_api(path: str | Path) -> dict:
    """Standalone code-identity diagnostic. Does not run the fusion module."""
    p = Path(path).expanduser().resolve()
    need(p.is_file(), f"Python module not found: {p}", AdapterAssetError)
    return _fusion_core_structure(p.read_bytes(), filename=str(p))


def _load_module(path: Path, expected_sha256: str | None = None, *, fusion: bool = False) -> Any:
    need(path.is_file(), f"Python module not found: {path}", AdapterAssetError)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None:
        need(digest == sha(expected_sha256, "expected module SHA256"),
             f"Module SHA256 mismatch: {path}; expected {expected_sha256}, got {digest}", AdapterAssetError)
    if fusion:
        audit = _fusion_core_structure(data, filename=str(path))
        need(audit["status"] == "PASS",
             "Fusion core differs from reviewed deployment API. "
             f"Python={audit['python_version']}; serializer={FUSION_CORE_AST_FORMAT}; "
             f"expected={FUSION_CORE_AST_SHA256}; actual={audit['actual_core_sha256']}; "
             f"missing={audit['missing_symbols']}; duplicate={audit['duplicate_symbols']}; "
             f"mismatched={audit['mismatched_symbols']}. "
             "Run --check-fusion-api --fusion-script <path> --output <new-report.json> "
             "for details. Do not disable checks or accept an unreviewed core.",
             AdapterAssetError)
    name = "_eav_qfa_" + digest + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    with _IMPORT_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            need(spec is not None and spec.loader is not None, "Cannot create module spec", AdapterAssetError)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                exec(compile(data, str(path), "exec"), module.__dict__)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        return sys.modules[name]


def _local_path(value: str | Path, base: Path) -> Path:
    s = str(value).replace("\\", "/")
    need(not (os.name != "nt" and PureWindowsPath(s).is_absolute()),
         f"Windows path on non-Windows host needs explicit local paths: {value}", AdapterAssetError)
    p = Path(s).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


@dataclass(frozen=True)
class CandidateQualityMapping:
    """Fixed, uncalibrated candidate. It is not the old q and not a probability."""
    name: str = "normalized_gaussian_floor_v1"
    floor: float = 1e-6

    def __post_init__(self) -> None:
        need(self.name == "normalized_gaussian_floor_v1" and type(self.floor) is float and self.floor == 1e-6,
             "V1 mapping is fixed; define/version/validate a different policy rather than silently retuning")

    def apply(self, B: Any, T: Any, available: bool) -> dict:
        need(type(available) is bool, "Mapping availability must be bool")
        t = real(T, "T")
        need(t > 0, "T must be strictly positive even for unavailable modalities")
        if not available:
            return {"available": False, "B": None, "T": t, "B_over_T": None,
                    "q_new": 0., "q_model_float32": 0., "floor_applied": False,
                    "reason": "EXTERNALLY_UNAVAILABLE"}
        b, t = real(B, "B"), real(T, "T")
        need(b >= 0 and t > 0, "B must be nonnegative and T strictly positive")
        r = b / t
        need(math.isfinite(r), "B/T overflow")
        # Log comparison avoids unnecessary exp underflow for extreme scores.
        at_floor = r >= math.sqrt(-2. * math.log(self.floor))
        q = self.floor if at_floor else math.exp(-.5 * r * r)
        return {"available": True, "B": b, "T": t, "B_over_T": r,
                "q_new": q, "q_model_float32": fp32(q), "floor_applied": at_floor,
                "reason": "CANDIDATE_NORMALIZED_ANOMALY_MAPPING"}

    def description(self) -> dict:
        return {"name": self.name, "formula_available": "max(1e-6, exp(-0.5*(B/T)**2))",
                "formula_unavailable": "0 with mask=0", "floor": self.floor,
                "q_at_B_zero": 1., "q_at_B_equals_T": math.exp(-.5),
                "floor_at_B_over_T": math.sqrt(-2. * math.log(self.floor)),
                "parameter_origin": "engineering_candidate_not_fitted",
                "monotonicity": "nonincreasing; floor causes saturation",
                "quality_is_correctness_probability": False,
                "quality_is_total_fusion_contribution": False,
                "AF4B_training_scale_compatibility_validated": False,
                "cross_modality_reliability_equivalence_validated": False,
                "branch_rule": "validated availability and strict float64 B>T, NOT q<0.80",
                "production_selected": False}


class FrozenBranchBackend:
    """Reuse only original core/loader functions; never call the original router.

    Only imports the fusion module, not main.py and not the seven upstream models.
    Without a supplied script byte hash, a reviewed numeric-core AST is checked;
    the remaining Python module must still be locally trusted. This is NOT a
    security sandbox. Model checkpoints always use the original fixed byte pins.
    """
    def __init__(self, *, root: str | Path | None = None,
                 system_config: str | Path | None = None,
                 fusion_script: str | Path | None = None,
                 fusion_script_sha256: str | None = None,
                 assets_dir: str | Path | None = None,
                 f4_checkpoint: str | Path | None = None,
                 af4b_checkpoint: str | Path | None = None,
                 manifest: str | Path | None = None, device: str | None = None) -> None:
        base = Path(root).resolve() if root else Path(__file__).resolve().parent.parent
        cfg_hash = None
        if system_config is not None:
            need(all(x is None for x in (fusion_script, assets_dir, f4_checkpoint, af4b_checkpoint, manifest)),
                 "Use system_config OR explicit model paths, not both", AdapterAssetError)
            cp = _local_path(system_config, Path.cwd())
            c = obj(read_json(cp), "system_config")
            need(c.get("schema") == "eav.system.config.v1", "Unsupported system config schema", AdapterAssetError)
            need(c.get("class_order") == list(EMOTIONS) and real(c.get("window_seconds"), "window_seconds") == 5.,
                 "Config class order or window mismatch", AdapterAssetError)
            spec = obj(obj(c.get("modules"), "modules").get("fusion"), "modules.fusion")
            k = obj(spec.get("kwargs"), "fusion.kwargs")
            base = cp.parent
            fusion_script = _local_path(text(spec.get("script"), "fusion.script"), base)
            pin = spec.get("sha256")
            if pin is not None:
                need(fusion_script_sha256 is None or fusion_script_sha256 == pin, "Conflicting fusion script pins")
                fusion_script_sha256 = pin
            for key in ("tau", "batch_size", "decision_window_seconds"):
                if key in k and key != "batch_size":
                    need(real(k[key], key) == (.80 if key == "tau" else 5.), "Unexpected original fusion config")
            assets_dir = _local_path(k["assets_dir"], base) if k.get("assets_dir") else None
            f4_checkpoint = _local_path(k["f4_checkpoint"], base) if k.get("f4_checkpoint") else None
            af4b_checkpoint = _local_path(k["af4b_checkpoint"], base) if k.get("af4b_checkpoint") else None
            manifest = _local_path(k["manifest"], base) if k.get("manifest") else None
            device = device or k.get("device", "cpu")
            cfg_hash = file_sha(cp)
        else:
            fusion_script = _local_path(fusion_script, Path.cwd()) if fusion_script else base / "fusion" / "fusion_af4c_deployment.py"
            assets_dir = _local_path(assets_dir, Path.cwd()) if assets_dir else None
            f4_checkpoint = _local_path(f4_checkpoint, Path.cwd()) if f4_checkpoint else None
            af4b_checkpoint = _local_path(af4b_checkpoint, Path.cwd()) if af4b_checkpoint else None
            manifest = _local_path(manifest, Path.cwd()) if manifest else None
        sp = Path(fusion_script).resolve()
        self.fm = _load_module(sp, fusion_script_sha256, fusion=True)
        fm = self.fm
        need(fm.VERSION == FUSION_VERSION and tuple(fm.EMOTIONS) == EMOTIONS
             and tuple(fm.MODALITIES) == MODALITIES and fm.REVIEWED_HASHES == CHECKPOINT_HASHES
             and fm.NUM_CLASSES == 5 and fm.NUM_MODALITIES == 3 and fm.INPUT_DIM == 15
             and fm.F4_NAME == F4_NAME and fm.AF4B_NAME == AF4B_NAME
             and fm.EXPECTED_AF4B_CANDIDATE == "af4b_qbeta_200",
             "Fusion module protocol/constants mismatch", AdapterAssetError)
        try:
            if f4_checkpoint is None and af4b_checkpoint is None and assets_dir is None:
                assets_dir = sp.parent
            f4_path, af4b_path = fm.resolve_assets(assets_dir, f4_checkpoint, af4b_checkpoint)
            self.device = fm.choose_device(device or "cpu")
            self.f4, self.af4b, _, _, identity = fm.load_pair(f4_path, af4b_path, self.device)
            candidate = Path(manifest) if manifest else f4_path.parent / fm.MANIFEST_NAME
            manifest_used = manifest is not None or (f4_path.parent == af4b_path.parent and candidate.is_file())
            if manifest_used:
                fm.validate_manifest(fm.read_json(candidate), identity)
        except Exception as exc:
            raise AdapterAssetError(f"Frozen fusion assets could not be loaded: {type(exc).__name__}: {exc}") from exc
        self._lock = threading.RLock()
        self._state = (fm.state_digest(self.f4), fm.state_digest(self.af4b))
        self._identity = {**identity, "backend_kind": "REVIEWED_FROZEN_CORE",
            "real_checkpoint_loaded": True, "fusion_script": str(sp),
            "fusion_script_byte_pin_verified": fusion_script_sha256 is not None,
            "numeric_core_AST_pin_verified": True, "numeric_core_AST_sha256": FUSION_CORE_AST_SHA256,
            "numeric_core_AST_format": FUSION_CORE_AST_FORMAT,
            "system_config_sha256": cfg_hash, "manifest_verified": manifest_used,
            "old_router_executed": False, "old_quality_calibrators_loaded": False,
            "old_q_used": False, "new_quality_mapping_compatibility_proven": False,
            "source_text_warning": "Local Python source is trusted code, not sandboxed by AST/hash checks."}
        self._calls = {"F4": 0, "AF4-B": 0}
        self.check_assets()

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)

    @property
    def calls(self) -> dict:
        return dict(self._calls)

    def check_assets(self) -> dict:
        fm = self.fm
        with self._lock:
            need((fm.state_digest(self.f4), fm.state_digest(self.af4b)) == self._state,
                 "Model state changed in memory", AdapterAssetError)
            need(all(not m.training and all(not p.requires_grad for p in m.parameters())
                     for m in (self.f4, self.af4b)), "Models are not frozen/eval", AdapterAssetError)
        return {"status": "PASS", "identity": self.identity, "model_state_unchanged": True,
                "forward_performed": False}

    def infer(self, branch: str, x15: list[float], q: list[float], mask: list[int]) -> dict:
        need(branch in ("F4", "AF4-B"), "Invalid branch", AdapterInferenceError)
        need(branch != "F4" or mask == [1, 1, 1], "Healthy F4 requires all three modalities", AdapterInferenceError)
        fm, np, torch = self.fm, self.fm.np, self.fm.torch
        with self._lock:
            self.check_assets()
            try:
                x, quality, masks = fm.canonical_arrays([x15], [q], [mask])
                need(bool(masks.any()), "All-missing input must bypass models", AdapterInferenceError)
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
                    if branch == "F4":
                        p = fm.infer_f4_core(self.f4, x, self.device, 1)
                        diagnostics = {"f4_probs": p[0].tolist(), "adaptive_branch_active": False}
                    else:
                        a = fm.infer_af4b(self.af4b, x, quality, masks, self.device, 1)
                        need(all(np.isfinite(v).all() for v in a.values()), "Nonfinite AF4-B diagnostic", AdapterInferenceError)
                        need(np.array_equal(a["available_count"], masks.sum(axis=1)) and not a["no_decision"].any(),
                             "AF4-B availability output mismatch", AdapterInferenceError)
                        for key in ("f4_probs", "adaptive_probs", "final_probs"):
                            fm.ensure_probs(key, a[key])
                        alpha, effective = a["alpha"], a["effective_weights"]
                        need(alpha.shape == (1, 3) and effective.shape == (1, 3, 5), "Wrong weight shapes", AdapterInferenceError)
                        need(np.all((alpha >= 0) & (alpha <= 1)) and np.allclose(alpha.sum(axis=1), 1., atol=1e-6),
                             "Invalid alpha", AdapterInferenceError)
                        need(np.all((effective >= 0) & (effective <= 1)) and np.allclose(effective.sum(axis=1), 1., atol=1e-6),
                             "Invalid class weights", AdapterInferenceError)
                        need(float(np.max(np.abs(alpha * (1-masks)))) <= 1e-7
                             and float(np.max(np.abs(effective * (1-masks[:, :, None])))) <= 1e-7,
                             "Unavailable modality retained adaptive weight", AdapterInferenceError)
                        need(np.all((a["gamma"] >= 0) & (a["gamma"] <= 1))
                             and np.all((a["class_reliability"] >= 0) & (a["class_reliability"] <= 1)),
                             "Invalid gates", AdapterInferenceError)
                        p = a["final_probs"]
                        diagnostics = {k: v[0].tolist() for k, v in a.items()}
                        diagnostics["adaptive_branch_active"] = True
                        diagnostics["weight_semantics"] = "Adaptive-branch weights, NOT total attribution through F4 residual"
                self._calls[branch] += 1
                seconds = time.perf_counter() - start
                self.check_assets()
                return {"probabilities": p[0].tolist(), "diagnostics": diagnostics,
                        "backend_identity": self.identity, "model_inference_seconds": seconds,
                        "model_quality_float32": quality[0].tolist(), "mask": mask,
                        "branch_executed": branch, "legacy_router_executed": False}
            except (AdapterContractError, AdapterAssetError, AdapterInferenceError):
                raise
            except Exception as exc:
                raise AdapterInferenceError(f"{branch} inference failed: {type(exc).__name__}: {exc}") from exc


class _UpstreamQualityError(AdapterContractError):
    pass


def _equivalent(observed: Any, expected: Any, name: str) -> None:
    """Compare only declared contract fields; never tolerate route/boolean changes."""
    if isinstance(expected, Mapping):
        current = obj(observed, name)
        need(set(current) == set(expected), f"{name}: key set mismatch")
        for key, value in expected.items():
            _equivalent(current[key], value, name + "." + key)
    elif isinstance(expected, list):
        need(isinstance(observed, (list, tuple)) and len(observed) == len(expected), f"{name}: sequence mismatch")
        for i, (a, b) in enumerate(zip(observed, expected)):
            _equivalent(a, b, f"{name}[{i}]")
    elif type(expected) is bool:
        need(type(observed) is bool and observed is expected, f"{name}: boolean mismatch")
    elif type(expected) is int:
        need(isinstance(observed, Integral) and not isinstance(observed, bool) and int(observed) == expected,
             f"{name}: integer mismatch")
    elif isinstance(expected, Real):
        value = real(observed, name)
        need(math.isclose(value, float(expected), rel_tol=1e-10, abs_tol=1e-10), f"{name}: recomputed numeric mismatch")
    else:
        need(observed == expected, f"{name}: mismatch")


def _probabilities(value: Any, name: str) -> list[float]:
    need(not isinstance(value, (str, bytes, Mapping)), f"{name}: explicit [5] probabilities required")
    try:
        values = list(value)
    except TypeError as exc:
        raise AdapterContractError(f"{name}: [5] probability sequence required") from exc
    need(len(values) == 5, f"{name}: expected five values in the fixed class order")
    p = [real(x, name) for x in values]
    need(all(0 <= x <= 1 for x in p), f"{name}: probabilities outside [0,1]")
    need(abs(math.fsum(p) - 1.) <= PROBABILITY_ATOL,
         f"{name}: probabilities must sum to 1; logits/arbitrary scores are not accepted")
    return p  # No new softmax, renormalization, temperature or clipping.


class QualityFusionAdapter:
    """New candidate quality policy + original frozen model cores.

    Construction loads/checks quality code/parameters ONLY. Fusion checkpoints are
    loaded lazily on the first valid decision or explicitly via check_assets().
    A missing checkpoint is an error, never a synthetic prediction fallback.
    Instances are serialized with a lock. Replay repetition is permitted;
    shadow publication ordering/deduplication belongs to the host coordinator.
    """
    def __init__(self, params_path: str | Path | None = None, *,
                 root: str | Path | None = None, system_config: str | Path | None = None,
                 fusion_script: str | Path | None = None, fusion_script_sha256: str | None = None,
                 assets_dir: str | Path | None = None,
                 f4_checkpoint: str | Path | None = None, af4b_checkpoint: str | Path | None = None,
                 manifest: str | Path | None = None, device: str | None = None,
                 max_age_seconds: float = 1., future_tolerance_seconds: float = .02,
                 newest_sample_tolerance_seconds: float = .1,
                 clock: Callable[[], float] = time.monotonic,
                 expected_clock_id: str | None = None) -> None:
        qc_path = Path(__file__).resolve().with_name("quality_controller.py")
        self._qc_module = _load_module(qc_path, CONTROLLER_SHA256)
        need(self._qc_module.VERSION == CONTROLLER_VERSION, "Controller API mismatch", AdapterAssetError)
        self.controller = self._qc_module.QualityController(params_path,
            max_age_seconds=max_age_seconds, future_tolerance_seconds=future_tolerance_seconds,
            newest_sample_tolerance_seconds=newest_sample_tolerance_seconds)
        self.mapping = CandidateQualityMapping()
        self._backend_settings = dict(root=root, system_config=system_config, fusion_script=fusion_script,
            fusion_script_sha256=fusion_script_sha256, assets_dir=assets_dir,
            f4_checkpoint=f4_checkpoint, af4b_checkpoint=af4b_checkpoint, manifest=manifest, device=device)
        self._backend: FrozenBranchBackend | None = None
        self._lock = threading.RLock()
        need(callable(clock), "clock must be callable")
        self.clock = clock
        self.expected_clock_id = text(expected_clock_id, "expected_clock_id") if expected_clock_id is not None else None
        self.max_age_seconds = real(max_age_seconds, "max_age_seconds")
        self.future_tolerance_seconds = real(future_tolerance_seconds, "future_tolerance_seconds")
        self.newest_sample_tolerance_seconds = real(newest_sample_tolerance_seconds, "newest_sample_tolerance_seconds")
        self._policy = {"schema": POLICY_SCHEMA, "version": VERSION,
            "mapping": self.mapping.description(), "thresholds": self.controller.thresholds,
            "modality_order": list(MODALITIES), "class_order": list(EMOTIONS), "window_seconds": 5.,
            "bundle_sha256": self.controller.identity["scorer"]["bundle_sha256"],
            "upstream_policy_fingerprint": self.controller.identity["scorer"]["policy_fingerprint_sha256"],
            "expected_frozen_checkpoint_sha256": CHECKPOINT_HASHES.copy(),
            "routing": {"HEALTHY_FUSION": "F4", "ROBUST_FUSION": "AF4-B",
                        "NO_DECISION": "NO_FORWARD", "QUALITY_CONTROL_ERROR": "NO_FORWARD"},
            "reference_threshold_transfer_validated": False,
            "full_system_accuracy_improvement_proven": False,
            "weights_retrained": False, "legacy_q_used": False, "production_approved": False}
        self._policy_hash = fingerprint(self._policy)

    @property
    def policy(self) -> dict:
        return {**deepcopy(self._policy), "policy_sha256": self._policy_hash}

    @property
    def identity(self) -> dict:
        return {"adapter_version": VERSION, "adapter_script_sha256": file_sha(Path(__file__)),
                "controller_script_sha256": CONTROLLER_SHA256,
                "scorer": self.controller.identity["scorer"], "policy_sha256": self._policy_hash,
                "fusion_backend_loaded": self._backend is not None,
                "fusion_backend": self._backend.identity if self._backend is not None else None}

    def _get_backend(self) -> FrozenBranchBackend:
        if self._backend is None:
            self._backend = FrozenBranchBackend(**self._backend_settings)
        return self._backend

    def check_assets(self) -> dict:
        with self._lock:
            report = self._get_backend().check_assets()
            return {"status": "PASS", "adapter_version": VERSION, "policy": self.policy,
                    "backend": report, "actual_forward_performed": False,
                    "legacy_q_used": False, "production_approved": False}

    def _validate_quality(self, result: Any, mode: str) -> tuple[Mapping, dict, dict, dict]:
        q = obj(result, "quality_result")
        need(q.get("schema") == CONTROLLER_SCHEMA and q.get("controller_version") == CONTROLLER_VERSION,
             "Expected reviewed quality_controller.result.v1")
        # In particular, three algorithm-error placeholders must not become NO_DECISION.
        if q.get("status") == "QUALITY_CONTROL_ERROR" or q.get("candidate_route") == "QUALITY_CONTROL_ERROR" or q.get("errors"):
            raise _UpstreamQualityError("Upstream quality controller reported an error; fusion withheld")
        need(mode in ("replay", "shadow") and q.get("mode") == mode,
             "Mode must match quality result; no live or silent replay/shadow relabelling")
        need(q.get("status") in ("OK", "NO_DECISION") and q.get("errors") == [], "Invalid quality status/errors")
        need(q.get("window_identity_verified") is True and q.get("availability_all_validated") is True,
             "Quality window identity or availability was not validated")
        identity = obj(q.get("identity"), "quality.identity")
        need(identity.get("controller_script_sha256") == CONTROLLER_SHA256
             and identity.get("controller_version") == CONTROLLER_VERSION, "Quality controller identity changed")
        src_id = obj(identity.get("scorer"), "quality.identity.scorer")
        local_id = self.controller.identity["scorer"]
        for key in ("runtime_version", "runtime_script_sha256", "bundle_sha256", "policy_fingerprint_sha256"):
            need(src_id.get(key) == local_id[key], "Quality identity mismatch: " + key)
        need(src_id.get("bundle_sha256_verified") is True and q.get("legacy_q_used") is False,
             "Unverified parameter bundle or legacy quality result")
        window = self.controller._dq.WindowIdentity.from_mapping(q.get("window")).to_dict()
        availability = obj(q.get("availability"), "quality.availability")
        need(set(availability) == set(MODALITIES) and all(type(availability[m]) is bool for m in MODALITIES),
             "Quality availability must be exactly three booleans")
        inp = obj(q.get("scorer_input"), "quality.scorer_input")
        need(self.controller._dq.WindowIdentity.from_mapping(inp.get("window")).to_dict() == window,
             "scorer_input.window: exact source-span identity mismatch")
        _equivalent(inp.get("availability"), availability, "scorer_input.availability")
        # Arithmetic replay uses installed identical references. Shadow freshness is
        # rechecked separately from the ACTUAL host clock, not a serialized now.
        replayed = self.controller._scorer.score_window(inp, mode="replay")
        need(not replayed["errors"] and replayed["candidate_route"] != "QUALITY_CONTROL_ERROR",
             "Stored raw evidence cannot be rescored with installed parameters")
        for key in ("status", "candidate_route", "any_quality_alarm"):
            _equivalent(q.get(key), replayed[key], "quality." + key)
        mods = obj(q.get("modalities"), "quality.modalities")
        need(set(mods) == set(MODALITIES), "Exactly three quality modality results required")
        numeric_keys = ("available", "status", "B", "T", "B_over_T", "exceeds_threshold", "comparison",
                        "required_feature_count", "scored_feature_count", "feature_order", "raw_features", "evidence", "errors")
        for m in MODALITIES:
            reported = obj(mods[m], "quality.modalities." + m)
            for key in numeric_keys:
                _equivalent(reported.get(key), replayed["modalities"][m][key], f"quality.{m}.{key}")
            need(real(reported.get("T"), m + ".T") == self.controller.thresholds[m],
                 f"{m}: threshold must exactly match the installed artifact")
            # Raw observations are not allowed to drift under a comparison tolerance.
            need(reported.get("raw_features") == replayed["modalities"][m]["raw_features"], f"{m}: raw features differ")
        if "distribution_result" in q:
            duplicate = obj(q["distribution_result"], "distribution_result")
            for key in ("status", "candidate_route", "availability", "any_quality_alarm", "bundle_sha256"):
                _equivalent(duplicate.get(key), replayed[key], "distribution_result." + key)
            if any(availability.values()):
                need(duplicate.get("window") == window, "Redundant distribution result has different window identity")
        audits = obj(q.get("adapter_audit"), "quality.adapter_audit")
        need(set(audits) == set(MODALITIES), "Exactly three source audits required")
        for m in MODALITIES:
            audit = obj(audits[m], "audit." + m)
            need(audit.get("modality") == m and audit.get("available") is availability[m], "Availability audit mismatch")
            need(audit.get("reported_status") == ("OK" if availability[m] else "UNAVAILABLE"), "Producer status mismatch")
            basis = text(audit.get("availability_basis"), m + ".availability_basis")
            reason = text(audit.get("availability_reason"), m + ".availability_reason")
            need(reason != "MODULE_ERROR_EXCLUDED", "An algorithm-error placeholder is not validated absence")
            if availability[m]:
                need(basis == "producer_available_with_consistent_source_and_report_contracts", "Unexpected available-evidence basis")
            else:
                need(basis in ("descriptor_present_false", "descriptor_capture_failure_or_noncontiguous",
                    "current_clock_confirms_stale_source", "explicit_quality_unavailable",
                    "explicit_emotion_unavailable", "explicit_quality_and_emotion_unavailable"),
                    "Unrecognized unavailable-evidence basis")
            if availability[m]:
                source = obj(audit.get("source"), m + ".source")
                text(source.get("source"), m + ".source.source")
                sha(source.get("source_sha256"), m + ".source_sha256")
                need(sha(source.get("window_sha256"), m + ".window_sha256") ==
                     sha(source.get("evidence_id"), m + ".evidence_id"), "Source evidence hash disagreement")
                need(source.get("declarations_consistent") is True, "Source declarations inconsistent")
        downstream = obj(q.get("downstream"), "quality.downstream")
        recommended = {"HEALTHY_FUSION": "F4", "ROBUST_FUSION": "AF4-B"}.get(replayed["candidate_route"])
        need(downstream.get("quality_decision_valid") is True
             and downstream.get("recommended_branch") == recommended
             and downstream.get("withhold_candidate_emotion") is (replayed["candidate_route"] == "NO_DECISION"),
             "Contradictory quality downstream recommendation")
        return q, window, dict(availability), replayed

    def _source_match(self, m: str, source: Any, audit_source: Mapping) -> None:
        s = obj(source, m + ".emotion.source")
        need(self._qc_module._same_path(text(s.get("source"), m + ".source.source"), audit_source["source"]),
             f"{m}: quality/emotion source path mismatch")
        sd = s.get("source_sha256", s.get("sha256"))
        need(sha(sd, m + ".source digest") == audit_source["source_sha256"], f"{m}: source digest mismatch")
        if "sha256" in s and "source_sha256" in s:
            need(sha(s["sha256"], "sha256") == sha(s["source_sha256"], "source_sha256"), "Contradictory source digests")
        wh = s.get("window_sha256", sd if s.get("source") == "adapter_array" else None)
        need(sha(wh, m + ".window digest") == audit_source["window_sha256"], f"{m}: source window digest mismatch")
        for key in ("window_index", "trial_index"):
            if key in s:
                need(key in audit_source and real(s[key], key) == audit_source[key], f"{m}: source {key} mismatch")

    def _emotion(self, m: str, report: Any, window: Mapping, available: bool,
                 audit: Mapping) -> tuple[list[float], dict]:
        r = obj(report, "emotion_reports." + m)
        need(r.get("modality") == m, f"{m}: emotion modality mismatch")
        need(r.get("status") == ("OK" if available else "UNAVAILABLE"), f"{m}: emotion status/availability mismatch")
        need("error" in r and r["error"] in (None, ""), f"{m}: emotion producer error/missing error declaration")
        need("algorithm_error" in r and not boolean(r["algorithm_error"], m + ".algorithm_error"),
             f"{m}: emotion algorithm error/missing declaration")
        schema = r.get("schema")
        source = audit.get("source")
        native_packet = schema == EMOTION_SCHEMA
        if native_packet:
            p = r
            need(self.controller._dq.WindowIdentity.from_mapping(p.get("window")).to_dict() == window,
                 f"{m}: emotion/quality source window mismatch")
            for key in ("session_id", "window_id"):
                if key in p:
                    need(p[key] == window[key], f"{m}: contradictory packet {key}")
            source_style = "EXPLICIT_SAME_WINDOW_EMOTION_PACKET"
        elif schema == NATIVE_REPORT_SCHEMA:
            need("packet" not in r, "Native report must not also contain legacy packet")
            need(self.controller._dq.WindowIdentity.from_mapping(r.get("window")).to_dict() == window,
                 f"{m}: native report window mismatch")
            need(r.get("session_id") == window["session_id"] and r.get("window_id") == window["window_id"],
                 f"{m}: native report identity mismatch")
            p = r
            source_style = "NATIVE_REPORT_BOUND_TO_CHECKED_SOURCE"
        else:
            need(schema is None, f"{m}: unsupported emotion report schema {schema!r}")
            p = obj(r.get("packet"), m + ".packet")
            need(p.get("schema") == MAIN_PACKET_SCHEMA, f"{m}: expected reviewed main.py packet")
            need(p.get("modality") == m and r.get("window_id") == window["window_id"], f"{m}: main report identity mismatch")
            for key in ("session_id", "window_id"):
                need(p.get(key) == window[key], f"{m}: main packet {key} mismatch")
                if key in r:
                    need(r[key] == window[key], f"{m}: main report {key} mismatch")
            need(real(p.get("window_seconds"), "packet.window_seconds") == 5., "Packet must be five seconds")
            source_style = "MAIN_PACKET_IDENTITY_AND_SAME_SOURCE_HASH"
        need(boolean(p.get("available"), m + ".available") == available, f"{m}: quality/emotion effective availability mismatch")
        if "available" in r:
            need(boolean(r["available"], m + ".available") == available, f"{m}: contradictory report availability")
        for part in (r, p):
            if "window_seconds" in part:
                need(real(part["window_seconds"], "window_seconds") == 5., "Not five seconds")
            if "window" in part:
                need(self.controller._dq.WindowIdentity.from_mapping(part["window"]).to_dict() == window, "Window mismatch")
        # Check errors/identity even for unavailable reports; never read their stale probabilities.
        er = r.get("emotion") if not native_packet else None
        if er is not None:
            self.controller._check_part(er, m + ".emotion", m, window)
            if m + "_available" in er:
                need(not available or boolean(er[m + "_available"], m + ".emotion.available"), "Head availability contradiction")
        if not available:
            return [.2] * 5, {"available": False, "probabilities_discarded": True,
                             "binding": source_style, "input_source_not_used": True}
        if not native_packet:
            self._source_match(m, r.get("source"), source)
        elif "source" in r:
            self._source_match(m, r["source"], source)
        need(sha(p.get("evidence_id"), m + ".evidence_id") == source["window_sha256"],
             f"{m}: probability/quality evidence ID mismatch")
        if schema == NATIVE_REPORT_SCHEMA:
            er = obj(er, m + ".emotion")
            order = er.get("class_order", r.get("class_order"))
            values = er.get(m + "_probs", er.get("probabilities"))
        else:
            order = p.get("class_order")
            values = p.get("probabilities")
        need(isinstance(order, (list, tuple)) and tuple(order) == EMOTIONS, f"{m}: class_order must be explicit and fixed")
        probs = _probabilities(values, m + ".probabilities")
        # Reject duplicated class/probability declarations that contradict the selected packet.
        if er is not None:
            if "class_order" in er:
                need(tuple(er["class_order"]) == EMOTIONS, f"{m}: contradictory head class order")
            if m + "_probs" in er:
                other = _probabilities(er[m + "_probs"], m + ".emotion probabilities")
                need([fp32(x) for x in other] == [fp32(x) for x in probs], f"{m}: packet/head probabilities differ")
        timing = audit.get("timing", {})
        if "timing" in p:
            tm = obj(p["timing"], "emotion timing")
            need(timing.get("checked") is True, f"{m}: unexpected probability timing without quality timing")
            for pk, ak in (("clock_id", "clock_id"), ("window_start_monotonic", "start_seconds"),
                           ("window_end_monotonic", "end_seconds"), ("newest_sample_monotonic", "newest_sample_seconds")):
                need(tm.get(pk) == timing.get(ak), f"{m}: probability timing mismatch: {pk}")
        return probs, {"available": True, "probabilities_discarded": False, "binding": source_style,
                       "evidence_id": source["window_sha256"], "source_declarations_matched": True,
                       "raw_bytes_independently_verified": False}

    def _freshness(self, q: Mapping, window: Mapping, availability: Mapping, now: float) -> dict:
        need(self.expected_clock_id is not None and self.expected_clock_id == window["clock_id"],
             "Shadow requires explicitly matching trusted expected_clock_id")
        need(window["end_seconds"] <= now + self.future_tolerance_seconds, "Source decision window is in the future")
        ages, limits = {}, []
        if any(availability.values()):
            need(q.get("freshness_checked") is True, "Quality result did not check shadow freshness")
        for m in MODALITIES:
            if not availability[m]:
                continue
            tm = obj(q["adapter_audit"][m].get("timing"), m + ".timing")
            need(tm.get("checked") is True and tm.get("freshness_checked") is True
                 and tm.get("clock_id") == self.expected_clock_id, "Missing/incorrect shadow quality timing")
            need(real(tm.get("start_seconds"), "start") == window["start_seconds"]
                 and real(tm.get("end_seconds"), "end") == window["end_seconds"], "Timing span mismatch")
            newest = real(tm.get("newest_sample_seconds"), m + ".newest_sample_seconds")
            need(window["start_seconds"] <= newest <= window["end_seconds"] + self.future_tolerance_seconds
                 and window["end_seconds"] - newest <= self.newest_sample_tolerance_seconds,
                 f"{m}: newest sample inconsistent with window")
            need(newest <= now + self.future_tolerance_seconds and now-newest <= self.max_age_seconds,
                 f"{m}: source expired or lies in the future; rerun the coordinator, do not reuse cached quality")
            ages[m] = now - newest
            limits.append(min(newest, window["end_seconds"]) + self.max_age_seconds)
        if limits:
            need(now <= min(limits), "Decision window expired")
        return {"checked": True, "clock_id": self.expected_clock_id, "checked_at_seconds": now,
                "ages_seconds": ages, "valid_until_seconds": min(limits) if limits else None}

    def _base(self, mode: str) -> dict:
        return {"schema": OUTPUT_SCHEMA, "adapter_version": VERSION, "status": "FUSION_INPUT_ERROR",
                "mode": mode, "candidate_only": True, "window": None,
                "class_order": list(EMOTIONS), "modality_order": list(MODALITIES),
                "candidate_route": None, "active_branch": None, "availability": None,
                "quality_for_fusion": None, "model_inputs": None, "fusion_input_ready": False,
                "fusion_attempted": False, "fusion_executed": False, "real_checkpoint_loaded": False,
                "legacy_q_used": False, "legacy_router_executed": False,
                "original_deployed_router_modified": False,
                "quality_scores_are_correctness_probabilities": False,
                "final": {"emotion": None, "label_id": None, "probabilities": None,
                          "confidence": None, "is_evidence": False},
                "source_bytes_independently_verified": False,
                "provenance_limit": "Raw-feature re-scoring + declared source identity comparison; not sensor authentication.",
                "policy": self.policy, "errors": [], "warnings": [
                    "CANDIDATE_MAPPING_NOT_CALIBRATED_TO_EMOTION_CORRECTNESS",
                    "AF4B_NEW_QUALITY_SCALE_COMPATIBILITY_NOT_VALIDATED",
                    "FULL_REFERENCE_THRESHOLD_TRANSFER_NOT_VALIDATED",
                    "FIVE_SECOND_SYSTEM_ACCURACY_NOT_ESTABLISHED_BY_THIS_SCRIPT"],
                "production_approved": False, "live_robot_control_authorized": False,
                "robot_action_performed": False}

    def _prepare(self, quality_result: Any, emotion_reports: Any, mode: str, out: dict) -> tuple[Mapping, dict, dict]:
        q, window, availability, rescored = self._validate_quality(quality_result, mode)
        route = rescored["candidate_route"]
        out.update(window=window, availability=availability, candidate_route=route,
                   quality_raw_evidence_recomputed=True, controller_result_consistency_checked=True)
        if mode == "shadow":
            out["freshness_before"] = self._freshness(q, window, availability, real(self.clock(), "clock()"))
        else:
            out["freshness_before"] = {"checked": False, "reason": "OFFLINE_REPLAY"}
        mapping = {m: self.mapping.apply(rescored["modalities"][m]["B"],
                   self.controller.thresholds[m], availability[m]) for m in MODALITIES}
        out["mapping_details"] = mapping
        out["quality_for_fusion"] = {m: mapping[m]["q_new"] for m in MODALITIES}
        mask = [int(availability[m]) for m in MODALITIES]
        if route == "NO_DECISION":
            out.update(status="NO_DECISION", emotion_reports_consumed=False,
                       model_inputs=None, fusion_input_ready=False,
                       reasons=["ALL_MODALITIES_VERIFIED_UNAVAILABLE"])
            out["final"]["emotion"] = "NO_DECISION"
            return q, window, availability
        reports = obj(emotion_reports, "emotion_reports")
        need(set(reports) == set(MODALITIES), "Exactly three explicit emotion reports required")
        x, audit = [], {}
        for m in MODALITIES:
            probs, audit[m] = self._emotion(m, reports[m], window, availability[m], q["adapter_audit"][m])
            x.extend(probs)
        mapped_q = [mapping[m]["q_new"] for m in MODALITIES]
        out.update(status="READY", fusion_input_ready=True, emotion_reports_consumed=True,
                   emotion_binding_audit=audit,
                   model_inputs={"x15_float32": [fp32(v) for v in x],
                                 "quality_float32": [fp32(v) for v in mapped_q], "mask": mask})
        out["input_binding_sha256"] = fingerprint({"window": window, "availability": availability,
            "evidence_ids": {m: audit[m].get("evidence_id") for m in MODALITIES},
            "model_inputs": out["model_inputs"], "policy_sha256": self._policy_hash})
        return q, window, availability

    def prepare_window(self, *, quality_result: Any, emotion_reports: Any, mode: str = "replay") -> dict:
        """Validate/map without model I/O. READY is NOT a prediction."""
        started = time.perf_counter()
        out = self._base(mode)
        try:
            with self._lock:
                self._prepare(quality_result, emotion_reports, mode, out)
        except Exception as exc:
            self._failure(out, exc)
        out["identity"] = self.identity
        out["processing_seconds"] = time.perf_counter() - started
        return out

    def predict_window(self, *, quality_result: Any, emotion_reports: Any, mode: str = "replay") -> dict:
        """Validate, map, execute the selected original model core, return JSON-safe output."""
        started = time.perf_counter()
        out = self._base(mode)
        try:
            with self._lock:
                q, window, availability = self._prepare(quality_result, emotion_reports, mode, out)
                if out["status"] != "NO_DECISION":
                    backend = self._get_backend()
                    out["real_checkpoint_loaded"] = backend.identity["real_checkpoint_loaded"]
                    # Model loading may take longer than the evidence lifetime.
                    if mode == "shadow":
                        out["freshness_at_forward"] = self._freshness(q, window, availability, real(self.clock(), "clock()"))
                    branch = "F4" if out["candidate_route"] == "HEALTHY_FUSION" else "AF4-B"
                    inputs = out["model_inputs"]
                    out["fusion_attempted"] = True
                    model = backend.infer(branch, inputs["x15_float32"], inputs["quality_float32"], inputs["mask"])
                    out["fusion_executed"] = True
                    out["executed_branch"] = branch
                    need(model.get("branch_executed") == branch and model.get("legacy_router_executed") is False,
                         "Backend branch mismatch/old router executed", AdapterInferenceError)
                    try:
                        probs = _probabilities(model.get("probabilities"), "fusion.probabilities")
                    except AdapterContractError as exc:
                        raise AdapterInferenceError(f"Invalid model output: {exc}") from exc
                    need(model.get("mask") == inputs["mask"], "Backend mask mismatch", AdapterInferenceError)
                    _equivalent(model.get("model_quality_float32"), inputs["quality_float32"], "backend.quality")
                    if mode == "shadow":
                        end = real(self.clock(), "clock()")
                        need(end >= out["freshness_at_forward"]["checked_at_seconds"], "Clock moved backwards")
                        out["freshness_after"] = self._freshness(q, window, availability, end)
                    index = max(range(5), key=probs.__getitem__)
                    out.update(status="OK", active_branch=branch,
                        final={"emotion": EMOTIONS[index], "label_id": index, "probabilities": probs,
                               "confidence": max(probs), "is_evidence": True,
                               "confidence_is_calibrated_correctness": False},
                        diagnostics=model["diagnostics"], backend_identity=model["backend_identity"],
                        model_inference_seconds=model["model_inference_seconds"])
        except Exception as exc:
            self._failure(out, exc)
        out["identity"] = self.identity
        out["processing_seconds"] = time.perf_counter() - started
        return out

    @staticmethod
    def _failure(out: dict, exc: Exception) -> None:
        if isinstance(exc, _UpstreamQualityError):
            status = "QUALITY_CONTROL_ERROR"
            out["candidate_route"] = "QUALITY_CONTROL_ERROR"
        elif isinstance(exc, AdapterAssetError):
            status = "FUSION_ASSET_ERROR"
        elif isinstance(exc, AdapterInferenceError):
            status = "FUSION_INFERENCE_ERROR"
        elif isinstance(exc, (ValueError, TypeError, OverflowError, KeyError)):
            status = "FUSION_INPUT_ERROR"
        else:
            status = "INTERNAL_ADAPTER_ERROR"
        out.update(status=status, active_branch=None)
        if status in ("QUALITY_CONTROL_ERROR", "FUSION_INPUT_ERROR", "INTERNAL_ADAPTER_ERROR"):
            out["fusion_input_ready"] = False
        out["final"] = {"emotion": None, "label_id": None, "probabilities": None, "confidence": None, "is_evidence": False}
        out["errors"].append({"code": status, "error_type": type(exc).__name__, "message": str(exc)[:2000]})
        if out["fusion_executed"]:
            out["computed_output_withheld"] = True

    def predict_input(self, payload: Any, *, mode: str = "replay", prepare_only: bool = False) -> dict:
        data = obj(payload, "adapter input")
        need(data.get("schema") == INPUT_SCHEMA, f"Input schema must be {INPUT_SCHEMA}")
        fn = self.prepare_window if prepare_only else self.predict_window
        return fn(quality_result=data.get("quality_result"), emotion_reports=data.get("emotion_reports"), mode=mode)

    def make_example(self, case: str = "healthy") -> dict:
        """Synthetic source reports and probabilities; no real measurement claim."""
        need(case in ("healthy", "degraded", "missing-video", "all-missing", "error"), "Unknown example case")
        src = self.controller.make_example(report_format="native")
        if case == "degraded":
            # Change raw audio evidence, then recompute B. Never edit a score/route.
            model = self.controller._scorer._policies["audio"]
            ref = next(x for x in model.features if x.path == "signal_metrics.rms_dbfs")
            # make_example nests raw metrics by source detector path.
            src["reports"]["audio"]["quality"]["signal_metrics"]["rms_dbfs"] = ref.mu - 6. * ref.sigma
        if case in ("missing-video", "all-missing"):
            targets = MODALITIES if case == "all-missing" else ("video",)
            for m in targets:
                src["source_window"]["modalities"][m] = {"present": False}
                r = src["reports"][m]
                r.update(available=False, status="UNAVAILABLE", reason="NO_CURRENT_SOURCE",
                         source=None, quality=None, emotion=None)
                r.pop("evidence_id", None)
        if case == "error":
            src["reports"]["audio"].update(algorithm_error=True, error="SYNTHETIC_DETECTOR_FAILURE", status="ERROR")
        q = self.controller.assess_input(src)
        vectors = {"eeg": [.15, .15, .1, .5, .1], "audio": [.1, .2, .1, .5, .1], "video": [.1, .1, .1, .6, .1]}
        packets = {}
        for m in MODALITIES:
            r = src["reports"][m]
            a = r["available"]
            packets[m] = {"schema": EMOTION_SCHEMA, "modality": m, "window": deepcopy(q["window"]),
                "available": a, "status": "OK" if a else "UNAVAILABLE", "error": None,
                "algorithm_error": False, "class_order": list(EMOTIONS),
                "probabilities": vectors[m] if a else None,
                "evidence_id": r.get("evidence_id") if a else None}
        return {"schema": INPUT_SCHEMA, "example_case": case, "real_measurements": False,
                "probabilities_are_synthetic": True,
                "note": "Contract/forward fixture only. Dummy source paths/hashes. Not accuracy or false-alarm evidence.",
                "quality_result": q, "emotion_reports": packets}


# -------------------------- Self-tests and CLI -------------------------------
class _TestBackend:
    """Logic-test double ONLY. Never selected by production/replay CLI fallback."""
    def __init__(self) -> None:
        self.calls = []

    @property
    def identity(self) -> dict:
        return {"backend_kind": "SELF_TEST_DOUBLE_NOT_A_MODEL", "real_checkpoint_loaded": False}

    def infer(self, branch: str, x: list, q: list, mask: list) -> dict:
        self.calls.append({"branch": branch, "x": x[:], "q": q[:], "mask": mask[:]})
        # Intentionally simple deterministic output; not an emotion model.
        p = [.1, .1, .1, .6, .1] if branch == "F4" else [.2, .1, .2, .4, .1]
        return {"probabilities": p, "diagnostics": {"test_double": True},
                "backend_identity": self.identity, "model_inference_seconds": 0.,
                "model_quality_float32": q[:], "mask": mask[:],
                "branch_executed": branch, "legacy_router_executed": False}


def _shadow_fixture(example: dict, end: float = 100.) -> dict:
    """Test conversion of a *synthetic* fixture, never relabel real measurements."""
    from copy import deepcopy as dc
    data = dc(example)
    q = data["quality_result"]
    w = {**q["window"], "clock_id": "SYNTHETIC_TEST_CLOCK", "start_seconds": end-5., "end_seconds": end}
    q.update(mode="shadow", window=w, freshness_checked=True)
    q["scorer_input"]["window"] = dc(w)
    for m in MODALITIES:
        q["scorer_input"]["modalities"][m]["window"] = dc(w)
        data["emotion_reports"][m]["window"] = dc(w)
        q["adapter_audit"][m]["timing"] = {"checked": True, "freshness_checked": True,
            "clock_id": w["clock_id"], "start_seconds": end-5., "end_seconds": end,
            "newest_sample_seconds": end, "age_seconds": .1}
    # distribution_result is redundant audit; only stored runtime identity changes.
    q["distribution_result"]["mode"] = "shadow"
    q["distribution_result"]["window"] = dc(w)
    q["distribution_result"]["freshness_checked"] = True
    return data


def run_self_tests(adapter: QualityFusionAdapter) -> dict:
    """No real model files required. Cases are reported individually, no fake totals."""
    import itertools
    import random
    checks = []
    original_backend = adapter._backend
    original_clock, original_cid = adapter.clock, adapter.expected_clock_id
    test_backend = _TestBackend()
    adapter._backend = test_backend

    def check(name: str, fn: Callable[[], Any]) -> None:
        try:
            value = fn()
            need(value is True, f"Test condition was not True: {value!r}")
            checks.append({"name": name, "passed": True})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    def rejects(data: dict, change: Callable[[dict], None], *, mode: str = "replay") -> bool:
        d = deepcopy(data)
        change(d)
        before = len(test_backend.calls)
        r = adapter.predict_input(d, mode=mode)
        return bool(r["errors"]) and r["final"]["probabilities"] is None and len(test_backend.calls) == before

    try:
        healthy = adapter.make_example("healthy")
        for case, status, branch in (("healthy", "OK", "F4"), ("degraded", "OK", "AF4-B"),
                                     ("missing-video", "OK", "AF4-B"), ("all-missing", "NO_DECISION", None),
                                     ("error", "QUALITY_CONTROL_ERROR", None)):
            def one(case=case, status=status, branch=branch):
                before = len(test_backend.calls)
                r = adapter.predict_input(adapter.make_example(case))
                return r["status"] == status and r["active_branch"] == branch and len(test_backend.calls)-before == int(branch is not None)
            check("route_" + case, one)
        for r in (0., .1, .5, 1., 2., 5., 6., 12.):
            check(f"mapping_r_{r}", lambda r=r: math.isclose(adapter.mapping.apply(r, 1., True)["q_new"],
                max(1e-6, math.exp(-.5*r*r)), rel_tol=1e-14))
        qs = [adapter.mapping.apply(i/100., 1., True)["q_new"] for i in range(1201)]
        check("mapping_nonincreasing", lambda: all(a >= b for a, b in zip(qs, qs[1:])))
        check("mapping_threshold_is_not_old_tau", lambda: math.isclose(adapter.mapping.apply(1., 1., True)["q_new"], math.exp(-.5)))
        check("mapping_missing_zero", lambda: adapter.mapping.apply(None, 1., False)["q_new"] == 0.)
        check("mapping_floor_explicit", lambda: adapter.mapping.apply(12., 1., True)["floor_applied"] is True)
        for bad in (True, "1", float("nan"), float("inf"), -1.):
            def map_bad(bad=bad):
                try:
                    adapter.mapping.apply(bad, 1., True)
                except AdapterContractError:
                    return True
                return False
            check("reject_mapping_B_" + repr(bad), map_bad)
        for t in (0., -1., True, "1", float("nan"), float("inf")):
            def map_t(t=t):
                try:
                    adapter.mapping.apply(0., t, True)
                except AdapterContractError:
                    return True
                return False
            check("reject_mapping_T_" + repr(t), map_t)
        # Values below the new anomaly threshold may map BELOW .80. They must
        # still select F4, proving that the old quality threshold is not used.
        src = adapter.controller.make_example(report_format="native")
        ref = next(x for x in adapter.controller._scorer._policies["audio"].features if x.path == "signal_metrics.rms_dbfs")
        target_r = .9
        src["reports"]["audio"]["quality"]["signal_metrics"]["rms_dbfs"] = ref.mu - math.sqrt(2.) * target_r * adapter.controller.thresholds["audio"] * ref.sigma
        mixed = deepcopy(healthy)
        mixed["quality_result"] = adapter.controller.assess_input(src)
        r = adapter.predict_input(mixed)
        check("healthy_even_when_new_q_below_0_80", lambda: r["active_branch"] == "F4" and r["quality_for_fusion"]["audio"] < .8)
        for bits in itertools.product((False, True), repeat=3):
            # Rebuild controller inputs, never alter validated B/availability manually.
            src = adapter.controller.make_example(report_format="native")
            for m, a in zip(MODALITIES, bits):
                if not a:
                    src["source_window"]["modalities"][m] = {"present": False}
                    src["reports"][m].update(available=False, status="UNAVAILABLE", source=None,
                        quality=None, emotion=None, reason="NO_CURRENT_SOURCE")
            data = deepcopy(healthy)
            data["quality_result"] = adapter.controller.assess_input(src)
            for m, a in zip(MODALITIES, bits):
                data["emotion_reports"][m].update(available=a, status="OK" if a else "UNAVAILABLE")
                if not a:
                    data["emotion_reports"][m]["probabilities"] = [float("nan")]  # intentionally irrelevant stale contents
            before = len(test_backend.calls)
            out = adapter.predict_input(data)
            expected = None if not any(bits) else ("F4" if all(bits) else "AF4-B")
            check("availability_" + "".join(str(int(a)) for a in bits), lambda out=out, expected=expected: out["active_branch"] == expected and not out["errors"])
            if any(bits):
                call = test_backend.calls[-1]
                check("mask_clearing_" + str(bits), lambda call=call, bits=bits: all(
                    a or (call["q"][i] == 0. and call["x"][i*5:(i+1)*5] == [fp32(.2)]*5)
                    for i, a in enumerate(bits)))
            else:
                check("no_decision_never_forwards", lambda before=before: len(test_backend.calls) == before)
        for m in MODALITIES:
            for field, bad in (("modality", "wrong"), ("class_order", list(reversed(EMOTIONS))),
                ("available", False), ("status", "ERROR"), ("error", "failure"),
                ("algorithm_error", True), ("evidence_id", "0"*64)):
                check(f"reject_{m}_{field}", lambda m=m, field=field, bad=bad: rejects(healthy,
                    lambda d: d["emotion_reports"][m].__setitem__(field, bad)))
            for key, bad in (("session_id", "wrong"), ("window_id", "wrong"), ("clock_id", "wrong"),
                             ("start_seconds", 0.), ("end_seconds", 20.)):
                check(f"reject_{m}_window_{key}", lambda m=m, key=key, bad=bad: rejects(healthy,
                    lambda d: d["emotion_reports"][m]["window"].__setitem__(key, bad)))
        for bad in (None, [1.]*5, [0.]*5, [.2]*4, [True, .2, .2, .2, .2], [float("nan")]*5,
                    [float("inf")]*5, [.2, .2, .2, .5, -.1], [".2"]*5):
            check("reject_probability_" + repr(bad), lambda bad=bad: rejects(healthy,
                lambda d: d["emotion_reports"]["audio"].__setitem__("probabilities", bad)))
        for field, bad in (("B", .1), ("T", 8.), ("exceeds_threshold", True), ("scored_feature_count", 1),
                           ("raw_features", {}), ("evidence", {})):
            check("reject_quality_tamper_" + field, lambda field=field, bad=bad: rejects(healthy,
                lambda d: d["quality_result"]["modalities"]["audio"].__setitem__(field, bad)))
        def tiny_shift(d):
            for key in ("start_seconds", "end_seconds"):
                d["quality_result"]["scorer_input"]["window"][key] += 1e-8
        check("reject_tiny_source_span_shift", lambda: rejects(healthy, tiny_shift))
        check("reject_tiny_threshold_change", lambda: rejects(healthy,
            lambda d: d["quality_result"]["modalities"]["audio"].__setitem__("T", adapter.controller.thresholds["audio"] + 1e-12)))
        check("reject_quality_route_tamper", lambda: rejects(healthy, lambda d: d["quality_result"].__setitem__("candidate_route", "ROBUST_FUSION")))
        check("reject_missing_emotion_report", lambda: rejects(healthy, lambda d: d["emotion_reports"].pop("video")))
        check("reject_missing_quality_identity", lambda: rejects(healthy, lambda d: d["quality_result"].pop("identity")))
        check("reject_bundle_identity_tamper", lambda: rejects(healthy, lambda d: d["quality_result"]["identity"]["scorer"].__setitem__("bundle_sha256", "0"*64)))
        check("reject_wrong_mode", lambda: rejects(healthy, lambda d: None, mode="live"))
        check("reject_replay_as_shadow", lambda: rejects(healthy, lambda d: None, mode="shadow"))
        before = fingerprint(healthy)
        adapter.predict_input(healthy)
        check("input_not_mutated", lambda: fingerprint(healthy) == before)
        count = len(test_backend.calls)
        p = adapter.predict_input(healthy, prepare_only=True)
        check("prepare_only_no_forward", lambda: p["status"] == "READY" and len(test_backend.calls) == count and not p["fusion_executed"])
        # Legacy fields must not even be READ, not merely ignored mathematically.
        class PoisonDict(dict):
            def __getitem__(self, key):
                if key in ("q_eeg", "q_audio", "q_video", "quality", "quality_state", "confidence"):
                    raise AssertionError("Legacy field was read: " + key)
                return super().__getitem__(key)
            def get(self, key, default=None):
                if key in ("q_eeg", "q_audio", "q_video", "quality", "quality_state", "confidence"):
                    raise AssertionError("Legacy field was read: " + key)
                return super().get(key, default)
        poisoned = {**healthy, "emotion_reports": {m: PoisonDict(r) for m, r in healthy["emotion_reports"].items()}}
        check("legacy_fields_never_read", lambda: adapter.predict_input(poisoned)["status"] == "OK")
        # Test main/native full reports, including the documented legacy packet shape.
        for fmt in ("main-v1", "native"):
            src = adapter.controller.make_example(report_format=fmt)
            for m in MODALITIES:
                em = src["reports"][m]["emotion"]
                em.update(class_order=list(EMOTIONS), **{m+"_probs": [.2]*5})
                if fmt == "main-v1":
                    src["reports"][m]["packet"].update(class_order=list(EMOTIONS), probabilities=[.2]*5)
                    src["reports"][m]["packet"] = PoisonDict(src["reports"][m]["packet"])
            qr = adapter.controller.assess_input(src)
            r = adapter.predict_window(quality_result=qr, emotion_reports=src["reports"])
            check("full_reports_" + fmt, lambda r=r: r["status"] == "OK")
        shadow = _shadow_fixture(healthy)
        adapter.expected_clock_id = "SYNTHETIC_TEST_CLOCK"
        adapter.clock = lambda: 100.1
        check("shadow_success", lambda: adapter.predict_input(shadow, mode="shadow")["status"] == "OK")
        adapter.clock = lambda: 102.
        check("shadow_expired_before_forward", lambda: rejects(shadow, lambda d: None, mode="shadow"))
        times = iter((100.1, 100.2, 102.))
        adapter.clock = lambda: next(times)
        r = adapter.predict_input(shadow, mode="shadow")
        check("shadow_expired_after_forward_withheld", lambda: r["fusion_executed"] and r["final"]["probabilities"] is None and r.get("computed_output_withheld") is True)
        adapter.clock = lambda: 100.1
        adapter.expected_clock_id = "WRONG_CLOCK"
        check("shadow_clock_mismatch", lambda: rejects(shadow, lambda d: None, mode="shadow"))
        adapter.clock, adapter.expected_clock_id = original_clock, original_cid
        # Randomized probability inputs: check strict FP32 canonicalization and branch selection.
        rng = random.Random(20260927)
        passed = 0
        for _ in range(128):
            d = deepcopy(healthy)
            for m in MODALITIES:
                v = [rng.random() for _ in range(5)]
                d["emotion_reports"][m]["probabilities"] = [x/sum(v) for x in v]
            r = adapter.predict_input(d)
            passed += int(r["status"] == "OK" and r["active_branch"] == "F4")
        check("128_randomized_windows", lambda: passed == 128)
    finally:
        adapter._backend = original_backend
        adapter.clock, adapter.expected_clock_id = original_clock, original_cid
    failed = [c for c in checks if not c["passed"]]
    return {"status": "PASS" if not failed else "FAIL", "n_checks": len(checks),
            "n_passed": len(checks)-len(failed), "checks": checks,
            "random_probability_windows": 128, "real_checkpoint_loaded": False,
            "backend": "SELF_TEST_DOUBLE_ONLY", "real_detector_data_used": False,
            "accuracy_evaluated": False}


def run_preflight(adapter: QualityFusionAdapter, *, real_models: bool = False) -> dict:
    quality_tests = adapter.controller.preflight()
    tests = run_self_tests(adapter)
    report = {"status": "PASS" if tests["status"] == quality_tests["status"] == "PASS" else "FAIL",
        "adapter_version": VERSION, "adapter_tests": tests, "upstream_quality_preflight": quality_tests,
        "policy": adapter.policy, "real_checkpoint_loaded": False,
        "real_model_forward_performed": False, "real_sensor_data_used": False,
        "accuracy_evaluated": False, "production_approved": False}
    if real_models:
        backend = adapter._get_backend()
        report["asset_check"] = backend.check_assets()
        checks = []
        for case in ("healthy", "degraded", "missing-video", "all-missing", "error"):
            data = adapter.make_example(case)
            before = backend.calls
            out = adapter.predict_input(data)
            expected_status = {"all-missing": "NO_DECISION", "error": "QUALITY_CONTROL_ERROR"}.get(case, "OK")
            expected_branch = {"healthy": "F4", "degraded": "AF4-B", "missing-video": "AF4-B"}.get(case)
            passed = out["status"] == expected_status and out["active_branch"] == expected_branch
            delta = {k: backend.calls[k]-before[k] for k in before}
            passed = passed and delta == {"F4": int(expected_branch == "F4"), "AF4-B": int(expected_branch == "AF4-B")}
            if expected_branch:
                inp = out["model_inputs"]
                direct = backend.infer(expected_branch, inp["x15_float32"], inp["quality_float32"], inp["mask"])
                maximum = max(abs(a-b) for a, b in zip(out["final"]["probabilities"], direct["probabilities"]))
                passed = passed and maximum <= 1e-7
            else:
                maximum = None
            checks.append({"case": case, "passed": passed, "selected_branch_call_delta": delta,
                           "max_abs_difference_from_direct_core": maximum, "result": out})
        report["real_model_checks"] = checks
        report["real_checkpoint_loaded"] = backend.identity["real_checkpoint_loaded"]
        report["real_model_forward_performed"] = True
        report["status"] = "PASS" if report["status"] == "PASS" and all(x["passed"] for x in checks) else "FAIL"
        report["asset_check_after"] = backend.check_assets()
    report["identity"] = adapter.identity
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--self-test", action="store_true", help="Adapter logic tests; no Torch/models")
    action.add_argument("--preflight-only", action="store_true", help="Adapter + controller/scorer tests; no fusion weights")
    action.add_argument("--preflight", action="store_true", help="Also load real frozen fusion weights and run synthetic-probability forward tests")
    action.add_argument("--check-fusion-api", action="store_true", help="Parse/check fusion core only; no module import or weights; requires --fusion-script")
    action.add_argument("--check-assets", action="store_true", help="Load/check real weights; no forward")
    action.add_argument("--write-example", metavar="PATH", help="Write explicitly synthetic complete input")
    action.add_argument("--input-json", metavar="PATH", help="Complete input or explicit batch envelope")
    action.add_argument("--export-policy", metavar="PATH", help="Write candidate mapping identity (does not approve it)")
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--params", help="Distribution bundle; defaults to sibling distribution_quality_params.json")
    p.add_argument("--output", help="New UTF-8 JSON output; never overwrites")
    p.add_argument("--prepare-only", action="store_true", help="With --input-json: no model loading/inference")
    p.add_argument("--example-case", choices=("healthy", "degraded", "missing-video", "all-missing", "error"), default="healthy")
    p.add_argument("--mode", choices=("replay", "shadow"), default="replay")
    p.add_argument("--clock-id", help="Trusted actual local monotonic clock ID for same-host shadow mode")
    p.add_argument("--system-config", help="Read ONLY existing system_config.json modules.fusion")
    p.add_argument("--fusion-script", help="Existing reviewed fusion module; default ../fusion/fusion_af4c_deployment.py")
    p.add_argument("--fusion-script-sha256", help="Optional full-byte script pin; numeric-core AST always checked")
    p.add_argument("--assets-dir", help="Directory containing the original two fusion .pt files")
    p.add_argument("--f4-checkpoint")
    p.add_argument("--af4b-checkpoint")
    p.add_argument("--manifest", help="Optional original fusion deployment manifest; auto-checked when beside both weights")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), help="Defaults to existing config device, otherwise CPU; no implicit CUDA fallback")
    args = p.parse_args(argv)
    need(not args.check_fusion_api or (args.fusion_script and not args.system_config),
         "--check-fusion-api requires --fusion-script; do not combine it with --system-config")
    need(not args.prepare_only or args.input_json, "--prepare-only requires --input-json")
    need(not args.output or not (args.write_example or args.export_policy), "Use action destination OR --output, not both")
    args.help_only = not any((args.self_test, args.preflight_only, args.preflight, args.check_assets,
                             args.write_example, args.input_json, args.export_policy, args.check_fusion_api))
    if args.help_only:
        p.print_help()
    return args


def _process_payload(adapter: QualityFusionAdapter, data: Any, args: Any) -> dict:
    data = obj(data, "input JSON")
    if data.get("schema") != BATCH_SCHEMA:
        return adapter.predict_input(data, mode=args.mode, prepare_only=args.prepare_only)
    samples = data.get("samples")
    need(isinstance(samples, list) and 0 < len(samples) <= 10000, "Batch requires 1..10000 samples")
    results = []
    for i, sample in enumerate(samples):
        try:
            result = adapter.predict_input(sample, mode=args.mode, prepare_only=args.prepare_only)
        except Exception as exc:
            result = adapter._base(args.mode)
            adapter._failure(result, exc)
        results.append({"index": i, "result": result})
    failures = sum(bool(x["result"]["errors"]) for x in results)
    return {"schema": BATCH_SCHEMA + ".result", "status": "PARTIAL_FAILURE" if failures else "OK",
            "n_samples": len(results), "n_errors": failures,
            "no_decision_count": sum(x["result"]["status"] == "NO_DECISION" for x in results),
            "results": results, "policy": adapter.policy, "failed_rows_retained": True}


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = None
    try:
        args = parse_args(argv)
        if args.help_only:
            return 0
        dest = args.output or args.write_example or args.export_policy
        if dest:
            need(not Path(dest).expanduser().exists(), f"Output exists, not overwritten: {dest}")
        if args.check_fusion_api:
            report = check_fusion_api(args.fusion_script)
            code = 0 if report["status"] == "PASS" else 2
            if args.output:
                write_json(args.output, report)
                print(json.dumps({"status": report["status"], "adapter_version": VERSION,
                                  "output": str(Path(args.output).resolve()), "exit_code": code},
                                 ensure_ascii=False, indent=2))
            else:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            return code
        adapter = QualityFusionAdapter(params_path=args.params, system_config=args.system_config,
            fusion_script=args.fusion_script, fusion_script_sha256=args.fusion_script_sha256,
            assets_dir=args.assets_dir, f4_checkpoint=args.f4_checkpoint, af4b_checkpoint=args.af4b_checkpoint,
            manifest=args.manifest, device=args.device, expected_clock_id=args.clock_id)
        if args.write_example:
            write_json(args.write_example, adapter.make_example(args.example_case))
            print(json.dumps({"status": "PASS", "output": str(Path(args.write_example).resolve()),
                              "real_measurements": False, "probabilities_are_synthetic": True}, indent=2))
            return 0
        if args.export_policy:
            write_json(args.export_policy, adapter.policy)
            print(json.dumps({"status": "PASS", "output": str(Path(args.export_policy).resolve()),
                              "production_approved": False}, indent=2))
            return 0
        if args.self_test:
            report = run_self_tests(adapter)
        elif args.preflight_only or args.preflight:
            report = run_preflight(adapter, real_models=args.preflight)
        elif args.check_assets:
            report = adapter.check_assets()
        else:
            report = _process_payload(adapter, read_json(args.input_json), args)
        code = 0 if report.get("status") in ("PASS", "OK", "READY", "NO_DECISION") else 2
        if args.output:
            write_json(args.output, report)
            summary = {"status": report.get("status"), "output": str(Path(args.output).resolve()), "exit_code": code}
            for field in ("active_branch", "fusion_executed"):
                if field in report:
                    summary[field] = report[field]
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
        return code
    except Exception as exc:
        error = {"status": "FUSION_ASSET_ERROR" if isinstance(exc, AdapterAssetError) else "FUSION_ADAPTER_ERROR",
                 "adapter_version": VERSION, "error_type": type(exc).__name__, "message": str(exc),
                 "fusion_executed": False, "production_approved": False, "exit_code": 2}
        if args is not None and args.output and not Path(args.output).exists():
            try:
                write_json(args.output, error)
            except OSError:
                pass
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
