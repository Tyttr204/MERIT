#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""System Test Runner -- evaluate the EXISTING EAV main.py, not a second model.

Entry points (run from the final deployment directory):
  python -X utf8 testing/system_test_runner.py --self-test
  python -X utf8 testing/system_test_runner.py --check-assets --config system_config.eav.json
  python -X utf8 testing/system_test_runner.py --suite contracts --config system_config.json
  python -X utf8 testing/system_test_runner.py --dry-run --suite robustness --config system_config.eav.json
  python -X utf8 testing/system_test_runner.py --preflight --config system_config.eav.json --eeg-unit uV --window-key KEY
  python -X utf8 testing/system_test_runner.py --suite smoke --config system_config.eav.json --eeg-unit uV --window-key KEY
  python -X utf8 testing/system_test_runner.py --suite reference --config system_config.eav.json --eeg-unit uV
  python -X utf8 testing/system_test_runner.py --suite robustness --config system_config.eav.json --eeg-unit uV
  python -X utf8 testing/system_test_runner.py --formal-balanced-val --config system_config.eav.json --eeg-unit uV

The EEG unit is an explicit, documented input assumption, NEVER an amplitude guess.
Default balanced selection: 1 complete trial/class/subject, the frozen 6 VAL
subjects -> 30 trials -> 120 windows. --window-key selects one explicit window;
--max-trials is a bounded debugging subset, never called a balanced population.
Test data remain opt-in. Existing training, calibration, tau=.80 and code stay
unchanged. Imports main.Runtime, main.SourceReader and the deployed fusion API.

Three comparisons use EXACTLY the same current probabilities and availability:
  fixed_f4: frozen F4 on canonical missing inputs, all-missing abstention.
  availability_only: AF4-C with q=1 for available, 0 for missing (ABLATION ONLY).
  quality_aware: main.py's actual quality-aware output.
No classifiers are re-run for baselines. No parameters are fitted or selected.

Raw perturbations are applied to private source copies BEFORE BOTH the affected
emotion head and its quality detector. EEG/audio doses use complete 20s trials
but inference is always 5s. All unchanged heads are also re-executed; no prediction
cache is used. Input preparation time, inference time and comparison overhead
are reported separately. No hardware timing or 20s fusion accuracy is claimed.

Robustness uses matched controls: audio common headroom is shared over one
trial's requested audio variants; RGB FFV1 is a matched video-filter reference;
CRF18 is the matched reference for optional CRF28/36/44 compression probes.
Centre-square partial occlusion is NOT guaranteed to cover the actual face.
Full black video and explicit missing input are distinct controls.

Scopes: contracts = synthetic probabilities/clock + real fusion only;
reference/smoke/robustness = real configured sources and real deployed modules.
Unavailable inputs may intentionally skip a model. No fallback to test doubles.
Raw modes are OFFLINE, not live sensor tests; identical-payload recovery is a
software probe. Primary performance excludes missing/zero/blackout/recovery
controls. Quality monotonicity/performance are observations, not PASS criteria.

Quality-distribution logging is observational only. It records each deployed
quality head's current payload plus canonical q/availability, exports clean
window- and trial-level descriptive distributions, and NEVER fits a Gaussian,
GMM, Beta model, changes q, or retunes the frozen tau=.80 router.

No installs, downloads, training, device activation, actuator commands, parameter
retuning, silent resume or overwrite. Use only trusted local Python/model files.
Large sources/models may take time to hash; hashes are rechecked on completion.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = 'EAV-SYSTEM-TEST.1.1-QDIST'
SCHEMA = 'eav.system_test.v1'
SCENARIO_SCHEMA = 'eav.system_test.scenarios.v1'
SUPPORTED_MAIN = 'EAV-MAIN-INTEGRATION.1.0'
EMOTIONS = ['Neutral', 'Sadness', 'Anger', 'Happiness', 'Calmness']
MODALITIES = ('eeg', 'audio', 'video')
METHODS = ('fixed_f4', 'availability_only', 'quality_aware')
SEVERITIES = ('mild', 'medium', 'severe')
EEG_FAMILIES = ('eeg_local_broadband', 'eeg_global_broadband', 'eeg_global_line',
                'eeg_global_slow_drift', 'eeg_channel_flatline', 'eeg_intermittent_hold')
AUDIO_FAMILIES = ('audio_attenuation', 'audio_white_noise', 'audio_pink_noise')
VIDEO_FAMILIES = ('video_blur', 'video_brightness', 'video_occlusion', 'video_compression')
FAMILIES = EEG_FAMILIES + AUDIO_FAMILIES + VIDEO_FAMILIES
DEFAULT_FAMILIES = ('audio_attenuation', 'audio_white_noise', 'video_blur',
                    'video_brightness', 'eeg_global_line', 'eeg_channel_flatline')
PRIMARY_KINDS = {'reference', 'matched_reference', 'corruption'}
VAL_SUBJECTS = {'subject08', 'subject09', 'subject10', 'subject13', 'subject14', 'subject33'}
TEST_SUBJECTS = {'subject03', 'subject05', 'subject20', 'subject31', 'subject35', 'subject39'}
DEFAULT_SEED = 20260919


class TestContractError(ValueError):
    """Invalid sources/configuration, not a failed emotion classification."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise TestContractError(message)


def finite(x: Any, name: str) -> float:
    require(isinstance(x, (float, int, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)),
            f'{name}: numeric scalar required')
    v = float(x)
    require(math.isfinite(v), f'{name}: finite value required')
    return v


def integer(x: Any, name: str) -> int:
    v = finite(x, name)
    require(v == math.floor(v), f'{name}: exact integer required')
    return int(v)


def boolean(x: Any, name: str) -> bool:
    require(isinstance(x, (bool, np.bool_)) or
            (isinstance(x, (int, float, np.integer, np.floating)) and x in (0, 1)),
            f'{name}: explicit bool or numeric 0/1 required')
    return bool(x)


def safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        require(math.isfinite(float(obj)), 'Refusing a nonfinite JSON output')
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def text_json(obj: Any, pretty: bool = False) -> str:
    return json.dumps(safe(obj), ensure_ascii=False, allow_nan=False, sort_keys=True,
                      indent=2 if pretty else None)


def object_hash(obj: Any) -> str:
    return hashlib.sha256(text_json(obj).encode('utf-8')).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def array_hash(x: np.ndarray) -> str:
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, f'Duplicate JSON key {k!r}: {path}')
            d[k] = v
        return d
    def bad(s):
        raise TestContractError(f'Nonfinite JSON constant {s}: {path}')
    result = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=pairs, parse_constant=bad)
    require(isinstance(result, dict), f'JSON object required: {path}')
    return result


def write_json(path: Path, obj: Any) -> None:
    """Exclusive output: never overwrite a prior run or user config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text_json(obj, True) + '\n')


def write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str] | None = None) -> None:
    keys = list(fields or dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: text_json(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def stable_seed(seed: int, *parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(text_json([int(seed), *parts]).encode()).digest()[:8], 'little')


def import_main(path: Path) -> Any:
    require(path.is_file(), f'main.py not found: {path}; use --main or put it beside the config')
    name = '_eav_test_main_' + digest(path)[:16]
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        require(spec is not None and spec.loader is not None, f'Cannot import main.py: {path}')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    require(getattr(mod, 'VERSION', None) == SUPPORTED_MAIN,
            f'Unsupported main version: {getattr(mod, "VERSION", None)}; expected {SUPPORTED_MAIN}')
    require(list(mod.EMOTIONS) == EMOTIONS and tuple(mod.MODALITIES) == MODALITIES, 'Class/modality order mismatch')
    for attr in ('Runtime', 'SourceReader', 'collect_descriptors', 'load_config', 'validate_descriptor',
                 'static_asset_audit', 'RunLog', 'find_executable', 'probe_video'):
        require(callable(getattr(mod, attr, None)), f'main API missing {attr}')
    return mod


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    family: str = ''
    severity: str = ''
    rank: int = 0
    affected: tuple[str, ...] = ()
    reference: str = 'reference'


def cases_for(suite: str, families: Sequence[str]) -> list[Case]:
    require(suite in ('reference', 'smoke', 'robustness'), 'Unknown raw suite')
    require(len(families) == len(set(families)) and set(families) <= set(FAMILIES), 'Unknown/duplicate corruption family')
    result = [Case('reference', 'reference')]
    if suite == 'robustness':
        if any(f in AUDIO_FAMILIES for f in families):
            result.append(Case('audio_matched_reference', 'matched_reference', affected=('audio',)))
        if any(f in VIDEO_FAMILIES[:-1] for f in families):
            result.append(Case('video_matched_reference', 'matched_reference', affected=('video',)))
        if 'video_compression' in families:
            result.append(Case('video_compression_reference', 'matched_reference', affected=('video',)))
        for family in families:
            modality = family.split('_', 1)[0]
            ref = ('audio_matched_reference' if modality == 'audio' else
                   'video_compression_reference' if family == 'video_compression' else
                   'video_matched_reference' if modality == 'video' else 'reference')
            for rank, sev in enumerate(SEVERITIES, 1):
                result.append(Case(f'{family}_{sev}', 'corruption', family, sev, rank, (modality,), ref))
        result += [Case('eeg_digital_zero', 'control', affected=('eeg',)),
                   Case('audio_digital_zero', 'control', affected=('audio',)),
                   Case('video_blackout', 'control', affected=('video',))]
    if suite != 'reference':
        # Every nonempty missing subset; all-missing is deliberately last.
        for n in (1, 2, 3):
            for missing in itertools.combinations(MODALITIES, n):
                result.append(Case('missing_' + '_'.join(missing), 'missing', affected=missing))
        result.append(Case('recovered', 'recovery'))
    require(len(result) == len({c.id for c in result}), 'Duplicate case ID')
    return result


def scenario_template() -> dict:
    return {'schema': SCENARIO_SCHEMA, 'families': list(DEFAULT_FAMILIES),
            'selection': 'balanced', 'trials_per_class': 1, 'seed': DEFAULT_SEED,
            'notes': ['No thresholds/weights are fitted by this file.',
                      'All supported optional families: ' + ', '.join(FAMILIES),
                      'Doses are fixed in the runner and exported in the plan.']}


def doses() -> dict:
    return {'audio_attenuation_db': [-6., -18., -30.],
            'injected_reference_to_noise_db': [20., 10., 0.],
            'eeg_line_hz': 50., 'eeg_slow_hz': .3, 'eeg_flat_channels': [3, 9, 18],
            'eeg_hold_seconds': [.25, 1., 2.5], 'eeg_local_channels': 6,
            'video_blur_sigma': [1., 2., 4.], 'video_brightness_factor': [.75, .50, .25],
            'video_center_occlusion_area_fraction': [.20, .40, .60],
            'video_h264_crf': [28, 36, 44], 'compression_reference_crf': 18,
            'audio_headroom': 'one gain per complete trial, shared by ALL selected audio variants and their matched reference',
            'audio_peak_limit': .999, 'video_filter_reference': 'RGB24 -> FFV1 bgr0 common decoded reference',
            'dose_scope': 'EEG/audio complete20s; Video individual5s; NOT true physiological/speech SNR'}


def group_key(d: Mapping) -> tuple[str, str]:
    ident = d.get('identity', {})
    return str(ident.get('subject', 'UNSPECIFIED')), str(ident.get('pair_key', d['window_id']))


def validate_pool(ds: Sequence[Mapping], main: Any, allow_test: bool) -> None:
    require(len(ds) > 0, 'No input windows')
    seen = set()
    for d in ds:
        main.validate_descriptor(d, live=False, allow_test=allow_test)
        require(d['window_id'] not in seen, 'Duplicate source window identity')
        seen.add(d['window_id'])
        require(d.get('split') != 'live', 'Use recorded files for OFFLINE tests, not live descriptors')
        for name in MODALITIES:
            item = d['modalities'][name]
            if item['present']:
                require('samples' not in item and bool(item.get('path')), 'Test runner source pool must use files')
    # Different labels within a declared trial cannot be treated as matched repeats.
    groups = defaultdict(list)
    for d in ds:
        groups[group_key(d)].append(d)
    for key, rows in groups.items():
        ys = {r.get('reference_label') for r in rows}
        require(len(ys) == 1, f'Trial has inconsistent reference labels: {key}')
        idx = [r.get('identity', {}).get('window_idx_0based') for r in rows]
        if all(i is not None for i in idx):
            require(len(idx) == len(set(idx)) and all(type(i) is int and 0 <= i < 4 for i in idx),
                    f'Trial has duplicate/invalid window indices: {key}')


def select_windows(ds: Sequence[dict], *, selection: str, trials_per_class: int,
                   seed: int, window_key: str | None, max_trials: int | None,
                   preflight: bool) -> tuple[list[dict], dict]:
    require(selection in ('balanced', 'all'), 'Selection must be balanced or all')
    require(1 <= trials_per_class <= 20, 'trials_per_class must be 1..20')
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in ds:
        groups[group_key(d)].append(d)
    if window_key:
        selected = [d for d in ds if d['window_id'] == window_key]
        require(len(selected) == 1, 'Requested window_key not unique/found')
        policy = 'EXPLICIT_ONE_WINDOW_NOT_BALANCED'
    elif preflight:
        selected = sorted(ds, key=lambda d: (group_key(d), d['window_id']))[:1]
        policy = 'DETERMINISTIC_ONE_WINDOW_PREFLIGHT'
    elif selection == 'all':
        selected = list(ds)
        policy = 'ALL_EXPLICIT_SOURCE_WINDOWS'
    else:
        subjects = {k[0] for k in groups}
        require(subjects in (VAL_SUBJECTS, TEST_SUBJECTS),
                'Balanced selection requires exactly the frozen six VAL or TEST subjects')
        selected = []
        for s in sorted(subjects):
            for y in range(5):
                keys = sorted(k for k, rows in groups.items() if k[0] == s and rows[0].get('reference_label') == y)
                require(len(keys) >= trials_per_class, f'Insufficient {s}/{EMOTIONS[y]} trials')
                # Stable hash order avoids dependence on dataframe order/RNG global state.
                keys.sort(key=lambda k: stable_seed(seed, 'trial_selection', *k))
                for key in keys[:trials_per_class]:
                    rows = groups[key]
                    require(len(rows) == 4 and sorted(r['identity']['window_idx_0based'] for r in rows) == [0, 1, 2, 3],
                            f'Balanced selection needs complete trials: {key}')
                    selected.extend(rows)
        policy = 'PREDECLARED_SUBJECT_CLASS_BALANCED_COMPLETE_TRIALS'
    if max_trials is not None:
        require(max_trials > 0, '--max-trials must be positive')
        keys = sorted({group_key(d) for d in selected})[:max_trials]
        selected = [d for d in selected if group_key(d) in set(keys)]
        policy += '_BOUNDED_DEBUG_SUBSET'
    selected = sorted(selected, key=lambda d: (group_key(d), d.get('identity', {}).get('window_idx_0based', 0), d['window_id']))
    require(selected, 'Empty selection')
    counts = Counter((group_key(d)[0], str(d.get('reference_label'))) for d in selected)
    return selected, {'policy': policy, 'seed': seed, 'source_windows': len(selected),
                      'complete_or_partial_trial_groups': len({group_key(d) for d in selected}),
                      'subject_class_window_counts': {f'{s}/{y}': n for (s, y), n in sorted(counts.items())},
                      'selection_uses_predictions_or_quality': False}


class IntegrityGuard:
    """Hash protected source/config/model files before and after, never write them."""
    def __init__(self):
        self.before: dict[str, dict] = {}

    def add(self, path: str | Path) -> None:
        p = Path(path).resolve()
        if str(p) in self.before:
            return
        require(p.is_file(), f'Protected file does not exist: {p}')
        st = p.stat()
        sha = digest(p)
        require((st.st_size, st.st_mtime_ns) == (p.stat().st_size, p.stat().st_mtime_ns), f'File changed while hashing: {p}')
        self.before[str(p)] = {'bytes': st.st_size, 'sha256': sha}

    def verify(self) -> dict:
        changes = []
        for value, old in self.before.items():
            p = Path(value)
            if not p.is_file() or p.stat().st_size != old['bytes'] or digest(p) != old['sha256']:
                changes.append(value)
        return {'status': 'PASS' if not changes else 'FAILED', 'files_checked': len(self.before),
                'changed_or_missing': changes, 'verification': 'SHA256_BYTES',
                'scope': 'enumerated files, not the whole Python environment or in-memory sensor-model state'}


def protect_assets(guard: IntegrityGuard, main: Any, config: dict, main_path: Path, fusion_only: bool) -> dict:
    guard.add(main_path); guard.add(config['_config_path']); guard.add(Path(__file__))
    if fusion_only:
        spec = config['modules']['fusion']; k = spec['kwargs']
        d = Path(k.get('assets_dir') or Path(spec['script']).parent)
        for p in (Path(spec['script']), Path(k.get('f4_checkpoint') or d/'best_validation_selected.pt'),
                  Path(k.get('af4b_checkpoint') or d/'best_validation_robustness.pt')):
            guard.add(p)
        mf = Path(k.get('manifest') or d/'fusion_deployment_manifest.json')
        if mf.is_file(): guard.add(mf)
        return {'status': 'FUSION_FILES_PRESENT', 'all_seven_assets_audited': False}
    audit = main.static_asset_audit(config)
    require(audit['status'] == 'PATHS_PASS', 'Missing assets. Run --check-assets first; no package installation is required.')
    for row in audit['assets']:
        paths = row.get('paths', [row['path']] if 'path' in row else [])
        for value in paths:
            p = Path(value)
            if p.is_file(): guard.add(p)
            elif p.is_dir():
                for f in p.rglob('*'):
                    if f.is_file() and f.suffix.lower() in ('.py', '.json', '.yml', '.yaml') and not ({'.git', '__pycache__'} & set(f.parts)):
                        guard.add(f)
    for spec in config['modules'].values():
        for k, v in spec['kwargs'].items():
            if k in main.PATH_ARGUMENTS and v and Path(v).is_file(): guard.add(v)
        d = Path(spec['kwargs'].get('assets_dir') or Path(spec['script']).parent)
        for p in d.glob('*.json'):
            guard.add(p)
    return audit


def checked_ffmpeg(cmd: list[str], timeout: float, command_log: list[dict]) -> None:
    start = time.perf_counter()
    record = {'command': cmd, 'timeout_seconds': timeout}
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='replace',
                           check=False, timeout=timeout, shell=False)
        record.update(returncode=r.returncode, stderr_tail=r.stderr[-8000:])
        require(r.returncode == 0, f'FFmpeg exited {r.returncode}: {r.stderr[-5000:]}')
    except subprocess.TimeoutExpired:
        record['timeout'] = True
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter() - start
        command_log.append(record)


def video_transform(main: Any, config: Mapping, source: Path, target: Path, case: str,
                    rank: int, timeout: float, command_log: list[dict]) -> dict:
    """Same geometry/frame count. Filter probes use lossless RGB; compression is explicit."""
    require(not target.exists() and source.resolve() != target.resolve(), 'Never overwrite video source/output')
    r = config['replay']; probe = main.probe_video(source, r['ffprobe'], count=True)
    require(probe['frames'] is not None and abs(probe['frames']/probe['fps']-5.) <= 1/probe['fps']+.001,
            'Video source is not a five-second window')
    base = [main.find_executable(r['ffmpeg']), '-nostdin', '-v', 'error', '-threads', '1',
            '-i', str(source), '-map', '0:v:0', '-an', '-filter_threads', '1']
    filt = 'format=rgb24'
    meta: dict[str, Any] = {'operation': case, 'rank': rank, 'input_probe': probe}
    if case == 'video_blur':
        sigma = (1., 2., 4.)[rank-1]; filt += f',gblur=sigma={sigma}:steps=1'; meta['sigma'] = sigma
    elif case == 'video_brightness':
        factor = (.75, .5, .25)[rank-1]
        # Multiplication remains inside the RGB value range: no problematic clip() expression.
        filt += f',lutrgb=r=val*{factor}:g=val*{factor}:b=val*{factor}'
        meta['brightness_factor'] = factor
    elif case == 'video_occlusion':
        frac = (.2, .4, .6)[rank-1]; w, h = probe['width'], probe['height']
        bw, bh = max(1, int(round(w*math.sqrt(frac)))), max(1, int(round(h*math.sqrt(frac))))
        x, y = (w-bw)//2, (h-bh)//2
        filt += f',drawbox=x={x}:y={y}:w={bw}:h={bh}:color=black:t=fill'
        meta.update(area_fraction_requested=frac, area_fraction_actual=bw*bh/(w*h),
                    box_xywh=[x, y, bw, bh], face_targeted=False)
    elif case == 'video_blackout':
        filt += ',lutrgb=r=0:g=0:b=0'
        meta['purpose'] = 'zero visual evidence control, not proof of camera failure'
    elif case not in ('video_matched_reference', 'video_compression', 'video_compression_reference'):
        raise TestContractError(f'Unknown video transform {case}')
    if case in ('video_compression', 'video_compression_reference'):
        require(probe['width'] % 2 == probe['height'] % 2 == 0,
                'H264 yuv420p probe needs even source dimensions; no implicit geometry padding')
        crf = 18 if case.endswith('reference') else (28, 36, 44)[rank-1]
        cmd = base + ['-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-threads', '1']
        meta.update(crf=crf, reference_crf=18, encoding='H264_yuv420p')
    else:
        cmd = base + ['-vf', filt, '-c:v', 'ffv1', '-level', '3', '-g', '1', '-pix_fmt', 'bgr0', '-threads', '1']
        meta['encoding'] = 'FFV1_bgr0'
    cmd += ['-fps_mode', 'passthrough', '-n', str(target)]
    checked_ffmpeg(cmd, timeout, command_log)
    out = main.probe_video(target, r['ffprobe'], count=True)
    require((out['width'], out['height'], out['frames']) == (probe['width'], probe['height'], probe['frames']) and
            abs(out['fps']-probe['fps']) < 1e-6, 'Video perturbation changed geometry/time sampling')
    meta.update(output_probe=out, source_sha256=digest(source), output_sha256=digest(target))
    return meta


def eeg_variants(x: np.ndarray, families: Sequence[str], seed: int, subject: str, pair: str) -> tuple[dict, dict]:
    a = np.asarray(x, np.float64)
    require(a.shape == (30, 10000) and np.isfinite(a).all(), 'EEG corruption needs full finite30x10000 trial')
    ac = np.sqrt(np.mean((a-a.mean(axis=1, keepdims=True))**2, axis=1))
    pos = ac[ac > 0]; require(len(pos) > 0, 'All-flat reference: relative EEG dose is undefined')
    scale = np.maximum(ac, np.median(pos))
    perm = np.random.default_rng(stable_seed(seed, subject, pair, 'eeg_channels')).permutation(30)
    local = perm[:6]; t = np.arange(10000)/500.
    phase = np.random.default_rng(stable_seed(seed, subject, pair, 'line_phase')).uniform(0, 2*np.pi)
    result = {}; metadata = {}
    for family in families:
        require(family in EEG_FAMILIES, 'Unexpected EEG family')
        noise = None
        if family in ('eeg_local_broadband', 'eeg_global_broadband'):
            noise = np.random.default_rng(stable_seed(seed, subject, pair, family)).standard_normal(a.shape)
            noise -= noise.mean(axis=1, keepdims=True)
            noise /= np.sqrt(np.mean(noise**2, axis=1, keepdims=True))
        elif family in ('eeg_global_line', 'eeg_global_slow_drift'):
            z = np.sin(2*np.pi*(50. if family.endswith('line') else .3)*t + (phase if family.endswith('line') else .7))
            z -= z.mean(); z /= np.sqrt(np.mean(z*z)); noise = np.broadcast_to(z, a.shape)
        for rank, sev in enumerate(SEVERITIES, 1):
            y = a.copy(); key = f'{family}_{sev}'
            m = {'generation_scope': '20s_trial_then_5s_slicing', 'seed': seed,
                 'source_trial_sha256': array_hash(a), 'reference_rms_rule': 'max(channel_AC_RMS,median_positive_AC_RMS)',
                 'true_neural_snr': False}
            if noise is not None:
                inds = local if family == 'eeg_local_broadband' else np.arange(30)
                ratio = (20., 10., 0.)[rank-1]
                y[inds] += noise[inds]*scale[inds, None]*10.**(-ratio/20.)
                m.update(reference_to_injected_db=ratio, target_channel_indices=inds.tolist())
            elif family == 'eeg_channel_flatline':
                inds = perm[:(3, 9, 18)[rank-1]]; y[inds] = 0.
                m['target_channel_indices'] = inds.tolist()
            elif family == 'eeg_intermittent_hold':
                width = int((.25, 1., 2.5)[rank-1]*500)
                for w in range(4):
                    start = w*2500+500; y[local, start:start+width] = a[local, start-1, None]
                m.update(target_channel_indices=local.tolist(), hold_seconds=width/500.)
            require(np.isfinite(y).all(), 'EEG transform overflow')
            result[key] = y; metadata[key] = m
    return result, metadata


def audio_variants(x: np.ndarray, families: Sequence[str], seed: int, subject: str, pair: str) -> tuple[dict, dict]:
    a = np.asarray(x, np.float64)
    require(a.shape == (320000,) and np.isfinite(a).all(), 'Audio corruption needs a full finite16k mono20s trial')
    rms = float(np.sqrt(np.mean(a*a)))
    require(rms > 0, 'Zero Audio reference: relative-noise dose undefined')
    waves = {'audio_matched_reference': a.copy()}; meta = {}
    for family in families:
        require(family in AUDIO_FAMILIES, 'Unexpected Audio family')
        if family != 'audio_attenuation':
            z = np.random.default_rng(stable_seed(seed, subject, pair, family)).standard_normal(a.size)
            if family == 'audio_pink_noise':
                spec = np.fft.rfft(z); freqs = np.fft.rfftfreq(a.size, 1/16000.)
                shape = np.zeros_like(freqs); shape[1:] = 1/np.sqrt(np.maximum(freqs[1:], 20.))
                z = np.fft.irfft(spec*shape, n=a.size)
            z -= z.mean(); z /= np.sqrt(np.mean(z*z))
        for rank, sev in enumerate(SEVERITIES, 1):
            key = f'{family}_{sev}'
            if family == 'audio_attenuation':
                db = (-6., -18., -30.)[rank-1]; waves[key] = a*10.**(db/20.)
                meta[key] = {'attenuation_db': db}
            else:
                db = (20., 10., 0.)[rank-1]; waves[key] = a+z*rms*10.**(-db/20.)
                meta[key] = {'reference_to_injected_db': db, 'true_speech_snr': False}
    peak = max(float(np.max(np.abs(v))) for v in waves.values())
    gain = min(1., .999/max(peak, 1e-15))
    common = {'generation_scope': '20s_trial_then_5s_slicing', 'common_headroom_gain': gain,
              'original_trial_rms': rms, 'seed': seed, 'requested_families': list(families),
              'no_per_condition_normalization_or_clipping': True,
              'reference_waveform_sha256': array_hash(a)}
    for key in waves:
        waves[key] = waves[key]*gain
        meta[key] = {**common, **meta.get(key, {})}
    return waves, meta


class PreparedTrial:
    """Private waveform/file snapshots. No predicted feature/probability caching."""
    def __init__(self, main: Any, config: dict, pool_rows: Sequence[dict], selected: Sequence[dict],
                 cases: Sequence[Case], work: Path, seed: int, timeout: float,
                 commands: list[dict], guard: IntegrityGuard, source_readers: Mapping | None = None):
        self.main, self.config, self.work = main, config, work
        self.timeout, self.commands = timeout, commands
        self.work.mkdir(parents=True, exist_ok=True)
        self.bases: dict[str, dict] = {}
        self.base_meta: dict[str, dict] = {}
        self.eeg_waves, self.eeg_meta, self.audio_waves, self.audio_meta = {}, {}, {}, {}
        self.prepared_ids: set[tuple[str, str]] = set()
        self.prep_seconds = 0.
        self.original_by_index = {}
        start = time.perf_counter()
        active_families = list(dict.fromkeys(c.family for c in cases if c.family))
        ef = [f for f in active_families if f in EEG_FAMILIES]
        af = [f for f in active_families if f in AUDIO_FAMILIES]
        needs_trial = bool(ef or af)
        rows = list(pool_rows) if needs_trial else list(selected)
        if needs_trial:
            require(len(rows) == 4 and sorted(r.get('identity', {}).get('window_idx_0based', -1) for r in rows) == [0, 1, 2, 3],
                    'EEG/Audio corruption requires the four original windows of each trial, even for --window-key')
        readers = source_readers if source_readers is not None else {m: main.SourceReader(config) for m in MODALITIES}
        audio_parts, eeg_parts = {}, {}
        selected_ids = {d['window_id'] for d in selected}
        for row in rows:
            idx = row.get('identity', {}).get('window_idx_0based', 0)
            wid = row['window_id']
            wd = work / ('w_' + object_hash(wid)[:12]); wd.mkdir()
            base = copy.deepcopy(row); base.pop('session_id', None)
            bm = {}
            for modality in MODALITIES:
                item = row['modalities'][modality]
                if not item['present']:
                    require(not (needs_trial and (modality == 'eeg' and ef or modality == 'audio' and af)),
                            f'{modality}: complete reference payload needed for relative doses')
                    continue
                if wid not in selected_ids and not ((modality == 'eeg' and ef) or (modality == 'audio' and af)):
                    continue
                guard.add(item['path'])
                if modality == 'eeg':
                    require(item.get('sample_rate') == 500, 'Source must explicitly declare EEG500Hz; shape alone does not establish rate')
                    require(item.get('channel_names') == list(main.CHANNELS),
                            'EAV tests require explicit30-channel E4 order; do not invent channel metadata')
                    x, meta = readers['eeg'].load_eeg(item)
                    require(x.shape == (30, 2500) and np.isfinite(x).all(), 'Invalid source EEG snapshot')
                    if ef: eeg_parts[idx] = np.asarray(x, np.float64)
                    target = wd/'reference_eeg.npy'
                    with target.open('xb') as f: np.save(f, x, allow_pickle=False)
                    base['modalities']['eeg'] = {'present': True, 'kind': 'eeg_window', 'path': str(target),
                        'sample_rate': 500, 'input_unit': item.get('input_unit', config['eeg']['input_unit']),
                        'channel_names': item['channel_names']}
                elif modality == 'audio':
                    import soundfile as sf
                    x, sr, meta = readers['audio'].load_audio(item)
                    require(np.isfinite(x).all(), 'Nonfinite source Audio snapshot')
                    if af:
                        require(sr == 16000 and x.shape == (80000,),
                                'Audio doses need A0-compatible16k mono windows; set raw EAV/A0 inputs explicitly')
                        audio_parts[idx] = np.asarray(x, np.float64)
                    target = wd/'reference_audio.wav'
                    require(not target.exists(), 'Audio snapshot already exists')
                    sf.write(str(target), x, sr, format='WAV', subtype='DOUBLE')
                    back, back_sr = sf.read(str(target), dtype='float64')
                    require(back_sr == sr and np.array_equal(back, np.asarray(x, np.float64)), 'Audio snapshot round-trip altered samples')
                    base['modalities']['audio'] = {'present': True, 'kind': 'audio_window', 'path': str(target),
                        'channel': item.get('channel', 'mean')}
                else:
                    vd = wd/'video_reference'; vd.mkdir()
                    target, meta = readers['video'].load_video(item, vd)
                    base['modalities']['video'] = {'present': True, 'kind': 'video_window', 'path': str(target)}
                bm[modality] = {**meta, 'private_snapshot': str(target), 'snapshot_sha256': digest(target)}
            if wid in selected_ids:
                self.bases[wid] = base; self.base_meta[wid] = bm
            self.original_by_index[idx] = row
        s, pair = group_key(rows[0])
        if ef:
            self.eeg_waves, self.eeg_meta = eeg_variants(np.concatenate([eeg_parts[i] for i in range(4)], axis=1), ef, seed, s, pair)
        if af:
            self.audio_waves, self.audio_meta = audio_variants(np.concatenate([audio_parts[i] for i in range(4)]), af, seed, s, pair)
        del readers
        self.prep_seconds = time.perf_counter() - start

    def build(self, source: dict, case: Case, session_id: str) -> tuple[dict, dict]:
        start = time.perf_counter(); wid = source['window_id']
        require((wid, case.id) not in self.prepared_ids, 'Scenario prepared twice; no implicit reuse/resume')
        self.prepared_ids.add((wid, case.id))
        d = copy.deepcopy(self.bases[wid]); idx = source.get('identity', {}).get('window_idx_0based', 0)
        # Opaque identifier: do not provide severity/class as a runtime predictor.
        d['window_id'] = 'CASE_' + object_hash([wid, case.id])[:24]
        d['session_id'] = session_id
        d.pop('source_manifest', None); d.pop('source_manifest_sha256', None)
        meta = {'case': asdict(case), 'source_window_id': wid, 'identity': source.get('identity'),
                'source_snapshot_metadata': self.base_meta[wid], 'original_samples_modified': False}
        if case.kind == 'missing':
            for modality in case.affected:
                d['modalities'][modality] = {'present': False, 'reason': 'EXPLICIT_SOFTWARE_LOSS_CONTROL'}
        elif case.id in self.eeg_waves or case.id == 'eeg_digital_zero':
            y = self.eeg_waves[case.id][:, idx*2500:(idx+1)*2500] if case.id in self.eeg_waves else np.zeros((30, 2500))
            path = self.work/('eeg_'+object_hash([wid, case.id])[:24]+'.npy')
            with path.open('xb') as f: np.save(f, y, allow_pickle=False)
            d['modalities']['eeg'] = {**d['modalities']['eeg'], 'path': str(path)}
            meta['perturbation'] = self.eeg_meta.get(case.id, {'digital_zero': True})
        elif case.id in self.audio_waves or case.id == 'audio_digital_zero':
            import soundfile as sf
            if case.id == 'audio_digital_zero':
                y, sr = np.zeros(80000), 16000
            else:
                y, sr = self.audio_waves[case.id][idx*80000:(idx+1)*80000], 16000
            path = self.work/('audio_'+object_hash([wid, case.id])[:24]+'.wav')
            require(not path.exists(), 'Audio variant already exists')
            sf.write(str(path), y, sr, format='WAV', subtype='DOUBLE')
            back, sr_back = sf.read(str(path), dtype='float64')
            require(sr_back == sr and np.array_equal(back, y), 'Audio variant round-trip changed samples')
            d['modalities']['audio'] = {**d['modalities']['audio'], 'path': str(path)}
            meta['perturbation'] = self.audio_meta.get(case.id, {'digital_zero': True})
        elif case.family in VIDEO_FAMILIES or case.id in ('video_matched_reference', 'video_compression_reference', 'video_blackout'):
            require(d['modalities']['video']['present'], 'Video transformation needs a real source clip')
            source_path = Path(self.bases[wid]['modalities']['video']['path'])
            # Filter variants consume the SAME decoded RGB reference when present.
            if case.family in VIDEO_FAMILIES[:-1]:
                key = self.work/('video_'+object_hash([wid, 'video_matched_reference'])[:24]+'.mkv')
                require(key.is_file(), 'Video matched reference must precede its filters')
                source_path = key
            suffix = '.mp4' if case.family == 'video_compression' or case.id == 'video_compression_reference' else '.mkv'
            target = self.work/('video_'+object_hash([wid, case.id])[:24]+suffix)
            meta['perturbation'] = video_transform(self.main, self.config, source_path, target,
                case.family or case.id, case.rank, self.timeout, self.commands)
            d['modalities']['video']['path'] = str(target)
        else:
            require(case.id in ('reference', 'recovered'), f'No preparation implementation for {case.id}')
        self.main.validate_descriptor(d, live=False, allow_test=source.get('split') == 'test')
        # Record the exact model-boundary input identities (not probabilities).
        meta['used_inputs'] = {m: {'present': bool(it['present']), **({'path': it['path'], 'sha256': digest(Path(it['path']))}
                              if it['present'] else {})} for m, it in d['modalities'].items()}
        meta['preparation_seconds'] = time.perf_counter() - start
        return d, meta


def prob(x: Any, name: str) -> np.ndarray:
    p = np.asarray(x, float)
    require(p.shape == (5,) and np.isfinite(p).all() and (p >= 0).all() and abs(float(p.sum())-1) <= 1e-5,
            f'{name}: invalid five-class probabilities')
    return p


def check_result(row: Mapping, descriptor: Mapping, case: Case, atol: float) -> list[dict]:
    checks = []
    def check(name, ok, details=None):
        checks.append({'name': name, 'passed': bool(ok), 'details': details})
    f = row['fusion']; fi = f['fusion_input']
    check('row_window_identity', row['window_id'] == descriptor['window_id'])
    check('row_session_identity', row['session_id'] == descriptor['session_id'])
    check('all_three_head_reports', set(row['heads']) == set(MODALITIES))
    quality, mask = [], []
    for m in MODALITIES:
        q = finite(fi['quality'][m], f'q_{m}'); av = boolean(fi['available'][m], f'{m}_available')
        p = prob(fi[m+'_probs'], m)
        check(m+'_quality_domain', 0 <= q <= 1)
        packet = row['heads'][m]['packet']
        check(m+'_packet_identity', packet['window_id'] == descriptor['window_id'] and
              packet['session_id'] == descriptor['session_id'] and packet['modality'] == m)
        check(m+'_packet_fusion_agree', av == bool(packet['available']) and abs(q-float(packet['quality'])) <= atol and
              np.max(np.abs(p-np.asarray(packet['probabilities']))) <= atol)
        check(m+'_no_algorithm_error', not row['heads'][m]['algorithm_error'])
        if not av:
            check(m+'_missing_canonical', q == 0 and np.max(np.abs(p-.2)) <= atol)
        if not descriptor['modalities'][m]['present']:
            check(m+'_explicit_loss_excluded', not av)
        quality.append(q); mask.append(av)
    if case.id in ('eeg_digital_zero', 'audio_digital_zero', 'video_blackout'):
        m = case.affected[0]
        check(case.id+'_no_evidence', not fi['available'][m])
    no = not any(mask); want = 'F4' if all(mask) and min(quality) >= .8 else 'AF4-B'
    check('route_matches_actual_input', f['route'] == want)
    check('all_missing_iff_no_decision', bool(f['no_decision']) == no)
    p = prob(f['final']['probabilities'], 'final')
    if no:
        check('no_decision_semantics', f['final']['emotion'] == 'NO_DECISION' and
              f['final']['confidence'] is None and np.max(np.abs(p-.2)) <= atol and f['modality_weights'] is None)
    else:
        check('final_class_and_confidence', f['final']['emotion'] == EMOTIONS[int(p.argmax())] and
              abs(float(f['final']['confidence'])-float(p.max())) <= atol)
    if want == 'AF4-B' and not no:
        weights = f['modality_weights']; effective = f['class_effective_weights']
        check('adaptive_weights_sum', abs(sum(weights.values())-1) <= atol)
        for j, m in enumerate(MODALITIES):
            if not mask[j]:
                check(m+'_adaptive_zero', abs(weights[m]) <= atol and all(abs(v) <= atol for v in effective[m].values()))
    elif not no:
        check('F4_has_no_invented_active_weights', f['modality_weights'] is None)
    return checks


def _prediction(p: np.ndarray, no: bool, route: str) -> dict:
    p = np.full(5, .2) if no else prob(p, 'comparison')
    return {'probabilities': p.tolist(), 'no_decision': bool(no), 'route': 'NO_DECISION' if no else route,
            'emotion': 'NO_DECISION' if no else EMOTIONS[int(p.argmax())],
            'confidence': None if no else float(p.max())}


def comparisons(runtime: Any, fused: Mapping, atol: float) -> dict:
    fi = fused['fusion_input']; fm, model = runtime.fm, runtime.fusion
    x = np.asarray([sum([list(fi[m+'_probs']) for m in MODALITIES], [])], np.float32)
    q = np.asarray([[fi['quality'][m] for m in MODALITIES]], np.float32)
    mask = np.asarray([[fi['available'][m] for m in MODALITIES]], np.float32)
    x, q, mask = fm.canonical_arrays(x, q, mask)
    no = bool(mask.sum() == 0)
    # One current forward for parity and F4 extraction. No mutation of learned modules.
    actual = model.predict_batch(x, q, mask)
    parity = float(np.max(np.abs(actual['router_output']['final_probs'][0]-np.asarray(fused['final']['probabilities']))))
    require(parity <= atol, f'Raw main output/recomputed fusion mismatch: {parity}')
    ablated = model.predict_batch(x, mask.copy(), mask)
    return {'fixed_f4': _prediction(actual['f4_probs'][0], no, 'F4'),
            'availability_only': _prediction(ablated['router_output']['final_probs'][0], no, str(ablated['route'][0])),
            'quality_aware': _prediction(np.asarray(fused['final']['probabilities']), no, fused['route']),
            'input_probability_sha256': array_hash(x), 'mask': mask[0].tolist(),
            'same_probability_and_availability_for_all_methods': True,
            'quality_ablation_applied_only_to_comparison': True,
            'main_recompute_max_abs_error': parity}


def sync_cuda() -> None:
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def reset_cuda_peaks() -> None:
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)


def resources() -> dict:
    result = {'process_peak_rss_bytes': None, 'cuda_allocator': [],
              'note': 'Process peak includes model loading; Torch peaks include comparisons, not non-Torch GPU allocations'}
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (k, ctypes.c_size_t) for k in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            c = Counters(); c.cb = ctypes.sizeof(c)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
                result['process_peak_rss_bytes'] = int(c.PeakWorkingSetSize)
        else:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['process_peak_rss_bytes'] = int(value*(1 if sys.platform == 'darwin' else 1024))
    except (OSError, ImportError, AttributeError) as exc:
        result['rss_measurement_error'] = str(exc)
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        for i in range(torch.cuda.device_count()):
            result['cuda_allocator'].append({'device': i, 'name': torch.cuda.get_device_name(i),
                'max_allocated_bytes': int(torch.cuda.max_memory_allocated(i)),
                'max_reserved_bytes': int(torch.cuda.max_memory_reserved(i))})
    return result


def environment() -> dict:
    import importlib.metadata
    values = {}
    for name in ('numpy', 'scipy', 'torch', 'pyprep', 'mne', 'soundfile', 'onnxruntime', 'funasr', 'decord', 'opencv-python'):
        try: values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: values[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': values}


def metrics(rows: Sequence[Mapping], method: str) -> dict:
    labelled = [r for r in rows if r.get('reference_label') is not None]
    cm = np.zeros((5, 6), np.int64); errors = abstained = answered = correct = 0
    for row in labelled:
        y = integer(row['reference_label'], 'reference_label'); require(0 <= y < 5, 'Invalid label')
        pred = row.get('comparisons', {}).get(method)
        if pred is None:
            errors += 1; j = 5
        elif pred['no_decision']:
            abstained += 1; j = 5
        else:
            j = EMOTIONS.index(pred['emotion']); answered += 1; correct += int(j == y)
        cm[y, j] += 1
    n = len(labelled); tp = np.diag(cm[:, :5]); fn = cm.sum(axis=1)-tp; fp = cm[:, :5].sum(axis=0)-tp
    f1 = np.divide(2*tp, 2*tp+fn+fp, out=np.zeros(5, float), where=2*tp+fn+fp > 0)
    # Metrics on answered samples exclude BOTH errors and deliberate abstentions.
    acm = cm[:, :5]; atp = np.diag(acm); afn = acm.sum(axis=1)-atp; afp = acm.sum(axis=0)-atp
    af1 = np.divide(2*atp, 2*atp+afn+afp, out=np.zeros(5, float), where=2*atp+afn+afp > 0)
    return {'attempted_labelled': n, 'answered': answered, 'software_errors': errors, 'no_decision': abstained,
            'coverage': answered/n if n else None, 'accuracy_all_attempts': correct/n if n else None,
            'accuracy_answered_only': correct/answered if answered else None,
            'macro_f1_all_attempts_errors_abstentions_as_FN': float(f1.mean()) if n else None,
            'macro_f1_answered_fixed_five_classes': float(af1.mean()) if answered else None,
            'missing_reference_classes': [EMOTIONS[i] for i in range(5) if cm[i].sum() == 0],
            'confusion_true5_pred5_plus_error_or_abstention': cm.tolist(),
            'classwise_f1_all_attempts': dict(zip(EMOTIONS, f1.tolist())) if n else {},
            'fixed_five_class_macro_average': True, 'rows_are_not_independent_subjects': True}


def latency_stats(values: Sequence[float]) -> dict:
    a = np.asarray(values, float)
    require(np.isfinite(a).all() and (a >= 0).all(), 'Bad latency value')
    return {'n': len(a), 'median_seconds': float(np.median(a)) if len(a) else None,
            'p95_seconds': float(np.percentile(a, 95)) if len(a) else None,
            'mean_seconds': float(a.mean()) if len(a) else None,
            'max_seconds': float(a.max()) if len(a) else None}


def aggregate(rows: Sequence[dict]) -> dict:
    primary = [r for r in rows if r['test_case']['kind'] in PRIMARY_KINDS]
    condition = []; subject = []; perclass = []; flat = []
    for field, dest in (('condition', condition), ('subject', subject)):
        groups = defaultdict(list)
        for row in rows:
            value = row['test_case']['id'] if field == 'condition' else str(row.get('identity', {}).get('subject', 'UNSPECIFIED'))
            if field == 'subject' and row['test_case']['kind'] not in PRIMARY_KINDS: continue
            groups[value].append(row)
        for key, group in sorted(groups.items()):
            for m in METHODS:
                mt = metrics(group, m)
                dest.append({field: key, 'method': m, 'attempted_rows': len(group),
                             'included_in_primary': all(r['test_case']['kind'] in PRIMARY_KINDS for r in group), **mt})
    for c in range(5):
        group = [r for r in primary if r.get('reference_label') == c]
        for m in METHODS:
            perclass.append({'reference_class': EMOTIONS[c], 'method': m, **metrics(group, m)})
    for r in rows:
        for m in METHODS:
            p = r.get('comparisons', {}).get(m)
            flat.append({'source_window_id': r['source_window_id'], 'window_id': r['window_id'],
                'subject': r.get('identity', {}).get('subject'), 'pair_key': r.get('identity', {}).get('pair_key'),
                'condition': r['test_case']['id'], 'kind': r['test_case']['kind'], 'method': m,
                'reference_label': r.get('reference_label'), 'status': r['status'],
                'emotion': p['emotion'] if p else None, 'confidence': p['confidence'] if p else None,
                'no_decision': p['no_decision'] if p else None, 'route': p['route'] if p else None,
                **{f'p_{label}': p['probabilities'][i] if p else None for i, label in enumerate(EMOTIONS)}})
    groups = defaultdict(dict)
    for r in rows:
        groups[r['source_window_id']][r['test_case']['id']] = r
    quality = []
    for family in FAMILIES:
        drops = []; fully_scored = []; nonincrease = []; strict = []; ties = []; states = []
        mod = family.split('_', 1)[0]
        for cases in groups.values():
            rr = [cases.get(f'{family}_{s}') for s in SEVERITIES]
            if not all(rr): continue
            ref = cases.get(rr[0]['test_case']['reference'])
            if ref is None: continue
            seq = [ref, *rr]
            if not all(r.get('fusion') for r in seq): continue
            vals = [float(r['fusion']['fusion_input']['quality'][mod]) for r in seq]
            av = [bool(r['fusion']['fusion_input']['available'][mod]) for r in seq]
            states.append(av)
            drops.append(vals[0]-vals[-1])
            # Missing controls are not treated as perfect continuous q detection.
            if all(av):
                fully_scored.append(vals); diffs = np.diff(vals)
                nonincrease.append(bool(np.all(diffs <= 1e-7)))
                strict.append(bool(np.all(diffs < -1e-7)))
                ties.append(bool(np.all(np.abs(diffs) <= 1e-7)))
        if drops:
            quality.append({'family': family, 'modality': mod, 'complete_numeric_pairs_including_unavailability': len(drops),
                'mean_reference_minus_severe_including_availability_zero': float(np.mean(drops)),
                'all_four_available_pairs': len(fully_scored),
                'nonincreasing_rate_among_all_available': float(np.mean(nonincrease)) if nonincrease else None,
                'strict_decrease_rate_among_all_available': float(np.mean(strict)) if strict else None,
                'all_tied_rate_among_all_available': float(np.mean(ties)) if ties else None,
                'available_rate_by_reference_mild_medium_severe': np.asarray(states, float).mean(axis=0).tolist(),
                'mean_q_when_all_four_available': np.asarray(fully_scored).mean(axis=0).tolist() if fully_scored else None,
                'is_hard_pass_criterion': False, 'quality_is_not_correctness_probability': True})
    timings = {}
    good = [r for r in rows if r.get('fusion')]
    timings['main_offline_pipeline'] = latency_stats([r['elapsed_seconds'] for r in good])
    timings['main_offline_pipeline_excluding_first'] = latency_stats([r['elapsed_seconds'] for r in good if not r.get('first_measured_inference')])
    timings['comparison_overhead'] = latency_stats([r.get('comparison_seconds', 0.) for r in good])
    for m in MODALITIES:
        timings[m+'_source_read_and_head_pair'] = latency_stats([r['heads'][m]['elapsed_seconds'] for r in good
                                                              if r.get('heads', {}).get(m, {}).get('source') is not None])
    timings['scope'] = 'OFFLINE_SERIAL, no5s_capture_wait; media preparation/model_loading/comparison measured separately'
    complete = [r for r in primary if r.get('comparisons') and all(m in r['comparisons'] for m in METHODS)]
    totals = {m: metrics(primary, m) for m in METHODS}
    paired = {m: metrics(complete, m) for m in METHODS}
    improvements = {}
    for m in METHODS[:-1]:
        a, b = paired['quality_aware'], paired[m]
        improvements[m] = {'delta_accuracy': a['accuracy_all_attempts']-b['accuracy_all_attempts'] if complete and a['attempted_labelled'] else None,
            'delta_macro_f1': a['macro_f1_all_attempts_errors_abstentions_as_FN']-b['macro_f1_all_attempts_errors_abstentions_as_FN']
            if complete and a['attempted_labelled'] else None, 'paired_rows': len(complete), 'significance_claimed': False}
    return {'primary_metrics_all_attempts': totals, 'paired_complete_primary_metrics': paired,
            'paired_complete_improvements': improvements, 'condition_rows': condition, 'subject_rows': subject,
            'per_class_rows': perclass, 'prediction_rows': flat, 'quality_response_rows': quality, 'latency': timings,
            'primary_scope': 'descriptive5s internal-development; excludes loss/digital-zero/blackout/recovery controls',
            'aggregate_condition_mix_is_suite_dependent': True}


def fusion_contract_tests(runtime: Any, main: Any, out: Path, seed: int, atol: float) -> dict:
    """Real frozen fusion, explicitly synthetic probability and clock fixtures."""
    system, fm = runtime.fusion, runtime.fm
    checks: dict[str, bool] = {}; details: dict[str, Any] = {}
    def check(name, value):
        checks[name] = bool(value)
    def rejected(name, fn):
        try: fn()
        except (ValueError, RuntimeError, TypeError): checks[name] = True
        else: checks[name] = False
    original = system.preflight()
    check('existing_fusion_preflight', original.get('status') == 'PASS')
    details['existing_fusion_preflight'] = original
    rng = np.random.default_rng(seed)
    x = rng.dirichlet(np.ones(5), size=(32, 3)).astype(np.float32).reshape(32, 15)
    ones = np.ones((32, 3), np.float32)
    initial = system.predict_batch(x, ones, ones)
    check('healthy_exactly_F4', np.array_equal(initial['f4_probs'], initial['router_output']['final_probs']))
    for bits in itertools.product((0, 1), repeat=3):
        mask = np.tile(np.array(bits, np.float32), (len(x), 1)); tag = ''.join(map(str, bits))
        q = .95*mask; a = system.predict_batch(x, q, mask)
        bad = x.copy()
        for j, active in enumerate(bits):
            if not active: bad[:, j*5:(j+1)*5] = np.nan
        b = system.predict_batch(bad, q, mask)
        err = float(np.max(np.abs(a['router_output']['final_probs']-b['router_output']['final_probs'])))
        check('mask_'+tag+'_old_or_nan_probabilities_irrelevant', err <= atol)
        check('mask_'+tag+'_no_decision', np.all((a['system_state'] == 'NO_DECISION') == (not any(bits))))
        check('mask_'+tag+'_normalized', np.allclose(a['router_output']['final_probs'].sum(axis=1), 1, atol=atol, rtol=0))
        details['mask_'+tag+'_max_abs_difference'] = err
    for j, m in enumerate(MODALITIES):
        q = ones.copy(); q[:, j] = np.float32(.8)
        check(m+'_threshold_equal_F4', np.all(system.predict_batch(x, q, ones)['route'] == 'F4'))
        q[:, j] = np.nextafter(np.float32(.8), np.float32(0))
        check(m+'_threshold_below_AF4B', np.all(system.predict_batch(x, q, ones)['route'] == 'AF4-B'))
    rejected('nonbinary_mask_rejected', lambda: system.predict_batch(x, ones, ones*.6))
    rejected('infinite_quality_rejected', lambda: system.predict_batch(x, ones*np.inf, ones))
    rejected('logits_rejected', lambda: system.predict_batch(x*2, ones, ones))
    # Timed interfaces are tested using a NEW fusion object and an explicit test clock.
    # No global timer, actual runtime configuration or learned state is patched.
    class Clock:
        def __init__(self): self.value = 100.
        def __call__(self): return self.value
    clock = Clock()
    timed = fm.FusionAF4C(f4_checkpoint=system.f4_checkpoint, af4b_checkpoint=system.af4b_checkpoint,
                         device=str(system.device), clock=clock)
    session, cid = 'CONTRACT_SYNTHETIC_SESSION', 'TEST_CLOCK_NOT_HARDWARE'
    def packets(wid, end):
        return {m: fm.make_modality_packet(m, session_id=session, window_id=wid, window_seconds=5.,
                class_order=EMOTIONS, probabilities=x[0,j*5:(j+1)*5], quality=.95, available=True,
                timing={'clock_id': cid, 'window_start_monotonic': end-5., 'window_end_monotonic': end,
                        'newest_sample_monotonic': end}, reason='SYNTHETIC_TEMPORAL_CONTRACT')
                for j,m in enumerate(MODALITIES)}
    ps = packets('W0', 100.)
    fresh = timed.predict_packets(ps, session_id=session, window_id='W0', window_end_monotonic=100., clock_id=cid, live=True)
    check('fresh_packets_accepted', not fresh['no_decision'])
    clock.value = 102.
    stale = timed.predict_packets(ps, session_id=session, window_id='W0', window_end_monotonic=100., clock_id=cid, live=True)
    check('stale_packets_all_rejected', stale['no_decision'] and stale['final']['confidence'] is None)
    rejected('wrong_clock_rejected', lambda: timed.predict_packets(ps, session_id=session, window_id='W0',
                window_end_monotonic=100., clock_id='OTHER_CLOCK', live=True))
    rejected('wrong_window_rejected', lambda: timed.predict_packets(ps, session_id=session, window_id='W1',
                window_end_monotonic=100., clock_id=cid, live=True))
    clock.value = 105.
    co = fm.FusionWindowCoordinator(timed, session_id=session, clock_id=cid, deadline_sec=.75)
    co.begin_window('W1', window_end_monotonic=105.)
    ps1 = packets('W1', 105.); co.submit(ps1['eeg']); co.submit(ps1['audio'])
    check('incomplete_window_waits', co.finalize('W1') is None)
    rejected('duplicate_modality_rejected', lambda: co.submit(ps1['audio']))
    clock.value = 105.76
    late = co.finalize('W1')
    check('deadline_missing_video', not late['video_available'] and not late['no_decision'])
    rejected('late_packet_rejected', lambda: co.submit(ps1['video']))
    rejected('duplicate_finalize_rejected', lambda: co.finalize('W1'))
    clock.value = 110.
    co.begin_window('W2', window_end_monotonic=110.)
    ps2 = packets('W2', 110.)
    for p in ps2.values(): co.submit(p)
    recovered = co.finalize('W2')
    check('new_full_window_recovers', not recovered['no_decision'] and all(recovered[m+'_available'] for m in MODALITIES))
    check('new_window_output_matches_fresh_probabilities', np.max(np.abs(np.asarray(recovered['final']['probabilities'])-
                                                                     np.asarray(fresh['final']['probabilities']))) <= atol)
    check('models_still_frozen', system.check_assets()['status'] == timed.check_assets()['status'] == 'PASS')
    # Exercise main.py's latest-state expiry without a physical sensor or patched clock.
    log = main.RunLog(out/'main_state_expiry_probe', runtime.config, 'live')
    try:
        r = copy.deepcopy(fresh)
        r['temporal']['valid_until_monotonic'] = time.monotonic()+60.
        log.result({'window_id': 'EXPLICIT_STATE_PROBE', 'fusion': r, 'status': 'COMPLETE'})
        check('latest_state_expiration_called', log.expire_current_state(now=time.monotonic()+61.))
        state = read_json(log.path/'latest_state.json')
        check('no_new_capture_clears_current_state', state['no_decision'] and state['final']['emotion'] == 'NO_DECISION')
    finally:
        log.finish(status='SYNTHETIC_STATE_PROBE_COMPLETE', extra={'real_capture_used': False})
    return {'status': 'PASS' if all(checks.values()) else 'FAILED', 'n_checks': len(checks),
            'checks': checks, 'details': details, 'synthetic_probabilities': True,
            'synthetic_clock': True, 'real_frozen_fusion_weights': True,
            'emotion_quality_models_executed': False, 'hardware_tested': False,
            'existing_preflight_checks_reported_separately': original.get('n_checks')}



# =============================================================================
# Quality-distribution logging (observational only; NO fitting / NO retuning)
# =============================================================================

QUALITY_LOG_SCHEMA = 'eav.system_test.quality_distribution.v1'


def _numeric_quality_features(obj: Any, prefix: str = '') -> dict[str, float]:
    """Flatten finite numeric scalars from a quality payload into path->value.

    Booleans are deliberately excluded: availability/state is logged separately.
    Lists/tuples are indexed so small diagnostic vectors remain recoverable.
    Strings, paths, None and non-numeric metadata remain in the lossless JSONL
    payload but are not copied to the numeric long-form CSV.
    """
    out: dict[str, float] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f'{prefix}.{key}' if prefix else str(key)
            out.update(_numeric_quality_features(value, child))
        return out
    if isinstance(obj, (list, tuple, np.ndarray)):
        # Avoid exploding very large tensors accidentally returned by a future module.
        flat = list(obj) if not isinstance(obj, np.ndarray) else obj.tolist()
        if len(flat) <= 128:
            for i, value in enumerate(flat):
                child = f'{prefix}[{i}]'
                out.update(_numeric_quality_features(value, child))
        return out
    if isinstance(obj, (bool, np.bool_)) or obj is None:
        return out
    if isinstance(obj, (int, float, np.integer, np.floating)):
        value = float(obj)
        if math.isfinite(value):
            out[prefix or 'value'] = value
    return out


def _modality_confidence_from_fusion_input(fused: Mapping, modality: str) -> float | None:
    try:
        p = np.asarray(fused['fusion_input'][modality + '_probs'], dtype=float)
        if p.shape == (5,) and np.isfinite(p).all():
            return float(p.max())
    except Exception:
        pass
    return None


def quality_samples_from_result(row: Mapping) -> list[dict]:
    """Create three modality quality records from one FULL raw runner row.

    This function must be called before compact(row), because compact output is
    intentionally small and does not duplicate the quality detector payloads.
    """
    fused = row.get('fusion')
    if not isinstance(fused, Mapping):
        return []
    fi = fused.get('fusion_input', {})
    if not isinstance(fi, Mapping):
        return []
    case = row.get('test_case', {}) or {}
    ident = row.get('identity', {}) or {}
    samples: list[dict] = []
    for modality in MODALITIES:
        head = (row.get('heads', {}) or {}).get(modality, {}) or {}
        payload = head.get('quality', {}) or {}
        if not isinstance(payload, Mapping):
            payload = {'_non_mapping_payload': payload}
        q = float(fi.get('quality', {}).get(modality, 0.0))
        available = bool(fi.get('available', {}).get(modality, False))
        # q has already passed check_result's domain contract in successful rows.
        sample = {
            'schema': QUALITY_LOG_SCHEMA,
            'source_window_id': row.get('source_window_id'),
            'runtime_window_id': row.get('window_id'),
            'subject': ident.get('subject'),
            'pair_key': ident.get('pair_key'),
            'window_idx_0based': ident.get('window_idx_0based'),
            'reference_label': row.get('reference_label'),
            'reference_emotion': (EMOTIONS[int(row['reference_label'])]
                                  if row.get('reference_label') is not None and 0 <= int(row['reference_label']) < 5
                                  else None),
            'condition': case.get('id'),
            'kind': case.get('kind'),
            'family': case.get('family') or '',
            'severity': case.get('severity') or '',
            'severity_rank': case.get('rank', 0),
            'affected_modalities': list(case.get('affected', ())),
            'matched_reference': case.get('reference'),
            'status': row.get('status'),
            'modality': modality,
            'available': available,
            'q': q,
            'route': fused.get('route'),
            'no_decision': bool(fused.get('no_decision', False)),
            'modality_confidence': _modality_confidence_from_fusion_input(fused, modality),
            'quality_state': payload.get('quality_state'),
            'quality_reason': payload.get('reason'),
            'quality_payload': safe(payload),
            'numeric_features': _numeric_quality_features(payload),
            'quality_is_not_emotion_correctness_probability': True,
            'distribution_model_fitted': False,
        }
        samples.append(sample)
    return samples


def _distribution_stats(values: Sequence[float]) -> dict:
    """Dependency-light descriptive shape statistics; no normality verdict."""
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {'n': 0}
    mean = float(a.mean())
    median = float(np.median(a))
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    centred = a - mean
    m2 = float(np.mean(centred**2))
    if m2 > 0:
        skew = float(np.mean(centred**3) / (m2**1.5))
        excess = float(np.mean(centred**4) / (m2**2) - 3.0)
    else:
        skew = 0.0
        excess = -3.0
    qs = np.quantile(a, [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
    return {
        'n': int(len(a)), 'mean': mean, 'median': median, 'std_sample': std,
        'min': float(qs[0]), 'q01': float(qs[1]), 'q05': float(qs[2]),
        'q10': float(qs[3]), 'q25': float(qs[4]), 'q50': float(qs[5]),
        'q75': float(qs[6]), 'q90': float(qs[7]), 'q95': float(qs[8]),
        'q99': float(qs[9]), 'max': float(qs[10]),
        'skewness_moment': skew, 'excess_kurtosis_moment': excess,
        'mean_minus_median': mean - median,
        'zero_rate': float(np.mean(np.isclose(a, 0.0, atol=1e-12))),
        'one_rate': float(np.mean(np.isclose(a, 1.0, atol=1e-12))),
    }


def _quality_summary_row(scope: str, modality: str, group: Sequence[Mapping], **tags: Any) -> dict:
    total = list(group)
    available = [r for r in total if bool(r['available'])]
    result = {
        'scope': scope, 'modality': modality, **tags,
        'n_total': len(total), 'n_available': len(available),
        'availability_rate': (len(available) / len(total)) if total else None,
        'q_all_including_unavailable_zero': _distribution_stats([r['q'] for r in total]),
        'q_available_only': _distribution_stats([r['q'] for r in available]),
        'window_rows_are_not_independent_subjects': True,
        'normality_claimed': False,
    }
    return result


def build_quality_trial_means(samples: Sequence[Mapping]) -> list[dict]:
    """Clean-reference trial aggregates: 4 windows/trial in the formal design.

    Availability remains separate. The available-only q mean is intended for
    later distribution auditing; it is NOT a replacement quality signal here.
    """
    clean = [r for r in samples if r.get('condition') == 'reference']
    groups: dict[tuple, list] = defaultdict(list)
    for r in clean:
        groups[(r.get('subject'), r.get('pair_key'), r.get('reference_label'), r.get('modality'))].append(r)
    rows = []
    for (subject, pair_key, label, modality), rr in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        av = [r for r in rr if bool(r['available'])]
        q_all = [float(r['q']) for r in rr]
        q_av = [float(r['q']) for r in av]
        rows.append({
            'subject': subject, 'pair_key': pair_key, 'reference_label': label,
            'reference_emotion': EMOTIONS[int(label)] if label is not None and 0 <= int(label) < 5 else None,
            'modality': modality, 'windows': len(rr), 'available_windows': len(av),
            'availability_rate': len(av)/len(rr) if rr else None,
            'mean_q_all_including_unavailable_zero': float(np.mean(q_all)) if q_all else None,
            'mean_q_available_only': float(np.mean(q_av)) if q_av else None,
            'min_q_available_only': float(np.min(q_av)) if q_av else None,
            'max_q_available_only': float(np.max(q_av)) if q_av else None,
            'distribution_model_fitted': False,
        })
    return rows


def export_quality_distribution(out: Path, samples: Sequence[Mapping], *, jsonl_exists: bool) -> dict:
    """Export distribution-ready observational logs without fitting any model."""
    ss = [safe(dict(s)) for s in samples]
    if not jsonl_exists:
        with (out/'quality_distribution_samples.jsonl').open('x', encoding='utf-8', newline='\n') as f:
            for row in ss:
                f.write(text_json(row)+'\n')

    sample_rows = []
    feature_rows = []
    for r in ss:
        sample_rows.append({
            'source_window_id': r['source_window_id'], 'runtime_window_id': r['runtime_window_id'],
            'subject': r['subject'], 'pair_key': r['pair_key'], 'window_idx_0based': r['window_idx_0based'],
            'condition': r['condition'], 'kind': r['kind'], 'family': r['family'], 'severity': r['severity'],
            'severity_rank': r['severity_rank'], 'modality': r['modality'],
            'reference_label': r['reference_label'], 'reference_emotion': r['reference_emotion'],
            'status': r['status'], 'available': r['available'], 'q': r['q'],
            'modality_confidence': r['modality_confidence'], 'quality_state': r['quality_state'],
            'quality_reason': r['quality_reason'], 'route': r['route'], 'no_decision': r['no_decision'],
        })
        for path, value in r.get('numeric_features', {}).items():
            feature_rows.append({
                'source_window_id': r['source_window_id'], 'subject': r['subject'], 'pair_key': r['pair_key'],
                'condition': r['condition'], 'family': r['family'], 'severity': r['severity'],
                'modality': r['modality'], 'available': r['available'], 'q': r['q'],
                'feature_path': path, 'value': value,
            })
    write_csv(out/'quality_distribution_samples.csv', sample_rows)
    write_csv(out/'quality_numeric_features.csv', feature_rows,
              fields=['source_window_id','subject','pair_key','condition','family','severity','modality','available','q','feature_path','value'])

    summary_rows = []
    for modality in MODALITIES:
        m = [r for r in ss if r['modality'] == modality]
        summary_rows.append(_quality_summary_row('all_logged_cases', modality, m,
            condition='', family='', severity=''))
        clean = [r for r in m if r['condition'] == 'reference']
        summary_rows.append(_quality_summary_row('clean_reference_windows', modality, clean,
            condition='reference', family='', severity=''))
        primary = [r for r in m if r['kind'] in PRIMARY_KINDS]
        summary_rows.append(_quality_summary_row('primary_reference_and_corruption', modality, primary,
            condition='', family='', severity=''))
        for condition in sorted({str(r['condition']) for r in m}):
            g = [r for r in m if str(r['condition']) == condition]
            family = g[0].get('family', '') if g else ''
            severity = g[0].get('severity', '') if g else ''
            summary_rows.append(_quality_summary_row('condition', modality, g,
                condition=condition, family=family, severity=severity))
    write_csv(out/'quality_distribution_summary.csv', summary_rows)

    trial_rows = build_quality_trial_means(ss)
    write_csv(out/'quality_distribution_trial_means.csv', trial_rows)
    trial_summary = []
    for modality in MODALITIES:
        rr = [r for r in trial_rows if r['modality'] == modality]
        vals = [r['mean_q_available_only'] for r in rr if r['mean_q_available_only'] is not None]
        trial_summary.append({'scope': 'clean_reference_trial_mean_available_q', 'modality': modality,
            'n_trials_total': len(rr), 'n_trials_with_available_q': len(vals),
            'all_four_windows_available_rate': float(np.mean([r['available_windows'] == r['windows'] for r in rr])) if rr else None,
            'stats': _distribution_stats(vals), 'normality_claimed': False,
            'trial_means_reduce_but_do_not_eliminate_subject_dependence': True})
    write_csv(out/'quality_distribution_trial_mean_summary.csv', trial_summary)

    clean_key = {}
    for modality in MODALITIES:
        rr = [r for r in ss if r['modality'] == modality and r['condition'] == 'reference']
        clean_key[modality] = _quality_summary_row('clean_reference_windows', modality, rr)
    audit = {
        'schema': QUALITY_LOG_SCHEMA,
        'status': 'OBSERVATIONAL_EXPORT_COMPLETE',
        'samples': len(ss), 'expected_modalities_per_successful_case': 3,
        'clean_reference': clean_key,
        'trial_mean_rows': len(trial_rows),
        'distribution_model_fitted': False,
        'gaussianity_or_other_distribution_family_selected': False,
        'router_threshold_retuned': False,
        'fusion_weights_changed': False,
        'quality_is_emotion_correctness_probability': False,
        'availability_is_modelled_separately_from_nonmissing_quality': True,
        'recommended_next_step': 'Audit clean VAL q/raw-feature distributions before fitting Gaussian/GMM/Beta/empirical candidates.',
        'limitations': [
            'Window rows from one subject/trial are correlated; do not treat 120 clean windows as 120 independent people.',
            'Canonical q values are already engineering-calibrated quality outputs; raw numeric detector features are exported separately.',
            'Moment skewness/kurtosis are descriptive and are not a normality hypothesis test.',
            'Corruption-condition distributions must not be mixed with clean distributions when estimating nominal sensor quality.',
        ],
    }
    write_json(out/'quality_distribution_audit.json', audit)
    return audit


def assert_formal_balanced_val(selected: Sequence[Mapping], cases: Sequence[Case], plan: Mapping,
                               options: Mapping, args: Any) -> None:
    """Hard guard for the predeclared formal development robustness run."""
    subjects = {group_key(d)[0] for d in selected}
    require(subjects == VAL_SUBJECTS, f'Formal balanced VAL requires exactly {sorted(VAL_SUBJECTS)}, got {sorted(subjects)}')
    require(len(selected) == 120, f'Formal balanced VAL requires 120 windows, got {len(selected)}')
    require(len({group_key(d) for d in selected}) == 30, 'Formal balanced VAL requires 30 complete trials')
    require(options['selection'] == 'balanced' and options['trials_per_class'] == 1,
            'Formal balanced VAL requires balanced selection and exactly 1 trial/class/subject')
    require(tuple(options['families']) == tuple(DEFAULT_FAMILIES),
            'Formal balanced VAL freezes the six default corruption families; use a separate exploratory run for --families all')
    require(options['seed'] == DEFAULT_SEED, f'Formal balanced VAL freezes seed={DEFAULT_SEED}')
    require(len(cases) == 32, f'Formal balanced VAL requires 32 conditions/window, got {len(cases)}')
    require(plan.get('planned_cases') == 3840, f'Formal balanced VAL requires 3840 case records, got {plan.get("planned_cases")}')
    require(not plan.get('test_data_requested'), 'Formal balanced VAL must not read TEST data')
    require(set(plan.get('split_scope', [])) <= {'val', 'validation'},
            f'Formal balanced VAL split scope is not VAL-only: {plan.get("split_scope")}')
    require(args.max_trials is None and args.window_key is None, 'Formal balanced VAL forbids debug truncation/window selection')
    require(not args.allow_test, 'Formal balanced VAL forbids --allow-test')
    require(not args.continue_on_error, 'Formal balanced VAL is fail-fast; do not use --continue-on-error')
    require(args.warmup == 0, 'Formal balanced VAL freezes --warmup 0')
    require(args.eeg_unit is not None, 'Formal balanced VAL requires an explicit --eeg-unit (current deployment: uV)')

def compact(row: Mapping) -> dict:
    keys = ('schema', 'window_id', 'source_window_id', 'identity', 'test_case', 'status', 'reference_label',
            'comparisons', 'elapsed_seconds', 'comparison_seconds', 'first_measured_inference',
            'hard_checks', 'error', 'case_metadata', 'recovery_probe')
    obj = {k: copy.deepcopy(row[k]) for k in keys if k in row}
    # Keep compact metadata, not complete source trace duplicated across all cases.
    if 'case_metadata' in obj:
        obj['case_metadata'] = {k: v for k,v in obj['case_metadata'].items() if k in ('preparation_seconds', 'perturbation')}
    if row.get('fusion'):
        f = row['fusion']
        obj['fusion'] = {k: copy.deepcopy(f[k]) for k in ('fusion_input', 'final', 'route', 'no_decision', 'modality_weights')}
    obj['heads'] = {m: {'elapsed_seconds': h['elapsed_seconds'], 'source': {} if h.get('source') is not None else None}
                    for m,h in row.get('heads', {}).items()}
    return obj


class Recorder:
    def __init__(self, out: Path):
        require(not out.exists(), f'Output exists; use a NEW directory: {out}')
        out.mkdir(parents=True)
        self.out = out; self.rows: list[dict] = []; self.errors: list[dict] = []; self.checks: list[dict] = []
        self.quality_samples: list[dict] = []
        self.stream = (out/'window_results.jsonl').open('x', encoding='utf-8', newline='\n')
        self.quality_stream = (out/'quality_distribution_samples.jsonl').open('x', encoding='utf-8', newline='\n')
        self.events = (out/'events.jsonl').open('x', encoding='utf-8', newline='\n')
        self.check_stream = (out/'contract_checks.jsonl').open('x', encoding='utf-8', newline='\n')

    def event(self, obj: Mapping):
        self.events.write(text_json(obj)+'\n'); self.events.flush()
        if obj.get('error'): self.errors.append(safe(obj))

    def record(self, row: dict):
        self.stream.write(text_json(row)+'\n'); self.stream.flush()
        for sample in quality_samples_from_result(row):
            self.quality_stream.write(text_json(sample)+'\n')
            self.quality_samples.append(sample)
        self.quality_stream.flush()
        small = compact(row); self.rows.append(small)
        for c in row.get('hard_checks', []):
            item = {'window_id': row['window_id'], 'source_window_id': row['source_window_id'],
                    'condition': row['test_case']['id'], **c}
            self.check_stream.write(text_json(item)+'\n'); self.checks.append(item)
        self.check_stream.flush()
        f = row.get('fusion')
        print(f"[{len(self.rows)}] {row['source_window_id']} / {row['test_case']['id']} | " +
              (f"{f['route']} | {f['final']['emotion']} | q="+
               ','.join(f"{m}:{f['fusion_input']['quality'][m]:.3f}" for m in MODALITIES) if f else row['status']), flush=True)

    def close(self):
        for f in (self.stream, self.quality_stream, self.events, self.check_stream):
            if not f.closed: f.close()


def export_reports(out: Path, rows: Sequence[dict]) -> dict:
    report = aggregate(rows)
    for filename, key in [('comparison_predictions.csv', 'prediction_rows'), ('condition_metrics.csv', 'condition_rows'),
                          ('subject_metrics.csv', 'subject_rows'), ('per_class_metrics.csv', 'per_class_rows'),
                          ('quality_response_summary.csv', 'quality_response_rows')]:
        write_csv(out/filename, report.pop(key))
    write_json(out/'latency_summary.json', report.pop('latency'))
    write_json(out/'comparison_summary.json', report)
    return report


def write_readable_summary(out: Path, summary: Mapping, report: Mapping) -> None:
    lines = ['# System test report', '', f"- Status: **{summary['status']}**",
             f"- Suite: `{summary['suite']}`", f"- Attempted / planned: {summary.get('attempted_cases', 0)} / {summary.get('planned_cases', 0)}",
             f"- Software/model errors: {summary.get('failed_cases', 0)}",
             f"- Failed hard checks: {summary.get('failed_hard_checks', 0)}", '',
             'Processing PASS is not an accuracy/quality superiority verdict.',
             'No training, threshold selection, hardware validation, or20s trial benchmark is performed.', '',
             '## Primary metrics (5s descriptive internal-development)', '',
             'Explicit loss, digital zero, blackout and identical-payload recovery controls are excluded here.',
             'Error and abstention rates remain visible; software errors are not converted into successful missing detections.', '',
             '| Method | Attempted labelled | Coverage | Accuracy, all attempts | Accuracy, answered | Macro-F1, all attempts |',
             '|---|---:|---:|---:|---:|---:|']
    def fmt(x): return 'N/A' if x is None else f'{x:.4f}'
    for m, v in report.get('primary_metrics_all_attempts', {}).items():
        lines.append(f"| {m} | {v['attempted_labelled']} | {fmt(v['coverage'])} | {fmt(v['accuracy_all_attempts'])} | "
                     f"{fmt(v['accuracy_answered_only'])} | {fmt(v['macro_f1_all_attempts_errors_abstentions_as_FN'])} |")
    lines += ['', 'See condition_metrics.csv before interpreting aggregate performance: the condition mix depends on the suite.',
              'Repeated windows/variants of one subject are not independent subjects. No significance test is asserted.',
              'The recovery case repeats the original payload with new inference; it is not a real disconnected-device experiment.',
              'Latency excludes capture waiting; raw preparation and baseline comparisons are separate.', '',
              '## Output files', '', '`window_results.jsonl`: current raw-module outputs and comparisons.',
              '`case_manifest.jsonl`: private inputs, transforms, seeds, hashes; files may have been cleaned up.',
              '`contract_checks.jsonl`: hard invariants; `quality_response_summary.csv`: measured, not required, trends.',
              '`quality_distribution_samples.jsonl/csv`: per-case, per-modality deployed quality payload/canonical q.',
              '`quality_numeric_features.csv`: long-form finite numeric fields from each quality detector payload.',
              '`quality_distribution_summary.csv` and `quality_distribution_trial_means.csv`: descriptive distribution audit; no model fitting.',
              '`protected_file_hashes.json` / `integrity_verification.json`: enumerated inputs/assets before and after.']
    with (out/'report.md').open('x', encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')


def defaults_root() -> Path:
    here = Path(__file__).resolve().parent
    # Known layouts only; do not discover newest projects or silently choose models.
    if (here/'main.py').is_file(): return here
    if here.name == 'testing' and (here.parent/'main.py').is_file(): return here.parent
    return Path.cwd().resolve()


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=VERSION)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--self-test', action='store_true')
    actions.add_argument('--init-scenarios', action='store_true')
    actions.add_argument('--check-assets', action='store_true')
    actions.add_argument('--dry-run', action='store_true')
    actions.add_argument('--preflight', action='store_true')
    actions.add_argument('--formal-balanced-val', action='store_true',
        help='Run the frozen 6-subject VAL robustness design: 30 complete trials, 120 windows, 32 conditions, 3840 records; no TEST.')
    actions.add_argument('--summarize', type=Path, help='Read a previous runner directory; no model or raw-source loading')
    p.add_argument('--suite', choices=('contracts', 'reference', 'smoke', 'robustness'))
    p.add_argument('--config', type=Path, help='Existing main configuration; default root/system_config.eav.json')
    p.add_argument('--main', type=Path, help='Explicit existing main.py; default config-directory/main.py')
    p.add_argument('--scenarios', type=Path, help='Optional test_scenarios.json; not a model/calibration file')
    p.add_argument('--selection', choices=('balanced', 'all'))
    p.add_argument('--trials-per-class', type=int)
    p.add_argument('--max-trials', type=int, help='Bounded debugging subset, NOT a balanced final set')
    p.add_argument('--families', help='Comma-separated supported families; use all to enable all13')
    p.add_argument('--seed', type=int)
    p.add_argument('--window-key')
    p.add_argument('--manifest'); p.add_argument('--stage0c-dir'); p.add_argument('--video-manifest')
    p.add_argument('--audio-manifest'); p.add_argument('--raw-root')
    p.add_argument('--confirm-legacy-speaking', action='store_true')
    p.add_argument('--allow-test', action='store_true')
    p.add_argument('--eeg-unit', choices=('V', 'mV', 'uV'))
    p.add_argument('--device', choices=('cpu', 'cuda'))
    p.add_argument('--output', type=Path, help='NEW output directory (or JSON path for --init-scenarios/--check-assets)')
    p.add_argument('--retain-inputs', action='store_true', help='Keep generated private media; may consume substantial disk space')
    p.add_argument('--continue-on-error', action='store_true', help='Record failed cases and continue; never report overall PASS')
    p.add_argument('--warmup', type=int, default=0, help='0..5 extra real-reference calls before measurement, recorded separately')
    p.add_argument('--atol', type=float, default=1e-5, help='Hard numerical contract tolerance, not an accuracy target')
    p.add_argument('--ffmpeg-timeout', type=float, default=180.)
    return p.parse_args(argv)


def resolve_test_options(args: Any) -> dict:
    template = scenario_template()
    if args.scenarios:
        supplied = read_json(args.scenarios.resolve())
        require(supplied.get('schema') == SCENARIO_SCHEMA, 'Wrong scenario schema')
        require(set(supplied) <= set(template), 'Unknown scenario fields')
        template.update(supplied)
    seed = args.seed if args.seed is not None else integer(template['seed'], 'seed')
    selection = args.selection or template['selection']
    n = args.trials_per_class if args.trials_per_class is not None else integer(template['trials_per_class'], 'trials_per_class')
    fams = template['families']
    if args.families is not None:
        fams = list(FAMILIES) if args.families == 'all' else [f.strip() for f in args.families.split(',') if f.strip()]
    require(isinstance(fams, list) and all(isinstance(f, str) for f in fams) and len(set(fams)) == len(fams) and set(fams) <= set(FAMILIES),
            'Invalid families')
    require(type(seed) is int and 0 <= seed < 2**63, 'seed must be a nonnegative63-bit integer')
    require(selection in ('balanced', 'all') and 1 <= n <= 20, 'Invalid selection/trials_per_class')
    require(math.isfinite(args.atol) and 0 < args.atol <= 1e-3, 'atol must be in (0,1e-3]')
    require(0 <= args.warmup <= 5, 'warmup must be 0..5')
    require(math.isfinite(args.ffmpeg_timeout) and 0 < args.ffmpeg_timeout <= 3600, 'Invalid FFmpeg timeout')
    return {'seed': seed, 'selection': selection, 'trials_per_class': n, 'families': fams}


def _source_plan(main: Any, config: dict, args: Any, options: dict, cases: Sequence[Case]) -> tuple[list, list, dict]:
    # Load full manifest pool without slicing so selection and dose preparation
    # see the same complete trial identities. Does NOT decode raw samples.
    desc_args = argparse.Namespace(allow_test=args.allow_test, window_key=None, limit=None, preflight=False)
    pool = main.collect_descriptors(config, desc_args)
    validate_pool(pool, main, args.allow_test)
    selected, audit = select_windows(pool, selection=options['selection'], trials_per_class=options['trials_per_class'],
        seed=options['seed'], window_key=args.window_key, max_trials=args.max_trials, preflight=args.preflight)
    families = {c.family for c in cases if c.family}
    if families:
        require(all(all(d['modalities'][m]['present'] for m in MODALITIES) for d in selected),
                'Robustness source reference must explicitly contain all three modalities; detector may later reject unusable evidence')
    allgroups = defaultdict(list)
    for d in pool: allgroups[group_key(d)].append(d)
    if families & set(EEG_FAMILIES+AUDIO_FAMILIES):
        for key in {group_key(d) for d in selected}:
            rr = allgroups[key]
            require(len(rr) == 4 and sorted(d.get('identity', {}).get('window_idx_0based', -1) for d in rr) == [0,1,2,3],
                    f'Perturbation dose requires full source trial {key}; do not supply only one JSONL window')
    plan = {'schema': SCHEMA, 'version': VERSION, 'suite': args.suite or 'reference', 'selection': audit,
            'conditions_per_window': len(cases), 'planned_cases': len(selected)*len(cases),
            'cases': [asdict(c) for c in cases], 'dose_policy': doses(),
            'source_window_ids': [d['window_id'] for d in selected],
            'source_descriptors_sha256': object_hash(selected), 'runtime_config_sha256': digest(Path(config['_config_path'])),
            'test_data_requested': any(d.get('split') == 'test' for d in selected),
            'split_scope': sorted({d.get('split', 'replay') for d in selected}),
            'selected_using_prediction_outcomes': False, 'raw_signals_decoded': False,
            'timing_mode': 'OFFLINE', 'new_test_population_claimed': False,
            'matched_controls_are_additional_records': True, 'cold_model_loading_in_latency': False}
    require(plan['planned_cases'] <= 200000, 'More than200000 cases: split into predeclared smaller runs')
    return pool, selected, plan


def _protect_sources(guard: IntegrityGuard, pool: Sequence[dict], selected: Sequence[dict], config: Mapping):
    # A referenced full trial may contribute RMS even when one window is selected.
    keys = {group_key(d) for d in selected}
    for d in pool:
        if group_key(d) not in keys: continue
        for item in d['modalities'].values():
            if item['present']: guard.add(item['path'])
        if d.get('source_manifest'): guard.add(d['source_manifest'])
    rp = config['replay']
    for key in ('manifest', 'video_manifest', 'audio_manifest'):
        if rp.get(key) and Path(rp[key]).is_file(): guard.add(rp[key])
    if rp.get('stage0c_dir'):
        p = Path(rp['stage0c_dir'])/'stage0c_summary.json'
        if p.is_file(): guard.add(p)


def run_raw(runtime: Any, main: Any, config: dict, pool: Sequence[dict], selected: Sequence[dict],
            cases: Sequence[Case], recorder: Recorder, guard: IntegrityGuard, args: Any, options: dict) -> dict:
    groups = defaultdict(list); allgroups = defaultdict(list)
    for d in selected: groups[group_key(d)].append(d)
    for d in pool: allgroups[group_key(d)].append(d)
    session = 'EAV_SUITE_' + object_hash([str(recorder.out), options['seed']])[:20]
    commands: list[dict] = []; prepared_seconds = 0.; warmup_rows = []; warmup_done = False
    n_completed = 0; expected = len(selected)*len(cases)
    started = time.perf_counter()
    manifest_stream = (recorder.out/'case_manifest.jsonl').open('x', encoding='utf-8', newline='\n')
    try:
        for key, sources in sorted(groups.items()):
            print(f'[PREPARE] {key[0]} / {key[1]} | selected windows={len(sources)}', flush=True)
            if args.retain_inputs:
                work = recorder.out/'cases'/object_hash(list(key))[:16]
                cm = contextlib.nullcontext(str(work))
            else:
                cm = tempfile.TemporaryDirectory(prefix='eav_system_cases_')
            with cm as temp:
                work = Path(temp)
                main.reject_test_path(work, args.allow_test)
                prep = PreparedTrial(main, config, allgroups[key], sources, cases, work, options['seed'],
                                     args.ffmpeg_timeout, commands, guard, source_readers=runtime.readers)
                prepared_seconds += prep.prep_seconds
                for source in sources:
                    reference_result = None
                    for case in cases:
                        d = None; meta = {}; inference_row = None
                        uid = 'CASE_' + object_hash([source['window_id'], case.id])[:24]
                        try:
                            d, meta = prep.build(source, case, session)
                            prepared_seconds += meta['preparation_seconds']
                            manifest_stream.write(text_json({'schema': SCHEMA, 'descriptor': d, 'metadata': meta,
                                'private_files_retained': args.retain_inputs})+'\n'); manifest_stream.flush()
                            if not warmup_done:
                                for i in range(args.warmup):
                                    wd = copy.deepcopy(d); wd['window_id'] = f'WARMUP_{i:02d}'
                                    sync_cuda(); t = time.perf_counter(); wr = runtime.process_offline(wd, allow_test=args.allow_test); sync_cuda()
                                    require(wr['status'] == 'COMPLETE', 'Warmup has model errors')
                                    warmup_rows.append({'index': i, 'source_window_id': source['window_id'],
                                                       'elapsed_seconds': time.perf_counter()-t, 'not_scored': True})
                                warmup_done = True; reset_cuda_peaks()
                            print(f'[RUN {n_completed+1}/{expected}] {source["window_id"]} / {case.id}', flush=True)
                            sync_cuda(); inference_row = runtime.process_offline(d, allow_test=args.allow_test); sync_cuda()
                            require(inference_row['status'] == 'COMPLETE', 'main.py returned an algorithm-error exclusion')
                            checks = check_result(inference_row, d, case, args.atol)
                            t = time.perf_counter(); comp = comparisons(runtime, inference_row['fusion'], args.atol); sync_cuda()
                            row = {**inference_row, 'schema': SCHEMA, 'source_window_id': source['window_id'],
                                'identity': source.get('identity', {}), 'test_case': asdict(case), 'case_metadata': meta,
                                'comparisons': comp, 'comparison_seconds': time.perf_counter()-t,
                                'first_measured_inference': n_completed == 0, 'hard_checks': checks,
                                'source_geometry': '5s_main_runtime', 'hardware_tested': False}
                            if case.id == 'reference': reference_result = compact(row)
                            if case.kind == 'recovery' and reference_result is not None:
                                f, old = row['fusion'], reference_result['fusion']
                                row['recovery_probe'] = {'scope': 'identical_input_replayed_after_explicit_loss; models_reexecuted',
                                    'same_availability_as_reference': f['fusion_input']['available'] == old['fusion_input']['available'],
                                    'max_probability_change': float(np.max(np.abs(np.asarray(f['final']['probabilities'])-
                                                                                  np.asarray(old['final']['probabilities'])))),
                                    'max_quality_change': max(abs(f['fusion_input']['quality'][m]-old['fusion_input']['quality'][m]) for m in MODALITIES),
                                    'restoration_to_F4_required': False, 'hardware_recovery_latency_measured': False}
                                # Input equality is hard; numeric recovery differences are visible observations.
                                refm = prep.base_meta[source['window_id']]
                                same = all((not d['modalities'][m]['present']) or
                                           meta['used_inputs'][m]['sha256'] == refm[m]['snapshot_sha256'] for m in MODALITIES)
                                row['hard_checks'].append({'name': 'recovery_uses_original_payload_bytes', 'passed': same, 'details': None})
                            recorder.record(row); n_completed += 1
                            failed = [c['name'] for c in row['hard_checks'] if not c['passed']]
                            if failed:
                                recorder.event({'event': 'HARD_CHECK_FAILURE', 'window_id': uid, 'error': ', '.join(failed)})
                                if not args.continue_on_error:
                                    raise AssertionError('Hard checks failed: '+', '.join(failed))
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as exc:
                            # Do not duplicate a successful output record that merely failed an assertion.
                            already = bool(recorder.rows and recorder.rows[-1]['window_id'] == uid)
                            err = {'event': 'CASE_FAILED', 'window_id': uid, 'source_window_id': source['window_id'],
                                   'condition': case.id, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
                            recorder.event(err)
                            if not already:
                                recorder.record({'schema': SCHEMA, 'window_id': uid, 'source_window_id': source['window_id'],
                                    'identity': source.get('identity', {}), 'reference_label': source.get('reference_label'),
                                    'test_case': asdict(case), 'status': 'ERROR', 'case_metadata': meta,
                                    'error': err['error'], 'fusion': None, 'comparisons': {}, 'hard_checks': []})
                                n_completed += 1
                            if not args.continue_on_error or 'out of memory' in str(exc).lower():
                                raise
                del prep
            gc.collect()
    finally:
        manifest_stream.close()
        write_json(recorder.out/'media_commands.json', {'commands': commands, 'source_clips_modified': False})
        write_json(recorder.out/'preparation_and_warmup.json', {'preparation_seconds': prepared_seconds,
                   'warmup': warmup_rows, 'wall_seconds_until_stop': time.perf_counter()-started,
                   'predicted_results_reused': False, 'retained_private_inputs': args.retain_inputs})
    return {'preparation_seconds': prepared_seconds, 'warmup_runs': len(warmup_rows),
            'raw_modality_calls': dict(runtime.calls), 'raw_models_requested': True,
            'input_data_not_claimed_unseen': True, 'offline_only': True}


def self_test() -> dict:
    checks: dict[str, bool] = {}
    def check(name, value):
        checks[name] = bool(value)
        require(checks[name], 'Self-test failed: '+name)
    def reject(name, fn):
        try: fn()
        except (TestContractError, ValueError, TypeError, FileExistsError): checks[name] = True
        else: raise AssertionError('Expected rejection: '+name)
    check('smoke_nine_conditions', len(cases_for('smoke', ())) == 9)
    check('default_robustness_32_conditions', len(cases_for('robustness', DEFAULT_FAMILIES)) == 32)
    cs = cases_for('robustness', FAMILIES)
    check('case_ids_unique', len(cs) == len({c.id for c in cs}))
    check('all_missing_before_recovery', cs[-2].affected == MODALITIES and cs[-1].kind == 'recovery')
    reject('unknown_family', lambda: cases_for('robustness', ['invented']))
    reject('duplicate_family', lambda: cases_for('robustness', ['video_blur', 'video_blur']))
    reject('boolean_string', lambda: boolean('false', 'available'))
    reject('fractional_mask', lambda: boolean(.5, 'available'))
    reject('nonfinite_quality', lambda: finite(float('inf'), 'quality'))
    reject('noninteger_index', lambda: integer(1.2, 'index'))
    reject('negative_prob', lambda: prob([-.1, .2, .3, .3, .3], 'p'))
    reject('logits', lambda: prob([1, 2, 3, 4, 5], 'p'))
    reject('nan_serialization', lambda: text_json({'x': np.nan}))
    pool = []
    for s in sorted(VAL_SUBJECTS):
        for label in range(5):
            for t in range(2):
                for w in range(4):
                    pool.append({'window_id': f'{s}_{label}_{t}_w{w}', 'reference_label': label,
                        'identity': {'subject': s, 'pair_key': f'p{label}_{t}', 'window_idx_0based': w}})
    sel, au = select_windows(pool, selection='balanced', trials_per_class=1, seed=77,
                             window_key=None, max_trials=None, preflight=False)
    rev, _ = select_windows(list(reversed(pool)), selection='balanced', trials_per_class=1, seed=77,
                           window_key=None, max_trials=None, preflight=False)
    check('balanced120_windows30trials', len(sel) == 120 and len({group_key(d) for d in sel}) == 30)
    check('selection_order_independent', sel == rev)
    check('balanced_each_subject_class_four_windows', set(au['subject_class_window_counts'].values()) == {4})
    one, oa = select_windows(pool, selection='balanced', trials_per_class=1, seed=77,
                            window_key=pool[13]['window_id'], max_trials=None, preflight=False)
    check('single_window_explicitly_not_balanced', len(one) == 1 and 'NOT_BALANCED' in oa['policy'])
    small, sa = select_windows(pool, selection='balanced', trials_per_class=1, seed=77,
                              window_key=None, max_trials=1, preflight=False)
    check('debug_limit_preserves_trial', len(small) == 4 and 'DEBUG' in sa['policy'])
    rng = np.random.default_rng(321)
    x = rng.normal(0, .4, 320000); initial = array_hash(x)
    aw, am = audio_variants(x, AUDIO_FAMILIES, 14, 'subject08', 'pair')
    aw2, _ = audio_variants(x, AUDIO_FAMILIES, 14, 'subject08', 'pair')
    check('audio_source_unchanged', array_hash(x) == initial)
    check('audio_deterministic', all(np.array_equal(aw[k], aw2[k]) for k in aw))
    check('headroom_shared_not_per_condition', len({v['common_headroom_gain'] for v in am.values()}) == 1)
    check('audio_no_new_clipping', max(np.max(np.abs(v)) for v in aw.values()) <= .999+1e-12)
    ref = aw['audio_matched_reference']; ref_rms = np.sqrt(np.mean(ref*ref))
    check('attenuation_exact_db', all(abs(20*np.log10(np.sqrt(np.mean(aw['audio_attenuation_'+s]**2))/ref_rms)-db) < 1e-9
                                    for s,db in zip(SEVERITIES, [-6, -18, -30])))
    for family in AUDIO_FAMILIES[1:]:
        check(family+'_actual_dose', all(abs(20*np.log10(ref_rms/np.sqrt(np.mean((aw[family+'_'+s]-ref)**2)))-db) < 1e-9
                                       for s,db in zip(SEVERITIES, [20,10,0])))
    eeg = rng.normal(size=(30, 10000)); eh = array_hash(eeg)
    ew, em = eeg_variants(eeg, EEG_FAMILIES, 14, 'subject08', 'pair')
    check('EEG_source_unchanged', array_hash(eeg) == eh)
    check('EEG_flat_nested', set(em['eeg_channel_flatline_mild']['target_channel_indices']).issubset(
                             em['eeg_channel_flatline_medium']['target_channel_indices']) and
                             set(em['eeg_channel_flatline_medium']['target_channel_indices']).issubset(
                             em['eeg_channel_flatline_severe']['target_channel_indices']))
    check('EEG_flat_channel_counts', [int(np.sum(np.all(ew['eeg_channel_flatline_'+s] == 0, axis=1))) for s in SEVERITIES] == [3,9,18])
    check('EEG_local_only_six_channels', sum(np.any(ew['eeg_local_broadband_severe'] != eeg, axis=1)) == 6)
    ids = em['eeg_intermittent_hold_severe']['target_channel_indices']
    check('EEG_hold_in_each_window', all(np.all(ew['eeg_intermittent_hold_severe'][ids,w*2500+500:w*2500+1750] ==
                                               eeg[ids,w*2500+499,None]) for w in range(4)))
    a = {'reference_label': 0, 'comparisons': {m: _prediction(np.array([1.,0,0,0,0]), False, 'F4') for m in METHODS}}
    b = {'reference_label': 1, 'comparisons': {m: _prediction(np.full(5,.2), True, 'NO_DECISION') for m in METHODS}}
    c = {'reference_label': 2, 'comparisons': {}}
    mt = metrics([a,b,c], 'quality_aware')
    check('errors_abstentions_separate', mt['software_errors'] == mt['no_decision'] == mt['answered'] == 1)
    check('coverage_denominator_all_attempts', abs(mt['coverage']-1/3) < 1e-12 and mt['accuracy_answered_only'] == 1.)
    check('no_label_metrics_none', metrics([], 'quality_aware')['accuracy_all_attempts'] is None)
    with tempfile.TemporaryDirectory(prefix='eav_runner_self_') as td:
        p = Path(td)/'配置.json'; write_json(p, {'你好': '世界'})
        check('unicode_json_roundtrip', read_json(p) == {'你好': '世界'})
        reject('no_overwrite', lambda: write_json(p, {}))
        d = Path(td)/'duplicate.json'; d.write_text('{"x":1,"x":2}')
        reject('duplicate_json', lambda: read_json(d))
        g = IntegrityGuard(); g.add(p); check('hash_guard_unchanged', g.verify()['status'] == 'PASS')
        p.write_text('{}'); check('hash_guard_detects_change', g.verify()['status'] == 'FAILED')
    # Quality-distribution logging is dependency-light and must not fit anything.
    qp = {'q_eeg': .7, 'available': True, 'nested': {'x': 1.25, 'flag': False}, 'vec': [1., 2.]}
    nf = _numeric_quality_features(qp)
    check('quality_numeric_flatten', nf['q_eeg'] == .7 and nf['nested.x'] == 1.25 and nf['vec[1]'] == 2. and 'nested.flag' not in nf)
    ds = _distribution_stats([.2, .4, .6, .8])
    check('quality_distribution_stats', ds['n'] == 4 and abs(ds['median']-.5) < 1e-12 and math.isfinite(ds['skewness_moment']))
    fake_samples=[]
    for m in MODALITIES:
        for w in range(4):
            fake_samples.append({'condition':'reference','subject':'subject08','pair_key':'p0','reference_label':0,
                                 'modality':m,'available':True,'q':.5+.1*w})
    tm=build_quality_trial_means(fake_samples)
    check('quality_trial_mean_rows', len(tm)==3 and all(r['windows']==4 for r in tm))
    return {'status': 'PASS', 'version': VERSION, 'n_checks': len(checks), 'checks': checks,
            'scope': 'NUMPY_LOGIC_AND_SYNTHETIC_WAVEFORMS_ONLY', 'trained_models_loaded': False,
            'raw_EAV_used': False, 'hardware_tested': False}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.formal_balanced_val:
        require(args.suite is None, '--formal-balanced-val selects robustness; do not also pass --suite')
        require(args.scenarios is None, '--formal-balanced-val uses the frozen default scenario design; no --scenarios override')
        require(args.selection in (None, 'balanced'), '--formal-balanced-val fixes --selection balanced')
        require(args.trials_per_class in (None, 1), '--formal-balanced-val fixes --trials-per-class 1')
        require(args.seed in (None, DEFAULT_SEED), f'--formal-balanced-val fixes --seed {DEFAULT_SEED}')
        require(args.families is None, '--formal-balanced-val fixes the six default robustness families')
        require(args.max_trials is None and args.window_key is None, '--formal-balanced-val forbids debug/window truncation')
        require(not args.allow_test, '--formal-balanced-val is VAL-only and forbids --allow-test')
        require(not args.continue_on_error, '--formal-balanced-val is fail-fast')
        require(args.warmup == 0, '--formal-balanced-val fixes --warmup 0')
        require(args.eeg_unit is not None, '--formal-balanced-val requires explicit --eeg-unit (use uV for the current EAV deployment)')
        args.suite = 'robustness'
        args.selection = 'balanced'
        args.trials_per_class = 1
        args.seed = DEFAULT_SEED
        args.families = ','.join(DEFAULT_FAMILIES)
    require(not (args.preflight and args.suite), '--preflight selects one reference case; do not combine with --suite')
    if args.self_test:
        print(text_json(self_test(), True)); return 0
    if args.init_scenarios:
        require(args.output is not None, '--init-scenarios needs --output testing/test_scenarios.json')
        write_json(args.output.resolve(), scenario_template()); print('SCENARIOS WRITTEN:', args.output.resolve()); return 0
    if args.summarize:
        src = args.summarize.resolve()
        require((src/'window_results.jsonl').is_file() and (src/'run_plan.json').is_file(), 'Expected a previous runner directory')
        plan = read_json(src/'run_plan.json'); require(plan.get('schema') == SCHEMA, 'Wrong source run schema')
        rows = []; full_rows = []; q_samples = []
        with (src/'window_results.jsonl').open(encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                if not line.strip(): continue
                row = json.loads(line)
                require(row.get('schema') == SCHEMA, f'Not a runner result on line {n}')
                full_rows.append(row); q_samples.extend(quality_samples_from_result(row)); rows.append(compact(row))
        require(rows, 'No raw result rows to summarize (contracts reports have no emotion metrics)')
        out = args.output.resolve() if args.output else src.parent/(src.name+'_summary_'+datetime.now().strftime('%H%M%S_%f'))
        require(not out.exists(), 'Summary output must be a NEW directory'); out.mkdir(parents=True)
        report = export_reports(out, rows)
        q_audit = export_quality_distribution(out, q_samples, jsonl_exists=False)
        summ = {'schema': SCHEMA, 'status': 'REPORT_ONLY_NO_INFERENCE', 'suite': plan['suite'],
                'source_run': str(src), 'source_results_sha256': digest(src/'window_results.jsonl'),
                'attempted_cases': len(rows), 'planned_cases': plan['planned_cases'],
                'failed_cases': sum(r['status'] == 'ERROR' for r in rows),
                'failed_hard_checks': sum(not c['passed'] for r in rows for c in r.get('hard_checks', [])),
                'quality_distribution_logging': q_audit}
        write_json(out/'run_summary.json', summ); write_readable_summary(out, summ, report)
        print('REPORT ONLY:', out/'run_summary.json'); return 0
    if not (args.suite or args.preflight or args.dry_run or args.check_assets):
        parse_args(['--help']); return 0
    options = resolve_test_options(args)
    root = defaults_root()
    config_path = (args.config or root/'system_config.eav.json').resolve()
    main_path = (args.main or config_path.parent/'main.py').resolve()
    require(config_path.is_file(), f'Configuration not found: {config_path}')
    app = import_main(main_path)
    args.error_policy = 'raise'  # Test failures must remain failures, not successful missing observations.
    config = app.load_config(config_path, args)
    if args.check_assets:
        audit = app.static_asset_audit(config)
        audit['unit_value_is_user_supplied_not_independently_verified'] = True
        print(text_json(audit, True))
        if args.output: write_json(args.output.resolve(), audit)
        return 0 if audit['status'] == 'PATHS_PASS' else 2
    suite = 'reference' if args.preflight else (args.suite or 'reference')
    if suite == 'contracts':
        pool, selected, cases = [], [], []
        plan = {'schema': SCHEMA, 'version': VERSION, 'suite': suite, 'planned_cases': 0,
                'scope': 'SYNTHETIC_PROBABILITIES_AND_CLOCK_REAL_FUSION_ONLY',
                'raw_EAV_read': False, 'source_models_used': False, 'hardware_tested': False}
    else:
        cases = cases_for(suite, options['families'])
        pool, selected, plan = _source_plan(app, config, args, options, cases)
        plan['suite'] = suite
        if args.formal_balanced_val:
            assert_formal_balanced_val(selected, cases, plan, options, args)
            plan['formal_balanced_val'] = True
            plan['formal_design'] = {'subjects': sorted(VAL_SUBJECTS), 'trials': 30, 'windows': 120,
                                     'conditions_per_window': 32, 'planned_records': 3840,
                                     'families': list(DEFAULT_FAMILIES), 'seed': DEFAULT_SEED}
    plan['options'] = {**options, 'atol': args.atol, 'continue_on_error': args.continue_on_error,
                       'warmup': args.warmup, 'retained_inputs': args.retain_inputs,
                       'ffmpeg_timeout': args.ffmpeg_timeout, 'preflight': args.preflight,
                       'formal_balanced_val': args.formal_balanced_val, 'quality_distribution_logging': True}
    print('='*100)
    print('EAV SYSTEM TEST RUNNER:', VERSION)
    print('Deployment configuration:', config_path)
    print('Existing main.py         :', main_path)
    print('Suite                    :', suite)
    if suite != 'contracts':
        print('Source windows           :', len(selected))
        print('Conditions per window    :', len(cases))
        print('Planned case records     :', plan['planned_cases'])
        print('Selection                :', plan['selection']['policy'])
        if args.formal_balanced_val:
            print('Formal VAL design        : 6 subjects / 30 trials / 120 windows / 32 conditions / 3840 records')
    print('Frozen tau               : 0.80; models/calibration NOT retrained')
    print('Input mode               : OFFLINE; NOT a live-device/20s benchmark')
    print('Quality distribution log : ENABLED; observational only, no distribution fit/threshold retuning')
    if args.dry_run:
        plan.update(status='PLAN_PASS', raw_signals_decoded=False, trained_models_loaded=False)
        if args.output: write_json(args.output.resolve(), plan)
        print(text_json(plan, True)); return 0
    out = args.output.resolve() if args.output else config_path.parent/'system_checks'/(
        'run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+suite)
    if suite != 'contracts':
        app.reject_test_path(out, args.allow_test)
    recorder = Recorder(out); guard = IntegrityGuard(); started = time.perf_counter()
    runtime = None; scope = {}; contracts = None; status = 'FAILED'; fatal = None; integrity = None; report = {}
    write_json(out/'run_plan.json', plan)
    write_json(out/'environment.json', environment())
    write_json(out/'run_config.json', {k:v for k,v in config.items() if not k.startswith('_')})
    write_json(out/'source_selection.json', {'windows': selected, 'selection': plan.get('selection'),
               'pool_size': len(pool), 'selected_before_model_loading': True})
    try:
        print('[AUDIT] Hashing configured source/model assets; no modification.', flush=True)
        asset_audit = protect_assets(guard, app, config, main_path, suite == 'contracts')
        write_json(out/'asset_audit.json', asset_audit)
        if args.scenarios: guard.add(args.scenarios)
        if suite != 'contracts': _protect_sources(guard, pool, selected, config)
        write_json(out/'protected_file_hashes.json', {'files': guard.before, 'hash': 'SHA256'})
        t = time.perf_counter(); runtime = app.Runtime(config, fusion_only=(suite == 'contracts')); sync_cuda()
        write_json(out/'loaded_asset_identity.json', {'modules': runtime.identities, 'model_load_seconds': time.perf_counter()-t})
        if suite == 'contracts':
            contracts = fusion_contract_tests(runtime, app, out, options['seed'], args.atol)
            write_json(out/'fusion_contract_results.json', contracts)
            status = 'PASS_CONTRACTS_ONLY' if contracts['status'] == 'PASS' else 'FAILED_CONTRACTS'
            scope = {'real_fusion_executed': True, 'raw_emotion_quality_modules_executed': False,
                     'synthetic_probability_input': True, 'hardware_tested': False}
        else:
            scope = run_raw(runtime, app, config, pool, selected, cases, recorder, guard, args, options)
            failed = any(r['status'] == 'ERROR' for r in recorder.rows)
            bad = any(not c['passed'] for c in recorder.checks)
            status = 'COMPLETE_WITH_ERRORS' if failed or bad else 'PASS_OFFLINE_PROCESSING'
        runtime.fusion.check_assets()
    except KeyboardInterrupt:
        status = 'INTERRUPTED'; fatal = 'KeyboardInterrupt'; recorder.event({'event': 'INTERRUPTED', 'error': fatal})
    except Exception as exc:
        status = 'FAILED'; fatal = f'{type(exc).__name__}: {exc}'
        recorder.event({'event': 'FATAL', 'error': fatal, 'traceback': traceback.format_exc()})
        print(f'\nSTOPPED: {fatal}', file=sys.stderr, flush=True)
    finally:
        recorder.close()
        try:
            integrity = guard.verify()
            if integrity['status'] != 'PASS': status = 'FAILED_INTEGRITY'
        except Exception as exc:
            integrity = {'status': 'VERIFICATION_ERROR', 'error': str(exc)}; status = 'FAILED_INTEGRITY'
        write_json(out/'integrity_verification.json', integrity)
        if not (out/'protected_file_hashes.json').exists():
            write_json(out/'protected_file_hashes.json', {'files': guard.before, 'collection_complete': False})
        report = export_reports(out, recorder.rows)
        q_audit = export_quality_distribution(out, recorder.quality_samples, jsonl_exists=True)
        used = resources(); write_json(out/'resource_usage.json', used)
        failed_cases = sum(r['status'] == 'ERROR' for r in recorder.rows)
        bad_checks = sum(not c['passed'] for c in recorder.checks)
        recovery = [r['recovery_probe'] for r in recorder.rows if 'recovery_probe' in r]
        summary = {'schema': SCHEMA, 'version': VERSION, 'status': status, 'suite': suite,
            'planned_cases': plan['planned_cases'], 'attempted_cases': len(recorder.rows),
            'completed_scope': (len(recorder.rows) == plan['planned_cases']) if suite != 'contracts' else bool(contracts and contracts['status'] == 'PASS'),
            'failed_cases': failed_cases, 'hard_checks_run': len(recorder.checks), 'failed_hard_checks': bad_checks,
            'fusion_contract_checks_run': contracts['n_checks'] if contracts else 0,
            'fusion_contract_checks_failed': [k for k,v in contracts['checks'].items() if not v] if contracts else [],
            'existing_fusion_preflight_checks': contracts.get('existing_preflight_checks_reported_separately', 0) if contracts else 0,
            'fatal_error': fatal, 'events_with_errors': len(recorder.errors),
            'elapsed_seconds': time.perf_counter()-started, 'scope': scope,
            'reference_recovery': {'n': len(recovery),
                'same_availability_count': sum(r['same_availability_as_reference'] for r in recovery),
                'max_probability_change': max((r['max_probability_change'] for r in recovery), default=None),
                'max_quality_change': max((r['max_quality_change'] for r in recovery), default=None),
                'not_a_hardware_recovery_test': True},
            'integrity': integrity, 'real_module_calls': dict(runtime.calls) if runtime else {},
            'frozen_model_training_performed': False, 'threshold_retuned': False, 'test_data_used': plan.get('test_data_requested', False),
            'original_20s_trial_accuracy_claimed': False, 'robot_actions_performed': False,
            'status_is_processing_verdict_not_accuracy_superiority': True,
            'formal_balanced_val': bool(args.formal_balanced_val),
            'quality_distribution_logging': q_audit,
            'primary_comparison': report['primary_metrics_all_attempts'],
            'limitations': ['5s internal-development replay, not unobserved-subject or20s benchmark evidence.',
                'Raw suite uses private waveform/codec copies; preparation overhead is not online throughput.',
                'No hardware drivers or live user trial are run.',
                'Partial central occlusion is not guaranteed to hide a face.',
                'Current dropout/zero controls do not cover every natural hardware fault.',
                'Mean metrics depend on condition mix; use condition/subject reports.',
                'Inference timeout/termination of native calls is NOT implemented here; Ctrl+C may wait for a native call.',
                'Peak CUDA usage is the PyTorch allocator, not total device memory.',
                'Source/model files are hashed; neural-module in-memory immutability is checked only for fusion.',
                'Quality distribution exports are descriptive only; no Gaussian/GMM/Beta family is selected and tau=0.80 is unchanged.']}
        write_json(out/'run_summary.json', summary)
        write_readable_summary(out, summary, report)
        print('\nSYSTEM TEST STATUS :', status)
        print('Attempted / planned:', len(recorder.rows), '/', plan['planned_cases'])
        print('Hard check failures:', bad_checks)
        if contracts:
            print('Fusion contract checks:', contracts['n_checks'], '| existing preflight checks:', contracts.get('existing_preflight_checks_reported_separately'))
        print('QUALITY DIST       :', out/'quality_distribution_audit.json')
        print('REPORT             :', out/'run_summary.json')
    return 0 if status in ('PASS_OFFLINE_PROCESSING', 'PASS_CONTRACTS_ONLY') else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (TestContractError, FileNotFoundError, FileExistsError) as exc:
        print(f'SYSTEM TEST INPUT ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
