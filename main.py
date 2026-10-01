#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EAV main entry: integrate the accepted KEEP_V1 distribution-quality policy.

No scoring formula, threshold, feature scale, class order or network weight is changed.
Raw readers and six original emotion/quality producers are retained. Every final
result uses QualityController.assess_window -> QualityFusionAdapter.predict_window.
No legacy router, Audio V2 fitting, auto-downloads or robot actuation.

The only TEST path is explicit, one-way evaluation after model/policy freeze.
--allow-test --split test --stage0c-raw may be used only for dry-run cohort
audit or one full replay benchmark. TEST cannot be subset with --limit,
--trial-key or --window-key, and must never be used for refitting/tuning.

--check-integration --cached-run DIR: reuse saved full reports; no upstream inference.
--preflight: one complete real source trial (four 5s windows when available).
--mode replay: real sources from existing config/manifest.
--mode live: same-host supplied windows, SHADOW quality only; no hardware guarantee.
--self-test / --dry-run / --check-assets / --check-models: explicit checks.

Pass --release with the frozen KEEP_V1 release candidate. Retain the old main.py
backup for historical hash-pinned experiments; historical plans are NOT rewritten.
Use locally trusted Python modules/checkpoints. SHA256 checks are not a sandbox.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

VERSION = "EAV-MAIN-INTEGRATION.2.1.0-FINAL-TEST"
CONFIG_SCHEMA = "eav.system.config.v1"
SOURCE_SCHEMA = "eav.system.source_window.v1"
EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
MODALITIES = ("eeg", "audio", "video")
CHANNELS = ["FP1","FP2","F7","F3","FZ","F4","F8","FC5","FC1","FC2","FC6",
            "T7","C3","CZ","C4","T8","CP5","CP1","CP2","CP6","P7","P3","PZ",
            "P4","P8","PO9","O1","OZ","O2","PO10"]
VAL_SUBJECTS = {"subject08","subject09","subject10","subject13","subject14","subject33"}
TEST_SUBJECTS = {"subject03","subject05","subject20","subject31","subject35","subject39"}
ALL_SUBJECTS = {f"subject{i:02d}" for i in range(1,43)}
SPLIT_SUBJECTS = {"val": VAL_SUBJECTS, "test": TEST_SUBJECTS,
                  "train": ALL_SUBJECTS - VAL_SUBJECTS - TEST_SUBJECTS}
CLASSES = {"eeg_emotion":"EEGEmotionE4", "eeg_quality":"EEGQualityV1",
           "audio_emotion":"AudioEmotionA1", "audio_quality":"AudioQualityV1",
           "video_emotion":"VideoEmotionV2B", "video_quality":"VideoQualityV1",
           "fusion":"FusionAF4C"}
PATH_ARGUMENTS = {"assets_dir","e4_run","checkpoint","normalization","frozen_report","contract",
    "a1_run","model_dir","head_checkpoint","feature_info","temp_dir","model_path","calibration",
    "dfew_checkpoint","yunet_model","project_root","dover_repo","dover_config","dover_checkpoint",
    "calibrator","f4_checkpoint","af4b_checkpoint","manifest"}
MEDIA_BASE_RE = r"(?P<instance>\d+)_Trial_(?P<trial>\d+)_(?P<task>Speaking|Listening)_(?P<emotion>Neutral|Sadness|Anger|Happiness|Calmness)"
AUDIO_MEDIA_RE = re.compile(rf"^{MEDIA_BASE_RE}(?:_aud)?\.wav$", re.I)
VIDEO_MEDIA_RE = re.compile(rf"^{MEDIA_BASE_RE}\.mp4$", re.I)


class ContractError(ValueError):
    """Wrong configuration/identity/geometry: never silently relabel inputs."""

class ModuleExecutionError(RuntimeError):
    """Sensor head / quality execution failure, not a diagnosed hardware fault."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ContractError(message)


def finite(value: Any, name: str) -> float:
    require(isinstance(value,(int,float,np.integer,np.floating)) and
            not isinstance(value,(bool,np.bool_)), f"{name}: numeric scalar required")
    v = float(value)
    require(math.isfinite(v), f"{name}: NaN/Inf not allowed")
    return v


def integer(value: Any, name: str) -> int:
    # CSVs carry strings. Booleans are never valid integer identifiers.
    require(not isinstance(value,(bool,np.bool_)), f"{name}: boolean not allowed")
    try:
        v = float(value)
    except (TypeError,ValueError) as exc:
        raise ContractError(f"{name}: integer required") from exc
    require(math.isfinite(v) and v == int(v), f"{name}: exact integer required")
    return int(v)


def boolean(value: Any, name: str) -> bool:
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,(int,np.integer)) and value in (0,1):
        return bool(value)
    raise ContractError(f"{name}: explicit bool or integer 0/1 required")


def ident(value: Any, name: str) -> str:
    require(isinstance(value,str) and value.strip() == value and 0 < len(value) <= 256,
            f"{name}: nonempty trimmed identifier (<=256 characters) required")
    return value


def safe_json(value: Any) -> Any:
    if isinstance(value,Mapping):
        return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,(int,np.integer)):
        return int(value)
    if isinstance(value,(float,np.floating)):
        # Diagnostics sometimes contain missing-face NaNs. Explicitly mark these
        # null in logs; numeric packets are validated separately and never patched.
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value,Path):
        return str(value)
    if value is None or isinstance(value,str):
        return value
    raise TypeError(f"Unserializable log value: {type(value).__name__}")


def parse_json_object(text: str) -> dict:
    def pairs(items):
        out = {}
        for key,value in items:
            require(key not in out, f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    def bad(s):
        raise ContractError(f"Nonstandard JSON constant: {s}")
    obj=json.loads(text,object_pairs_hook=pairs,parse_constant=bad)
    require(isinstance(obj,dict),"Expected a JSON object")
    return obj


def read_json(path: str | Path) -> dict:
    p=Path(path)
    require(p.is_file() and p.stat().st_size <= 8*1024*1024, f"Missing/oversized JSON: {p}")
    return parse_json_object(p.read_text(encoding="utf-8-sig"))


def write_new_json(path: str | Path, obj: Any) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation: do not overwrite a previous experiment/configuration.
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(safe_json(obj),f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write("\n")


def digest(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def signature(path: Path) -> tuple[int,int]:
    s=path.stat()
    return s.st_size,s.st_mtime_ns


def array_digest(x: np.ndarray) -> str:
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def resolve_path(value: str | Path, base: Path, maps: Sequence[Mapping] = ()) -> Path:
    require(isinstance(value,(str,Path)) and str(value).strip(),"Nonempty path required")
    s=str(value).replace("\\","/")
    for pair in sorted(maps,key=lambda x:len(x["old"]),reverse=True):
        old=str(pair["old"]).replace("\\","/").rstrip("/")
        if s.casefold()==old.casefold() or s.casefold().startswith(old.casefold()+"/"):
            s=str(pair["new"]).rstrip("/\\")+s[len(old):]
            break
    require(not (os.name!="nt" and PureWindowsPath(s).is_absolute()),
            f"Windows source path needs an explicit replay.path_maps entry on this OS: {value}")
    p=Path(s).expanduser()
    return (p if p.is_absolute() else base/p).resolve()


def normalized_subject(value: Any) -> str:
    m=re.fullmatch(r"(?:subject)?0*(\d+)",str(value).strip(),re.I)
    require(m is not None,"Invalid subject identity")
    return f"subject{int(m[1]):02d}"


def reject_test_path(path: str | Path, allow_test: bool) -> None:
    if not allow_test:
        require(not any(s=="test" or s.startswith("test_") for s in
                        re.split(r"[\\/]",str(path).lower())), f"Test source requires --allow-test: {path}")


def module_from_path(path: Path, role: str) -> Any:
    require(path.is_file(),f"Module missing: {path}")
    name="_eav_main_"+role+"_"+digest(path)[:12]
    if name in sys.modules:
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return mod


def default_config() -> dict:
    """No unknown physical unit is guessed. User edits paths, not source code."""
    modules={
      "eeg_emotion":{"script":"eeg/eeg_emotion_e4_deployment.py","kwargs":{"assets_dir":"eeg","device":"cuda"}},
      "eeg_quality":{"script":"eeg/eeg_quality_v1_deployment.py","kwargs":{"assets_dir":"eeg","candidate":"pyprep_physical"}},
      "audio_emotion":{"script":"audio/audio_emotion_a1_deployment.py","kwargs":{"assets_dir":"audio","device":"cuda"}},
      "audio_quality":{"script":"audio/audio_quality_v1_deployment.py","kwargs":{"assets_dir":"audio","candidate":"ovrl_level_cap"}},
      "video_emotion":{"script":"video/video_emotion_v2b_deployment.py","kwargs":{
        "dfew_checkpoint":"video/DFEW-set1-model.pth",
        "head_checkpoint":"video/best_v2b_frozen_dferclip_head_validation_selected.pt",
        "yunet_model":"video/face_detection_yunet_2023mar.onnx","device":"cuda","strict":True}},
      "video_quality":{"script":"video/video_quality_v1_deployment.py","kwargs":{
        "project_root":".","dover_repo":"video/DOVER_official","dover_config":"video/dover.yml",
        "dover_checkpoint":"video/DOVER.pth","calibrator":"video/video_quality_calibrator.joblib",
        "yunet_model":"video/face_detection_yunet_2023mar.onnx","device":"cuda","strict":True}},
      "fusion":{"script":"Fusion/融合层部署模块.py","kwargs":{"assets_dir":"Fusion","device":"cpu"}}}
    return {"schema":CONFIG_SCHEMA,"class_order":EMOTIONS,"window_seconds":5.0,"modules":modules,
      "eeg":{"training_unit":None,"input_unit":"training_native","unit_evidence":"USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED"},
      "runtime":{"error_policy":"raise","serialize_heads":True,"stale_after_sec":1.0,
          "alignment_tolerance_sec":0.10,"future_tolerance_sec":0.02,"deadline_sec":0.75,
          "max_pending":8,"poll_interval_sec":0.01,"shutdown_grace_sec":2.0},
      "replay":{"manifest":None,"stage0c_dir":None,"video_manifest":None,"audio_manifest":None,
          "raw_root":None,"project_root":".","split":"val","confirm_legacy_speaking":False,
          "path_maps":[],"audio_policy":"a0_pcm16","ffmpeg":"ffmpeg","ffprobe":"ffprobe"},
      "live":{"adapter":None,"settings":{},"clock_id":None},
      "output_root":"runs"}


def load_config(path: str | Path, args: Any = None) -> dict:
    cp=Path(path).expanduser().resolve(); c=read_json(cp)
    require(c.get("schema")==CONFIG_SCHEMA,"Wrong main config schema")
    require(c.get("class_order")==EMOTIONS and c.get("window_seconds")==5.,
            "Current integrated raw-data pipeline is five seconds / fixed five classes only")
    require(set(c.get("modules",{}))==set(CLASSES),"Configuration needs all seven deployment modules")
    base=cp.parent
    for role,spec in c["modules"].items():
        require(isinstance(spec,dict) and isinstance(spec.get("kwargs"),dict),f"Bad module config: {role}")
        spec["script"]=str(resolve_path(spec["script"],base))
        if "sha256" in spec:
            require(re.fullmatch(r"[0-9a-f]{64}",spec["sha256"]) is not None,"Invalid script hash")
        for k,v in list(spec["kwargs"].items()):
            if k in PATH_ARGUMENTS and v is not None:
                spec["kwargs"][k]=str(resolve_path(v,base))
    c.setdefault("runtime",{}); c.setdefault("replay",{}); c.setdefault("live",{})
    for section in ("runtime","replay","live","eeg"):
        template=default_config()[section]
        for k,v in template.items(): c[section].setdefault(k,copy.deepcopy(v))
    rt=c["runtime"]
    require(rt["error_policy"] in ("raise","exclude"),"runtime.error_policy must be raise/exclude")
    rt["serialize_heads"]=boolean(rt["serialize_heads"],"serialize_heads")
    for k in ("stale_after_sec","alignment_tolerance_sec","future_tolerance_sec","deadline_sec",
              "poll_interval_sec","shutdown_grace_sec"):
        rt[k]=finite(rt[k],"runtime."+k)
    require(0<rt["stale_after_sec"]<=30 and 0<=rt["deadline_sec"]<=rt["stale_after_sec"],"Invalid live deadline/staleness")
    require(0<=rt["alignment_tolerance_sec"]<=.5 and 0<=rt["future_tolerance_sec"]<=.1,"Invalid time tolerances")
    require(.001<=rt["poll_interval_sec"]<=.25 and 0<=rt["shutdown_grace_sec"]<=30,"Invalid poll/shutdown settings")
    rt["max_pending"]=integer(rt["max_pending"],"max_pending"); require(1<=rt["max_pending"]<=32,"max_pending must be 1..32")
    require(c["eeg"]["training_unit"] in (None,"V","mV","uV"),"Set a documented EEG training unit, not a guessed scale")
    require(c["eeg"]["input_unit"] in ("training_native","V","mV","uV"),"Unsupported input EEG unit")
    rp=c["replay"]
    require(rp["split"] in SPLIT_SUBJECTS,"replay.split must be train/val/test")
    require(rp["audio_policy"] in ("a0_pcm16","float_window"),"Unsupported raw Audio replay policy")
    require(isinstance(rp["path_maps"],list),"path_maps must be a list of old/new mappings")
    for m in rp["path_maps"]:
        require(isinstance(m,dict) and set(m)=={"old","new"} and m["old"] and m["new"],"Bad path map")
        m["new"]=str(resolve_path(m["new"],base))
    if args:
        for attr,key in (("manifest","manifest"),("stage0c_dir","stage0c_dir"),("video_manifest","video_manifest"),
                         ("audio_manifest","audio_manifest"),("raw_root","raw_root")):
            value=getattr(args,attr,None)
            if value: rp[key]=str(Path(value).expanduser().resolve())
        if getattr(args,"confirm_legacy_speaking",False):rp["confirm_legacy_speaking"]=True
        if getattr(args,"eeg_unit",None):c["eeg"]["training_unit"]=args.eeg_unit
        if getattr(args,"error_policy",None):rt["error_policy"]=args.error_policy
        if getattr(args,"split",None):rp["split"]=args.split
        if getattr(args,"stage0c_raw",False):
            # Final benchmark source rule: build the cohort from the Stage0C split
            # manifest and exact raw Audio/Video instance matching. Do not inherit
            # a VAL-only selected_sources/video/audio manifest from an old run.
            rp["manifest"]=None;rp["video_manifest"]=None;rp["audio_manifest"]=None
        if getattr(args,"device",None):
            for key in ("eeg_emotion","audio_emotion","video_emotion","video_quality","fusion"):
                c["modules"][key]["kwargs"]["device"]=args.device
    for key in ("manifest","stage0c_dir","video_manifest","audio_manifest","raw_root","project_root"):
        if rp.get(key):rp[key]=str(resolve_path(rp[key],base,rp["path_maps"]))
    for tool in ("ffmpeg","ffprobe"):
        v=rp[tool]
        if "/" in v or "\\" in v:rp[tool]=str(resolve_path(v,base))
    c["output_root"]=str(resolve_path(c.get("output_root","runs"),base))
    c["_config_path"]=str(cp); c["_base"]=str(base)
    return quality_config(c, args)


def static_asset_audit(c: Mapping) -> dict:
    """No constructors/imports. A path PASS is deliberately not inference PASS."""
    rows=[]
    def add(role,label,p,kind="file"):
        path=Path(p); exists=path.is_dir() if kind=="directory" else path.is_file()
        row={"role":role,"asset":label,"path":str(path),"kind":kind,"exists":exists}
        if exists and kind=="file":row["bytes"]=path.stat().st_size
        rows.append(row)
    for role,s in c["modules"].items():
        add(role,"runtime_script",s["script"])
        if Path(s["script"]).is_file():
            actual=digest(s["script"]); rows[-1]["sha256"]=actual
            if s.get("sha256"): require(actual==s["sha256"],f"Pinned script changed: {role}")
        k=s["kwargs"]; d=Path(k.get("assets_dir",Path(s["script"]).parent))
        def asset(key,default):return k.get(key) or d/default
        if role=="eeg_emotion":
            add(role,"E4",asset("checkpoint","best_multiscale_dilated_tcn_eegnet_validation_selected.pt"))
            add(role,"Train channel normalization",asset("normalization","train_channel_normalization.json"))
        elif role=="eeg_quality":add(role,"EQ2",asset("calibration","eeg_quality_calibration.json"))
        elif role=="audio_emotion":
            add(role,"A1",asset("head_checkpoint","best_a1_embedding_scores_validation_selected.pt"))
            md=Path(k.get("model_dir") or d/"emotion2vec_plus_large")
            for n in ("model.pt","config.yaml","configuration.json","tokens.txt"):add(role,"emotion2vec/"+n,md/n)
            opts=[Path(k[x]) for x in ("contract","feature_info") if k.get(x)]
            if not opts:opts=[d/"audio_emotion_a1_contract.json",d/"train_feature_cache_info.json",d/"val_feature_cache_info.json"]
            rows.append({"role":role,"asset":"nine-score order metadata","paths":[str(p) for p in opts],
                         "exists":any(p.is_file() for p in opts)})
        elif role=="audio_quality":
            add(role,"DNSMOS",asset("model_path","sig_bak_ovr.onnx"))
            add(role,"AQ3",asset("calibration","audio_quality_calibration.json"))
        elif role=="video_emotion":
            for key,n in (("dfew_checkpoint","DFEW-set1-model.pth"),("head_checkpoint","best_v2b_frozen_dferclip_head_validation_selected.pt"),
                          ("yunet_model","face_detection_yunet_2023mar.onnx")):add(role,key,asset(key,n))
        elif role=="video_quality":
            for key in ("dover_repo","dover_config","dover_checkpoint","calibrator","yunet_model"):
                require(k.get(key),f"video_quality.kwargs.{key} must be explicit")
                add(role,key,k[key],"directory" if key=="dover_repo" else "file")
            add(role,"DOVER package",Path(k["dover_repo"])/"dover","directory")
        elif role=="fusion":
            add(role,"F4",asset("f4_checkpoint","best_validation_selected.pt"))
            add(role,"AF4-B",asset("af4b_checkpoint","best_validation_robustness.pt"))
    return {"status":"PATHS_PASS" if all(r["exists"] for r in rows) else "MISSING_ASSETS",
            "model_inference_performed":False,"physical_units_confirmed":c["eeg"]["training_unit"] is not None,
            "assets":rows,"note":"File existence is not model identity or forward validation. No assets downloaded."}


# ------------------------ Explicit EAV manifest adapters -----------------------

def read_csv(path: Path, allow_test: bool) -> list[dict]:
    reject_test_path(path,allow_test)
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)),"Empty/duplicate CSV header")
        rows=list(reader)
    require(rows and len(rows)<=200000,f"Empty/oversized manifest: {path}")
    return rows


def validate_eav_row(r: Mapping, split: str, confirm_legacy: bool) -> tuple[str,str,str,int,int]:
    subject=normalized_subject(r.get("subject")); pair=ident(r.get("pair_key"),"pair_key")
    window=ident(r.get("window_key"),"window_key"); w=integer(r.get("window_idx_0based"),"window index")
    require(str(r.get("split","")).strip().lower()==split,"Mixed/wrong split in manifest")
    require(subject in SPLIT_SUBJECTS[split],f"Unexpected {split} subject: {subject}")
    if "task_condition" in r:
        require(str(r["task_condition"]).strip().lower()=="speaking","Listening/mixed task refused")
    else:require(confirm_legacy,"Legacy task-less manifest needs --confirm-legacy-speaking")
    label=integer(r.get("label_id"),"label_id")
    require(0<=label<5 and r.get("emotion")==EMOTIONS[label],"Emotion order/label mismatch")
    require(0<=w<4,"EAV window index must be 0..3")
    return subject,pair,window,w,label


def _media_match(path: Path):
    suffix=path.suffix.lower()
    if suffix==".wav":
        return AUDIO_MEDIA_RE.fullmatch(path.name)
    if suffix==".mp4":
        return VIDEO_MEDIA_RE.fullmatch(path.name)
    return None


def media_identity(path: Path) -> tuple[int,int,str,str]:
    m=_media_match(path)
    require(m is not None,f"Unexpected EAV media name (no fuzzy matching): {path.name}")
    emotion=next(e for e in EMOTIONS if e.casefold()==m.group("emotion").casefold())
    return int(m.group("instance")),int(m.group("trial")),m.group("task").casefold(),emotion


def media_by_instance(directory: Path, instance: int, suffix: str, emotion: str) -> Path:
    require(directory.is_dir(),f"Media directory missing: {directory}")
    hits=[]
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower()==suffix:
            m=_media_match(p)
            if m and int(m.group("instance"))==instance and m.group("task").casefold()=="speaking":hits.append(p)
    require(len(hits)==1,f"Expected ONE Speaking {suffix} for instance {instance} in {directory}, got {len(hits)}")
    p=hits[0].resolve(); require(media_identity(p)[3]==emotion,"Cross-modal emotion mismatch")
    return p


def build_eav_descriptors(c: Mapping, *, allow_test: bool = False) -> list[dict]:
    r=c["replay"]; split=r["split"]
    require(split!="test" or allow_test,"Test replay requires explicit --allow-test")
    require(r.get("stage0c_dir"),"Set replay.stage0c_dir or pass --stage0c-dir; no newest-directory discovery")
    stage=Path(r["stage0c_dir"]); summary=read_json(stage/"stage0c_summary.json")
    require(str(summary.get("status","")).upper()=="PASS","Stage0C must be PASS")
    if summary.get("task_condition"):
        require(str(summary["task_condition"]).lower()=="speaking","Listening Stage0C refused")
    confirm=boolean(r["confirm_legacy_speaking"],"confirm_legacy_speaking")
    manifest=stage/f"{split}_window_manifest.csv"
    rows=read_csv(manifest,allow_test)
    manifest_sha=digest(manifest)
    root=Path(r["project_root"]); maps=r["path_maps"]
    def load_index(path_value):
        if not path_value:return {}
        rr=read_csv(Path(path_value),allow_test); idx={}
        for row in rr:
            s,p,w,wi,y=validate_eav_row(row,split,confirm)
            key=(s,w); require(key not in idx,"Duplicate joined window")
            idx[key]=row
        return idx
    videos=load_index(r.get("video_manifest")); audios=load_index(r.get("audio_manifest"))
    groups={}; seen=set(); result=[]; subject_dirs={}
    for row in rows:
        s,p,win,w,y=validate_eav_row(row,split,confirm)
        require(win not in seen,"Duplicate Stage0C window ID"); seen.add(win)
        require(integer(row["eeg_fs_hz"],"EEG rate")==500 and
                integer(row["eeg_start_sample"],"EEG start")==w*2500 and
                integer(row["eeg_end_sample_exclusive"],"EEG end")== (w+1)*2500,"EEG window boundary mismatch")
        trial=integer(row["eeg_trial_idx_0based"],"EEG trial"); require(0<=trial<200,"Bad EEG trial index")
        eegpath=resolve_path(row["eeg_path"],root,maps)
        require(all(normalized_subject(v)==s for v in re.findall(r"subject0*(\d+)(?!\d)",str(eegpath),re.I)),"EEG path subject mismatch")
        vr=videos.get((s,win),row)
        if videos:require((s,win) in videos,"Video manifest misses a Stage0C window")
        for jr in [vr]+([audios[(s,win)]] if (s,win) in audios else []):
            ss,pp,ww,wi,yy=validate_eav_row(jr,split,confirm)
            require((ss,pp,ww,wi,yy)==(s,p,win,w,y),"Cross-modal identity mismatch")
        if vr.get("video_path"):
            vp=resolve_path(vr["video_path"],root,maps)
        else:
            require(r.get("raw_root"),"No video_path: supply --video-manifest or --raw-root for exact instance matching")
            if s not in subject_dirs:
                dd=[d for d in Path(r["raw_root"]).iterdir() if d.is_dir() and
                    re.fullmatch(r"subject\d+",d.name,re.I) and normalized_subject(d.name)==s]
                require(len(dd)==1,f"Expected unique raw subject directory: {s}"); subject_dirs[s]=dd[0]
            mi=re.fullmatch(r"subject0*\d+_instance(\d+)",p,re.I)
            require(mi is not None,"Raw-root fallback needs original subjectXX_instanceNNN pair_key")
            vp=media_by_instance(subject_dirs[s]/"Video",int(mi[1]),".mp4",EMOTIONS[y])
        inst,_,task,emo=media_identity(vp)
        require(task=="speaking" and emo==EMOTIONS[y],"Video task/label mismatch")
        # When present, the numeric instance in the pair ID must match media ID.
        mi=re.search(r"_instance(\d+)$",p,re.I)
        if mi:require(int(mi[1])==inst,"Video and pair_key have different EAV instance IDs")
        for key,expected in (("video_start_sec",5*w),("video_end_sec_exclusive",5*(w+1))):
            if vr.get(key) not in (None,""):require(float(vr[key])==expected,"Video boundaries disagree")
        ar=audios.get((s,win),row)
        if audios:require((s,win) in audios,"Audio manifest misses a Stage0C window")
        if ar.get("audio_processed_path"):
            ap=resolve_path(ar["audio_processed_path"],root,maps); audio_kind="audio_window"
        elif ar.get("audio_path") or ar.get("source_audio_path"):
            ap=resolve_path(ar.get("audio_path") or ar["source_audio_path"],root,maps); audio_kind="audio_trial"
        else:
            require(vp.parent.name.lower()=="video","Automatic paired WAV lookup requires a Video/ sibling Audio/ layout")
            ap=media_by_instance(vp.parent.parent/"Audio",inst,".wav",EMOTIONS[y]); audio_kind="audio_trial"
        if audio_kind=="audio_trial":
            ai,_,at,ae=media_identity(ap)
            require(ai==inst and at==task and ae==emo,"Audio/Video leading instance/task/emotion mismatch")
        g=groups.setdefault((s,p),[]); g.append((w,y,str(eegpath),trial,str(row["eeg_variable"])))
        for path in (eegpath,ap,vp):
            reject_test_path(path,allow_test)
            require(path.is_file(),f"Missing source file: {path}")
            ss={normalized_subject(v) for v in re.findall(r"subject0*(\d+)(?!\d)",str(path),re.I)}
            require(not ss or ss=={s},f"Cross-subject path: {path}")
        result.append({"schema":SOURCE_SCHEMA,"window_id":win,"window_seconds":5.,"task_condition":"Speaking",
          "split":split,"identity":{"subject":s,"pair_key":p,"window_idx_0based":w},"reference_label":y,
          "source_manifest":str(manifest),"source_manifest_sha256":manifest_sha,"task_evidence":"DECLARED" if "task_condition" in row else "CALLER_CONFIRMED_LEGACY",
          "modalities":{
            "eeg":{"present":True,"path":str(eegpath),"kind":"eav_mat","variable":row["eeg_variable"],
                   "trial_index":trial,"window_index":w,"sample_rate":500,"input_unit":"training_native","channel_names":CHANNELS},
            "audio":{"present":True,"path":str(ap),"kind":audio_kind,"window_index":w,"audio_policy":r["audio_policy"]},
            "video":{"present":True,"path":str(vp),"kind":"video_trial","window_index":w}}})
    for key,g in groups.items():
        require(sorted(v[0] for v in g)==[0,1,2,3] and len({v[1:] for v in g})==1,f"Incomplete/inconsistent trial: {key}")
    return result


def validate_descriptor(d: Mapping, *, live: bool, allow_test: bool = False) -> dict:
    require(isinstance(d,Mapping) and d.get("schema")==SOURCE_SCHEMA,"Wrong raw-window schema")
    x=dict(d); ident(x.get("window_id"),"window_id")
    require(x.get("window_seconds")==5.,"All modalities must describe one complete 5-second source window")
    require(str(x.get("task_condition","")).lower()=="speaking","This integration is Speaking-only")
    split=x.get("split","live" if live else "replay")
    require(split in ("train","val","test","replay","live","synthetic"),"Unknown source split")
    require(split!="test" or allow_test,"Test input requires --allow-test")
    if not allow_test:
        ss=x.get("identity",{}).get("subject")
        if ss:require(normalized_subject(ss) not in TEST_SUBJECTS,"Test subject requires --allow-test")
    require(isinstance(x.get("modalities"),Mapping) and set(x["modalities"])==set(MODALITIES),
            "Exactly three explicit raw modality descriptors required; use present=false for absence")
    for name,item in x["modalities"].items():
        require(isinstance(item,Mapping),f"Bad {name} descriptor")
        present=boolean(item.get("present"),name+".present")
        if present:
            has_array="samples" in item
            require(has_array != bool(item.get("path")),f"{name}: provide samples OR explicit path")
            if has_array:require(name in ("eeg","audio"),"Video uses a completed immutable clip path")
            if item.get("path"):reject_test_path(item["path"],allow_test)
            if live:
                tm=item.get("timing"); require(isinstance(tm,Mapping),f"{name}: actual capture timing required")
                require(tm.get("clock_id")==x.get("clock_id"),"Capture clock domains disagree")
                start=finite(tm.get("window_start_monotonic"),"capture start")
                end=finite(tm.get("window_end_monotonic"),"capture end")
                newest=finite(tm.get("newest_sample_monotonic"),"latest capture")
                require(end>start and start<=newest<=end+.02,"Invalid capture interval")
            if "continuous" in item:boolean(item["continuous"],"continuous")
            if "capture_ok" in item:boolean(item["capture_ok"],"capture_ok")
    if "reference_label" in x:
        require(integer(x["reference_label"],"reference_label") in range(5),"Invalid reference label")
    if live:
        ident(x.get("session_id"),"source session"); ident(x.get("clock_id"),"source clock")
        finite(x.get("window_end_monotonic"),"decision window end")
    return x


def load_jsonl_descriptors(path: Path, *, allow_test: bool = False) -> list[dict]:
    reject_test_path(path,allow_test); out=[]; seen=set()
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():continue
            require(len(line)<=2*1024*1024,"Oversized source JSONL line")
            d=parse_json_object(line)
            validate_descriptor(d,live=False,allow_test=allow_test)
            require(d["window_id"] not in seen,"Duplicate replay window ID");seen.add(d["window_id"])
            for name,item in d["modalities"].items():
                if boolean(item["present"],"present"):
                    require("samples" not in item,"Replay JSONL uses files, not JSON-encoded sample arrays")
                    p=resolve_path(item["path"],path.parent)
                    reject_test_path(p,allow_test); require(p.is_file(),f"Missing {name} source: {p}")
                    item["path"]=str(p)
            out.append(d)
    require(out,"No replay windows")
    return out


# ---------------------------- Raw window readers ------------------------------

def find_executable(value: str) -> str:
    found=shutil.which(value)
    if not found and Path(value).is_file():found=str(Path(value).resolve())
    require(found is not None,f"Executable not found: {value}. Set its absolute path; do not reinstall Python.")
    return str(found)


def checked_command(cmd: list[str], *, timeout: float = 90.) -> str:
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",errors="replace",
                        timeout=timeout,check=False,env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONUTF8":"1"})
    if proc.returncode:
        raise RuntimeError(f"Command exited {proc.returncode}: {cmd[0]}\n{proc.stderr[-5000:]}")
    return proc.stdout


def probe_video(path: Path, ffprobe: str, *, count: bool = False) -> dict:
    cmd=[find_executable(ffprobe),"-v","error","-select_streams","v:0"]
    if count:cmd += ["-count_frames"]
    cmd += ["-show_entries","stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
            "-of","json",str(path)]
    streams=json.loads(checked_command(cmd))["streams"]; require(len(streams)==1,"Expected one selected video stream")
    s=streams[0]
    def rate(v):
        a,b=v.split("/"); require(float(b)!=0,"Invalid video frame rate");return float(a)/float(b)
    fps=rate(s["avg_frame_rate"]); require(fps>0 and math.isfinite(fps),"Invalid average video FPS")
    frame_value=s.get("nb_read_frames") if count else s.get("nb_frames")
    n=int(frame_value) if frame_value not in (None,"N/A") else None
    return dict(fps=fps,nominal_fps=rate(s["r_frame_rate"]),frames=n,width=int(s["width"]),height=int(s["height"]))


class SourceReader:
    """One instance per modality worker: bounded one-MAT/one-Audio cache."""
    def __init__(self, c: Mapping):
        self.config=c; self.mat_cache=None; self.audio_cache=None

    def load_eeg(self, item: Mapping) -> tuple[np.ndarray,dict]:
        if "samples" in item:
            x=np.array(item["samples"],copy=True)
            return x,{"source":"adapter_array","sha256":array_digest(x)}
        p=Path(item["path"]); before=signature(p)
        kind=item.get("kind","eeg_window")
        if kind=="eav_mat":
            variable=item.get("variable","seg"); key=(str(p),variable,before)
            if self.mat_cache is None or self.mat_cache[0]!=key:
                from scipy.io import loadmat
                obj=loadmat(str(p),variable_names=[variable]); actual=variable
                if variable not in obj and variable in ("seg","seg1"):
                    actual="seg1" if variable=="seg" else "seg"; obj=loadmat(str(p),variable_names=[actual])
                require(actual in obj,f"Missing EEG variable: {variable}")
                a=np.asarray(obj[actual]);require(a.ndim==3 and set(a.shape)=={10000,30,200},"EAV MAT axes mismatch")
                a=a.transpose(a.shape.index(10000),a.shape.index(30),a.shape.index(200))
                self.mat_cache=(key,a,digest(p),actual)
            _,a,sha,actual=self.mat_cache
            trial=integer(item["trial_index"],"trial_index");w=integer(item["window_index"],"window_index")
            require(0<=trial<200 and 0<=w<4,"Bad MAT trial/window")
            x=np.array(a[w*2500:(w+1)*2500,:,trial].T,copy=True)
            meta={"source":str(p),"sha256":sha,"variable":actual,"trial_index":trial,"window_index":w}
        else:
            require(kind=="eeg_window" and p.suffix.lower() in (".npy",".npz"),"Window EEG must be numeric NPY/NPZ")
            if p.suffix.lower()==".npy":x=np.load(p,allow_pickle=False)
            else:
                with np.load(p,allow_pickle=False) as z:
                    require("eeg" in z,"NPZ needs eeg key");x=z["eeg"].copy()
                    if "sample_rate" in z:require(float(np.asarray(z["sample_rate"]).item())==500,"NPZ EEG rate mismatch")
                    if "channel_names" in z:require(z["channel_names"].astype(str).tolist()==item.get("channel_names"),"NPZ channel metadata disagrees")
            meta={"source":str(p),"sha256":digest(p)}
        require(signature(p)==before,"EEG file changed while reading")
        require(x.dtype.kind in "fiu" and x.shape==(30,2500),"Raw EEG must be exactly [30,2500]")
        meta["window_sha256"]=array_digest(x)
        return x,meta

    def load_audio(self, item: Mapping) -> tuple[np.ndarray,int,dict]:
        if "samples" in item:
            x=np.array(item["samples"],copy=True);sr=integer(item["sample_rate"],"sample_rate")
            return x,sr,{"source":"adapter_array","sha256":array_digest(x)}
        import soundfile as sf
        p=Path(item["path"]);before=signature(p);kind=item.get("kind","audio_window")
        if kind=="audio_trial":
            key=(str(p),before)
            if self.audio_cache is None or self.audio_cache[0]!=key:
                from scipy.signal import resample_poly
                x,sr=sf.read(str(p),dtype="float32",always_2d=False)
                require(x.ndim in (1,2) and np.isfinite(x).all(),"Invalid raw WAV")
                if x.ndim==2:x=x.mean(axis=1).astype(np.float32)
                if sr!=16000:
                    g=math.gcd(int(sr),16000);x=resample_poly(x,16000//g,int(sr)//g).astype(np.float32)
                require(x.shape[0]>=320000,"Raw WAV shorter than20s; no padding is invented")
                self.audio_cache=(key,x[:320000].copy(),digest(p),int(sr))
            _,full,sha,source_sr=self.audio_cache
            w=integer(item["window_index"],"window_index");require(0<=w<4,"Bad Audio trial window")
            x=full[w*80000:(w+1)*80000].copy()
            policy=item.get("audio_policy",self.config["replay"]["audio_policy"])
            require(policy in ("a0_pcm16","float_window"),"Unknown Audio replay quantization policy")
            if policy=="a0_pcm16":
                buf=io.BytesIO();sf.write(buf,x,16000,format="WAV",subtype="PCM_16");buf.seek(0)
                x,_=sf.read(buf,dtype="float64")
            sr=16000
            meta={"source":str(p),"sha256":sha,"source_sample_rate":source_sr,"replay_policy":policy,
                  "resample_scope":"whole source trial then first20s","window_index":w}
        else:
            require(kind=="audio_window","Unknown Audio source kind")
            x,sr=sf.read(str(p),dtype="float64",always_2d=False)
            require(len(x)==int(sr)*5,"Audio window is not exactly5s; no implicit trimming/padding")
            meta={"source":str(p),"sha256":digest(p),"source_sample_rate":int(sr),"replay_policy":"existing_window"}
        require(signature(p)==before,"Audio file changed while reading")
        meta["window_sha256"]=array_digest(x)
        return x,int(sr),meta

    def load_video(self, item: Mapping, destination: Path) -> tuple[Path,dict]:
        p=Path(item["path"]);before=signature(p);sha=digest(p)
        r=self.config["replay"];kind=item.get("kind","video_window")
        if kind=="video_trial":
            probe=probe_video(p,r["ffprobe"])
            require(abs(probe["fps"]-30)<1e-6 and abs(probe["nominal_fps"]-30)<1e-6,
                    "EAV trial trimming requires recorded 30-fps protocol; no time-warping")
            w=integer(item["window_index"],"window_index");require(0<=w<4,"Bad Video trial window")
            target=destination/"current_window.mkv"
            vf=f"trim=start_frame={w*150}:end_frame={(w+1)*150},setpts=PTS-STARTPTS"
            cmd=[find_executable(r["ffmpeg"]),"-nostdin","-v","error","-i",str(p),"-map","0:v:0",
                 "-vf",vf,"-an","-c:v","ffv1","-level","3","-g","1","-threads","1",
                 "-fps_mode","passthrough",str(target)]
            checked_command(cmd)
            mode="exact150frame_trim_then_lossless_FFV1_container; no H264 quality corruption"
        else:
            require(kind=="video_window","Unknown Video source kind")
            # Freeze one PRIVATE file for BOTH emotion and quality. A camera writer
            # must never continue modifying the original after committing its JSON.
            target=destination/("current_window"+p.suffix)
            shutil.copyfile(p,target);mode="byte_copy_of_completed_window"
        require(signature(p)==before and digest(p)==sha,"Video changed during snapshot")
        actual=probe_video(target,r["ffprobe"],count=True)
        require(actual["frames"] is not None and abs(actual["frames"]/actual["fps"]-5.)<=1/actual["fps"]+.001,
                "Video window duration/frame count not5s")
        if kind=="video_trial":require(actual["frames"]==150,"Incomplete EAV150frame trim")
        return target,{"source":str(p),"source_sha256":sha,"window_sha256":digest(target),"snapshot_policy":mode,"video":actual}


# --------------------- Seven-module runtime and head adapters ------------------

# -------------------------- Fixed KEEP_V1 integration --------------------------
# These are the reviewed bytes used by the accepted pipeline, not newly trained
# assets. Hash checks detect changes; they are not authenticity signatures.
QUALITY_PINS = {
    'quality_fusion_adapter.py': '878fe610877dc8c3a1e5431b4ffb9c5db4db65cf983493a03c327dfe6bf437b8',
    'quality_controller.py': '7feead21763a4f0b7e3c05ae1fda3d1d7bca200a97d67208ba23dcd6dac28e72',
    'distribution_quality.py': '23dd2d8074ce2adb548e3c8d71ac0ee2fb92b6cb88d24337aae600d2607c1d28',
    'distribution_quality_params.json': 'b9fd67533622c0de733ec7e5c0f4b5a80cb9819b74b07f57700b623c5a79d503',
}
POLICY_PIN = 'a75f43a6a3589247af565e457b6be89441cfaa5ba57e1228a3857bec7a46f088'
CHECKPOINT_PINS = {
    'f4': '3baee8225f871591f1f28da39717553c8d7a2fcc879dcd6de9297d485629d3d1',
    'af4b': 'de1056dd1a2e31778443a47cc8229911ce573f5be3d63886d10d22382dfb088b',
}
PROBABILITY_ATOL = 1e-6
PROBABILITY_RTOL = 1e-5
VALID_STATUSES = ('OK', 'NO_DECISION')


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def quality_config(c: dict, args: Any = None) -> dict:
    """Add only integration settings; never change scoring/fusion parameters."""
    base = Path(c['_base'])
    q = c.setdefault('quality_layer', {})
    allowed = {'variant', 'adapter_script', 'params', 'release'}
    require(set(q) <= allowed, 'Unknown quality_layer setting: ' + repr(set(q)-allowed))
    q.setdefault('variant', 'KEEP_V1')
    q.setdefault('adapter_script', str(Path(__file__).resolve().parent/'Quality'/'quality_fusion_adapter.py'))
    q.setdefault('params', str(Path(__file__).resolve().parent/'Quality'/'distribution_quality_params.json'))
    q.setdefault('release', None)
    for attr, key in [('quality_adapter', 'adapter_script'), ('quality_params', 'params'), ('release', 'release')]:
        v = getattr(args, attr, None)
        if v:
            q[key] = str(Path(v).expanduser().resolve())
    require(q['variant'] == 'KEEP_V1', 'This main.py integrates ONLY frozen KEEP_V1; no Audio V2 selection')
    for key in ('adapter_script', 'params', 'release'):
        if q[key]:
            q[key] = str(resolve_path(q[key], base))
    fs = c['modules']['fusion']
    if getattr(args, 'fusion_script', None):
        fs['script'] = str(Path(args.fusion_script).expanduser().resolve())
    if getattr(args, 'assets_dir', None):
        fs['kwargs']['assets_dir'] = str(Path(args.assets_dir).expanduser().resolve())
    if getattr(args, 'fusion_device', None):
        fs['kwargs']['device'] = args.fusion_device
    if getattr(args, 'unit_evidence', None):
        c['eeg']['unit_evidence'] = args.unit_evidence
    for key in ('tau', 'decision_window_seconds'):
        if key in fs['kwargs']:
            require(finite(fs['kwargs'][key], key) == (.80 if key == 'tau' else 5.),
                    'Original fusion metadata changed: '+key)
    return c


def verify_frozen_identity(actual: Mapping, expected: Mapping | None = None, *, weights: bool) -> None:
    require(actual.get('adapter_version') == 'EAV-QUALITY-FUSION-ADAPTER.1.1', 'Adapter version mismatch')
    require(actual.get('adapter_script_sha256') == QUALITY_PINS['quality_fusion_adapter.py'], 'Adapter bytes changed')
    require(actual.get('controller_script_sha256') == QUALITY_PINS['quality_controller.py'], 'Controller bytes changed')
    s = actual.get('scorer', {})
    require(s.get('runtime_script_sha256') == QUALITY_PINS['distribution_quality.py'], 'Scorer bytes changed')
    require(s.get('bundle_sha256') == QUALITY_PINS['distribution_quality_params.json'], 'Quality parameters changed')
    require(actual.get('policy_sha256') == POLICY_PIN, 'KEEP_V1 formula/threshold/policy identity changed')
    if expected is not None:
        for k in ('adapter_version', 'adapter_script_sha256', 'controller_script_sha256', 'policy_sha256'):
            require(actual.get(k) == expected.get(k), 'Release/source identity differs: '+k)
        for k in ('runtime_script_sha256', 'bundle_sha256'):
            require(s.get(k) == expected.get('scorer', {}).get(k), 'Release/source scorer differs: '+k)
    if weights:
        b = actual.get('fusion_backend') or {}
        require(b.get('real_checkpoint_loaded') is True, 'Expected real frozen checkpoints')
        for role, sha in CHECKPOINT_PINS.items():
            require(b.get(role, {}).get('sha256') == sha, 'Frozen checkpoint changed: '+role)
            if expected is not None:
                require(b[role]['sha256'] == expected.get('fusion_backend', {}).get(role, {}).get('sha256'),
                        'Release/source checkpoint mismatch: '+role)
        require(b.get('old_router_executed') is False and b.get('old_q_used') is False,
                'Legacy fusion router/quality cannot be used by the integrated path')


def load_quality_release(path: str | Path) -> dict:
    r = read_json(path)
    payload = {k:v for k,v in r.items() if k != 'release_sha256'}
    require(canonical_hash(payload) == r.get('release_sha256'), 'Release content/hash mismatch')
    require(r.get('schema') == 'eav.quality_layer.release_candidate.v1', 'Wrong quality release schema')
    require(r.get('variant') == 'KEEP_V1' and r.get('policy') is None and r.get('policy_sha256') is None,
            'Only KEEP_V1 with no new fitted Audio parameters is permitted')
    require(r.get('structural_search_closed') is True and r.get('parameters_may_not_change_in_frozen_evaluation') is True,
            'Release is not frozen')
    require(r.get('production_approved') is False and r.get('robot_actions_authorized') is False,
            'This entry does not accept/issue robot actuation approvals')
    verify_frozen_identity(r.get('frozen_dependency_identity', {}), weights=True)
    return r


def make_quality_adapter(config: Mapping, *, clock_id: str | None = None, allow_test: bool = False):
    q = config['quality_layer']
    require(q.get('release'), 'Pass --release quality_layer_release_candidate.json or set quality_layer.release')
    release = load_quality_release(q['release'])
    sp = Path(q['adapter_script'])
    for name in ('quality_fusion_adapter.py', 'quality_controller.py', 'distribution_quality.py'):
        p = sp if name == 'quality_fusion_adapter.py' else sp.with_name(name)
        require(p.is_file() and digest(p) == QUALITY_PINS[name], 'Reviewed quality module missing/changed: '+str(p))
    pp = Path(q['params'])
    require(pp.is_file() and digest(pp) == QUALITY_PINS['distribution_quality_params.json'],
            'Frozen quality parameter file missing/changed: '+str(pp))
    am = module_from_path(sp, 'distribution_quality_adapter')
    fs = config['modules']['fusion']; k = fs['kwargs']; rt = config['runtime']
    require(Path(fs['script']).is_file(), 'Fusion script not found: '+fs['script'])
    if fs.get('sha256'):
        require(digest(fs['script']) == fs['sha256'], 'Configured fusion script pin mismatch')
    expected_fusion = release['frozen_dependency_identity']['fusion_backend']
    require(digest(fs['script']) == expected_fusion['runtime_script_sha256'], 'Fusion source bytes differ from accepted release')
    adapter = am.QualityFusionAdapter(params_path=pp, root=Path(config['_base']),
        fusion_script=fs['script'], fusion_script_sha256=expected_fusion['runtime_script_sha256'],
        assets_dir=k.get('assets_dir'), f4_checkpoint=k.get('f4_checkpoint'), af4b_checkpoint=k.get('af4b_checkpoint'),
        manifest=k.get('manifest'), device=k.get('device', 'cpu'),
        max_age_seconds=rt['stale_after_sec'], future_tolerance_seconds=rt['future_tolerance_sec'],
        newest_sample_tolerance_seconds=rt['alignment_tolerance_sec'], clock=time.monotonic,
        expected_clock_id=clock_id)
    verify_frozen_identity(adapter.identity, release['frozen_dependency_identity'], weights=False)
    if allow_test:
        # quality_fusion_adapter.py remains byte-for-byte frozen. Its reviewed
        # QualityController already contains an explicit TEST gate; the adapter
        # constructor simply did not expose that flag. Enable ONLY that gate here.
        # Scorer bytes, parameter bundle, thresholds, mapping, routing and policy
        # hash are verified unchanged before and after this state change.
        before_policy=adapter.policy
        before_scorer=adapter.controller.identity['scorer']
        before_thresholds=copy.deepcopy(adapter.controller.thresholds)
        require(adapter.controller.allow_test is False, 'Unexpected pre-enabled TEST controller')
        adapter.controller.allow_test=True
        if hasattr(adapter.controller,'_identity'):
            adapter.controller._identity['allow_test']=True
        require(adapter.policy==before_policy and adapter.controller.identity['scorer']==before_scorer
                and adapter.controller.thresholds==before_thresholds,
                'TEST gate changed frozen quality policy/scorer state')
        verify_frozen_identity(adapter.identity, release['frozen_dependency_identity'], weights=False)
    return adapter, release


def fixed_fuse(adapter: Any, descriptor: Mapping, reports: Mapping, *, mode: str = 'replay') -> tuple[dict, dict]:
    """The only quality-to-fusion call site, shared by raw and cached main input."""
    require(mode in ('replay', 'shadow'), 'Unsupported quality execution mode')
    kw = {'now_seconds': adapter.clock(), 'clock_id': adapter.expected_clock_id} if mode == 'shadow' else {}
    quality = adapter.controller.assess_window(source_window=descriptor, reports=reports, mode=mode, **kw)
    result = adapter.predict_window(quality_result=quality, emotion_reports=reports, mode=mode)
    return quality, result


def checked_final(fusion: Mapping) -> list[float] | None:
    status = fusion.get('status'); f = fusion.get('final', {})
    if status != 'OK':
        require(not f.get('is_evidence') and f.get('probabilities') is None,
                'Error/NO_DECISION must not carry a published probability vector')
        return None
    require(f.get('is_evidence') is True, 'OK result needs evidence')
    p = f.get('probabilities')
    require(isinstance(p, (list, tuple)) and len(p) == 5, 'Five probabilities required')
    p = [finite(v, 'final.probability') for v in p]
    require(all(0 <= v <= 1 for v in p) and abs(sum(p)-1) <= 1e-5, 'Invalid final probabilities; no normalization applied')
    y = max(range(5), key=p.__getitem__)
    require(f.get('label_id') == y and f.get('emotion') == EMOTIONS[y], 'Final class/probability mismatch')
    return p


def main_window_result(d: Mapping, reports: Mapping, quality: Mapping, fusion: Mapping,
                       elapsed: float, *, mode: str) -> dict:
    checked_final(fusion)
    return {'schema': 'eav.main.window.v2', 'main_version': VERSION, 'status': fusion['status'],
        'window_id': d['window_id'], 'session_id': d['session_id'], 'source_window': copy.deepcopy(dict(d)),
        'identity': copy.deepcopy(d.get('identity')), 'reference_label': d.get('reference_label'),
        'mode': mode, 'reports': reports, 'quality_result': quality, 'fusion': fusion,
        'selected_variant': 'KEEP_V1', 'elapsed_seconds': elapsed,
        'quality_formula_changed': False, 'thresholds_changed': False, 'weights_trained': False,
        'legacy_router_executed': False, 'legacy_q_used_by_candidate': False,
        'upstream_original_wrappers_may_compute_legacy_q': True,
        'freshness_policy': 'CHECKED_SHADOW' if mode == 'LIVE_SHADOW' else 'NOT_APPLIED_TO_OFFLINE_DATA',
        'source_split': d.get('split', 'replay'), 'is_original_20s_trial_benchmark': False,
        'production_approved': False, 'robot_action_performed': False}


def integration_error_result(d: Mapping, reports: Mapping, error: str, *, mode: str) -> dict:
    # A host deadline, worker failure or malformed source is NOT sensor absence.
    q = {'status': 'QUALITY_CONTROL_ERROR', 'candidate_route': 'QUALITY_CONTROL_ERROR',
         'availability': {m:None for m in MODALITIES}, 'errors':[{'code':'HOST_INTEGRATION_ERROR','message':error}]}
    f = {'status':'QUALITY_CONTROL_ERROR', 'active_branch':None, 'candidate_route':'QUALITY_CONTROL_ERROR',
         'final':{'is_evidence':False, 'probabilities':None, 'emotion':None, 'label_id':None, 'confidence':None},
         'fusion_executed':False, 'quality_for_fusion':None, 'availability':q['availability'], 'errors':q['errors']}
    return main_window_result(d, reports, q, f, 0., mode=mode)


class Runtime:
    def __init__(self, config: Mapping, *, fusion_only: bool = False, clock_id: str | None = None, allow_test: bool = False):
        self.config = config; self.allow_test = bool(allow_test)
        self.modules, self.instances, self.identities = {}, {}, {}
        self.lock = threading.RLock()
        self.readers = {m:SourceReader(config) for m in MODALITIES}
        self.calls = Counter(); self.counter_lock = threading.Lock()
        self.quality_adapter, self.release = make_quality_adapter(config, clock_id=clock_id, allow_test=self.allow_test)
        print('[LOAD] frozen KEEP_V1 quality + F4/AF4-B cores (no legacy router)', flush=True)
        self.quality_adapter.check_assets()
        verify_frozen_identity(self.quality_adapter.identity, self.release['frozen_dependency_identity'], weights=True)
        self.fm = self.quality_adapter._get_backend().fm  # legacy packet FORMAT only, not its router
        self.fusion = self.quality_adapter  # No predict_packets compatibility fallback.
        self.identities['quality_fusion'] = self.quality_adapter.identity
        self.identities['test_access']={'allow_test':self.allow_test,'controller_allow_test':self.quality_adapter.controller.allow_test,'policy_modified':False}
        unit = config['eeg']['training_unit']; rt = config['runtime']
        if not fusion_only:
            require(unit in ('V','mV','uV'), 'Set eeg.training_unit/--eeg-unit from documented source units')
        roles = [] if fusion_only else ['eeg_emotion','eeg_quality','audio_quality','audio_emotion','video_emotion','video_quality']
        for role in roles:
            print(f'[LOAD] {role}', flush=True)
            spec = config['modules'][role]; p = Path(spec['script'])
            if spec.get('sha256'):
                require(digest(p) == spec['sha256'], f'Pinned script changed: {role}')
            mod = module_from_path(p, role); ctor = getattr(mod, CLASSES[role]); k = copy.deepcopy(spec['kwargs'])
            if role in ('eeg_emotion','eeg_quality','audio_emotion','audio_quality'):
                k['stale_after_sec'] = rt['stale_after_sec']
            if role == 'eeg_emotion':
                k.update(training_eeg_unit=unit, unit_evidence=config['eeg']['unit_evidence'], on_inference_error='raise')
            elif role == 'eeg_quality':
                input_unit = config['eeg']['input_unit']
                k.update(eeg_unit=unit if input_unit == 'training_native' else input_unit,
                         unit_evidence=config['eeg']['unit_evidence'], on_quality_error='raise')
            elif role == 'audio_emotion': k.update(error_policy='raise')
            elif role == 'audio_quality': k.update(quality_error_policy='raise')
            elif role in ('video_emotion','video_quality'): k['strict'] = True
            inspect.signature(ctor).bind(**k)
            obj = ctor(**k); self.modules[role] = mod; self.instances[role] = obj
            self.identities[role] = {'script':str(p), 'script_sha256':digest(p), 'constructor':CLASSES[role],
                'model_identity':safe_json(getattr(obj,'identity',getattr(obj,'model_identity',{})))}
        locked = [Path(config['quality_layer']['release']), Path(config['quality_layer']['params']), Path(config['quality_layer']['adapter_script']),
                  Path(config['quality_layer']['adapter_script']).with_name('quality_controller.py'),
                  Path(config['quality_layer']['adapter_script']).with_name('distribution_quality.py')]
        locked += [Path(spec['script']) for role,spec in config['modules'].items() if role in roles or role == 'fusion']
        for role in ('f4','af4b'):
            filename = (self.quality_adapter.identity.get('fusion_backend') or {}).get(role,{}).get('file')
            if filename: locked.append(Path(filename))
        if config.get('_config_path'): locked.append(Path(config['_config_path']))
        self._file_snapshot = {str(p):digest(p) for p in locked}

    def make_packet(self, modality: str, d: Mapping, payload: Mapping, *, reason: str,
                    evidence: str | None=None, live: bool=False) -> dict:
        item=d["modalities"][modality]
        return self.fm.make_modality_packet(modality,session_id=d["session_id"],window_id=d["window_id"],
            window_seconds=5.,class_order=EMOTIONS,probabilities=payload.get(modality+"_probs"),
            quality=payload["q_"+modality],available=payload[modality+"_available"],
            timing=item.get("timing") if live else None,reason=reason[:256],evidence_id=evidence)

    def missing(self, modality: str, d: Mapping, reason: str, *, status: str="UNAVAILABLE",error: str | None=None,live: bool=False) -> dict:
        payload={modality+"_probs":[.2]*5,"q_"+modality:0.,modality+"_available":False}
        return {"modality":modality,"window_id":d["window_id"],"status":status,"reason":reason,"error":error,
                "packet":self.make_packet(modality,d,payload,reason=reason,live=live),"source":None,"emotion":None,
                "quality":None,"elapsed_seconds":0.,"algorithm_error":status=="ERROR"}

    def process(self, modality: str, d: Mapping, *, live: bool=False) -> dict:
        start=time.perf_counter();item=d["modalities"][modality]
        if not boolean(item["present"],"present"):
            return self.missing(modality,d,str(item.get("reason","NO_CURRENT_SOURCE")),live=live)
        capture_ok=boolean(item.get("capture_ok",True),"capture_ok")
        continuous=boolean(item.get("continuous",True),"continuous")
        if not capture_ok or not continuous:
            return self.missing(modality,d,"CAPTURE_ERROR" if not capture_ok else "NONCONTIGUOUS_SOURCE",live=live)
        if live:
            now=time.monotonic();latest=finite(item["timing"]["newest_sample_monotonic"],"latest sample")
            if now-latest>self.config["runtime"]["stale_after_sec"]:
                return self.missing(modality,d,"STALE_BEFORE_SOURCE_READ",live=True)
        else:latest=None
        try:
            reader=self.readers[modality]
            with tempfile.TemporaryDirectory(prefix="eav_window_") as work:
                if modality=="eeg":
                    x,source=reader.load_eeg(item)
                elif modality=="audio":
                    x,sr,source=reader.load_audio(item)
                else:
                    path,source=reader.load_video(item,Path(work))
                guard=self.lock if self.config["runtime"]["serialize_heads"] else _NullLock()
                with guard:
                    # Cold loads are outside the loop; at most one call per modality
                    # is active. GPU calls may still exceed a live deadline.
                    with self.counter_lock:self.calls[modality]+=1
                    if modality=="eeg":
                        er=self.instances["eeg_emotion"];qr=self.instances["eeg_quality"]
                        expected=self.config["eeg"]["input_unit"]
                        actual=item.get("input_unit",expected)
                        require(actual==expected,"Per-window EEG unit differs from configured input unit")
                        names=item.get("channel_names")
                        require(names is not None,"EEG source must carry explicit30channel names")
                        report=er.process_with_quality_array(x,quality_runtime=qr,sample_rate=item.get("sample_rate",500),
                            channel_names=names,input_unit=actual,window_id=d["window_id"],live=live,
                            newest_sample_monotonic=latest,contiguous=True,backend_error=False)
                        payload=report["fusion_eeg_input"];emotion=report["emotion"];quality=report["quality"]
                    elif modality=="audio":
                        report=self.instances["audio_emotion"].process_with_quality_array(x,sr,
                            quality_runtime=self.instances["audio_quality"],window_id=d["window_id"],channel=item.get("channel","mean"),
                            live=live,newest_sample_monotonic=latest,capture_ok=True,continuous=True)
                        payload=report["fusion_audio_input"];emotion=report["audio_emotion"];quality=report["audio_quality"]
                    else:
                        # Models retain their own frozen face sampling/preprocessing.
                        # Exactly the same completed clip supplies both branches.
                        quality=self.instances["video_quality"].assess(path)
                        if quality.get("error"):raise ModuleExecutionError("Video quality: "+str(quality["error"]))
                        if boolean(quality["video_available"],"video quality available"):
                            emotion=self.instances["video_emotion"].predict(path)
                            if emotion.get("error"):raise ModuleExecutionError("Video emotion: "+str(emotion["error"]))
                            available=boolean(emotion["video_available"],"video emotion available")
                        else:
                            emotion={"video_probs":[.2]*5,"video_available":False,"reason":"NO_FACE_QUALITY_BRANCH"};available=False
                        payload={"video_probs":emotion["video_probs"] if available else [.2]*5,
                                 "q_video":quality["q_video"] if available else 0.,"video_available":available}
                        require(digest(path)==source["window_sha256"],"Video snapshot changed between two branches")
                for part in (emotion,quality):
                    if part.get("status")=="ERROR" or part.get("error"):
                        raise ModuleExecutionError(str(part.get("error") or part.get("reason")))
                av=boolean(payload[modality+"_available"],"effective availability")
                reason="CURRENT_CLASSIFIER_AND_QUALITY" if av else str(quality.get("reason") or emotion.get("reason") or "NO_USABLE_EVIDENCE")
                evidence=source.get("window_sha256",source.get("sha256"))
                packet=self.make_packet(modality,d,payload,reason=reason,evidence=evidence,live=live)
                return {"modality":modality,"window_id":d["window_id"],"status":"OK" if av else "UNAVAILABLE",
                        "reason":reason,"error":None,"packet":packet,"source":source,"emotion":emotion,"quality":quality,
                        "elapsed_seconds":time.perf_counter()-start,"algorithm_error":False}
        except Exception as exc:
            if self.config["runtime"]["error_policy"]=="raise":
                raise ModuleExecutionError(f"{modality}/{d['window_id']}: {type(exc).__name__}: {exc}") from exc
            result=self.missing(modality,d,"MODULE_ERROR_EXCLUDED",status="ERROR",
                                error=f"{type(exc).__name__}: {exc}",live=live)
            result["elapsed_seconds"]=time.perf_counter()-start
            return result

    def fuse_reports(self, d: Mapping, reports: Mapping, *, mode: str = 'replay') -> dict:
        """Both real processing and cache regression call this identical path."""
        start = time.perf_counter()
        q, f = fixed_fuse(self.quality_adapter, d, reports, mode=mode)
        return main_window_result(d, reports, q, f, time.perf_counter()-start,
                                  mode='LIVE_SHADOW' if mode == 'shadow' else 'OFFLINE_REPLAY')

    def process_offline(self, d: Mapping, *, allow_test: bool = False) -> dict:
        require(bool(allow_test)==self.allow_test, 'Runtime/source TEST authorization mismatch')
        require(d.get('split')!='test' or allow_test, 'TEST source requires explicit --allow-test')
        validate_descriptor(d, live=False, allow_test=allow_test)
        start = time.perf_counter(); reports = {}
        for m in MODALITIES:
            try:
                reports[m] = self.process(m, d, live=False)
            except Exception as exc:
                reports[m] = self.missing(m, d, 'MAIN_CAUGHT_PRODUCER_ERROR', status='ERROR',
                    error=f'{type(exc).__name__}: {exc}', live=False)
        # ERROR placeholders are passed honestly: the controller withholds fusion.
        result = self.fuse_reports(d, reports)
        result['elapsed_seconds'] = time.perf_counter()-start
        result['raw_sensor_models_invoked'] = True
        return result

    def verify_unchanged(self) -> dict:
        for filename, sha in self._file_snapshot.items():
            require(Path(filename).is_file() and digest(filename) == sha, 'Loaded dependency changed during run: '+filename)
        r = self.quality_adapter.check_assets()
        verify_frozen_identity(self.quality_adapter.identity, self.release['frozen_dependency_identity'], weights=True)
        return r


class _NullLock:
    def __enter__(self):return self
    def __exit__(self,*args):return False


# ------------------------- Bounded foreground live loop -----------------------

@dataclass
class JobResult:
    modality: str
    window_id: str
    report: dict | None
    error: BaseException | None


class SingleWorker:
    """Capacity=one running job, ZERO waiting jobs; does not kill native calls."""
    def __init__(self,name: str,runtime: Runtime):
        self.name=name;self.runtime=runtime;self.jobs=queue.Queue(maxsize=1);self.results=queue.Queue(maxsize=1)
        self.busy=False;self.closed=False
        self.thread=threading.Thread(target=self._run,name="eav-"+name,daemon=True);self.thread.start()

    def submit(self,d: Mapping) -> bool:
        if self.busy or self.closed:return False
        self.busy=True;self.jobs.put_nowait(d);return True

    def _run(self):
        while True:
            d=self.jobs.get()
            if d is None:return
            try:r=self.runtime.process(self.name,d,live=True);error=None
            except BaseException as exc:r=None;error=exc
            self.results.put(JobResult(self.name,d["window_id"],r,error))

    def take(self) -> JobResult | None:
        try:r=self.results.get_nowait()
        except queue.Empty:return None
        self.busy=False
        return r

    def close(self,grace: float) -> bool:
        self.closed=True
        try:self.jobs.put_nowait(None)
        except queue.Full:pass
        self.thread.join(timeout=grace)
        return not self.thread.is_alive()


class LiveScheduler:
    """Existing same-host acquisition support, full-report SHADOW publication only.

    No legacy coordinator or q router. Busy/late/failed jobs withhold the entire
    window as a host error, not fabricated sensor absence. A native call is not
    force-cancelled. This software path is NOT a live hardware validation result.
    """
    def __init__(self, runtime: Runtime, *, session_id: str, clock_id: str):
        self.runtime = runtime; self.session_id = session_id; self.clock_id = clock_id
        require(runtime.quality_adapter.expected_clock_id == clock_id, 'Runtime shadow clock mismatch')
        self.events = []; self.windows = OrderedDict(); self.reports = {}; self.host_errors = {}
        self.workers = {m:SingleWorker(m,runtime) for m in MODALITIES}
        self.last_end = -math.inf; self.seen = set()

    def submit_window(self, d: Mapping) -> None:
        validate_descriptor(d, live=True)
        require(d['session_id'] == self.session_id and d['clock_id'] == self.clock_id, 'Wrong capture session/clock')
        wid = d['window_id']; end = finite(d['window_end_monotonic'], 'window end')
        rt = self.runtime.config['runtime']; now = time.monotonic()
        require(wid not in self.seen and len(self.seen) < 10000, 'Duplicate window ID/session window limit reached')
        require(end > self.last_end, 'Capture windows must have strictly increasing end times')
        require(end <= now+rt['future_tolerance_sec'], 'Future capture window')
        require(len(self.windows) < rt['max_pending'], 'Pending window limit reached')
        snap = copy.deepcopy(dict(d)); errors = []
        # Inspect every modality before dispatching any workers.
        for m, item in snap['modalities'].items():
            if item['present']:
                tm = item['timing']
                require(tm['clock_id'] == self.clock_id, 'Capture clock mismatch')
                # The controller requires the actual SAME span, not timestamp relabelling.
                require(tm['window_start_monotonic'] == end-5. and tm['window_end_monotonic'] == end,
                        m+': capture span differs from decision window')
        self.windows[wid] = snap; self.reports[wid] = {}; self.host_errors[wid] = errors
        self.last_end = end; self.seen.add(wid)
        for m, item in snap['modalities'].items():
            if not item['present']:
                self.reports[wid][m] = self.runtime.missing(m, snap, str(item.get('reason','NO_CURRENT_SOURCE')), live=True)
            elif not self.workers[m].submit(snap):
                errors.append(m+': WORKER_BUSY_NO_CURRENT_REPORT')
        if errors:
            self.events.append({'event':'HOST_WINDOW_ERROR','window_id':wid,'errors':errors[:]})

    def poll(self) -> list[dict]:
        for m, worker in self.workers.items():
            jr = worker.take()
            if jr is None: continue
            wid = jr.window_id
            if wid not in self.windows:
                self.events.append({'event':'FINALIZED_WINDOW_RESULT_DISCARDED','window_id':wid,'modality':m})
                continue
            if jr.error is not None:
                self.reports[wid][m] = self.runtime.missing(m, self.windows[wid], 'WORKER_PRODUCER_ERROR',
                    status='ERROR', error=repr(jr.error), live=True)
            else:
                self.reports[wid][m] = jr.report
        out = []; rt = self.runtime.config['runtime']
        # Ordered finalization prevents an old result overwriting a newer state.
        for wid in list(self.windows):
            d = self.windows[wid]; reports = self.reports[wid]; now = time.monotonic()
            deadline = d['window_end_monotonic'] + rt['deadline_sec']
            errors = self.host_errors[wid]
            if now > deadline: errors.append('HOST_DEADLINE_EXCEEDED')
            if errors:
                row = integration_error_result(d, reports, '; '.join(errors), mode='LIVE_SHADOW')
            elif set(reports) != set(MODALITIES):
                break
            else:
                row = self.runtime.fuse_reports(d, reports, mode='shadow')
                if time.monotonic() > deadline:
                    prior = row
                    row = integration_error_result(d, reports, 'DEADLINE_EXCEEDED_DURING_FUSION', mode='LIVE_SHADOW')
                    row['withheld_computation'] = prior['fusion']
                    row['computed_output_withheld'] = True
            row['elapsed_since_capture_end_seconds'] = time.monotonic()-d['window_end_monotonic']
            out.append(row)
            del self.windows[wid]; del self.reports[wid]; del self.host_errors[wid]
        return out

    def close(self) -> dict:
        invalid = list(self.windows)
        self.windows.clear(); self.reports.clear(); self.host_errors.clear()
        grace = self.runtime.config['runtime']['shutdown_grace_sec']
        ended = {m:w.close(grace) for m,w in self.workers.items()}
        return {'invalidated_windows':invalid, 'workers_stopped':ended, 'native_calls_force_cancelled':False,
                'note':'Pending evidence discarded. Running native calls may require process termination.'}


class FileInboxSource:
    """Non-destructive, bounded same-host IPC: collector atomically commits *.json.

    JSONs contain SOURCE_SCHEMA plus complete file descriptors and REAL capture
    timing. No user waveform is uploaded; files are not deleted or rewritten.
    Session-filtered filenames are ordered by declared end, not modification time.
    A stop.json containing {session_id, stop:true} is an explicit producer EOF.
    """
    def __init__(self,directory: Path,*,session_id: str,clock_id: str):
        require(directory.is_dir(),f"Live inbox missing: {directory}")
        self.directory=directory;self.session_id=session_id;self.clock_id=clock_id;self.seen=set();self.ended=False

    def poll(self,timeout_sec: float=0.) -> dict | None:
        candidates=[]
        paths=list(self.directory.glob("*.json"));require(len(paths)<=10000,"Inbox file count limit; rotate capture directories")
        for p in paths:
            if str(p) in self.seen:continue
            d=read_json(p)
            if d.get("session_id")!=self.session_id:
                self.seen.add(str(p));continue
            if d.get("stop") is True:
                continue
            validate_descriptor(d,live=True)
            require(d["clock_id"]==self.clock_id,"Inbox clock ID mismatch")
            candidates.append((d["window_end_monotonic"],str(p),d))
        if candidates:
            _,name,d=min(candidates,key=lambda v:(v[0],v[1]));self.seen.add(name)
            for item in d["modalities"].values():
                if item.get("present"):
                    require("samples" not in item,"Inbox samples must use NPY/WAV paths, not JSON arrays")
                    p=resolve_path(item["path"],self.directory);require(p.is_file(),f"Committed raw input missing: {p}")
                    item["path"]=str(p)
            return d
        stop=self.directory/"stop.json"
        if stop.is_file():
            st=read_json(stop)
            if st.get("session_id")==self.session_id and st.get("stop") is True:self.ended=True
        if timeout_sec>0:time.sleep(min(timeout_sec,.05))
        return None

    def close(self):pass


# -------------------------------- Run outputs --------------------------------

def metric_counts(labels: Sequence[int], predictions: Sequence[int | None]) -> dict:
    require(len(labels) == len(predictions), 'Metric length mismatch')
    cm = [[0]*6 for _ in range(5)]
    for y, p in zip(labels, predictions):
        require(type(y) is int and y in range(5) and (p is None or type(p) is int and p in range(5)), 'Invalid metric class')
        cm[y][5 if p is None else p] += 1
    n = len(labels); decided = sum(p is not None for p in predictions); correct = sum(y==p for y,p in zip(labels,predictions))
    f1 = []
    for k in range(5):
        tp = cm[k][k]; den = sum(cm[k])+sum(cm[j][k] for j in range(5))
        f1.append(2*tp/den if den else 0.)
    return {'n_planned':n,'n_decisions':decided,'n_correct':correct,'decision_coverage':decided/n if n else None,
            'accuracy_all_planned':correct/n if n else None,'accuracy_on_decisions':correct/decided if decided else None,
            'macro_f1_all_planned_abstentions_as_FN':sum(f1)/5 if n else None,
            'confusion_matrix':cm,'class_order':EMOTIONS,'confusion_columns':EMOTIONS+['NO_VALID_PREDICTION']}


def result_key(d: Mapping, condition: str = 'clean') -> tuple[str,str,str]:
    return condition, str(d.get('session_id','')), str(d['window_id'])


class RunLog:
    """Append-only results, explicit errors, atomic latest state and full denominators."""
    def __init__(self, path: Path, config: Mapping, mode: str):
        require(not path.exists(), 'Output exists; use a NEW run directory: '+str(path))
        path.mkdir(parents=True); self.path=path; self.mode=mode; self.config=config
        self.count=0; self.error_windows=0; self.error_events=0; self.no_decision=0
        self.routes=Counter(); self.started=time.monotonic(); self.current_valid_until=None; self.current_window_id=None
        self.plan=OrderedDict(); self.observed={}; self.rows=(path/'window_results.jsonl').open('x',encoding='utf-8',newline='\n')
        self.events_file=(path/'events.jsonl').open('x',encoding='utf-8',newline='\n')
        write_new_json(path/'run_config.json', {k:v for k,v in config.items() if not k.startswith('_')})
        write_new_json(path/'run_manifest.json', {'main_version':VERSION,'main_sha256':digest(Path(__file__)),
            'config_sha256':digest(config['_config_path']) if config.get('_config_path') else None,
            'python':sys.version,'executable':sys.executable,'started_utc':datetime.now(timezone.utc).isoformat(),
            'selected_variant':'KEEP_V1','mode':mode,'quality_and_weights_modified':False,
            'source_split':config.get('replay',{}).get('split'),'test_data_used':config.get('replay',{}).get('split')=='test',
            'historical_plan_hashes_not_rewritten':True,'robot_actions_authorized':False})
        self._write_current_state('WAITING_FOR_INPUT')

    def register(self, descriptors: Sequence[Mapping], *, condition: str = 'clean') -> None:
        for d in descriptors:
            k = result_key(d,condition)
            require(k not in self.plan, 'Duplicate planned source window: '+repr(k))
            y = d.get('reference_label')
            if y is not None: require(integer(y,'reference_label') in range(5), 'Invalid reference label')
            self.plan[k] = {'condition':condition,'session_id':d.get('session_id'), 'window_id':d['window_id'],
                'identity':copy.deepcopy(d.get('identity',{})), 'label':y}

    def _write_current_state(self, reason: str, fusion: Mapping | None = None) -> None:
        p = checked_final(fusion) if fusion is not None else None
        state = {'status':fusion.get('status') if fusion is not None else 'NO_CURRENT_DECISION',
            'reason':reason,'mode':self.mode,'source_window_id':self.current_window_id,
            'has_evidence':p is not None,'final':fusion['final'] if fusion is not None else
                {'emotion':None,'confidence':None,'probabilities':None,'label_id':None,'is_evidence':False},
            'active_branch':fusion.get('active_branch') if fusion else None,
            'quality_for_fusion':fusion.get('quality_for_fusion') if fusion else None,
            'availability':fusion.get('availability') if fusion else None,
            'valid_until_monotonic':self.current_valid_until, 'updated_monotonic':time.monotonic(),
            'offline_result_not_live_evidence':self.mode != 'live',
            'robot_action_authorized':False,'state_update_is_not_a_new_model_prediction':fusion is None}
        # Publish the current-state snapshot atomically. On Windows, Defender,
        # Search indexing, editors, or other readers can transiently hold the
        # destination path and make os.replace() raise WinError 5. Use a unique
        # temporary file plus bounded retry without deleting/truncating the
        # previously valid state. This changes only state-file publication; it
        # does not affect quality scoring, routing, model inputs, or predictions.
        target=self.path/'latest_state.json'
        tmp=self.path/(f'.latest_state.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp')
        payload=json.dumps(safe_json(state),ensure_ascii=False,allow_nan=False,indent=2)+'\n'
        try:
            with tmp.open('x',encoding='utf-8',newline='\n') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            last_error=None
            for attempt in range(20):
                try:
                    os.replace(tmp,target)
                    last_error=None
                    break
                except PermissionError as exc:
                    last_error=exc
                    if attempt==19:
                        break
                    time.sleep(min(.02*(attempt+1),.20))
            if last_error is not None:
                raise PermissionError(
                    f'Unable to atomically publish current state after 20 attempts: {target}'
                ) from last_error
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def expire_current_state(self, now: float | None = None) -> bool:
        now=time.monotonic() if now is None else now
        if self.current_valid_until is not None and now > self.current_valid_until:
            self.current_valid_until=None
            self._write_current_state('LIVE_EVIDENCE_EXPIRED_NO_CURRENT_DECISION')
            self.event({'event':'CURRENT_DECISION_EXPIRED','window_id':self.current_window_id,'stale_probability_reused':False})
            return True
        return False

    def event(self, obj: Mapping) -> None:
        if 'ERROR' in str(obj.get('event','')) or obj.get('event')=='FATAL': self.error_events+=1
        self.events_file.write(json.dumps(safe_json(obj),ensure_ascii=False,allow_nan=False)+'\n'); self.events_file.flush()

    def result(self, row: Mapping) -> None:
        d=row['source_window']; condition=row.get('condition_name','clean'); k=result_key(d,condition)
        require(k in self.plan and k not in self.observed, 'Unplanned/duplicate result: '+repr(k))
        fusion=row['fusion']; p=checked_final(fusion); status=fusion['status']
        self.observed[k]={'status':status,'probabilities':p,'prediction':max(range(5),key=p.__getitem__) if p is not None else None}
        self.rows.write(json.dumps(safe_json(row),ensure_ascii=False,allow_nan=False)+'\n'); self.rows.flush()
        self.count+=1; self.error_windows+=int(status not in VALID_STATUSES); self.no_decision+=int(status=='NO_DECISION')
        route=fusion.get('active_branch') or status; self.routes[route]+=1
        self.current_window_id=d['window_id']
        self.current_valid_until=fusion.get('freshness_after',{}).get('valid_until_seconds') if self.mode=='live' else None
        self._write_current_state('CURRENT_WINDOW_RESULT',fusion)
        self.expire_current_state()
        if self.mode != 'cached' or self.count <= 3 or self.count % 50 == 0 or status not in VALID_STATUSES:
            print(f"[{self.count}] {condition}/{d['window_id']} | {status} | {route} | "
                  f"{fusion['final'].get('emotion')} | q_new={fusion.get('quality_for_fusion')}",flush=True)

    def metrics(self) -> tuple[dict,list[dict]]:
        groups={}; trial_rows=[]
        for cond in dict.fromkeys(x['condition'] for x in self.plan.values()):
            planned=[(k,x) for k,x in self.plan.items() if x['condition']==cond and x['label'] is not None]
            ys=[x['label'] for _,x in planned]
            ps=[self.observed.get(k,{}).get('prediction') for k,_ in planned]
            m=metric_counts(ys,ps); trials=OrderedDict()
            for key, row in planned:
                ident_=row['identity']
                if 'pair_key' not in ident_ or 'window_idx_0based' not in ident_: continue
                tk=(row['session_id'],ident_.get('subject'),ident_['pair_key'])
                trials.setdefault(tk,[]).append((key,row))
            tys,tps=[],[]
            for (sid,subject,pair), items in trials.items():
                items=sorted(items,key=lambda v:v[1]['identity']['window_idx_0based'])
                labels={x['label'] for _,x in items}; require(len(labels)==1,'Trial labels disagree')
                indices=[x['identity']['window_idx_0based'] for _,x in items]
                probs=[self.observed.get(k,{}).get('probabilities') for k,_ in items]
                avg=None
                if indices==[0,1,2,3] and all(p is not None for p in probs):
                    avg=[sum(p[j] for p in probs)/4. for j in range(5)]
                pred=max(range(5),key=avg.__getitem__) if avg is not None else None
                y=next(iter(labels)); tys.append(y); tps.append(pred)
                trial_rows.append({'condition':cond,'session_id':sid,'subject':subject,'pair_key':pair,'label':y,
                    'planned_windows':len(items),'valid_windows':sum(p is not None for p in probs),
                    'probabilities':avg,'prediction':pred,'aggregation':'mean of four final 5s probability vectors; no partial trial'})
            m['trial_aggregate']=metric_counts(tys,tps)
            m['trial_aggregate']['original_20s_trial_benchmark']=False
            groups[cond]=m
        return groups,trial_rows

    def finish(self, *, status: str, error: str | None = None, extra: Mapping | None = None) -> dict:
        # No stale evidence survives process exit; detailed final result stays in JSONL.
        self.current_valid_until=None; self._write_current_state('SYSTEM_STOPPED_NO_CURRENT_DECISION')
        self.rows.close(); self.events_file.close(); metrics,trials=self.metrics()
        with (self.path/'trial_results.jsonl').open('x',encoding='utf-8') as f:
            for r in trials:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        summary={'schema':'eav.main.run.v2','version':VERSION,'status':status,'mode':self.mode,'selected_variant':'KEEP_V1',
            'n_planned':len(self.plan),'n_processed':self.count,'n_unprocessed':len(self.plan)-self.count,
            'error_windows':self.error_windows,'no_decision_windows':self.no_decision,'error_events':self.error_events,
            'route_counts':dict(self.routes),'metrics_by_condition':metrics,
            'elapsed_seconds':time.monotonic()-self.started,'error':error,
            'main_integration_code_path_used':True,'quality_formulas_modified':False,'thresholds_modified':False,
            'training_performed':False,'legacy_router_executed':False,'legacy_q_used_by_candidate':False,
            'production_approved':False,'robot_actions_performed':False,'live_hardware_validated':False,
            'incomplete_and_error_records_retained_in_denominators':True,
            'performance_status':'NOT_SPECIFIED','is_independent_accuracy_experiment':False,**dict(extra or {})}
        write_new_json(self.path/'run_summary.json',summary)
        print(json.dumps({'status':status,'main_version':VERSION,'n_processed':self.count,
            'output_directory':str(self.path),'report':'run_summary.json'},ensure_ascii=False,indent=2),flush=True)
        return summary


def integration_cache(path: Path, *, allow_synthetic: bool = False) -> tuple[dict,list[dict],dict]:
    """Audit existing report records. No raw files/arrays are opened here."""
    sp, lp = path/'run_summary.json', path/'windows.jsonl'
    summary=read_json(sp)
    require(summary.get('schema')=='eav.quality_control_validation.v1' and summary.get('status')=='PASS',
            'Cache must be an intact PASS raw_validation directory, not a failed wrapper summary')
    require(summary.get('version')=='EAV-QUALITY-CONTROL-VALIDATION.1.0','Unsupported cache producer version')
    require(summary.get('acceptance',{}).get('engineering_pass') is True,'Cached raw run did not pass engineering checks')
    for key in ('test_data_used','training_performed','thresholds_or_mapping_fitted','candidate_uses_old_q'):
        require(summary.get(key) is False,'Cache has non-frozen/unknown behavior: '+key)
    require(summary.get('development_fixture_only') is not True or allow_synthetic,
            'Synthetic cache requires explicit --allow-synthetic-source; not real acceptance evidence')
    require(lp.is_file(),'windows.jsonl missing in '+str(path))
    hashes={'run_summary':digest(sp),'windows_jsonl':digest(lp)}
    sel=summary['selection']; specs={c['name']:c for c in sel['conditions']}
    require(len(specs)==len(sel['conditions']),'Duplicate condition names')
    trials={(r['subject'],r['pair_key']):r['label'] for r in sel['selected_trial_keys']}
    require(len(trials)==sel['selected_trials'],'Duplicate/inconsistent trial list')
    expected={(c,s,p,w) for c in specs for s,p in trials for w in range(4)}
    require(len(expected)==summary['n_planned']==summary['n_processed'], 'Cache is incomplete')
    rows=[]; seen=set()
    with lp.open(encoding='utf-8-sig') as f:
        for ln,line in enumerate(f,1):
            if not line.strip():continue
            require(len(line)<=8*1024*1024,'Oversized cache record')
            r=parse_json_object(line)
            require(r.get('schema')=='eav.quality_control_validation.v1.window','Wrong cached window schema')
            d=r['source_window']; c=r['condition']['name']; ident_=d['identity']
            key=(c,ident_['subject'],ident_['pair_key'],integer(ident_['window_idx_0based'],'window index'))
            require(key in expected and key not in seen,'Unknown/duplicate cached window: '+repr(key))
            require(r['condition']==specs[c],'Condition differs from cache summary')
            require(d['reference_label']==trials[(key[1],key[2])],'Cached reference label mismatch')
            require(d.get('split')=='val' and key[1] in VAL_SUBJECTS,'Only existing VAL regression cache accepted')
            require(d.get('schema')==SOURCE_SCHEMA and d.get('window_seconds')==5. and str(d.get('task_condition')).lower()=='speaking',
                    'Wrong cached source protocol')
            ident(d.get('session_id'),'session_id');ident(d.get('window_id'),'window_id')
            require(r.get('numeric_parity',{}).get('status')=='PASS','Cache contains failed numeric comparison')
            require(r.get('legacy_q_used_by_candidate') is False and r.get('robot_action_performed') is False,
                    'Cache behavior inconsistent')
            require(r['candidate'].get('status') in VALID_STATUSES,'Cached failed prediction must not be accepted as a baseline')
            checked_final(r['candidate'])
            require(set(r.get('reports',{}))==set(MODALITIES),'Cache must include all three full reports')
            seen.add(key); rows.append(r)
            require(len(rows)<=100000,'Cache record limit exceeded')
    require(seen==expected,'Missing cached windows')
    return summary,rows,hashes


def compare_integration(row: Mapping, logged: Mapping) -> dict:
    f=row['fusion']; old=logged['candidate']; quality=row['quality_result']; oldq=logged['quality_result']
    checks={}; delta=None
    for key in ('status','active_branch','candidate_route','availability','model_inputs'):
        checks['fusion_'+key]=(f.get(key)==old.get(key))
    for key in ('status','candidate_route','availability','any_quality_alarm'):
        checks['quality_'+key]=(quality.get(key)==oldq.get(key))
    for m in MODALITIES:
        for key in ('B','T','B_over_T','exceeds_threshold','status','available'):
            checks[m+'_'+key]=(quality.get('modalities',{}).get(m,{}).get(key)==oldq.get('modalities',{}).get(m,{}).get(key))
    checks['q_new_identical']=f.get('quality_for_fusion')==old.get('quality_for_fusion')
    a,b=checked_final(f),checked_final(old)
    if a is None or b is None:
        checks['probabilities']=a is None and b is None
    else:
        delta=max(abs(x-y) for x,y in zip(a,b))
        checks['probabilities']=all(abs(x-y)<=PROBABILITY_ATOL+PROBABILITY_RTOL*abs(y) for x,y in zip(a,b))
        checks['predicted_class']=max(range(5),key=a.__getitem__)==max(range(5),key=b.__getitem__)
    return {'passed':all(checks.values()),'maximum_probability_difference':delta,
            'failed_checks':[k for k,v in checks.items() if not v], 'checks':checks,
            'source_window_id':row['window_id'],'condition':logged['condition']['name']}


def run_cached(runtime: Runtime, log: RunLog, source: Path, summary: Mapping, rows: list[dict], hashes: Mapping) -> dict:
    verify_frozen_identity(runtime.quality_adapter.identity, summary['candidate_identity_before'], weights=True)
    if not log.plan:
        for r in rows:log.register([r['source_window']],condition=r['condition']['name'])
    else:
        require(set(log.plan)=={result_key(r['source_window'],r['condition']['name']) for r in rows},
                'Registered regression plan differs from cache')
    failures=0; maximum=0.; counts=Counter()
    with (log.path/'integration_comparison.jsonl').open('x',encoding='utf-8') as out:
        for r in rows:
            row=runtime.fuse_reports(r['source_window'],r['reports'])
            row['condition_name']=r['condition']['name']; row['cached_reports_reused']=True; row['raw_sensor_models_invoked']=False
            audit=compare_integration(row,r);failures+=not audit['passed'];counts[r['condition']['name']]+=1
            if audit['maximum_probability_difference'] is not None: maximum=max(maximum,audit['maximum_probability_difference'])
            # A mismatch is fully logged, not turned into a purported successful prediction.
            row['integration_comparison']=audit
            out.write(json.dumps(audit,ensure_ascii=False,allow_nan=False)+'\n');out.flush();log.result(row)
    require(digest(source/'windows.jsonl')==hashes['windows_jsonl'] and digest(source/'run_summary.json')==hashes['run_summary'],
            'Regression cache changed during read')
    calculated,_=log.metrics(); summaries=[]
    for condition, actual in calculated.items():
        expected=summary['metrics_by_condition'][condition]
        for level in ('window','trial'):
            a=actual if level=='window' else actual['trial_aggregate']
            b=expected if level=='window' else expected['trial_aggregate']
            good=all(a[k]==b[k] for k in ('n_planned','n_decisions','n_correct','confusion_matrix'))
            summaries.append({'condition':condition,'level':level,'passed':good})
            failures+=not good
    result={'status':'PASS' if not failures else 'INTEGRATION_MISMATCH', 'n_checked':len(rows),
        'failed_checks':failures,'maximum_reproduction_probability_difference':maximum,
        'summary_checks':summaries,'condition_counts':dict(counts),'cached_run':str(source),'source_hashes':dict(hashes),
        'raw_cache_reused':True,'upstream_inference_launched':False,'new_accuracy_experiment':False,
        'same_quality_and_fusion_entry_as_raw_mode':True,'historical_main_hash_intentionally_different':True,
        'no_dependency_on_analysis_or_validation_scripts':True,
        'synthetic_source':summary.get('development_fixture_only') is True}
    write_new_json(log.path/'integration_check.json',result)
    return result


def collect_descriptors(c: Mapping, args: Any) -> list[dict]:
    r=c['replay']; allow_test=bool(args.allow_test)
    ds=load_jsonl_descriptors(Path(r['manifest']),allow_test=allow_test) if r.get('manifest') else build_eav_descriptors(c,allow_test=allow_test)
    if args.trial_key:
        ds=[d for d in ds if d.get('identity',{}).get('pair_key')==args.trial_key]
        require(len(ds)==4 and sorted(d['identity']['window_idx_0based'] for d in ds)==[0,1,2,3],
                'Requested trial must contain exactly four source windows')
    if args.window_key:
        ds=[d for d in ds if d['window_id']==args.window_key];require(len(ds)==1,'Requested window not unique/found')
    if args.preflight and not args.window_key:
        # Only an entrypoint check, not a new performance experiment.
        pair=ds[0].get('identity',{}).get('pair_key')
        ds=[d for d in ds if d.get('identity',{}).get('pair_key')==pair] if pair else ds[:1]
        if pair: require(len(ds)==4,'Preflight selected an incomplete source trial')
    if args.limit:
        require(args.limit>0,'Positive limit required');ds=ds[:args.limit]
    require(ds,'No selected source windows')
    return ds


def cohort_summary(ds: Sequence[Mapping]) -> dict:
    trials=OrderedDict(); subject_windows=Counter(); class_windows=Counter()
    for d in ds:
        ident_=d.get('identity',{}); subject=ident_.get('subject'); pair=ident_.get('pair_key'); wi=ident_.get('window_idx_0based')
        y=integer(d.get('reference_label'),'reference_label')
        require(subject and pair is not None and wi is not None,'Final benchmark requires EAV subject/pair/window identity')
        subject_windows[subject]+=1; class_windows[EMOTIONS[y]]+=1
        key=(subject,pair); trials.setdefault(key,[]).append((integer(wi,'window index'),y))
    class_trials=Counter(); subject_trials=Counter()
    for (subject,pair), items in trials.items():
        require(sorted(i for i,_ in items)==[0,1,2,3] and len({y for _,y in items})==1,
                'Benchmark cohort contains incomplete/inconsistent trial: '+repr((subject,pair)))
        subject_trials[subject]+=1; class_trials[EMOTIONS[items[0][1]]]+=1
    return {'selected_windows':len(ds),'selected_trials':len(trials),'subjects':sorted(subject_trials),
            'subject_window_counts':dict(sorted(subject_windows.items())),
            'subject_trial_counts':dict(sorted(subject_trials.items())),
            'class_window_counts':dict(class_windows),'class_trial_counts':dict(class_trials),
            'all_trials_have_four_windows':True,'selection_used_outcomes':False}


def run_demo(runtime: Runtime, log: RunLog, session: str) -> None:
    # Examples go through the frozen controller, not hand-crafted q or routes.
    am=runtime.quality_adapter
    for case in ('healthy','degraded','missing-video','all-missing','error'):
        # make_example returns quality + emotion; obtain the source-report fixture
        # directly, with the same documented native contract.
        raw=am.controller.make_example(report_format='main-v1')
        d,reports=raw['source_window'],raw['reports']
        d['window_id']='synthetic_'+case;d['session_id']=session
        for m,r in reports.items():
            r['window_id']=d['window_id'];r['packet']['window_id']=d['window_id'];r['packet']['session_id']=session
            r['packet'].update(class_order=EMOTIONS,probabilities=[.1,.1,.1,.6,.1])
            r['emotion'].update(class_order=EMOTIONS,**{m+'_probs':[.1,.1,.1,.6,.1]})
        if case=='degraded':reports['audio']['quality']['signal_metrics']['rms_dbfs']=-100.
        if case in ('missing-video','all-missing'):
            for m in (MODALITIES if case=='all-missing' else ('video',)):
                d['modalities'][m]={'present':False}
                reports[m]=runtime.missing(m,d,'SYNTHETIC_DECLARED_ABSENCE')
        if case=='error':reports['eeg'].update(status='ERROR',algorithm_error=True,error='SYNTHETIC_PRODUCER_ERROR')
        log.register([d]);row=runtime.fuse_reports(d,reports);row['explicit_synthetic_demo']=True;log.result(row)





def self_test() -> dict:
    checks={}
    def test(name,fn):
        fn();checks[name]=True
    def rejects(fn):
        try:fn()
        except (ContractError,ValueError,TypeError):return
        raise AssertionError("Invalid input was accepted")
    template={"schema":SOURCE_SCHEMA,"window_id":"test_window","window_seconds":5.,"task_condition":"Speaking", "split":"replay",
              "modalities":{m:{"present":False} for m in MODALITIES}}
    test("explicit_missing_descriptor",lambda:validate_descriptor(template,live=False))
    test("all_three_modalities_required",lambda:rejects(lambda:validate_descriptor({**template,"modalities":{}},live=False)))
    test("listening_rejected",lambda:rejects(lambda:validate_descriptor({**template,"task_condition":"Listening"},live=False)))
    test("20s_not_silently_aggregated",lambda:rejects(lambda:validate_descriptor({**template,"window_seconds":20},live=False)))
    test("test_split_opt_in",lambda:rejects(lambda:validate_descriptor({**template,"split":"test"},live=False)))
    test("test_split_explicit_allow",lambda:validate_descriptor({**template,"split":"test","identity":{"subject":"subject03"}},live=False,allow_test=True))
    test("boolean_strings_rejected",lambda:rejects(lambda:boolean("false","mask")))
    test("fractional_mask_rejected",lambda:rejects(lambda:boolean(.6,"mask")))
    test("nan_config_number_rejected",lambda:rejects(lambda:finite(float("nan"),"quality")))
    test("instance_media_parser",lambda:require(media_identity(Path("010_Trial_6_Speaking_Anger.wav"))==(10,6,"speaking","Anger"),"parser"))
    test("instance_media_parser_aud_suffix",lambda:require(media_identity(Path("002_Trial_02_Speaking_Neutral_aud.wav"))==(2,2,"speaking","Neutral"),"audio suffix parser"))
    test("instance_media_parser_video_strict",lambda:require(media_identity(Path("002_Trial_01_Speaking_Neutral.mp4"))==(2,1,"speaking","Neutral"),"video parser"))
    test("subject_padding",lambda:require(normalized_subject("subject8")=="subject08","subject"))
    test("test_named_path_guard",lambda:rejects(lambda:reject_test_path("/data/test_window.wav",False)))
    with tempfile.TemporaryDirectory(prefix="eav_main_self_") as td:
        root=Path(td);jp=root/"config.json"
        write_new_json(jp,default_config())
        test("config_load_relative_paths",lambda:require(Path(load_config(jp)["modules"]["fusion"]["script"]).is_absolute(),"path"))
        test("no_output_overwrite",lambda:rejects_existing(jp))
        dup=root/"duplicate.json";dup.write_text('{"a":1,"a":2}')
        test("duplicate_json_key_rejected",lambda:rejects(lambda:read_json(dup)))
        amap=[{"old":"C:/old","new":str(root)}]
        test("explicit_path_relocation",lambda:require(resolve_path("C:/old/a.npy",root,amap)==root/"a.npy","map"))
    return {"status":"PASS","version":VERSION,"checks":checks,"n_checks":len(checks),
            "scope":"LOGIC_ONLY","trained_models_loaded":False,"raw_EAV_used":False,"hardware_tested":False}


def rejects_existing(path: Path):
    try:write_new_json(path,{})
    except FileExistsError:return
    raise AssertionError("Overwrite was not blocked")


def parse_args(argv: Sequence[str] | None = None) -> Any:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version',action='version',version=VERSION)
    g=p.add_mutually_exclusive_group()
    for opt in ('init-config','self-test','check-assets','check-models','dry-run','preflight','check-integration'):
        g.add_argument('--'+opt,action='store_true')
    g.add_argument('--mode',choices=('replay','demo','live'))
    p.add_argument('--config','--system-config',default='system_config.json')
    p.add_argument('--output',help='JSON for audits/config; NEW directory for inference/regression')
    p.add_argument('--release','--quality-release',help='Frozen quality_layer_release_candidate.json (KEEP_V1)')
    p.add_argument('--quality-adapter');p.add_argument('--quality-params')
    p.add_argument('--fusion-script');p.add_argument('--assets-dir');p.add_argument('--fusion-device',choices=('cpu','cuda'))
    p.add_argument('--cached-run',help='Existing raw_validation directory; --check-integration only, no raw inference')
    p.add_argument('--manifest',help='Existing source-window JSONL, not windows.jsonl result records')
    for name in ('stage0c-dir','video-manifest','audio-manifest','raw-root'):
        p.add_argument('--'+name)
    p.add_argument('--confirm-legacy-speaking',action='store_true')
    p.add_argument('--allow-test',action='store_true',help='Explicit one-way independent TEST evaluation gate; never enables fitting/tuning')
    p.add_argument('--split',choices=('train','val','test'),help='Explicit replay split override; TEST also requires --allow-test')
    p.add_argument('--stage0c-raw',action='store_true',help='Force Stage0C + raw-root source construction; clears inherited replay/media manifests')
    p.add_argument('--eeg-unit',choices=('V','mV','uV'));p.add_argument('--unit-evidence')
    p.add_argument('--device',choices=('cpu','cuda'),help='Explicit override ALL neural devices; use --fusion-device for fusion only')
    p.add_argument('--error-policy',choices=('raise','exclude'),help='raise stops after logging an error; exclude continues but never fuses a producer error')
    p.add_argument('--window-key');p.add_argument('--trial-key');p.add_argument('--limit',type=int)
    p.add_argument('--inbox');p.add_argument('--adapter',help='Trusted same-host acquisition adapter; used only with --mode live')
    p.add_argument('--session-id');p.add_argument('--clock-id')
    p.add_argument('--allow-synthetic-source',action='store_true',help='Explicit software fixtures; never hardware/accuracy evidence')
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for s in (sys.stdout,sys.stderr):
        if hasattr(s,'reconfigure'):s.reconfigure(encoding='utf-8',errors='backslashreplace')
    args=parse_args(argv)
    if args.limit is not None:require(args.limit>0,'--limit must be positive')
    require(not (args.trial_key and args.window_key),'Choose --trial-key OR --window-key')
    require(not args.cached_run or args.check_integration,'--cached-run belongs to --check-integration only')
    if args.init_config:
        require(args.output,'--init-config requires --output; never overwrite an existing configuration')
        write_new_json(args.output,default_config());print('CONFIG TEMPLATE WRITTEN: '+str(Path(args.output).resolve()));return 0
    if args.self_test:
        report=self_test();print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.output:write_new_json(args.output,report)
        return 0
    if not any((args.mode,args.check_assets,args.check_models,args.dry_run,args.preflight,args.check_integration)):
        parse_args(['--help']);return 0
    c=load_config(args.config,args)
    is_test=bool(args.allow_test)
    if is_test:
        require(c['replay']['split']=='test','--allow-test requires --split test / replay.split=test')
        require(args.dry_run or args.mode=='replay','TEST is one-way: use --dry-run cohort audit, then one full --mode replay benchmark')
        require(args.stage0c_raw,'Final TEST benchmark requires --stage0c-raw to avoid inherited VAL manifests')
        require(not any((args.limit,args.window_key,args.trial_key,args.manifest,args.preflight)),
                'Final TEST benchmark cannot be subset/preflighted or supplied from an ad-hoc manifest')
        require(c['replay'].get('stage0c_dir') and c['replay'].get('raw_root'),
                'Final TEST benchmark requires explicit --stage0c-dir and --raw-root')
        if args.mode=='replay':
            require(c['runtime']['error_policy']=='exclude',
                    'Final TEST benchmark must use --error-policy exclude so failures remain in the full denominator without aborting the cohort')
    else:
        require(c['replay']['split']!='test','TEST split requires explicit --allow-test')
    ds=None;cache=None;sid=None;cid=None
    if args.check_integration:
        require(args.cached_run,'--check-integration needs --cached-run (existing raw_validation directory)')
        require(not any((args.limit,args.window_key,args.trial_key)), 'Regression checks the whole provided cache; do not silently subset it')
        cache=integration_cache(Path(args.cached_run).resolve(),allow_synthetic=args.allow_synthetic_source)
    if args.check_assets:
        report=static_asset_audit(c)
        adapter,release=make_quality_adapter(c)
        report.update(main_version=VERSION,quality_identity=adapter.identity,release_sha256=release['release_sha256'],
                      selected_variant='KEEP_V1',old_router_executed=False,model_inference_performed=False)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.output:write_new_json(args.output,report)
        return 0 if report['status']=='PATHS_PASS' else 2
    if args.dry_run or args.preflight or args.mode=='replay':
        ds=collect_descriptors(c,args)
        if args.dry_run:
            cohort=cohort_summary(ds)
            if is_test:
                require(set(cohort['subjects'])==TEST_SUBJECTS,'TEST cohort must contain exactly the six frozen TEST subjects')
                require(all(d.get('split')=='test' for d in ds),'Mixed/non-TEST descriptor in final TEST cohort')
            report={'status':'PASS','scope':'FINAL_TEST_COHORT_AUDIT' if is_test else 'SOURCE_IDENTITIES_AND_PATHS_ONLY',
                    'selected_windows':len(ds),'main_version':VERSION,'selected_variant':'KEEP_V1',
                    'raw_samples_decoded':False,'models_loaded':False,'first_window':ds[0],
                    'test_used':is_test,'full_quality_integration_verified':False,'cohort':cohort,
                    'parameters_fitted':False,'post_test_tuning_authorized':False}
            print(json.dumps(report,ensure_ascii=False,indent=2))
            if args.output:write_new_json(args.output,report)
            return 0
        if is_test:
            cohort=cohort_summary(ds)
            require(set(cohort['subjects'])==TEST_SUBJECTS and all(d.get('split')=='test' for d in ds),
                    'Final TEST cohort identity mismatch')
        sid=args.session_id or 'main_replay_'+uuid.uuid4().hex
        for d in ds:
            # Preserve a recorded session. Never relabel an explicit quality span.
            if args.session_id and d.get('session_id') not in (None,args.session_id):
                raise ContractError('--session-id conflicts with recorded source session')
            d.setdefault('session_id',sid)
    if args.mode=='live':
        require(not (args.inbox and args.adapter),'Choose inbox OR acquisition adapter')
        sid=ident(args.session_id or 'capture_'+uuid.uuid4().hex,'session_id')
        cid=ident(args.clock_id or c['live'].get('clock_id'),'explicit same-host clock_id')
        require(args.inbox or args.adapter or c['live'].get('adapter'),'Live requires an explicit source; no automatic device activation')
    mode='cached' if args.check_integration else 'preflight' if args.preflight else 'model_load' if args.check_models else args.mode
    out=Path(args.output).expanduser().resolve() if args.output else Path(c['output_root'])/('main_keep_v1_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+str(mode))
    if args.cached_run:
        cp=Path(args.cached_run).resolve();require(not out.is_relative_to(cp) and not cp.is_relative_to(out),'Output must not contain or modify the regression cache')
    log=RunLog(out,c,mode);runtime=None;source=None;scheduler=None;extra={};status='FAILED';err=None
    if ds is not None:
        log.register(ds);write_new_json(out/'source_index.json',{'count':len(ds),'windows':ds})
        if is_test:
            write_new_json(out/'final_test_cohort.json',cohort_summary(ds))
    if cache is not None:
        for record in cache[1]:log.register([record['source_window']],condition=record['condition']['name'])
    try:
        runtime=Runtime(c,fusion_only=args.check_integration or args.mode=='demo',clock_id=cid,allow_test=is_test)
        write_new_json(out/'loaded_asset_identity.json',runtime.identities)
        extra.update(release_sha256=runtime.release['release_sha256'],policy_sha256=runtime.quality_adapter.identity['policy_sha256'])
        if args.check_models:
            status='MODELS_LOADED_NOT_END_TO_END_VALIDATED'
        elif args.check_integration:
            source_summary,records,hashes=cache
            check=run_cached(runtime,log,Path(args.cached_run).resolve(),source_summary,records,hashes)
            status=check['status'];extra.update(check)
        elif args.mode=='demo':
            run_demo(runtime,log,args.session_id or 'explicit_demo')
            status='PASS_DEMO_ONLY';extra['synthetic_source']=True
        elif args.preflight or args.mode=='replay':
            for d in ds:
                try:row=runtime.process_offline(d,allow_test=is_test)
                except Exception as exc:
                    row=integration_error_result(d,{},f'{type(exc).__name__}: {exc}',mode='OFFLINE_REPLAY')
                log.result(row)
                if row['status'] not in VALID_STATUSES and c['runtime']['error_policy']=='raise':
                    raise ModuleExecutionError('Window failed; original error is logged. No partial-modality fallback: '+d['window_id'])
            status='PASS' if not log.error_windows else 'COMPLETE_WITH_ERRORS'
            extra.update(raw_sensor_models_invoked=True,entrypoint_end_to_end_exercised=True,
                         integration_regression_requires_cached_check=not is_test,preflight_full_trial=(args.preflight and len(ds)==4),
                         test_data_used=is_test,final_test_benchmark=(is_test and args.mode=='replay'),
                         is_independent_accuracy_experiment=(is_test and args.mode=='replay'),
                         independent_test_subjects=sorted(TEST_SUBJECTS) if is_test else [],
                         benchmark_cohort=cohort_summary(ds) if is_test else None,
                         post_test_tuning_authorized=False,test_results_must_not_be_used_for_parameter_selection=is_test)
        elif args.mode=='live':
            print(f'[SHADOW INPUT] session_id={sid} clock_id={cid}; no robot control',flush=True)
            if args.inbox:source=FileInboxSource(Path(args.inbox).resolve(),session_id=sid,clock_id=cid)
            else:
                name=args.adapter or c['live']['adapter'];mod=module_from_path(resolve_path(name,Path(c['_base'])),'acquisition')
                synthetic=getattr(mod,'SYNTHETIC_SOURCE',False) is True
                require(not synthetic or args.allow_synthetic_source,'Synthetic acquisition needs --allow-synthetic-source')
                extra['synthetic_source']=synthetic
                source=mod.open_source(copy.deepcopy(c['live']['settings']),session_id=sid,clock_id=cid,clock=time.monotonic)
            require(callable(getattr(source,'poll',None)) and callable(getattr(source,'close',None)),'Source needs poll(timeout_sec)/close()')
            scheduler=LiveScheduler(runtime,session_id=sid,clock_id=cid)
            while True:
                log.expire_current_state()
                for row in scheduler.poll():log.result(row)
                for e in scheduler.events:log.event(e)
                scheduler.events.clear()
                if args.limit and log.count>=args.limit:break
                if getattr(source,'ended',False) and not scheduler.windows:break
                if len(scheduler.windows)<c['runtime']['max_pending'] and not getattr(source,'ended',False):
                    d=source.poll(timeout_sec=0.)
                    if d is not None:
                        scheduler.submit_window(d);log.register([d])
                time.sleep(c['runtime']['poll_interval_sec'])
            status='PASS_SHADOW_SOFTWARE_ONLY' if not log.error_windows and not log.error_events else 'COMPLETE_WITH_ERRORS'
            extra.update(live_quality_mode='shadow',robot_actions_authorized=False)
        extra['frozen_assets_after']=runtime.verify_unchanged()
    except KeyboardInterrupt:
        status='INTERRUPTED';err='KeyboardInterrupt; pending evidence discarded'
    except Exception as exc:
        status='FAILED';err=f'{type(exc).__name__}: {exc}'
        log.event({'event':'FATAL','error':err,'traceback':traceback.format_exc()})
        write_new_json(out/'fatal_error.json',{'error':err,'traceback':traceback.format_exc()})
        print(err,file=sys.stderr)
    finally:
        if scheduler is not None:
            extra['shutdown']=scheduler.close()
            if not all(extra['shutdown']['workers_stopped'].values()) and status=='PASS_SHADOW_SOFTWARE_ONLY':status='COMPLETE_WITH_ACTIVE_WORKERS'
        if source is not None:
            try:source.close()
            except Exception as exc:
                log.event({'event':'SOURCE_CLOSE_ERROR','error':repr(exc)});status='COMPLETE_WITH_ERRORS'
        if runtime is not None:extra['raw_modality_invocation_counts']={m:runtime.calls[m] for m in MODALITIES}
        assets_verified = 'frozen_assets_after' in extra
        extra.update(quality_modules_modified=False if assets_verified else None,
                     weights_modified=False if assets_verified else None,
                     assets_verified_after_run=assets_verified,parameters_fitted=False,
                     performance_status='FINAL_TEST_REPORTED_NO_TUNING' if is_test and args.mode=='replay' else 'NOT_SPECIFIED',
                     test_controller_gate_enabled=is_test,quality_policy_modified_for_test=False,
                     main_integrated_code=True)
        # Status cannot be overridden by an earlier intermediate success report.
        extra.pop('status',None)
        log.finish(status=status,error=err,extra=extra)
    return 0 if status in ('PASS','PASS_DEMO_ONLY','PASS_SHADOW_SOFTWARE_ONLY','MODELS_LOADED_NOT_END_TO_END_VALIDATED') else 130 if status=='INTERRUPTED' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status':'MAIN_INTEGRATION_ERROR','version':VERSION,'error_type':type(exc).__name__,
            'error':str(exc),'production_approved':False},ensure_ascii=False,indent=2),file=sys.stderr)
        raise SystemExit(2)






