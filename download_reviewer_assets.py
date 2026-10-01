#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Download and verify MERIT GitHub Release assets.
#
# Run from the MERIT repository root:
#   python download_reviewer_assets.py
#
# Options:
#   --check          Verify local assets only; perform no downloads.
#   --base-url URL   Override the release base URL.
#   --only NAME      Download/check one release asset filename (repeatable).
#
# Downloads use a .part file, verify byte size and SHA-256, then atomically
# replace the final repository-relative target.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

VERSION = "MERIT-ASSET-DOWNLOADER.1.0"


class AssetError(RuntimeError):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise AssetError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, size: int, sha: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(size)
        and digest(path).lower() == str(sha).lower()
    )


def safe_target(root: Path, rel: str) -> Path:
    p = Path(rel)
    require(not p.is_absolute() and ".." not in p.parts, f"Unsafe asset path: {rel}")
    target = (root / p).resolve()
    require(root == target or root in target.parents, f"Asset escaped repository root: {rel}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="RELEASE_ASSETS_MANIFEST.json")
    ap.add_argument("--base-url")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    require(manifest_path.is_file(), f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(manifest.get("schema") == "merit.github_release_assets.v2",
            f"Unsupported manifest schema: {manifest.get('schema')!r}")
    assets = manifest.get("assets")
    require(isinstance(assets, list) and assets, "Manifest has no assets")

    selected = set(args.only)
    if selected:
        known = {str(x.get("release_name")) for x in assets}
        unknown = selected - known
        require(not unknown, "Unknown --only release asset(s): " + ", ".join(sorted(unknown)))
        assets = [x for x in assets if x.get("release_name") in selected]

    base = (args.base_url or manifest.get("release_base_url") or "").rstrip("/")
    if not args.check:
        require(base.startswith("https://"), "A HTTPS release base URL is required")

    passed = downloaded = missing = 0

    for i, asset in enumerate(assets, 1):
        rel = str(asset["path"])
        name = str(asset["release_name"])
        size = int(asset["size_bytes"])
        sha = str(asset["sha256"])
        target = safe_target(root, rel)

        if verify(target, size, sha):
            print(f"[{i:02d}/{len(assets):02d}] PASS  {rel}")
            passed += 1
            continue

        if args.check:
            print(f"[{i:02d}/{len(assets):02d}] MISS  {rel}")
            missing += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        part.unlink(missing_ok=True)

        url = base + "/" + urllib.parse.quote(name, safe="-._~")
        print(f"[{i:02d}/{len(assets):02d}] GET   {name} -> {rel}")

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MERIT-reviewer-asset-downloader/1.0"},
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    total += len(block)
                    if total > size:
                        raise AssetError(f"Download exceeds expected size for {name}")
        except Exception:
            part.unlink(missing_ok=True)
            raise

        require(part.stat().st_size == size,
                f"Downloaded size mismatch for {name}: {part.stat().st_size} != {size}")
        require(digest(part).lower() == sha.lower(),
                f"SHA-256 mismatch for downloaded asset: {name}")

        os.replace(part, target)
        require(verify(target, size, sha), f"Post-install verification failed: {rel}")
        downloaded += 1
        print(f"             VERIFIED SHA-256 {sha}")

    print()
    print("=" * 78)
    print("MERIT RELEASE ASSET STATUS")
    print("=" * 78)
    print(f"Verified existing : {passed}")
    print(f"Downloaded        : {downloaded}")
    print(f"Missing (--check) : {missing}")
    print(f"Total selected    : {len(assets)}")
    print("=" * 78)

    if args.check and missing:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ASSET ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
