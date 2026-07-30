"""Download one CPython Linux wheel with resumable HTTP range requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "BetAIPlatform/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("package index returned an invalid response")
    return payload


def _select_wheel(payload: dict[str, Any], version: str) -> dict[str, Any]:
    candidates = payload.get("urls")
    if not isinstance(candidates, list):
        releases = payload.get("releases")
        candidates = releases.get(version) if isinstance(releases, dict) else None
    if not isinstance(candidates, list):
        raise RuntimeError(f"package index has no release for {version}")

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine().lower()
    supported_machine = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(machine)
    if supported_machine is None:
        raise RuntimeError(f"unsupported wheel architecture: {machine}")

    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("packagetype") == "bdist_wheel"
        and python_tag in str(candidate.get("filename", ""))
        and "manylinux" in str(candidate.get("filename", ""))
        and supported_machine in str(candidate.get("filename", ""))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {python_tag}/{supported_machine} wheel, found {len(matches)}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    *,
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    attempts: int,
) -> None:
    partial = destination.with_suffix(f"{destination.suffix}.part")
    partial.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        current_size = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "BetAIPlatform/1.0"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if current_size and status != 206:
                    current_size = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if current_size else "wb"
                with partial.open(mode) as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
        except (OSError, urllib.error.URLError) as exc:
            if attempt == attempts:
                raise RuntimeError("wheel download failed after retries") from exc
            time.sleep(min(attempt * 2, 10))
            continue

        actual_size = partial.stat().st_size
        if actual_size == expected_size:
            if _sha256(partial) != expected_sha256:
                partial.unlink(missing_ok=True)
                raise RuntimeError("downloaded wheel checksum mismatch")
            os.replace(partial, destination)
            return
        if actual_size > expected_size:
            partial.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(attempt * 2, 10))

    raise RuntimeError("wheel download remained incomplete after retries")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("version")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--attempts", type=int, default=12)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")

    payload = _json(f"https://pypi.org/pypi/{args.package}/{args.version}/json")
    wheel = _select_wheel(payload, args.version)
    filename = str(wheel.get("filename", ""))
    url = str(wheel.get("url", ""))
    size = wheel.get("size")
    digests = wheel.get("digests")
    sha256 = digests.get("sha256") if isinstance(digests, dict) else None
    if (
        not filename.endswith(".whl")
        or not url.startswith("https://")
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
    ):
        raise RuntimeError("package index wheel metadata is incomplete")

    destination = args.destination.resolve() / filename
    _download(
        url=url,
        destination=destination,
        expected_size=size,
        expected_sha256=sha256,
        attempts=args.attempts,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
