from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SECRET_NAMES = (
    "API_FOOTBALL_KEY",
    "SPORTMONKS_API_TOKEN",
    "JWT_SECRET_KEY",
    "JWT_REFRESH_SECRET_KEY",
    "MODEL_SIGNING_KEY",
    "ADMIN_PASSWORD",
)
ASSIGNMENT = re.compile(
    rf"(?P<name>{'|'.join(SECRET_NAMES)})\s*[:=]\s*[\"']?(?P<value>[^\s\"',}}]+)"
)
SAFE_MARKERS = (
    "${",
    "${{",
    "settings.",
    "os.environ",
    "Field(",
    "DEMO_KEY",
    "change-this",
    "development-",
    "your_",
    "replace_with",
    "replace-with",
    "minimum_",
    "minimum-",
    "live_api_key",
    "ci-",
)


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def find_exposed_secrets(root: Path) -> list[str]:
    findings: list[str] = []
    for path in tracked_files(root):
        if not path.is_file() or path.suffix.lower() in {
            ".png",
            ".jpg",
            ".ico",
            ".lock",
        }:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for match in ASSIGNMENT.finditer(line):
                value = match.group("value").strip()
                if not value or any(
                    marker in value or marker in line for marker in SAFE_MARKERS
                ):
                    continue
                if len(value) >= 20:
                    findings.append(
                        f"{path.relative_to(root)}:{line_number}:{match.group('name')}"
                    )
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = find_exposed_secrets(root)
    if findings:
        print("Potential committed secrets detected:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("No committed secrets detected in tracked text files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
