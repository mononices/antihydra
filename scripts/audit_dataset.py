#!/usr/bin/env python3
"""Audit challenge data for flag leaks into the agent sandbox.

Run: python scripts/audit_dataset.py [root]   (default: competition/2025)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_ROOT = "competition/2025"


def _inside(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def audit(root: Path) -> int:
    root = root.resolve()
    problems: list[tuple[str, str]] = []
    meta_files = sorted(root.rglob("challenge.json"))
    for meta_file in meta_files:
        base = meta_file.parent.resolve()
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append((str(meta_file), f"unreadable challenge.json: {exc}"))
            continue
        name = str(metadata.get("name") or base.name)
        flag = str(metadata.get("flag") or "")
        description = str(metadata.get("description") or "")

        if flag and flag in description:
            problems.append((name, "flag value appears in the description"))

        for entry in metadata.get("files") or []:
            target = (base / str(entry)).resolve()
            if not _inside(target, base):
                problems.append(
                    (name, f"files entry escapes the challenge directory: {entry}")
                )
                continue
            if not target.exists():
                problems.append((name, f"files entry missing on disk: {entry}"))
                continue

            candidates = (
                [target]
                if target.is_file()
                else [p for p in target.rglob("*") if p.is_file()]
            )
            for path in candidates:
                relative = path.relative_to(base)
                if path.name == "challenge.json":
                    problems.append(
                        (name, f"copied path would include challenge.json: {relative}")
                    )
                if flag:
                    try:
                        data = path.read_bytes()
                    except OSError:
                        continue
                    if flag.encode("utf-8") in data:
                        problems.append(
                            (
                                name,
                                "flag string found in a file that gets copied into the "
                                f"sandbox: {relative}",
                            )
                        )

    print(f"audited {len(meta_files)} challenge(s) under {root}")
    if not problems:
        print("ok: no flag leaks in the files copied into the sandbox")
        return 0
    for name, issue in problems:
        print(f"  [{name}] {issue}")
    print(f"{len(problems)} issue(s) found")
    return 1


def main(argv: list[str]) -> int:
    return audit(Path(argv[1] if len(argv) > 1 else DEFAULT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
