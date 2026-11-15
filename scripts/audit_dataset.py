#!/usr/bin/env python3
"""Audit challenge data for flag leaks into the agent sandbox.

Run: python scripts/audit_dataset.py [--root competition/2026]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOT = "competition/2025"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root_positional",
        nargs="?",
        default=None,
        help="challenge root (kept for compatibility; prefer --root)",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="challenge root (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.root_positional:
        args.root = args.root_positional
    return args


def _inside(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def audit(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        print(f"error: no such challenge root: {root}", file=sys.stderr)
        return 2
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
    if not meta_files:
        # An empty audit must never read as a clean one.
        print(
            f"error: no challenge.json under {root}; pass --root with the set you "
            "mean to audit",
            file=sys.stderr,
        )
        return 2
    if not problems:
        print("ok: no flag leaks in the files copied into the sandbox")
        return 0
    for name, issue in problems:
        print(f"  [{name}] {issue}")
    print(f"{len(problems)} issue(s) found")
    return 1


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    return audit(Path(args.root))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
