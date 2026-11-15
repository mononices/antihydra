#!/usr/bin/env python3
"""Command-line entry point for parallel racing."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from antihydra.challenge import ChallengeError, resolve, safe_dir_name
from antihydra.compaction import CompactionConfig
from antihydra.env import load_dotenv
from antihydra.llm import REASONING_EFFORTS, LLMError
from antihydra.overseer import OverseerConfig
from antihydra.runtime import Runtime
from antihydra.racing import Lane, RaceCoordinator, lane_label, load_lane_specs
from antihydra.sandbox import SandboxError, find_docker
from antihydra.stalls import StallConfig

DEFAULT_ROOT = "competition/2025"
REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antihydra-race",
        description="Run several solver lanes against one challenge; first flag wins",
    )
    parser.add_argument(
        "--root", default=DEFAULT_ROOT, help="challenge root (default: %(default)s)"
    )
    parser.add_argument(
        "--challenge", required=True, help="challenge name or directory"
    )
    parser.add_argument("--lanes", nargs="+", default=[], help="model id per lane")
    parser.add_argument(
        "--lanes-file",
        help="TOML or JSON roster (default: lanes.toml when present)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="default lane temperature (default: 0.7 or lanes file)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="reasoning effort for all lanes (default: lanes file, else max)",
    )
    parser.add_argument(
        "--flag-hint",
        choices=["generic", "branded"],
        default=None,
        help="how prompts describe the flag: 'generic' shape or the literal "
        "'branded' prefix (default: lanes file, else generic)",
    )
    parser.add_argument(
        "--lane-max-cost",
        type=float,
        default=None,
        help="USD cap per lane (default: 4.0 or lanes file)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="round cap per lane (default: 200 or lanes file)",
    )
    parser.add_argument(
        "--no-subtasks",
        action="store_true",
        default=None,
        help="disable subtasks in all lanes",
    )
    parser.add_argument("--image", default="ctfenv:antihydra", help="sandbox image")
    parser.add_argument("--platform", default="linux/amd64", help="sandbox platform")
    parser.add_argument("--network", default="ctfnet", help="docker network")
    parser.add_argument("--lane-memory", default="4g", help="memory limit per lane")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="command timeout per lane"
    )
    parser.add_argument(
        "--max-command-timeout",
        type=float,
        default=600.0,
        help=(
            "hard ceiling for one run_command, even when the model asks for more "
            "(default 600s); commands are also clamped to the remaining race budget"
        ),
    )
    parser.add_argument(
        "--max-minutes", type=float, default=None, help="stop the race after N minutes"
    )
    parser.add_argument(
        "--race-dir", default="logs/races", help="where race artifacts go"
    )
    parser.add_argument(
        "--api-key", default=None, help="OpenRouter key (or OPENROUTER_API_KEY)"
    )
    parser.add_argument(
        "--compact-model",
        default=None,
        help="compaction model (or OPENROUTER_COMPACT_MODEL)",
    )
    parser.add_argument(
        "--no-compaction", action="store_true", help="disable context compaction"
    )
    parser.add_argument(
        "--no-stall-detection", action="store_true", help="disable stall detection"
    )
    parser.add_argument(
        "--no-overseer", action="store_true", help="disable the overseer"
    )
    parser.add_argument(
        "--overseer-model",
        default=None,
        help="overseer model (or OPENROUTER_OVERSEER_MODEL; default: z-ai/glm-5.3-flash)",
    )
    parser.add_argument(
        "--overseer-effort",
        default=None,
        help="overseer reasoning effort (or OPENROUTER_OVERSEER_EFFORT; default: max)",
    )
    parser.add_argument(
        "--overseer-interval",
        type=float,
        default=120.0,
        help="seconds between scheduled reviews (default: %(default)s)",
    )
    parser.add_argument(
        "--overseer-max-cost",
        type=float,
        default=1.0,
        help="USD budget for overseer calls (default: %(default)s)",
    )
    parser.add_argument(
        "--overseer-max-calls",
        type=int,
        default=25,
        help="max overseer LLM calls per race (default: %(default)s)",
    )
    parser.add_argument(
        "--no-service", action="store_true", help="never start the challenge service"
    )
    parser.add_argument(
        "--no-follow", action="store_true", help="do not stream lane output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    try:
        challenge = resolve(args.root, args.challenge)
        challenge.require_delivered_files()
        lanes_file = args.lanes_file
        if not lanes_file and not args.lanes and (REPO_ROOT / "lanes.toml").is_file():
            lanes_file = REPO_ROOT / "lanes.toml"
        specs = load_lane_specs(
            args.lanes,
            lanes_file,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            flag_hint=args.flag_hint,
            max_cost=args.lane_max_cost,
            max_rounds=args.max_rounds,
            no_subtasks=args.no_subtasks,
        )
    except (ChallengeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "error: no API key: pass --api-key or set OPENROUTER_API_KEY",
            file=sys.stderr,
        )
        return 2
    compact_model = args.compact_model or os.environ.get("OPENROUTER_COMPACT_MODEL")

    try:
        docker = find_docker()
    except SandboxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    runtime = Runtime(api_key)
    compaction_llm = runtime.compaction_llm(compact_model) if compact_model else None

    race_dir = Path(args.race_dir) / (
        f"{time.strftime('%Y%m%d-%H%M%S')}__{safe_dir_name(challenge.name)}"
    )
    compaction = CompactionConfig(enabled=not args.no_compaction)
    stall = StallConfig(enabled=not args.no_stall_detection)

    overseer = None
    if not args.no_overseer:
        overseer_model = args.overseer_model or os.environ.get(
            "OPENROUTER_OVERSEER_MODEL", "z-ai/glm-5.3-flash"
        )
        overseer_effort = args.overseer_effort or os.environ.get(
            "OPENROUTER_OVERSEER_EFFORT", "max"
        )
        if overseer_effort not in REASONING_EFFORTS:
            print(
                f"error: invalid --overseer-effort {overseer_effort!r}; "
                f"use one of: {', '.join(REASONING_EFFORTS)}",
                file=sys.stderr,
            )
            return 2
        overseer = runtime.overseer(
            OverseerConfig(
                model=overseer_model,
                reasoning_effort=overseer_effort,
                interval_seconds=args.overseer_interval,
                max_calls=args.overseer_max_calls,
                max_cost=args.overseer_max_cost,
            )
        )

    def lane_factory(spec, lane_dir, emit, hints):
        return Lane(
            spec,
            challenge,
            llm=runtime.llm(
                spec.model, spec.reasoning_effort, temperature=spec.temperature
            ),
            compaction_llm=compaction_llm,
            docker=docker,
            image=args.image,
            network=args.network,
            memory=args.lane_memory,
            platform=args.platform,
            timeout=args.timeout,
            max_command_timeout=args.max_command_timeout,
            log_dir=lane_dir,
            sandbox_label=lane_label(race_dir.name, spec),
            emit=emit,
            hints=hints,
            compaction=compaction,
            stall=stall,
        )

    coordinator = RaceCoordinator(
        challenge,
        specs,
        race_dir=race_dir,
        lane_factory=lane_factory,
        docker=docker,
        network=args.network,
        memory=args.lane_memory,
        start_service=not args.no_service,
        follow=not args.no_follow,
        overseer=overseer,
        max_minutes=args.max_minutes,
    )
    try:
        report = coordinator.run()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ChallengeError, SandboxError, LLMError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0 if report.get("solved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
