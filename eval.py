#!/usr/bin/env python3
"""Command-line entry point for the evaluation harness."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from antihydra.challenge import Challenge, ChallengeError, discover, resolve
from antihydra.compaction import CompactionConfig
from antihydra.env import load_dotenv
from antihydra.evaluation import EvalConfig, prebuild_images, run_eval
from antihydra.llm import REASONING_EFFORTS, LLMError
from antihydra.overseer import OverseerConfig
from antihydra.racing import Lane, RaceCoordinator, lane_label, load_lane_specs
from antihydra.runtime import Runtime
from antihydra.sandbox import SandboxError, find_docker, warn_memory
from antihydra.stalls import StallConfig

DEFAULT_ROOT = "competition/2025"
REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antihydra-eval",
        description="Run the racing solver across a challenge set and aggregate results",
    )
    parser.add_argument(
        "--root", default=DEFAULT_ROOT, help="challenge root (default: %(default)s)"
    )
    parser.add_argument(
        "--eval-dir", default="logs/evals", help="where eval artifacts go"
    )
    parser.add_argument(
        "--challenge",
        action="append",
        default=[],
        help="limit to a named challenge (repeatable)",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="limit to categories (repeatable)",
    )
    parser.add_argument("--limit", type=int, default=None, help="max challenges to run")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="challenges to run in parallel (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="reuse existing race summaries"
    )
    parser.add_argument(
        "--skip-build-errors",
        action="store_true",
        help=(
            "continue when a challenge image fails to build instead of aborting; "
            "the affected challenges stop before spending anything"
        ),
    )
    parser.add_argument(
        "--include-remote-only",
        action="store_true",
        help="run challenges that declare a remote service but ship no local compose file",
    )
    parser.add_argument(
        "--comment", default="", help="leaderboard comment (default: pass@k)"
    )
    parser.add_argument("--lanes", nargs="+", default=[], help="model id per lane")
    parser.add_argument(
        "--lanes-file", help="TOML or JSON roster (default: lanes.toml)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="default lane temperature"
    )
    parser.add_argument(
        "--reasoning-effort", default=None, help="default lane reasoning effort"
    )
    parser.add_argument(
        "--flag-hint",
        choices=["generic", "branded"],
        default=None,
        help="how prompts describe the flag: 'generic' shape (default) or the "
        "literal 'branded' prefix from challenge.json",
    )
    parser.add_argument(
        "--lane-max-cost", type=float, default=None, help="USD cap per lane"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None, help="round cap per lane"
    )
    parser.add_argument(
        "--no-subtasks",
        action="store_true",
        default=None,
        help="disable subtasks in all lanes",
    )
    parser.add_argument(
        "--api-key", default=None, help="OpenRouter key (or OPENROUTER_API_KEY)"
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
        "--max-minutes", type=float, default=None, help="per-challenge time cap"
    )
    parser.add_argument(
        "--max-total-cost",
        type=float,
        default=None,
        help="stop the eval after this spend",
    )
    parser.add_argument(
        "--compact-model", default=None, help="compaction model (or env)"
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
        "--overseer-model", default=None, help="overseer model (or env)"
    )
    parser.add_argument(
        "--overseer-effort", default=None, help="overseer reasoning effort (or env)"
    )
    parser.add_argument(
        "--overseer-interval",
        type=float,
        default=120.0,
        help="seconds between overseer reviews",
    )
    parser.add_argument(
        "--overseer-max-cost",
        type=float,
        default=1.0,
        help="overseer USD cap per challenge",
    )
    parser.add_argument(
        "--overseer-max-calls",
        type=int,
        default=25,
        help="overseer call cap per challenge",
    )
    parser.add_argument(
        "--no-follow", action="store_true", help="do not stream lane output"
    )
    return parser


def select_challenges(args: argparse.Namespace) -> list[Challenge]:
    if args.challenge:
        return [resolve(args.root, name) for name in args.challenge]
    challenges = discover(args.root)
    if args.category:
        wanted = {category.lower() for category in args.category}
        challenges = [
            challenge
            for challenge in challenges
            if challenge.category.lower() in wanted
        ]
    if not args.include_remote_only:
        runnable = []
        for challenge in challenges:
            if challenge.is_remote_only():
                print(
                    f"[skip] {challenge.name}: declares a public service but ships no "
                    f"local compose file; pass --include-remote-only to run it anyway",
                    file=sys.stderr,
                )
                continue
            runnable.append(challenge)
        challenges = runnable
    if args.limit:
        challenges = challenges[: args.limit]
    return challenges


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    try:
        challenges = select_challenges(args)
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
    if not challenges:
        print("error: no challenges selected", file=sys.stderr)
        return 2

    broken = False
    for challenge in challenges:
        try:
            challenge.require_delivered_files()
        except ChallengeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            broken = True
    if broken:
        return 3

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "error: no API key: pass --api-key or set OPENROUTER_API_KEY",
            file=sys.stderr,
        )
        return 2
    try:
        docker = find_docker()
    except SandboxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    runtime = Runtime(api_key)
    warn_memory(docker, args.lane_memory, len(specs) * args.concurrency, print)
    compact_model = args.compact_model or os.environ.get("OPENROUTER_COMPACT_MODEL")
    compaction_llm = runtime.compaction_llm(compact_model) if compact_model else None
    compaction = CompactionConfig(enabled=not args.no_compaction)
    stall = StallConfig(enabled=not args.no_stall_detection)

    overseer_config = None
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
        overseer_config = OverseerConfig(
            model=overseer_model,
            reasoning_effort=overseer_effort,
            interval_seconds=args.overseer_interval,
            max_calls=args.overseer_max_calls,
            max_cost=args.overseer_max_cost,
        )

    stop_event = threading.Event()
    eval_dir = Path(args.eval_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}__eval"
    build_failures = prebuild_images(
        docker, challenges, eval_dir, skip_existing=args.skip_existing
    )
    if build_failures and not args.skip_build_errors:
        print(
            f"error: {len(build_failures)} challenge image(s) failed to build:",
            file=sys.stderr,
        )
        for directory, message in build_failures.items():
            print(f"  {directory}: {message}", file=sys.stderr)
        return 3
    if build_failures:
        print(
            "warning: continuing past "
            f"{len(build_failures)} failed build(s):",
            file=sys.stderr,
        )
        for directory in build_failures:
            print(f"  {directory}", file=sys.stderr)
        print("  see build.log in each race dir", file=sys.stderr)

    def coordinator_factory(challenge: Challenge, run_dir: Path) -> RaceCoordinator:
        if str(challenge.directory) in build_failures:
            raise ChallengeError(
                f"challenge images were not built; see {run_dir / 'build.log'}"
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
                sandbox_label=lane_label(run_dir.name, spec),
                emit=emit,
                hints=hints,
                compaction=compaction,
                stall=stall,
            )

        return RaceCoordinator(
            challenge,
            specs,
            race_dir=run_dir,
            lane_factory=lane_factory,
            docker=docker,
            network=args.network,
            memory=args.lane_memory,
            start_service=True,
            build_images=False,
            follow=not args.no_follow,
            overseer=runtime.overseer(overseer_config) if overseer_config else None,
            max_minutes=args.max_minutes,
            prefix=f"[{challenge.name}] ",
            stop_event=stop_event,
        )

    try:
        summary = run_eval(
            challenges,
            eval_dir=eval_dir,
            coordinator_factory=coordinator_factory,
            stop_event=stop_event,
            config=EvalConfig(
                lane_models=[spec.model for spec in specs],
                max_minutes_per_challenge=args.max_minutes,
                max_total_cost=args.max_total_cost,
                skip_existing=args.skip_existing,
                concurrency=args.concurrency,
                comment=args.comment,
                root=str(args.root),
                network=args.network,
                flag_hint=specs[0].flag_hint if specs else "generic",
            ),
        )
    except (ChallengeError, SandboxError, LLMError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    errors = [record for record in summary["results"] if record.get("error")]
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
