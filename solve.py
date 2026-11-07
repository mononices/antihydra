#!/usr/bin/env python3
"""Command-line entry point for the antihydra solver."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from antihydra.agent import Agent
from antihydra.blackboard import Blackboard
from antihydra.challenge import ChallengeError, discover, resolve
from antihydra.compaction import CompactionConfig
from antihydra.env import load_dotenv
from antihydra.llm import (
    REASONING_EFFORTS,
    LLMError,
    load_catalog,
    normalize_model,
)
from antihydra.prompts import initial_prompt, system_prompt
from antihydra.runtime import Runtime
from antihydra.sandbox import (
    LEASE_LABEL,
    Sandbox,
    SandboxError,
    cleanup_orphans,
    daemon_version,
    ensure_network,
    find_docker,
)
from antihydra.stalls import StallConfig
from antihydra.tools import CostLedger, build_main_tools

DEFAULT_ROOT = "competition/2025"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antihydra-solve",
        description="Independent LLM CTF solver for CSAW/NYU-style challenges",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="challenge root directory (default: %(default)s)",
    )
    parser.add_argument("--challenge", help="challenge name or directory")
    parser.add_argument("--list", action="store_true", help="list challenges and exit")
    parser.add_argument(
        "--show",
        action="store_true",
        help="print challenge details and agent prompts, then exit (no docker)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="start the sandbox, copy files, print prompts; no model calls",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list OpenRouter models and prices, then exit",
    )
    parser.add_argument("--grep", help="filter for --list / --list-models")
    parser.add_argument(
        "--limit", type=int, default=30, help="max rows for --list-models"
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="force a refresh of the cached model price catalog",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove orphan sandbox containers and exit",
    )

    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL"),
        help="model id, e.g. openrouter/anthropic/claude-opus-4.8 (or OPENROUTER_MODEL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or OPENROUTER_API_KEY)",
    )
    parser.add_argument("--image", default="ctfenv:antihydra", help="sandbox image")
    parser.add_argument("--platform", default="linux/amd64", help="sandbox platform")
    parser.add_argument(
        "--network", default="ctfnet", help="docker network for the sandbox"
    )
    parser.add_argument("--memory", default="8g", help="sandbox memory limit")
    parser.add_argument(
        "--sandbox-label",
        default=LEASE_LABEL,
        help="docker label key for the sandbox container (default: %(default)s)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=200, help="model calls per run"
    )
    parser.add_argument(
        "--max-cost", type=float, default=3.0, help="USD budget per run"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENROUTER_REASONING_EFFORT", "max"),
        help="reasoning effort (or OPENROUTER_REASONING_EFFORT; default: %(default)s)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="default command timeout (s)"
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--no-service",
        action="store_true",
        help="never start the challenge docker-compose service",
    )
    parser.add_argument(
        "--no-subtasks",
        action="store_true",
        help="disable spawn_subtask (fresh-context helper agents)",
    )
    parser.add_argument(
        "--max-subtasks",
        type=int,
        default=4,
        help="max subtask workers per run (default: %(default)s)",
    )
    parser.add_argument(
        "--subtask-rounds",
        type=int,
        default=30,
        help="max model calls per subtask worker (default: %(default)s)",
    )
    parser.add_argument(
        "--subtask-cost",
        type=float,
        default=0.75,
        help="USD cap per subtask worker (default: %(default)s)",
    )
    parser.add_argument(
        "--no-compaction",
        action="store_true",
        help="disable context compaction (summarising old history)",
    )
    parser.add_argument(
        "--compact-model",
        default=os.environ.get("OPENROUTER_COMPACT_MODEL"),
        help="compaction model (or OPENROUTER_COMPACT_MODEL; default: same as --model)",
    )
    parser.add_argument(
        "--no-stall-detection",
        action="store_true",
        help="disable the repeated-call stall detector",
    )
    parser.add_argument(
        "--stall-repeat",
        type=int,
        default=3,
        help="identical calls before the stall detector intervenes (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _list_models(args: argparse.Namespace) -> int:
    catalog = load_catalog(refresh=args.refresh_catalog)
    if not catalog:
        print("could not load the OpenRouter model catalog", file=sys.stderr)
        return 3
    rows = sorted(catalog.values(), key=lambda price: (price.prompt, price.id))
    if args.grep:
        needle = args.grep.lower()
        rows = [row for row in rows if needle in row.id.lower()]
    label = f" matching {args.grep!r}" if args.grep else ""
    print(f"{len(rows)} model(s){label}:")
    for row in rows[: args.limit]:
        context = f"{row.context_length:>9,}" if row.context_length else "        ?"
        print(
            f"  {row.id:<66} ctx={context}"
            f"  ${row.prompt * 1e6:>8.2f}/M in  ${row.completion * 1e6:>8.2f}/M out"
        )
    if len(rows) > args.limit:
        print(f"  ... {len(rows) - args.limit} more (use --grep or --limit)")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        if args.cleanup:
            docker = find_docker()
            removed = cleanup_orphans(docker)
            print(f"removed {removed} orphan sandbox container(s)")
            return 0

        if args.list_models:
            return _list_models(args)

        challenges = discover(args.root)
        if args.list:
            if args.grep:
                needle = args.grep.lower()
                challenges = [
                    challenge
                    for challenge in challenges
                    if needle in challenge.name.lower()
                    or needle in challenge.category.lower()
                ]
            for challenge in challenges:
                service = "service" if challenge.compose_file() else "files"
                print(
                    f"{challenge.category:<10} {challenge.name:<34} "
                    f"[{service:<7}] {challenge.directory}"
                )
            print(f"\n{len(challenges)} challenge(s) under {Path(args.root).resolve()}")
            return 0

        if not args.challenge:
            print(
                "nothing to do: pass --challenge, or use --list / --list-models / --cleanup",
                file=sys.stderr,
            )
            return 2
        challenge = resolve(args.root, args.challenge)

        if args.show:
            print(f"challenge: {challenge.name} [{challenge.category}]")
            print(f"directory: {challenge.directory}")
            print(f"target:    {challenge.target_line() or '-'}")
            print(f"files:     {', '.join(challenge.files) or '-'}")
            print(f"compose:   {challenge.compose_file() or '-'}")
            print(f"flag hint: {challenge.flag_format_hint() or '-'}")
            print("\n----- system prompt -----\n")
            print(system_prompt(challenge, args.timeout))
            print("\n----- initial prompt -----\n")
            print(initial_prompt(challenge))
            return 0

        challenge.require_delivered_files()
        docker = find_docker()
        daemon_version(docker)
        ensure_network(docker, args.network)

        service = None
        if not args.no_service and not args.dry_run and challenge.compose_file():
            print(f"starting challenge service ({challenge.compose_file()})")
            service = challenge.start_service(docker)
            target = challenge.discover_local_target(docker)
            if target:
                print(f"service reachable from the sandbox at {target}")

        sandbox = Sandbox(
            image=args.image,
            docker=docker,
            network=args.network,
            platform=args.platform,
            memory=args.memory,
            label=args.sandbox_label,
        )
        try:
            print(f"starting sandbox from {args.image} ...")
            sandbox.start()
            challenge.install_files(sandbox)

            if args.dry_run:
                print("\n----- system prompt -----\n")
                print(system_prompt(challenge, args.timeout))
                print("\n----- initial prompt -----\n")
                print(initial_prompt(challenge))
                probe = sandbox.exec("pwd && uname -m && ls -la ctf_files", timeout=60)
                print("\n----- sandbox probe -----\n")
                print(probe.stdout)
                if probe.stderr:
                    print(probe.stderr, file=sys.stderr)
                print("dry run complete (no model calls)")
                return 0

            if not args.model:
                print(
                    "no model selected: pass --model or set OPENROUTER_MODEL",
                    file=sys.stderr,
                )
                return 2
            if not args.api_key:
                print(
                    "no API key: pass --api-key or set OPENROUTER_API_KEY",
                    file=sys.stderr,
                )
                return 2
            if args.reasoning_effort not in REASONING_EFFORTS:
                print(
                    f"invalid --reasoning-effort {args.reasoning_effort!r}; "
                    f"use one of: {', '.join(REASONING_EFFORTS)}",
                    file=sys.stderr,
                )
                return 2

            runtime = Runtime(args.api_key)
            llm = runtime.llm(
                args.model,
                args.reasoning_effort,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            blackboard = Blackboard(sandbox)
            ledger = CostLedger()
            compaction_llm = (
                runtime.compaction_llm(args.compact_model)
                if args.compact_model
                else None
            )
            agent = Agent(
                challenge,
                sandbox,
                llm,
                build_main_tools(blackboard, enable_subtasks=not args.no_subtasks),
                max_rounds=args.max_rounds,
                max_cost=args.max_cost,
                timeout=args.timeout,
                log_dir=args.log_dir,
                quiet=args.quiet,
                ledger=ledger,
                blackboard=blackboard,
                max_subtasks=args.max_subtasks,
                subtask_rounds=args.subtask_rounds,
                subtask_cost=args.subtask_cost,
                compaction=CompactionConfig(enabled=not args.no_compaction),
                compaction_llm=compaction_llm,
                stall=StallConfig(
                    enabled=not args.no_stall_detection,
                    repeat=args.stall_repeat,
                ),
            )
            print(f"model: {normalize_model(args.model)}")
            result = agent.run()
            status = "SOLVED" if result.solved else "NOT SOLVED"
            print(
                f"\nresult: {status} ({result.reason}) "
                f"rounds={result.rounds} cost=${ledger.total:.4f} "
                f"notes={len(blackboard.search())} "
                f"stalls={result.stall_events} compactions={result.compactions}"
            )
            if result.flag:
                print(f"flag: {result.flag}")
            if result.transcript_path:
                print(f"transcript: {result.transcript_path}")
            return 0 if result.solved else 1
        finally:
            sandbox.stop()
            if service:
                service.stop()
    except (ChallengeError, SandboxError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
