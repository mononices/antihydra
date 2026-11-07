"""Prompts for the main agent and spawned subtask workers."""

from __future__ import annotations

from .challenge import Challenge

_SANDBOX_FACTS = """\
Sandbox facts:
- Home directory: /home/ctfplayer
- Challenge files: /home/ctfplayer/ctf_files/ (read them; do not modify originals)
- Scratch space: /home/ctfplayer/workspace/
- The sandbox runs Linux x86-64 with common CTF tooling (see /tools.txt)
- The default run_command timeout is {timeout:.0f} seconds
{target_block}"""

SYSTEM_TEMPLATE = """\
You are an autonomous CTF player solving {category} challenges. You work inside \
an isolated Linux sandbox and interact with it exclusively through tools.

{sandbox_facts}{bus_block}Rules:
1. Think before acting. Prefer a few purposeful commands over many guesses.
2. Inspect every output carefully; errors often reveal how the challenge works.
3. Build exploits and payloads in the scratch directory.
4. Use record_finding for durable facts, artifacts and dead ends; use get_findings
   to reload them after long or complex stretches of work.
5. For context-heavy work (deep reversing, exploit development, long multi-step
   investigation), call spawn_subtask with a precise task. Helpers share the
   sandbox and the findings board but not your conversation, and return a summary.
6. When you are confident you have the flag, call submit_flag. Never invent flags.
7. Never repeat an identical command expecting a different result; change approach
   instead. Treat an "[Earlier context summary]" message as ground truth for older work.
8. If you are completely stuck after exhausting your ideas, call giveup.

{flag_block}"""

WORKER_SYSTEM_TEMPLATE = """\
You are a focused CTF worker inside an isolated Linux sandbox. Another agent \
assigned you exactly one subtask; complete it and report back.

{sandbox_facts}{bus_block}Workflow:
1. Call get_findings first to load what is already known (skip only if the task
   is trivial).
2. Work the subtask step by step with tools. Stay focused on it; do not wander.
3. Record durable findings as you go with record_finding: facts, artifacts
   (paths you created), and dead ends worth remembering.
4. End by calling finish_task with a compact summary: what you found, artifacts
   created, what is still unknown, and the recommended next step. If the task
   cannot be completed, still call finish_task with the blocker and what you
   tried.
5. Never repeat an identical command expecting a different result; change approach
   instead. Treat an "[Earlier context summary]" message as ground truth for older work.

{flag_block}"""


def _sandbox_facts(challenge: Challenge, timeout: float) -> str:
    target_block = _target_block(challenge, prefix="- ")
    return _SANDBOX_FACTS.format(timeout=timeout, target_block=target_block)


def _target_block(challenge: Challenge, *, prefix: str = "") -> str:
    target = challenge.reachable_target()
    if target:
        return f"{prefix}Remote service: {target} (reachable from inside the sandbox)\n"
    if challenge.is_remote_only():
        return (
            f"{prefix}No local service is running for this challenge and the public "
            f"address ({challenge.target_line()}) is not reachable from the sandbox; "
            "work from the provided files alone.\n"
        )
    return ""


def _flag_block(challenge: Challenge) -> str:
    hint = challenge.flag_format_hint()
    return f"The flag format is {hint}.\n" if hint else ""


BUS_BLOCK = (
    "Other attempts run on this challenge in parallel: check_hints reads their "
    "findings, publish_insight shares yours. Share durable facts and dead ends; "
    "never publish flags or speculation.\n\n"
)


def system_prompt(
    challenge: Challenge, timeout: float, *, coordinated: bool = False
) -> str:
    """System prompt for the main agent."""
    return SYSTEM_TEMPLATE.format(
        category=challenge.category,
        sandbox_facts=_sandbox_facts(challenge, timeout),
        bus_block=BUS_BLOCK if coordinated else "",
        flag_block=_flag_block(challenge),
    )


def worker_system_prompt(
    challenge: Challenge, timeout: float, *, coordinated: bool = False
) -> str:
    """System prompt for a spawned subtask worker."""
    return WORKER_SYSTEM_TEMPLATE.format(
        sandbox_facts=_sandbox_facts(challenge, timeout),
        bus_block=BUS_BLOCK if coordinated else "",
        flag_block=_flag_block(challenge),
    )


def initial_prompt(challenge: Challenge) -> str:
    files = ", ".join(challenge.files) if challenge.files else "none provided"
    target_line = _target_block(challenge)
    return (
        f"Challenge: {challenge.name} ({challenge.category})\n"
        f"{target_line}"
        f"Files: {files}\n\n"
        f"{challenge.description_text()}\n\n"
        "Begin. Inspect the provided files and make a plan."
    )


def worker_initial_prompt(challenge: Challenge, task: str) -> str:
    return (
        f"Challenge: {challenge.name} ({challenge.category})\n\n"
        f"Subtask:\n{task.strip()}\n\n"
        "Begin. Check the findings board, then execute the subtask and end with "
        "finish_task."
    )


KEEP_GOING = (
    "Continue. Use a tool to make concrete progress, or submit the flag "
    "if you already have it."
)

WORKER_KEEP_GOING = (
    "Continue the subtask. Use a tool to make progress, and end with "
    "finish_task(summary=...) when done or blocked."
)
