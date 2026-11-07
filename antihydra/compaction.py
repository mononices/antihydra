"""Summarize old conversation history when the model context fills up."""

from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN = 4.0


@dataclass
class CompactionConfig:
    enabled: bool = True
    at_fraction: float = 0.7  # trigger at this share of the model context
    tokens_fallback: int = 100_000  # trigger threshold when context size is unknown
    keep_recent: int = 12  # messages kept verbatim at the tail
    min_dropped: int = 6  # never compact fewer messages than this
    min_interval: int = 5  # rounds between compactions
    input_clip: int = 2500  # chars per dropped message in the summary prompt
    max_input_chars: int = 150_000  # total prompt budget for the dropped history
    max_output_tokens: int = 1200


PROMPT_TEMPLATE = """\
You are compacting the working memory of a CTF-solving agent so it can continue \
with a smaller context window.

Write a compact progress report covering, in this order:
1. Challenge facts learned (files, formats, keys, offsets, addresses, credentials).
2. Actions already taken and their outcomes, including failed attempts and why.
3. Artifacts created (absolute paths) and the current state of the exploit or solution.
4. Open threads and recommended next steps.

Be specific and factual; never invent details. Use short bullet lines. \
Output only the report, no preamble.

Previous summary (merge it if present):
{previous}

Conversation to compress:
{history}
"""


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate from total character count."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            total += len(str(function.get("arguments") or ""))
    return int(total / CHARS_PER_TOKEN)


def safe_cut_index(body: list[dict], keep_recent: int) -> int | None:
    """Index of the first turn to drop; cuts only at assistant-message boundaries."""
    target = len(body) - keep_recent
    if target <= 0:
        return None
    turn_starts = [
        i for i, message in enumerate(body) if message.get("role") == "assistant"
    ]
    candidates = [i for i in turn_starts if i <= target]
    if not candidates:
        return None
    return candidates[-1]


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _render_message(message: dict, clip_chars: int) -> str:
    role = str(message.get("role") or "?")
    content = message.get("content")
    content_text = content if isinstance(content, str) else ""
    parts = [f"{role.upper()}: {clip(content_text.strip(), clip_chars)}"]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        parts.append(
            f"  TOOL_CALL {function.get('name', '?')}: "
            f"{clip(str(function.get('arguments') or ''), 300)}"
        )
    return "\n".join(part for part in parts if part.strip())


def build_prompt(
    previous_summary: str | None, dropped: list[dict], config: CompactionConfig
) -> str:
    """Render the compaction prompt, tail-biased when history is too large."""
    rendered = "\n".join(
        _render_message(message, config.input_clip) for message in dropped
    )
    if len(rendered) > config.max_input_chars:
        omitted = len(rendered) - config.max_input_chars
        rendered = (
            f"... [{omitted} chars of older history omitted] ...\n"
            + rendered[-config.max_input_chars :]
        )
    return PROMPT_TEMPLATE.format(
        previous=clip(previous_summary or "(none)", config.max_input_chars // 3),
        history=rendered,
    )


def archive_messages(messages: list[dict], clip_chars: int = 4000) -> list[dict]:
    """Copy dropped messages for the transcript, clipping huge contents."""
    archived: list[dict] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = content[:clip_chars]
        archived.append(item)
    return archived
