"""Sandbox images as OpenRouter multimodal content parts.

Follows OpenRouter's image-input rules: base64 ``data:`` URLs, only png/jpeg/
webp/gif, a text part before the image parts, system messages stay text-only.
The transport stays on ``/chat/completions`` (``ChatContentImage``) because the
whole agent loop is built on that message shape; the payload is identical to
the Responses API ``input_image`` form.
"""

from __future__ import annotations

import base64
import re
import shlex
from dataclasses import dataclass

SUPPORTED_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_IMAGE_BYTES = 900_000  # decoded bytes actually sent to the model
MAX_IMAGES_PER_AGENT = 12
DEFAULT_MAX_DIMENSION = 1200
MAX_READ_BYTES = 12_000_000  # hard cap before we refuse to pull a file out

ATTACHMENT_DIR = "/home/ctfplayer/workspace/.attachments"

_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def sniff_media_type(raw: bytes) -> str | None:
    """Identify an image from magic bytes; the extension is never trusted."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


def data_url(media_type: str, payload: str) -> str:
    """Assemble the ``data:`` URL for a base64 payload (without header)."""
    return f"data:{media_type};base64,{payload}"


def encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def extension_for(media_type: str) -> str:
    return _EXTENSIONS.get(media_type, ".png")


def parse_crop(value: str) -> str | None:
    """Normalise ``x,y,w,h`` into ImageMagick's ``WxH+X+Y`` (or None)."""
    if not value:
        return None
    parts = [part.strip() for part in str(value).replace("x", ",").split(",")]
    parts = [part for part in parts if part]
    if len(parts) != 4:
        raise ValueError("crop must be x,y,w,h")
    x, y, w, h = (int(part) for part in parts)
    if min(x, y) < 0 or w <= 0 or h <= 0:
        raise ValueError("crop needs non-negative x,y and positive w,h")
    return f"{w}x{h}+{x}+{y}"


def slug(value: str, *, limit: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return clean[-limit:] or "image"


@dataclass(frozen=True)
class AttachedImage:
    """One image queued for the next model call."""

    path: str
    media_type: str
    payload: str  # base64, no data: header
    detail: str = "auto"
    question: str = ""

    @property
    def bytes(self) -> int:
        return len(self.payload) * 3 // 4

    def content_part(self) -> dict:
        return {
            "type": "image_url",
            "image_url": {
                "url": data_url(self.media_type, self.payload),
                "detail": self.detail,
            },
        }

    def summary(self) -> str:
        size = f"{self.bytes / 1024:.0f} kB" if self.bytes >= 1024 else f"{self.bytes} B"
        question = f' question: "{self.question}"' if self.question else ""
        return f"{self.path} ({self.media_type}, {size}, detail={self.detail}){question}"


def render_attachment_text(images: list[AttachedImage]) -> str:
    """Text part that precedes the image parts in the user message."""
    listing = "\n".join(f"- {image.summary()}" for image in images)
    return (
        "Attached image(s) rendered from the sandbox, shown to you at the "
        f"requested detail level:\n{listing}\n"
        "Read them directly: describe any text, symbols or layout you can see. "
        "If something is unreadable, call view_image again with a crop or a "
        "higher detail instead of guessing."
    )


def text_only_parts(parts: list) -> list:
    """Same content parts without image payloads (for older requests)."""
    kept: list = []
    omitted = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            omitted += 1
            continue
        kept.append(part)
    if omitted:
        kept.append(
            {
                "type": "text",
                "text": (
                    f"[{omitted} image(s) omitted from this older message; call "
                    "view_image again if you need to look again]"
                ),
            }
        )
    return kept or [{"type": "text", "text": "[images omitted]"}]


def redact_image_parts(messages: list[dict]) -> list[dict]:
    """Copy messages for JSON storage without megabyte-sized base64 payloads."""
    redacted: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    size = len(url) * 3 // 4
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:<omitted {size} bytes of image>"
                            },
                        }
                    )
                else:
                    parts.append(part)
            message = {**message, "content": parts}
        redacted.append(message)
    return redacted


def prepare_script(
    src: str,
    dst: str,
    *,
    fmt: str = "png",
    crop: str | None = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    upscale: bool = False,
) -> str:
    """Shell command normalising ``src`` into ``dst`` (convert -> magick -> ffmpeg)."""
    dimension = max(int(max_dimension), 64)
    resize = f"{dimension}x{dimension}" + ("" if upscale else ">")
    ops = []
    if crop:
        ops.append(f"-crop {crop} +repage")
    ops.append(f"-resize {resize}")
    ops.append("-strip")
    if fmt == "jpeg":
        ops.append("-quality 88")
    convert_ops = " ".join(ops)

    filters = []
    if crop:
        width, rest = crop.split("x", 1)
        height, offset = rest.split("+", 1)
        x, y = offset.split("+", 1)
        filters.append(f"crop={width}:{height}:{x}:{y}")
    scale = (
        f"scale='min({dimension},iw)':'min({dimension},ih)'"
        ":force_original_aspect_ratio=decrease"
    )
    filters.append(scale)
    vf = ",".join(filters)
    if fmt == "jpeg":
        vf += " -q:v 3"

    return (
        "set -e; "
        f"src={shlex.quote(src)}; dst={shlex.quote(dst)}; "
        'mkdir -p "$(dirname "$dst")"; '
        'if command -v convert >/dev/null 2>&1; then '
        f'convert "$src" {convert_ops} "$dst"; '
        'elif command -v magick >/dev/null 2>&1; then '
        f'magick "$src" {convert_ops} "$dst"; '
        'elif command -v ffmpeg >/dev/null 2>&1; then '
        f'ffmpeg -v error -y -i "$src" -vf "{vf}" -frames:v 1 "$dst"; '
        'else cp "$src" "$dst"; fi; '
        'wc -c < "$dst"'
    )
