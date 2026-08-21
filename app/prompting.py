"""Prompt loading.

Prompts live as files rather than as string literals buried in the code, so
that a change to the wording is a visible diff, and so the report can show a
before-and-after between versions. PROMPT_VERSION picks which file is used.

Placeholders are double-braced ({{QUESTION}}) and filled by plain string
replacement rather than str.format, because the prompts contain JSON examples
and every { in them would otherwise have to be escaped.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config import settings

PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM_MARKER = "# System"
USER_MARKER = "# User"


@lru_cache(maxsize=16)
def load_named_prompt(name: str) -> tuple[str, str]:
    """Return (system_prompt, user_template) for a prompt file stem."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.md")))
        raise FileNotFoundError(f"No prompt file {path.name}. Available: {available or 'none'}")

    text = path.read_text(encoding="utf-8")
    if SYSTEM_MARKER not in text or USER_MARKER not in text:
        raise ValueError(f"{path.name} must contain '{SYSTEM_MARKER}' and '{USER_MARKER}' headings")

    _, remainder = text.split(SYSTEM_MARKER, 1)
    system, user = remainder.split(USER_MARKER, 1)
    return system.strip(), user.strip()


def load_prompt(version: str | None = None) -> tuple[str, str]:
    """The answering prompt for a version, defaulting to PROMPT_VERSION."""
    return load_named_prompt(f"answer_{version or settings.prompt_version}")


def render(template_name: str, **values: str) -> tuple[str, str]:
    """Load a prompt and fill its {{PLACEHOLDER}} slots."""
    system, user = load_named_prompt(template_name)
    for key, value in values.items():
        user = user.replace("{{" + key + "}}", value)
    return system, user


def render_user_prompt(question: str, excerpts: str, version: str | None = None) -> str:
    _, template = load_prompt(version)
    return template.replace("{{EXCERPTS}}", excerpts).replace("{{QUESTION}}", question)


def available_versions() -> list[str]:
    """Answer prompt versions on disk, newest naming aside."""
    return sorted(path.stem.removeprefix("answer_") for path in PROMPTS_DIR.glob("answer_*.md"))
