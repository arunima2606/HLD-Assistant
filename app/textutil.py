"""Small text helpers shared across modules."""
from __future__ import annotations

import re

_EMPTY = {"", "-", "—", "–", "n/a", "na", "none", "null", "tbd", "?"}


def clean(value: object) -> str:
    """Collapse whitespace; map None/placeholder-ish values to ''."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in _EMPTY else text


def norm_key(name: str) -> str:
    """Canonical comparison key: lowercase, no spaces/underscores/hyphens/dots."""
    return re.sub(r"[\s_\-\.]+", "", (name or "").lower())


def split_identifier(text: str) -> list[str]:
    """Split camelCase / snake_case identifiers into lowercase words."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", spaced.lower())


def word_count(text: str) -> int:
    return len(text.split())
