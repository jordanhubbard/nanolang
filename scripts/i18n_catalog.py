#!/usr/bin/env python3
"""Validate and render NanoLang's small UTF-8 message catalogs."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CATALOGS = ROOT / "catalogs"
LANGUAGES = ("en", "zh", "hi", "es", "ar", "fr")
PLACEHOLDER = re.compile(r"\{[a-z_][a-z0-9_]*\}")


def load_catalog(language: str) -> dict[str, str]:
    if language not in LANGUAGES:
        language = "en"
    data = json.loads((CATALOGS / f"{language}.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError(f"catalogs/{language}.json: expected string keys and values")
    return data


def validate_catalogs() -> None:
    english = load_catalog("en")
    for language in LANGUAGES:
        catalog = load_catalog(language)
        if catalog.keys() != english.keys():
            missing = sorted(english.keys() - catalog.keys())
            extra = sorted(catalog.keys() - english.keys())
            raise ValueError(f"catalogs/{language}.json: missing={missing}, extra={extra}")
        for message_id, text in catalog.items():
            if set(PLACEHOLDER.findall(text)) != set(PLACEHOLDER.findall(english[message_id])):
                raise ValueError(f"catalogs/{language}.json: incompatible placeholders for {message_id}")


def render(language: str, message_id: str, **parameters: str) -> str:
    english = load_catalog("en")
    catalog = load_catalog(language)
    template = catalog.get(message_id, english.get(message_id))
    if template is None:
        return english["catalog.missing"].format(message_id=message_id)
    return template.format(**parameters)
