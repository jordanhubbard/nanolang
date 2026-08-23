#!/usr/bin/env python3
"""Generate root dependency metadata for GitHub's dependency graph."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "vscode/package.json"
DEFAULT_OUTPUT = ROOT / "package.json"


def generate(version: str, source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError(f"invalid release version: {version}")
    extension = json.loads(source.read_text())
    package = {
        "name": "nanolang-repository",
        "version": version,
        "private": True,
        "description": "NanoLang repository dependency metadata for GitHub dependency analysis",
        "dependencies": extension.get("dependencies", {}),
        "devDependencies": extension.get("devDependencies", {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(package, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.version, args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
