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


def generate(version: str, source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT,
             version_header: Path | None = None) -> None:
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
    if version_header is not None:
        core, separator, suffix = version.partition("-")
        major, minor, patch = (int(part) for part in core.split("."))
        header = f'''#ifndef NANOLANG_VERSION_H
#define NANOLANG_VERSION_H

#define NANOLANG_VERSION_MAJOR {major}
#define NANOLANG_VERSION_MINOR {minor}
#define NANOLANG_VERSION_PATCH {patch}
#define NANOLANG_VERSION_SUFFIX "{separator}{suffix}"

#define NANOLANG_VERSION "{version}"
#define NANOLANG_BUILD_DATE __DATE__
#define NANOLANG_BUILD_TIME __TIME__

#endif /* NANOLANG_VERSION_H */
'''
        version_header.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=version_header.parent, delete=False
        ) as handle:
            handle.write(header)
            temporary = Path(handle.name)
        os.replace(temporary, version_header)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--version-header", type=Path, help="I also update the public C version header")
    args = parser.parse_args()
    generate(args.version, args.source, args.output, args.version_header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
