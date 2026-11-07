#!/usr/bin/env python3
"""I reject test prerequisites that have neither inputs nor build rules."""
from pathlib import Path
import os
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def database(directory, makefile="Makefile.gnu", extra=""):
    # I select an empty target: -q alone could inspect a real build graph.
    source = f"include {makefile}\n{extra}\n.PHONY: nano-prerequisite-probe\nnano-prerequisite-probe:\n"
    result = subprocess.run(
        shlex.split(os.environ.get("MAKE", "make"))
        + ["--no-print-directory", "-rRqp", "-f", "-", "nano-prerequisite-probe"],
        input=source, text=True, capture_output=True, cwd=directory, timeout=30,
    )
    if result.returncode not in (0, 1) or "# Files" not in result.stdout:
        raise RuntimeError(f"I could not inspect my Make rules: {result.stderr}")
    return result.stdout


def missing_prerequisites(text, directory):
    rules = {}
    for block in text.split("\n\n"):
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if not re.match(r"^[^#\s:=]+:(?![=])", line):
                continue
            if not index or lines[index - 1] != "# Not a target:":
                name, dependencies = line.split(":", 1)
                rules[name] = dependencies.lstrip(":").split()
            break
    patterns = [re.compile(re.escape(name).replace("%", ".*"))
                for name in rules if "%" in name]
    missing = []
    for name, dependencies in rules.items():
        if not name.startswith("test-"):
            continue
        for dependency in dependencies:
            if dependency == "|" or dependency in rules:
                continue
            if (Path(directory) / dependency).exists():
                continue
            if any(pattern.fullmatch(dependency) for pattern in patterns):
                continue
            missing.append((name, dependency))
    return sorted(set(missing))


def main():
    missing = missing_prerequisites(database(ROOT), ROOT)
    for target, dependency in missing:
        print(f"I cannot build {target}: prerequisite {dependency} has no input or rule.")
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())
