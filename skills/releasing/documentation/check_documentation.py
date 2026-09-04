#!/usr/bin/env python3
"""Pre-tag check for my user-facing prose.

This file is a manifest, not an algorithm. The question it asks lives in
`lib/staleness.py`; what is here is the mapping from each document to the part
of me it claims to describe. Adding an artifact means adding a Watch.

Run directly, or through `make release-docs-check`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from releasing.lib import report, staleness  # noqa: E402
from releasing.lib.staleness import Watch  # noqa: E402

WATCHES = [
    Watch(
        document="README.md",
        describes=[
            "docs/RELEASE_*.md",
            "spec/nanoisa.yaml",
            "src/main.c",
            "src/nanovirt/main.c",
            "src/nanovm/main.c",
        ],
        why="README states what I am and how to run me. A new release document, "
        "a changed instruction set, or a changed CLI entry point can all "
        "make it wrong.",
    ),
    Watch(
        document="CONTRIBUTING.md",
        describes=[
            "src/nanoisa/verifier.c",
            "src/nanoisa/verifier_types.c",
            "docs/NANOISA_OPTIMIZATION_POLICY.md",
            "docs/NANOISA_MEASUREMENTS.md",
            "tests/*fuzz*",
            "tests/test_verify_all_programs.sh",
        ],
        why="CONTRIBUTING states the rules a contributor is held to. In 4.0 the "
        "verifier, the optimization policy and the fuzzing surfaces all "
        "changed and it was never opened -- the case this watch exists for.",
    ),
    Watch(
        document="userguide/",
        describes=[
            "src/parser.c",
            "src/typechecker.c",
            "src/lexer.c",
            "examples/language/",
        ],
        why="The user guide teaches the surface language. A change to how "
        "source is parsed, typed, or exemplified can strand a lesson.",
    ),
    Watch(
        document="docs/ROADMAP.md",
        describes=["docs/RELEASE_*.md"],
        why="A release document that ships work the roadmap still lists as "
        "pending leaves the roadmap describing a future that already happened.",
    ),
]


def main() -> int:
    findings = staleness.evaluate(WATCHES)
    text, code = report.resolve(findings, "Documentation freshness")
    sys.stdout.write(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
