#!/usr/bin/env python3
"""Pre-tag check for the deck and its narrative.

Two questions, both of which have already been answered wrong here:

1. Did the deck's sources move without the deck moving? Same staleness check
   every other artifact gets, from the same library.

2. Does the deck agree with its own specification? `deck-specification.md`
   enumerates the slide sequence one per line and `build_deck.py` implements
   it, so the two counts must match. They silently did not: `main()` printed
   "built 12 slides" through the entire 14-slide 4.0 edition, and the manifest
   hardcoded a count beside it. A constant that describes a list is a claim,
   and claims get checked.

Publishing is deliberately not here. It pushes to a personal Google Drive and
is outward-facing, so it stays an explicit human step -- see SKILL.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from releasing.lib import repo, report, staleness  # noqa: E402
from releasing.lib.report import Finding  # noqa: E402
from releasing.lib.staleness import Watch  # noqa: E402

WATCHES = [
    Watch(
        document="docs/presentation/",
        describes=[
            "docs/RELEASE_*.md",
            "docs/ROADMAP.md",
            "docs/NANOISA_MEASUREMENTS.md",
            "spec/nanoisa.yaml",
            "src/nanoisa/verifier.c",
        ],
        why="The deck and narrative are built from these. A release that "
        "changes what I prove, what I measured, or what I plan leaves the "
        "deck describing the previous release.",
    ),
]

SPEC = "docs/presentation/deck-specification.md"
BUILDER = "docs/presentation/build_deck.py"


def _specified_slides(root: Path) -> int | None:
    """Length of the numbered slide sequence in the specification."""
    text = (root / SPEC).read_text(encoding="utf-8")
    match = re.search(r"^## Slide sequence\s*$(.*?)^## ", text, re.M | re.S)
    if not match:
        return None
    return len(re.findall(r"^\d+\. ", match.group(1), re.M))


def _implemented_slides(root: Path) -> int:
    """Slides the builder actually emits, counted from its own markers."""
    text = (root / BUILDER).read_text(encoding="utf-8")
    return len(re.findall(r"^    # \d+ — ", text, re.M))


def check_specification_agrees(root: Path) -> list[Finding]:
    specified = _specified_slides(root)
    implemented = _implemented_slides(root)
    if specified is None:
        return [
            Finding(
                subject=SPEC,
                summary="no '## Slide sequence' section found",
                detail="The specification is the deck's authority. Without the "
                "sequence there is nothing for the builder to be checked against.",
            )
        ]
    if specified != implemented:
        return [
            Finding(
                subject=BUILDER,
                summary=f"builds {implemented} slides, specification lists {specified}",
                detail="The specification is the authority. Either implement the "
                "missing slide or amend the sequence -- do not let the two drift.",
            )
        ]
    return []


def main() -> int:
    root = repo.repo_root()
    findings = staleness.evaluate(WATCHES) + check_specification_agrees(root)
    text, code = report.resolve(findings, "Presentation freshness")
    sys.stdout.write(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
