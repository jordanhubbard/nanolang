"""Tests for the pre-tag release gates.

The load-bearing test is `test_catches_the_contributing_regression`. It replays
the 3.5 -> 4.0 release exactly as it shipped and requires the gate to name
CONTRIBUTING.md, which really did go that whole release untouched while the
verifier, the optimization policy and the fuzzing surfaces changed underneath
it. A gate nobody has watched fail is not evidence of anything.

Stdlib `unittest`, matching `tests/test_nanoisa_schema.py`, so this runs
wherever the rest of my suite runs rather than only where pytest happens to be
installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
sys.path.insert(0, str(SKILLS))

from releasing.lib import repo, report, staleness  # noqa: E402
from releasing.lib.report import Finding  # noqa: E402
from releasing.lib.staleness import Watch  # noqa: E402

DOCUMENTATION = SKILLS / "releasing" / "documentation" / "check_documentation.py"
PRESENTATION = SKILLS / "releasing" / "presentation" / "check_presentation.py"


def _run(script: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, **env},
    )


def _has_tag(name: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{name}^{{}}"],
            capture_output=True,
            cwd=str(ROOT),
        ).returncode
        == 0
    )


HAS_HISTORY = _has_tag("v3.5.0") and _has_tag("v4.0.0")
NEEDS_HISTORY = "shallow clone: release tags unavailable"


class ReleaseGateTest(unittest.TestCase):
    def setUp(self) -> None:
        # Each test picks its own release boundary, and the lookups are cached.
        repo.latest_tag.cache_clear()
        repo.changed_since.cache_clear()

    tearDown = setUp

    @unittest.skipUnless(HAS_HISTORY, NEEDS_HISTORY)
    def test_catches_the_contributing_regression(self):
        """The defect that motivated the gate must trip it."""
        result = _run(DOCUMENTATION, RELEASE_BASE_TAG="v3.5.0", RELEASE_HEAD_REF="v4.0.0")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("CONTRIBUTING.md", result.stdout)
        # A finding has to carry proof, not just an accusation.
        self.assertIn("src/nanoisa/verifier.c", result.stdout)

    @unittest.skipUnless(HAS_HISTORY, NEEDS_HISTORY)
    def test_acknowledgement_requires_a_reason_not_a_boolean(self):
        """`=1` says nothing to a later reader, so it must not unlock the gate."""
        boolean = _run(
            DOCUMENTATION,
            RELEASE_BASE_TAG="v3.5.0",
            RELEASE_HEAD_REF="v4.0.0",
            RELEASE_DOCS_ACK="1",
        )
        self.assertEqual(boolean.returncode, 1)

        reasoned = _run(
            DOCUMENTATION,
            RELEASE_BASE_TAG="v3.5.0",
            RELEASE_HEAD_REF="v4.0.0",
            RELEASE_DOCS_ACK="rules verified unchanged by hand",
        )
        self.assertEqual(reasoned.returncode, 0, reasoned.stdout)
        self.assertIn("rules verified unchanged by hand", reasoned.stdout)

    def test_current_tree_is_releasable(self):
        """Whatever else is true, main must not be shipping stale prose."""
        for script in (DOCUMENTATION, PRESENTATION):
            with self.subTest(gate=script.name):
                result = _run(script)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_deck_matches_its_specification(self):
        """The slide count is a claim the specification makes; check it."""
        sys.path.insert(0, str(PRESENTATION.parent))
        from check_presentation import _implemented_slides, _specified_slides

        specified = _specified_slides(ROOT)
        self.assertIsNotNone(specified, "deck-specification.md lost its slide sequence")
        self.assertEqual(specified, _implemented_slides(ROOT))

    def test_a_document_that_moved_with_its_subject_is_not_reported(self):
        """The gate asks for attention, not for a commit to every file."""
        watch = Watch(document="CONTRIBUTING.md", describes=["src/"], why="test")
        original = repo.changed_since
        try:
            staleness.repo.changed_since = lambda _tag: (
                "CONTRIBUTING.md",
                "src/nanoisa/verifier.c",
            )
            self.assertEqual(staleness.evaluate([watch], tag="v0.0.0"), [])
        finally:
            staleness.repo.changed_since = original

    def test_a_neglected_document_is_reported(self):
        """The same watch, with the document left alone."""
        watch = Watch(document="CONTRIBUTING.md", describes=["src/"], why="test")
        original = repo.changed_since
        try:
            staleness.repo.changed_since = lambda _tag: ("src/nanoisa/verifier.c",)
            findings = staleness.evaluate([watch], tag="v0.0.0")
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].subject, "CONTRIBUTING.md")
        finally:
            staleness.repo.changed_since = original

    def test_report_is_quiet_when_there_is_nothing_to_say(self):
        text, code = report.resolve([], "Documentation freshness")
        self.assertEqual(code, 0)
        self.assertIn("nothing stale", text)

    def test_report_names_the_document_and_its_evidence(self):
        finding = Finding(
            subject="README.md",
            summary="subject moved",
            detail="why it matters",
            evidence=["src/main.c"],
            evidence_total=3,
        )
        text, code = report.resolve([finding], "Documentation freshness")
        self.assertEqual(code, 1)
        self.assertIn("README.md", text)
        self.assertIn("src/main.c", text)
        self.assertIn("and 2 more", text)


if __name__ == "__main__":
    unittest.main()
