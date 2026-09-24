#!/usr/bin/env python3
"""Unit tests for i18n generated-guide fallback in the markdown link checker."""

from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_markdown_links as checker  # noqa: E402


class TestI18nGeneratedFallback(unittest.TestCase):
    def test_published_chapter_fallback_and_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "userguide"
            (source / "guide").mkdir(parents=True)
            (source / "nav.txt").write_text(
                "guide/runtime.md | Runtime | Use\n"
                "guide/missing.md | Missing | Use\n")
            (source / "guide/runtime.md").write_text("# Runtime\n")
            (source / "guide/unpublished.md").write_text("# Draft\n")
            (root / "outside.md").write_text("# Outside\n")
            rel = Path("userguide/i18n/fr/guide/start.md")
            draft = root / rel
            draft.parent.mkdir(parents=True)
            draft.write_text("[Runtime](runtime.md) [HTML](runtime.html)\n"
                             "[Missing](missing.md) [Draft](unpublished.md)\n"
                             "[Outside](../../outside.md)\n")
            broken = checker.find_broken_links_in_file(root, draft)
            self.assertEqual([link.target for link in broken],
                             ["missing.md", "unpublished.md", "../../outside.md"])
            self.assertFalse(checker.i18n_published_fallback(
                root, Path("docs/start.md"), "guide/runtime.md"))
            self.assertFalse(checker.i18n_published_fallback(
                root, Path("userguide/i18n/unknown/guide/start.md"), "runtime.md"))

    def test_accepts_english_generated_page(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            generated = root / "userguide" / "generated"
            generated.mkdir(parents=True)
            (generated / "cli.md").write_text("# CLI\n", encoding="utf-8")
            rel = Path("userguide/i18n/zh/README.md")
            self.assertTrue(checker.i18n_generated_fallback(root, rel, "generated/cli.md"))

    def test_rejects_missing_generated_page(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rel = Path("userguide/i18n/zh/README.md")
            self.assertFalse(checker.i18n_generated_fallback(root, rel, "generated/missing.md"))

    def test_ignores_non_i18n_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            generated = root / "userguide" / "generated"
            generated.mkdir(parents=True)
            (generated / "cli.md").write_text("# CLI\n", encoding="utf-8")
            rel = Path("docs/README.md")
            self.assertFalse(checker.i18n_generated_fallback(root, rel, "generated/cli.md"))


if __name__ == "__main__":
    unittest.main()
