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
