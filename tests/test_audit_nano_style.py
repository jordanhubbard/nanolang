#!/usr/bin/env python3
"""Tests for scripts/audit_nano_style.py."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_nano_style.py"


class AuditNanoStyleTest(unittest.TestCase):
    def run_audit(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def write(self, root: Path, relative: str, content: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_reports_objective_findings_and_exempts_harness_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "examples/demo.nano",
                '# Example: Demo\nfrom "stdlib/option.nano" import Option\n'
                "fn unchecked() -> int { return 1 }\n"
                "shadow unchecked { assert (== 1 1) }\n// no\n",
            )
            self.write(root, "examples/check_harness.nano", "fn helper() -> int { return 1 }\n")
            result = self.run_audit(root, "--no-baseline")
            self.assertEqual(result.returncode, 0, result.stderr)
            for category in (
                "example-metadata",
                "root-stdlib-dialect",
                "stale-path",
                "unsupported-comment",
                "vacuous-shadow",
                "missing-shadow",
            ):
                self.assertIn(category, result.stdout)
            self.assertIn("metadata:harness-or-fixture=1", result.stdout)

    def test_strings_and_supported_comments_do_not_trigger_comment_findings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "src/good.nano",
                '# https://example.invalid/a--b\n/* // masked */\n'
                'fn url() -> string { return "https://example.invalid/a--b" }\n'
                'shadow url { assert (== (url) "https://example.invalid/a--b") }\n',
            )
            result = self.run_audit(root, "--strict", "--no-baseline")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("unsupported-comment", result.stdout)

    def test_strict_mode_accepts_baseline_and_rejects_new_debt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "src/debt.nano", "fn debt() -> int { return 1 }\n")
            written = self.run_audit(root, "--write-baseline")
            self.assertEqual(written.returncode, 0, written.stderr)
            accepted = self.run_audit(root, "--strict")
            self.assertEqual(accepted.returncode, 0, accepted.stdout)
            self.write(root, "src/new.nano", "fn new_debt() -> int { return 2 }\n")
            rejected = self.run_audit(root, "--strict")
            self.assertEqual(rejected.returncode, 1, rejected.stdout)
            self.assertIn("NEW missing-shadow src/new.nano", rejected.stdout)

    def test_resolves_supported_std_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "modules/std/fs.nano", "extern fn read(path: string) -> string\n")
            self.write(
                root,
                "src/tool.nano",
                'from "std/fs.nano" import read\nfn ok() -> int { return 0 }\nshadow ok { assert (== (ok) 0) }\n',
            )
            result = self.run_audit(root, "--strict", "--no-baseline")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("stale-path", result.stdout)

    def test_reports_retired_root_stdlib_module_even_without_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "stdlib/list.nano", "# retired dialect module\n")
            result = self.run_audit(root, "--no-baseline")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "NEW root-stdlib-dialect stdlib/list.nano:1: retired root stdlib module remains",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
