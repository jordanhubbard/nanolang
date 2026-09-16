"""I check C-seed destination identity before any artifact or report is written."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

COMPILER = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else Path("bin/nanoc_c").resolve()
FLAGS = ("--profile-output", "--profile-runtime-output", "--llm-diags-json",
         "--llm-diags-toon", "--llm-shadow-json", "--reflect", "--bench-json")
SOURCE = "fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n"


class Destinations(unittest.TestCase):
    def invoke(self, directory, artifact, report, flag):
        source = directory / "input.nano"
        source.write_text(SOURCE)
        result = subprocess.run([str(COMPILER), str(source), "-o", str(artifact),
                                 flag, str(report)], cwd=directory, capture_output=True,
                                text=True, timeout=60, env=dict(os.environ, TMPDIR=str(directory)))
        self.assertEqual(source.read_text(), SOURCE)
        return result

    def test_aliases_preserve_existing_artifact(self):
        for flag in FLAGS:
            for kind in ("same", "hardlink", "symlink", "relative"):
                with self.subTest(flag=flag, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    artifact = directory / "artifact"
                    artifact.write_bytes(b"prior artifact")
                    report = directory / "report"
                    if kind == "same": report = artifact
                    elif kind == "hardlink": os.link(artifact, report)
                    elif kind == "symlink": report.symlink_to(artifact)
                    else: report = Path("artifact")
                    result = self.invoke(directory, artifact, report, flag)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("same destination", result.stderr)
                    self.assertEqual(artifact.read_bytes(), b"prior artifact")

    def test_absent_names_follow_filesystem_identity(self):
        for same in (True, False):
            with self.subTest(same=same), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                artifact = directory / "Artifact"
                report = artifact if same else directory / "artifact"
                artifact.touch()
                aliases = report.exists() and os.path.samefile(artifact, report)
                artifact.unlink()
                result = self.invoke(directory, artifact, report, "--llm-diags-json")
                if aliases:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("same destination", result.stderr)
                    self.assertFalse(artifact.exists())
                    self.assertFalse(report.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(artifact.is_file())
                    self.assertTrue(report.is_file())
                    self.assertFalse(os.path.samefile(artifact, report))

    def test_dangling_report_fails_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            artifact = directory / "artifact"
            report = directory / "report"
            missing = directory / "missing"
            report.symlink_to(missing)
            result = self.invoke(directory, artifact, report, "--llm-diags-json")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot verify", result.stderr)
            self.assertFalse(artifact.exists())
            self.assertFalse(missing.exists())
            self.assertTrue(report.is_symlink())

    def test_lookup_failure_preserves_existing_artifact(self):
        for flag in FLAGS:
            for loop in (True, False):
                with self.subTest(flag=flag, loop=loop), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    artifact = directory / "artifact"
                    artifact.write_bytes(b"prior artifact")
                    report = directory / "report"
                    report.symlink_to(report if loop else directory / "missing")
                    result = self.invoke(directory, artifact, report, flag)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("cannot verify", result.stderr)
                    self.assertEqual(artifact.read_bytes(), b"prior artifact")
                    self.assertTrue(report.is_symlink())
                    self.assertFalse((directory / "missing").exists())


if __name__ == "__main__":
    unittest.main()
