"""I query the real make rules in an isolated, already-built fixture."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BootstrapDependencies(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nanolang-bootstrap-deps-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(ROOT / "Makefile.gnu", self.root / "Makefile.gnu")
        self.sources = ["src_nano/parser.nano", "src_nano/compiler/module_loader.nano",
                        "src_nano/compiler/nested/new_import.nano"]
        for name in self.sources:
            self.file(name, 100)
        os.utime(self.root / "Makefile.gnu", (100, 100))
        for name, stamp in [("bin/nanoc_c", 110), (".bootstrap0.built", 120),
                            (".bootstrap1.built", 130), ("bin/nanoc_stage1", 130),
                            (".bootstrap2.built", 140), ("bin/nanoc_stage2", 140),
                            (".bootstrap3.built", 150), (".stage1.built", 160),
                            (".stage2.built", 170), (".stage3.built", 180)]:
            self.file(name, stamp)
        (self.root / "bin/nanoc").symlink_to("nanoc_stage2")

    def file(self, name, stamp):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.utime(path, (stamp, stamp))

    def query(self, target, changed=None):
        command = [os.environ.get("MAKE_BIN", "make"), "-f", "Makefile.gnu",
                   "--no-print-directory", "-q", "-o", "bin/nanoc_c",
                   "-o", ".bootstrap0.built", "-o", ".stage1.built",
                   "UNAME_S=Linux", target]
        if changed:
            command += ["-W", changed]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertIn(result.returncode, (0, 1), result.stdout + result.stderr)
        return result.returncode

    def test_up_to_date_and_source_invalidation(self):
        targets = [".bootstrap1.built", ".bootstrap2.built", ".bootstrap3.built",
                   ".stage2.built", ".stage3.built", "bin/nanoc"]
        for target in targets:
            with self.subTest(target=target, changed=None):
                self.assertEqual(self.query(target), 0)
            for source in self.sources + ["Makefile.gnu"]:
                with self.subTest(target=target, changed=source):
                    self.assertEqual(self.query(target, source), 1)

    def test_unrelated_program_does_not_invalidate_bootstrap(self):
        self.file("examples/unrelated.nano", 100)
        self.assertEqual(self.query(".bootstrap3.built", "examples/unrelated.nano"), 0)

    def test_missing_stage_one_invalidates_later_stages(self):
        (self.root / "bin/nanoc_stage1").unlink()
        for target in [".bootstrap1.built", ".bootstrap2.built", ".bootstrap3.built", "bin/nanoc"]:
            with self.subTest(target=target):
                self.assertEqual(self.query(target), 1)

    def test_missing_stage_two_invalidates_validation(self):
        (self.root / "bin/nanoc_stage2").unlink()
        self.assertEqual(self.query(".bootstrap1.built"), 0)
        self.assertEqual(self.query(".bootstrap3.built"), 1)


if __name__ == "__main__":
    unittest.main()
