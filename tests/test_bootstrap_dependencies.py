import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def bootstrap_prerequisites(files, directories):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in directories:
            (root / name).mkdir(parents=True, exist_ok=True)
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        makefile = root / "GNUmakefile"
        makefile.write_text((ROOT / "Makefile.gnu").read_text(encoding="utf-8"), encoding="utf-8")
        result = subprocess.run(
            ["make", "-f", str(makefile), "-pn", "bootstrap"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ, "MAKEFLAGS": ""},
        )
        rule = next(
            line for line in result.stdout.splitlines()
            if line.startswith(".bootstrap0.built:")
        )
        return set(rule.split(":", 1)[1].split())


class BootstrapDependencyTests(unittest.TestCase):
    def setUp(self):
        self.directories = ["src_nano", "src", "modules/std", "std", "stdlib"]

    def prerequisites(self, *files):
        return bootstrap_prerequisites(files, self.directories)

    def test_tracks_nano_sources(self):
        self.assertIn("src_nano/compiler.nano", self.prerequisites("src_nano/compiler.nano"))

    def test_tracks_c_sources(self):
        self.assertIn("modules/std/fs.c", self.prerequisites("modules/std/fs.c"))

    def test_tracks_headers(self):
        self.assertIn("modules/std/fs.h", self.prerequisites("modules/std/fs.h"))

    def test_tracks_json_inputs(self):
        self.assertIn("modules/std/module.manifest.json", self.prerequisites("modules/std/module.manifest.json"))

    def test_tracks_directory_membership(self):
        prerequisites = self.prerequisites()
        self.assertIn("modules/std/.", prerequisites)
        self.assertIn("stdlib/.", prerequisites)

    def test_excludes_documentation(self):
        self.assertNotIn("modules/std/README.md", self.prerequisites("modules/std/README.md"))

    def test_excludes_objects_and_archives(self):
        prerequisites = self.prerequisites("src/runtime.o", "modules/std/libstd.a")
        self.assertNotIn("src/runtime.o", prerequisites)
        self.assertNotIn("modules/std/libstd.a", prerequisites)

    def test_excludes_hidden_cache_contents(self):
        directories = self.directories + ["modules/.cache"]
        prerequisites = bootstrap_prerequisites(["modules/.cache/result.json"], directories)
        self.assertNotIn("modules/.cache/result.json", prerequisites)
        self.assertNotIn("modules/.cache/.", prerequisites)


if __name__ == "__main__":
    unittest.main()
