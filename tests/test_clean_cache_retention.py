"""I exercise real clean recipes only in disposable fixture workspaces."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture(directory):
    for name in ("Makefile.gnu", "examples/Makefile", "scripts/clean_build_trees.py"):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name in ("tests", "formal", "modules", "std", "bin"):
        (directory / name).mkdir(exist_ok=True)


class CleanCacheRetention(unittest.TestCase):
    def test_clean_preserves_cache_artifacts(self):
        locations = ("obj/custom/cache", "obj/build_bootstrap", "bin/gpu/cache", "coverage/cache",
                     "tests/cache", ".test_output/cache")
        for example_only, configured in ((example, path) for example in (False, True) for path in locations):
            with self.subTest(example_only=example_only, configured=configured), tempfile.TemporaryDirectory(prefix="nano-clean-make-") as tmp:
                directory = Path(tmp)
                fixture(directory)
                cache_names = ("obj/module_cache", configured, "obj/former/cache", "modules/a/.build", "std/b/.build")
                retained = []
                for name in cache_names:
                    generation = directory / name / ".nano-gen-ABC123"
                    generation.mkdir(parents=True)
                    artifact = generation / "library.out"
                    artifact.write_bytes(b"I remain available to copied bytecode.")
                    (generation.parent / "current").symlink_to(generation.name)
                    retained.append(artifact)
                disposable = directory / "obj/stale.o"
                disposable.write_bytes(b"old object")
                orphan = directory / "obj/old-location/.nano-build-ABC123/partial.o"
                orphan.parent.mkdir(parents=True)
                orphan.write_bytes(b"possibly still in use by a compiler child")
                gpu_object = directory / "bin/gpu/stale.o"
                gpu_object.parent.mkdir(parents=True, exist_ok=True)
                gpu_object.write_bytes(b"old GPU object")
                untouched = directory / "notes.out"
                untouched.mkdir()
                (untouched / "keep").write_bytes(b"not a generated output file")
                cache_setting = os.path.relpath(directory / configured, directory / "examples") if example_only else str(directory / configured)
                env = dict(os.environ, NANO_BUILD_CACHE=cache_setting)
                command = ["make", "-f", "Makefile" if example_only else "Makefile.gnu", "clean"]
                result = subprocess.run(command, cwd=directory / "examples" if example_only else directory,
                                        env=env, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                for artifact in retained:
                    self.assertEqual(artifact.read_bytes(), b"I remain available to copied bytecode.")
                    self.assertEqual((artifact.parent.parent / "current").resolve(), artifact.parent.resolve())
                if not example_only: self.assertFalse(disposable.exists())
                self.assertFalse(gpu_object.exists())
                self.assertTrue((untouched / "keep").is_file())
                self.assertEqual(orphan.read_bytes(), b"possibly still in use by a compiler child")

    def test_unsafe_root_refuses_before_removal(self):
        with tempfile.TemporaryDirectory(prefix="nano-clean-root-") as tmp:
            directory = Path(tmp)
            fixture(directory)
            artifact = directory / "obj/keep"
            artifact.parent.mkdir()
            artifact.write_bytes(b"I survive a rejected request.")
            for unsafe in (".", "..", "/"):
                result = subprocess.run(["python3", "scripts/clean_build_trees.py", "--root", "obj",
                                         "--root", unsafe], cwd=directory, capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(artifact.read_bytes(), b"I survive a rejected request.")

    def test_cache_markers_retain_direct_cleanup_roots(self):
        for name in (".nano-gen-ABC123", ".nano-build-ABC123", "old-cache"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="nano-clean-marker-") as tmp:
                directory = Path(tmp)
                fixture(directory)
                cache = directory / name
                cache.mkdir()
                artifact = cache / "artifact"
                artifact.write_bytes(b"I remain retained.")
                if name == "old-cache": (cache / ".build.lock").touch()
                result = subprocess.run(["python3", "scripts/clean_build_trees.py", "--root", name],
                                        cwd=directory, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(artifact.read_bytes(), b"I remain retained.")

    def test_symlink_ancestor_refuses_before_removal(self):
        with tempfile.TemporaryDirectory(prefix="nano-clean-link-") as tmp:
            directory = Path(tmp)
            fixture(directory)
            external = directory / "external/obj"
            external.mkdir(parents=True)
            artifact = external / "keep"
            artifact.write_bytes(b"I am outside the requested tree.")
            (directory / "alias").symlink_to(external.parent, target_is_directory=True)
            result = subprocess.run(["python3", "scripts/clean_build_trees.py", "--root", "alias/obj"],
                                    cwd=directory, capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(artifact.read_bytes(), b"I am outside the requested tree.")


if __name__ == "__main__":
    unittest.main()
