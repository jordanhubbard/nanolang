"""I exercise the module and emitted-runtime walker with real filesystem trees."""
import errno
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DirectoryWalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="nano-walk-build-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.programs = []
        for header in (False, True):
            output = Path(cls.build.name) / ("header" if header else "module")
            command = [*shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-g",
                       "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra", "-Werror", "-Isrc",
                       "tests/fs_walk_probe.c", "src/runtime/dyn_array.c", "src/runtime/gc.c",
                       "src/runtime/gc_struct.c", "-o", str(output)]
            command += ["-DWALK_HEADER"] if header else ["modules/std/fs.c"]
            built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
            if built.returncode:
                raise AssertionError(built.stderr.decode(errors="replace"))
            cls.programs.append(output)

    def walk(self, root):
        results = []
        for program in self.programs:
            result = subprocess.run([program, str(root)], capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(sorted(os.fsdecode(path) for path in result.stdout.split(b"\0") if path))
        self.assertEqual(results[0], results[1])
        return results[0]

    def test_cycles_aliases_and_broken_links(self):
        with tempfile.TemporaryDirectory(prefix="nano-walk-cycle-") as tmp:
            root = Path(tmp)
            child = root / "child"
            child.mkdir()
            (child / "leaf.txt").write_text("leaf")
            (root / "top.txt").write_text("top")
            (child / "back").symlink_to(root, target_is_directory=True)
            (root / "alias").symlink_to(child, target_is_directory=True)
            (root / "file-alias").symlink_to(root / "top.txt")
            (root / "broken").symlink_to(root / "absent")
            paths = self.walk(root)
            self.assertEqual(len(paths), 3)
            self.assertEqual(sum(path.endswith("leaf.txt") for path in paths), 1)
            self.assertIn(str(root / "file-alias"), paths)

    def test_deep_tree_under_small_descriptor_limit(self):
        with tempfile.TemporaryDirectory(prefix="nano-walk-depth-") as tmp:
            leaf = Path(tmp)
            for _ in range(120):
                leaf /= "d"
                leaf.mkdir()
            target = leaf / "file"
            target.write_text("deep")
            self.assertEqual(self.walk(tmp), [str(target)])

    def test_queue_growth_and_exact_file_names(self):
        with tempfile.TemporaryDirectory(prefix="nano-walk-wide-") as tmp:
            expected = []
            for i in range(40):
                directory = Path(tmp) / str(i)
                directory.mkdir()
                target = directory / "spaces é\nfile"
                target.write_text("wide")
                expected.append(str(target))
            self.assertEqual(self.walk(tmp), sorted(expected))

    def test_paths_beyond_old_buffer_when_host_supports_them(self):
        with tempfile.TemporaryDirectory(prefix="nano-walk-long-") as tmp:
            leaf = Path(tmp)
            try:
                while len(os.fsencode(leaf)) < 2200:
                    leaf /= "d" * 80
                    leaf.mkdir()
                target = leaf / "file"
                target.write_text("long")
            except OSError as error:
                if error.errno == errno.ENAMETOOLONG:
                    self.skipTest("Host path limit is below the former 2048-byte buffer")
                raise
            self.assertEqual(self.walk(tmp), [str(target)])

    def test_missing_empty_and_unreadable(self):
        self.assertEqual(self.walk(""), [])
        with tempfile.TemporaryDirectory(prefix="nano-walk-errors-") as tmp:
            root = Path(tmp)
            self.assertEqual(self.walk(root / "missing"), [])
            target = root / "visible"
            target.write_text("visible")
            self.assertEqual(self.walk(target), [])
            if os.geteuid() == 0:
                return
            hidden = root / "hidden"
            hidden.mkdir()
            (hidden / "secret").write_text("hidden")
            hidden.chmod(0)
            try:
                self.assertEqual(self.walk(root), [str(target)])
            finally:
                hidden.chmod(0o700)

    def test_language_native_and_vm_entry_points(self):
        with tempfile.TemporaryDirectory(prefix="nano-walk-language-") as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            leaf = root
            for _ in range(50):
                leaf /= "d"
                leaf.mkdir()
            (leaf / "one").write_text("one")
            (leaf / "cycle").symlink_to(root, target_is_directory=True)
            source = Path(tmp) / "walk.nano"
            source.write_text(
                f'module "{ROOT}/modules/std/fs.nano" as fs\n'
                'fn main() -> int {\n'
                f'    let files: array<string> = (fs.walkdir {json.dumps(str(root))})\n'
                '    assert (== (array_length files) 1)\n'
                '    return 0\n}\nshadow main { assert (== (main) 0) }\n')
            for name, variable, bytecode in (("nanoc_c", "NANO_TEST_NATIVE_COMPILER", False),
                                             ("nano_virt", "NANO_TEST_VM_COMPILER", True)):
                with self.subTest(backend=name):
                    compiler = os.environ.get(variable, str(ROOT / "bin" / name))
                    output = Path(tmp) / ("walk.nvm" if bytecode else "walk")
                    command = [compiler, str(source), "-o", str(output)]
                    if bytecode:
                        command.append("--emit-nvm")
                    built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                    self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
                    command = [str(ROOT / "bin/nano_vm"), str(output)] if bytecode else [str(output)]
                    ran = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=15)
                    self.assertEqual(ran.returncode, 0, ran.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
