"""I test complete-file publication and executable generic-list output."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_list.sh"


class ListGenerator(unittest.TestCase):
    def generate(self, directory, name="Point", definition="int", **kwargs):
        return subprocess.run(["bash", str(GENERATOR), name, str(directory), definition],
                              text=True, capture_output=True, timeout=20, **kwargs)

    def test_generated_runtime_executes_with_space_in_path(self):
        with tempfile.TemporaryDirectory(prefix="nanolang list 'test-") as directory:
            path = Path(directory)
            result = self.generate(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            driver = path / "driver.c"
            driver.write_text('#include "list_Point.h"\n#include <assert.h>\n'
                              'int main(void) { List_Point *xs = nl_list_Point_new(); '
                              'nl_list_Point_push(xs, 42); assert(nl_list_Point_pop(xs) == 42); '
                              'nl_list_Point_free(xs); return 0; }\n')
            subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                            str(driver), str(path / "list_Point.c"), "-o", str(path / "test")],
                           check=True, capture_output=True, timeout=20)
            subprocess.run([str(path / "test")], check=True, timeout=10)
            self.assertFalse(list(path.glob(".nanolang-list.*")))

    def test_invalid_type_name_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../escape", "a/b", "bad-name", "x;echo", ""):
                with self.subTest(name=name):
                    result = self.generate(directory, name)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_generated_capacity_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = self.generate(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            driver = path / "capacity.c"
            driver.write_text(r'''
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
static void *checked_malloc(size_t size) {
    assert(size != 0);
    return malloc(size);
}
static void *checked_realloc(void *ptr, size_t size) {
    if (size > 1048576) {
        assert(size == sizeof(int) * ((size_t)INT_MAX / 2 + 2));
        fputs("checked large growth\n", stderr);
        exit(77);
    }
    return realloc(ptr, size);
}
#define malloc checked_malloc
#define realloc checked_realloc
#include "list_Point.c"
#undef malloc
#undef realloc
int main(int argc, char **argv) {
    assert(argc == 2);
    int mode = atoi(argv[1]);
    if (mode == 1) { nl_list_Point_with_capacity(-1); return 10; }
    List_Point full = {NULL, INT_MAX, INT_MAX};
    if (mode == 2) { nl_list_Point_push(&full, 1); return 10; }
    if (mode == 3) { nl_list_Point_insert(&full, 0, 1); return 10; }
    if (mode == 4) {
        List_Point large = {NULL, 0, INT_MAX / 2 + 1};
        ensure_capacity_Point(&large, INT_MAX / 2 + 2);
        return 10;
    }
    List_Point *xs = nl_list_Point_with_capacity(0);
    assert(xs && xs->data == NULL && xs->capacity == 0);
    for (int i = 0; i < 33; i++) nl_list_Point_push(xs, i);
    assert(xs->length == 33 && xs->capacity >= 33);
    nl_list_Point_insert(xs, 0, 99);
    assert(nl_list_Point_remove(xs, 0) == 99);
    for (int i = 32; i >= 0; i--) assert(nl_list_Point_pop(xs) == i);
    nl_list_Point_free(xs);
    return 0;
}
''')
            subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                            "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
                            str(driver), "-o", str(path / "capacity")],
                           check=True, capture_output=True, timeout=20)
            for mode, status, diagnostic in ((0, 0, ""), (1, 1, "capacity"),
                                            (2, 1, "length"), (3, 1, "length"),
                                            (4, 77, "checked large growth")):
                with self.subTest(mode=mode):
                    run = subprocess.run([str(path / "capacity"), str(mode)],
                                         text=True, capture_output=True, timeout=10)
                    self.assertEqual(run.returncode, status, run.stderr)
                    self.assertIn(diagnostic, run.stderr)
                    self.assertNotIn("runtime error:", run.stderr)

    def test_type_definition_is_not_a_sed_replacement_program(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = "int /* & | \\ */"
            result = self.generate(directory, definition=definition)
            self.assertEqual(result.returncode, 0, result.stderr)
            source = (Path(directory) / "list_Point.c").read_text()
            self.assertIn(definition + " *new_data", source)
            self.assertNotIn("$TYPE_DEF", source)

    def test_substitution_failure_preserves_published_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            header, source = path / "list_Point.h", path / "list_Point.c"
            header.write_text("old header")
            source.write_text("old source")
            shim = path / "sed"
            shim.write_text("#!/bin/sh\nexit 7\n")
            shim.chmod(0o755)
            result = self.generate(path, env=dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"]))
            self.assertEqual(result.returncode, 7)
            self.assertEqual(header.read_text(), "old header")
            self.assertEqual(source.read_text(), "old source")
            self.assertFalse(list(path.glob(".nanolang-list.*")))

    def test_overlapping_generators_do_not_expose_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            header, source = path / "list_Point.h", path / "list_Point.c"
            header.write_text("old header")
            source.write_text("old source")
            shims = path / "shims"
            shims.mkdir()
            shim = shims / "sed"
            shim.write_text(f"#!{sys.executable}\n" + '''import os, pathlib, sys, time
root = pathlib.Path(os.environ["PROBE_ROOT"])
identity = os.environ["PROBE_ID"]
ready = root / (identity + ".ready")
if not ready.exists():
    ready.touch()
    deadline = time.monotonic() + 30
    while not (root / (identity + ".release")).exists():
        if time.monotonic() > deadline: sys.exit(8)
        time.sleep(0.01)
os.execv(os.environ["REAL_SED"], [os.environ["REAL_SED"]] + sys.argv[1:])
''')
            shim.chmod(0o755)
            env = dict(os.environ, PATH=str(shims) + os.pathsep + os.environ["PATH"],
                       REAL_SED=shutil.which("sed"), PROBE_ROOT=directory)
            processes = []
            try:
                for identity in ("one", "two"):
                    processes.append(subprocess.Popen(
                        ["bash", str(GENERATOR), "Point", directory, "int"],
                        env=dict(env, PROBE_ID=identity), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, start_new_session=True))
                deadline = time.monotonic() + 20
                while not all((path / (name + ".ready")).exists() for name in ("one", "two")):
                    self.assertTrue(all(p.poll() is None for p in processes))
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                self.assertEqual(header.read_text(), "old header")
                self.assertEqual(source.read_text(), "old source")
                self.assertEqual(len(list(path.glob(".nanolang-list.*"))), 2)
                for identity, process in zip(("one", "two"), processes):
                    (path / (identity + ".release")).touch()
                    output, _ = process.communicate(timeout=20)
                    self.assertEqual(process.returncode, 0, output)
                    self.assertIn("List_Point", header.read_text())
                    self.assertNotIn("TYPENAME", source.read_text())
                    self.assertNotIn("$TYPE_DEF", source.read_text())
                self.assertFalse(list(path.glob(".nanolang-list.*")))
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
