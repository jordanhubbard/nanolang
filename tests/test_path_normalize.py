"""I exercise public module and generated-native normalizers beyond old limits."""
import ast
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PathNormalize(unittest.TestCase):
    def test_dynamic_paths(self):
        source = (ROOT / "src/stdlib_runtime.c").read_text()
        emitted = "".join(ast.literal_eval(item) for item in re.findall(
            r'sb_append\(sb, ("(?:[^"\\]|\\.)*")\);', source))
        match = re.search(r'static char\* nl_os_path_normalize\([^)]*\) \{.*?\n\}', emitted, re.S)
        self.assertIsNotNone(match)
        for module in (False, True):
            with self.subTest(module=module), tempfile.TemporaryDirectory(prefix="nano-normalize-") as tmp:
                prefix = '#include <stdlib.h>\n#include <string.h>\n#include <assert.h>\n'
                if module:
                    prefix += '#include "modules/std/fs.h"\n#define normalize path_normalize\n'
                else:
                    prefix += ('#include "runtime/path_normalize.h"\n'
                               'static char *gc_alloc_string(size_t size) { return calloc(size + 1, 1); }\n' +
                               match.group(0) + '\n#define normalize nl_os_path_normalize\n')
                program = prefix + r'''
static void check(const char *input, const char *expected) {
    const char *result = normalize(input);
    assert(result && strcmp(result, expected) == 0);
    free((void *)result);
}
int main(void) {
    check("", "."); check("///", "/"); check("a//b/./..", "a");
    check("../../a/../b", "../../b"); check("/../../a/..", "/");
    check("a/../../b", "../b"); check(".../..", ".");
    char parents[2101], components[1401], cancelled[3501], long_name[5001];
    for (int i = 0; i < 700; ++i) {
        memcpy(parents + i * 3, "../", 3);
        memcpy(components + i * 2, "x/", 2);
    }
    parents[2099] = 0; components[1399] = 0;
    strcpy(cancelled, components);
    for (int i = 0; i < 700; ++i) memcpy(cancelled + 1399 + i * 3, "/..", 3);
    cancelled[3499] = 0;
    memset(long_name, 'x', 5000); long_name[5000] = 0;
    check(parents, parents); check(components, components);
    check(cancelled, "."); check(long_name, long_name);
    return 0;
}
'''
                if module:
                    program = program.replace('    return 0;\n}', r'''
    const char *relative = path_relpath("/a/b/c", "/a");
    assert(relative && strcmp(relative, "b/c") == 0); free((void *)relative);
    relative = path_relpath("/a", "/a/b/c");
    assert(relative && strcmp(relative, "../..") == 0); free((void *)relative);
    relative = path_relpath(components, components);
    assert(relative && strcmp(relative, ".") == 0); free((void *)relative);
    char shared[1500];
    strcpy(shared, components); strcat(shared, "/leaf");
    relative = path_relpath(shared, components);
    assert(relative && strcmp(relative, "leaf") == 0); free((void *)relative);
    char absolute_long[5002]; absolute_long[0] = '/'; strcpy(absolute_long + 1, long_name);
    relative = path_relpath(absolute_long, "/");
    assert(relative && strcmp(relative, long_name) == 0); free((void *)relative);
    char many[3001], expected[4505];
    for (int i = 0; i < 1500; ++i) {
        memcpy(many + i * 2, "x/", 2);
        memcpy(expected + i * 3, "../", 3);
    }
    many[2999] = 0; memcpy(expected + 4500, "leaf", 5);
    relative = path_relpath("leaf", many);
    assert(relative && strcmp(relative, expected) == 0); free((void *)relative);
    return 0;
}''')
                work = Path(tmp)
                (work / "probe.c").write_text(program)
                command = [*shlex.split(os.environ.get("CC", "cc")), "-std=c99", "-D_GNU_SOURCE",
                           "-Wall", "-Wextra", "-Werror", "-Isrc", "-I.", str(work / "probe.c")]
                if module:
                    command += ["modules/std/fs.c", "src/runtime/dyn_array.c",
                                "src/runtime/gc.c", "src/runtime/gc_struct.c"]
                command += ["-o", str(work / "probe")]
                built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stderr.decode())
                run = subprocess.run([str(work / "probe")], cwd=work, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr.decode())

    def test_relative_anchors(self):
        with tempfile.TemporaryDirectory(prefix="nano-relpath-anchors-") as tmp:
            work = Path(tmp)
            source = work / "probe.c"
            source.write_text(r'''
#include "modules/std/fs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
int main(int argc, char **argv) {
    if (argc == 2) {
        if (mkdir("vanished", 0700) || chdir("vanished") || rmdir("../vanished")) return 2;
        const char *result = path_relpath("/a/b", "/a");
        if (!result || strcmp(result, "b")) return 3;
        free((void *)result);
        result = path_relpath("leaf", ".");
        if (result) { free((void *)result); return 4; }
        return 0;
    }
    if (argc != 4) return 2;
    const char *result = path_relpath(argv[1], argv[2]);
    if (!result || strcmp(result, argv[3])) {
        fprintf(stderr, "I got %s, expected %s\n", result ? result : "NULL", argv[3]);
        free((void *)result); return 1;
    }
    free((void *)result); return 0;
}
''')
            built = subprocess.run([*shlex.split(os.environ.get("CC", "cc")),
                                    "-std=c99", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-Werror",
                                    "-Isrc", "-I.", str(source), "modules/std/fs.c",
                                    "src/runtime/dyn_array.c", "src/runtime/gc.c", "src/runtime/gc_struct.c",
                                    "-o", str(work / "probe")], cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr.decode())
            cases = [(".", "."), ("leaf", "."), (".", "leaf"), ("../leaf", "."),
                     ("leaf", "../a"), ("/a/b", "."), ("leaf", "/a"), ("", ""),
                     ("", "a"), ("/a/../b", "/"), ("../..", "../../x"),
                     ("a/./b/../../c", "a/../b"), ("missing/target", "missing/base")]
            anchor = str(work.resolve())
            for target, base in cases:
                with self.subTest(target=target, base=base):
                    target_absolute = os.path.normpath(os.path.join(anchor, target))
                    base_absolute = os.path.normpath(os.path.join(anchor, base))
                    expected = os.path.relpath(target_absolute, base_absolute)
                    run = subprocess.run([str(work / "probe"), target, base, expected],
                                         cwd=work, capture_output=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stderr.decode())
                    self.assertEqual(os.path.normpath(os.path.join(base_absolute, expected)), target_absolute)
            deep = work
            for _ in range(3):
                deep = deep / ("d" * 100)
                deep.mkdir()
            expected = str(deep.resolve() / "leaf").lstrip("/")
            run = subprocess.run([str(work / "probe"), "leaf", "/", expected], cwd=deep,
                                 capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr.decode())
            run = subprocess.run([str(work / "probe"), "deleted-cwd"], cwd=work,
                                 capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr.decode())


if __name__ == "__main__":
    unittest.main()
