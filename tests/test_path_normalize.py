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


if __name__ == "__main__":
    unittest.main()
