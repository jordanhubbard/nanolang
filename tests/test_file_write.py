"""I execute production writer bodies with controlled stdio failures."""
import ast
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileWrite(unittest.TestCase):
    def test_production_wrappers(self):
        cases = [
            ("src/eval/eval_io.c", "builtin_file_write", "w", True),
            ("src/eval/eval_io.c", "builtin_file_append", "a", True),
            ("src/nanovm/vm_builtins.c", "vm_file_write", "w", False),
            ("modules/std/fs.c", "file_write", "w", False),
            ("modules/std/fs.c", "file_append", "a", False),
            ("src/stdlib_runtime.c", "nl_os_file_write", "w", False),
            ("src/stdlib_runtime.c", "nl_os_file_append", "a", False),
        ]
        for filename, name, mode, interpreter in cases:
            with self.subTest(wrapper=name), tempfile.TemporaryDirectory(prefix="nano-write-errors-") as tmp:
                source = (ROOT / filename).read_text()
                if filename == "src/stdlib_runtime.c":
                    source = "".join(ast.literal_eval(item) for item in re.findall(
                        r'sb_append\(sb, ("(?:[^"\\]|\\.)*")\);', source))
                match = re.search(r'(?:static )?(?:Value|int64_t) ' + name +
                                  r'\([^)]*\) \{.*?\n\}', source, re.S)
                self.assertIsNotNone(match, "I require the production function body")
                body = match.group(0)
                call = f"{name}(args).as.int_val" if interpreter else f'{name}("fixture", text)'
                setup = ('Value args[2] = {0}; args[0].as.string_val = "fixture"; '
                         'args[1].as.string_val = text;' if interpreter else '')
                program = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int failure, opens, writes, closes;
static const char *wanted_mode;
static FILE *probe_open(const char *path, const char *mode) {
    ++opens; assert(strcmp(path, "fixture") == 0);
    assert(strcmp(mode, wanted_mode) == 0);
    return failure == 1 ? NULL : tmpfile();
}
static size_t probe_write(const void *data, size_t size, size_t count, FILE *file) {
    ++writes;
    if (failure == 2) return count ? count - 1 : 0;
    return fwrite(data, size, count, file);
}
static int probe_close(FILE *file) {
    ++closes; int result = fclose(file);
    return failure == 3 ? EOF : result;
}
#define fopen probe_open
#define fwrite probe_write
#define fclose probe_close
#include "runtime/file_write.h"
#undef fopen
#undef fwrite
#undef fclose
'''
                if interpreter:
                    program += ('typedef struct { union { const char *string_val; int64_t int_val; } as; } Value;\n'
                                'static Value create_int(int64_t i) { Value v; v.as.int_val = i; return v; }\n')
                program += body
                program += f'''
int main(void) {{
    wanted_mode = "{mode}";
    for (failure = 0; failure < 4; ++failure) {{
        opens = writes = closes = 0;
        const char *text = "payload";
        {setup}
        assert(({call}) == (failure ? -1 : 0));
        assert(opens == 1);
        assert(writes == (failure == 1 ? 0 : 1));
        assert(closes == (failure == 1 ? 0 : 1));
    }}
    failure = 0; opens = writes = closes = 0;
    const char *text = "";
    {setup}
    assert(({call}) == 0);
    assert(opens == 1 && writes == 1 && closes == 1);
    return 0;
}}
'''
                work = Path(tmp)
                (work / "probe.c").write_text(program)
                built = subprocess.run([*shlex.split(os.environ.get("CC", "cc")),
                                        "-std=c99", "-Wall", "-Wextra", "-Werror", "-Isrc",
                                        str(work / "probe.c"), "-o", str(work / "probe")],
                                       cwd=ROOT, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stderr.decode())
                run = subprocess.run([str(work / "probe")], cwd=work, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr.decode())


if __name__ == "__main__":
    unittest.main()
