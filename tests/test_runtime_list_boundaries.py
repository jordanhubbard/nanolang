"""I compile each checked-in list implementation and exercise its boundaries."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeListBoundaries(unittest.TestCase):
    def test_string_copy_failure_and_aliasing(self):
        with tempfile.TemporaryDirectory(prefix="nanolang-string-list-") as directory:
            binary = str(Path(directory) / "test")
            compile = subprocess.run([
                "cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                str(ROOT / "tests/test_string_list_failure.c"), "-o", binary],
                text=True, capture_output=True, timeout=30)
            self.assertEqual(compile.returncode, 0, compile.stderr)
            run = subprocess.run([binary], text=True, capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_every_checked_in_list(self):
        sources = sorted((ROOT / "src/runtime").glob("list_*.c"))
        self.assertGreaterEqual(len(sources), 39)
        with tempfile.TemporaryDirectory(prefix="nanolang-list-boundaries-") as directory:
            path = Path(directory)
            for source in sources:
                with self.subTest(source=source.name):
                    text = source.read_text()
                    constructor = re.search(r"(List_\w+)\*\s+(\w+)_with_capacity\(int capacity\)", text)
                    self.assertIsNotNone(constructor)
                    list_type, prefix = constructor.groups()
                    ensure = re.search(r"static void (ensure_capacity\w*)\(", text).group(1)
                    value_type = re.search(r"void \w+_push\([^,]+, (.*?)\bvalue\)", text).group(1).strip()
                    string = source.name == "list_string.c"
                    value = '"value"' if string else f"({value_type}){{0}}"
                    discard = "free((void *)VALUE);" if string else "(void)VALUE;"
                    driver = path / "driver.c"
                    driver.write_text(f'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
static size_t element_size;
static void *checked_malloc(size_t size) {{ assert(size); return malloc(size); }}
static void *checked_realloc(void *pointer, size_t size) {{
    if (size > 1048576) {{
        assert(size == element_size * ((size_t)INT_MAX / 2 + 2));
        fputs("checked large growth\\n", stderr);
        exit(77);
    }}
    return realloc(pointer, size);
}}
#define malloc checked_malloc
#define realloc checked_realloc
#include "{source}"
#undef malloc
#undef realloc
int main(int argc, char **argv) {{
    assert(argc == 2);
    int mode = atoi(argv[1]);
    element_size = sizeof(*(({list_type} *)0)->data);
    if (mode == 1) {{ {prefix}_with_capacity(-1); return 10; }}
    {list_type} full = {{NULL, INT_MAX, INT_MAX}};
    if (mode == 2) {{ {prefix}_push(&full, {value}); return 10; }}
    if (mode == 3) {{ {prefix}_insert(&full, 0, {value}); return 10; }}
    if (mode == 4) {{
        {list_type} large = {{NULL, 0, INT_MAX / 2 + 1}};
        {ensure}(&large, INT_MAX / 2 + 2);
        return 10;
    }}
    {list_type} *xs = {prefix}_with_capacity(0);
    assert(xs && !xs->data && xs->capacity == 0);
    for (int i = 0; i < 33; i++) {prefix}_push(xs, {value});
    assert(xs->length == 33 && xs->capacity >= 33);
    {prefix}_insert(xs, 0, {value});
    {discard.replace('VALUE', prefix + '_remove(xs, 0)')}
    for (int i = 0; i < 33; i++) {{ {discard.replace('VALUE', prefix + '_pop(xs)')} }}
    assert(xs->length == 0);
    {prefix}_free(xs);
    return 0;
}}
''')
                    compile = subprocess.run([
                        "cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                        "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
                        "-I", str(ROOT / "src"),
                        str(driver), "-o", str(path / "test")],
                        text=True, capture_output=True, timeout=30)
                    self.assertEqual(compile.returncode, 0, compile.stderr)
                    for mode, status, diagnostic in ((0, 0, ""), (1, 1, "capacity"),
                                                    (2, 1, "length"), (3, 1, "length"),
                                                    (4, 77, "checked large growth")):
                        run = subprocess.run([str(path / "test"), str(mode)],
                                             text=True, capture_output=True, timeout=10)
                        self.assertEqual(run.returncode, status, run.stderr)
                        self.assertIn(diagnostic, run.stderr)
                        self.assertNotIn("runtime error:", run.stderr)


if __name__ == "__main__":
    unittest.main()
