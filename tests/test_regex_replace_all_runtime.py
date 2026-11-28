"""I run actual shared regex growth and ownership cases with native sanitizers."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_numeric_builtin_selection as numeric

ROOT = Path(__file__).resolve().parents[1]


class RegexReplacementRuntime(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command

    def test_actual_shared_runtime_replacement_matrix(self):
        output = ROOT / '.test_output/regex-runtime-regressions'
        output.mkdir(parents=True, exist_ok=True)
        names = ['tests/test_regex_replace_all.c', 'src/runtime/regex.c',
                 'std/regex/regex_impl.c',
                 'src/runtime/dyn_array.c', 'src/runtime/gc.c',
                 'src/runtime/gc_struct.c', 'src/utf8.c']
        before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
        for entry in ('src/runtime/regex.c', 'std/regex/regex_impl.c'):
            with self.subTest(entry=entry):
                work = Path(tempfile.mkdtemp(prefix='replacement-', dir=output))
                product = work / 'replacement'
                try:
                    self.command(work, 'build', [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                        '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                        ROOT / names[0], ROOT / entry, *[ROOT / name for name in names[3:]],
                        '-lm', '-o', product], build=True)
                    pin = hashlib.sha256(product.read_bytes()).hexdigest()
                    result = self.command(work, 'run', [product])
                    self.assertEqual(result.stdout, 'I passed 24 regex replacement cases.\n')
                    self.assertEqual(result.stderr, '')
                    self.assertEqual(hashlib.sha256(product.read_bytes()).hexdigest(), pin)
                finally:
                    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
                    (work / 'inputs.json').write_text(json.dumps({'entry': entry, 'before': before, 'after': after}, indent=2) + '\n')
                    self.assertEqual(before, after)
