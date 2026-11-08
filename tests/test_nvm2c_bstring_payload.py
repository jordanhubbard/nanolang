"""I check my mechanically shared payload against the complete original string units."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest

from scripts.gen_nvm2c_bstring_payload import payload_source
from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class NativeBstringPayload(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command

    def test_original_string_units_with_generated_payload(self):
        output = ROOT/'.test_output/bstring-native-payload'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='original-', dir=output))
        paths = [ROOT/'src/runtime/nl_string.h', ROOT/'src/runtime/nl_string.c',
                 ROOT/'src/utf8.c', ROOT/'src/utf8.h', ROOT/'tests/test_nl_string.c',
                 ROOT/'scripts/gen_nvm2c_bstring_payload.py',
                 ROOT/'src/nanoisa/nvm2c_bstring_payload.inc']
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        source, product = work/'unit.c', work/'unit'
        original = paths[4].read_text()
        self.assertEqual(original.count('#include "../src/runtime/nl_string.h"'), 1)
        source.write_text(original.replace('#include "../src/runtime/nl_string.h"', payload_source())
                          .replace('nl_string_', 'nbpayload_'))
        pin = None
        try:
            self.command(work, 'generation-check', ['python3', paths[5], '--check'])
            self.command(work, 'build', [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
                '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                source, paths[2], '-o', product], build=True)
            pin = hashlib.sha256(product.read_bytes()).hexdigest()
            run = self.command(work, 'run', [product])
            self.assertNotIn('FAILED', run.stdout)
            self.assertEqual(run.stdout.count('Testing '), 16)
            self.assertIn('All tests passed!', run.stdout)
        finally:
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            observed = hashlib.sha256(product.read_bytes()).hexdigest() if product.exists() else None
            (work/'inputs.json').write_text(json.dumps({'scope':
                'I test only the derived C payload; native BSTRING imports and tagged owner translation remain unimplemented.',
                'before': before, 'after': after, 'product_before_run': pin,
                'product_after_run': observed}, indent=2)+'\n')
            self.assertEqual(before, after)
            if pin:
                self.assertEqual(pin, observed)
