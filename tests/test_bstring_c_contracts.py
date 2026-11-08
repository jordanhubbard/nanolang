"""I exercise generated C bounds; interpreter builtin shadows remain a separate gate."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class BstringCContracts(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command

    def test_extreme_generated_c_bounds_under_sanitizers(self):
        source = '''fn main()->int{
 let value:bstring=(bstr_new "abc")
 let tail:bstring=(bstr_substring value 1 9223372036854775807)
 let beyond:bstring=(bstr_substring value 9223372036854775807 9223372036854775807)
 let negative:bstring=(bstr_substring value -1 9223372036854775807)
 let negative_length:bstring=(bstr_substring value 1 -1)
 let empty:bstring=(bstr_substring value 3 9223372036854775807)
 assert (== (bstr_length tail) 2)
 assert (== (bstr_to_cstr tail) "bc")
 assert (== (bstr_length beyond) 0)
 assert (== (bstr_length negative) 0)
 assert (== (bstr_length negative_length) 0)
 assert (== (bstr_length empty) 0)
 assert (== (bstr_byte_at value -1) 0)
 assert (== (bstr_byte_at value 9223372036854775807) 0)
 (bstr_free tail) (bstr_free beyond) (bstr_free negative)
 (bstr_free negative_length) (bstr_free empty) (bstr_free value)
 return 0
}
# I check the production native body; useful interpreter builtin shadows remain open.
shadow main{assert true}
'''
        output = ROOT/'.test_output/bstring-c-contracts'
        output.mkdir(parents=True, exist_ok=True)
        for optimization in ['-O0', '-O2']:
            with self.subTest(optimization=optimization):
                work = Path(tempfile.mkdtemp(prefix=optimization[1:]+'-', dir=output))
                path, product = work/'main.nano', work/'native'
                path.write_text(source)
                compiler = ROOT/'bin/nanoc_c'
                before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, compiler]}
                cflags = os.environ.get('NANO_CFLAGS', '')+' '+optimization+' -fsanitize=address,undefined -fno-sanitize-recover=all'
                product_before = None
                try:
                    with patch.dict(os.environ, {'NANO_CFLAGS': cflags}):
                        self.command(work, 'build', [compiler, path, '--keep-c', '-o', product], build=True)
                    self.assertIn(product.read_bytes()[:4], [b'\x7fELF', bytes.fromhex('cffaedfe'), bytes.fromhex('feedfacf')])
                    product_before = hashlib.sha256(product.read_bytes()).hexdigest()
                    self.assertEqual(self.command(work, 'run', [product]).stdout, '')
                finally:
                    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, compiler]}
                    product_after = hashlib.sha256(product.read_bytes()).hexdigest() if product.exists() else None
                    (work/'inputs.json').write_text(json.dumps({'scope': 'I test generated C production bounds, not interpreter builtin parity.',
                        'native_build_cflags': cflags, 'before': before, 'after': after,
                        'product_before_run': product_before, 'product_after_run': product_after}, indent=2)+'\n')
                    self.assertEqual(before, after)
                    if product_before:
                        self.assertEqual(product_before, product_after)
