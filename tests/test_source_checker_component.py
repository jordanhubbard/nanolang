"""I execute my Nano checker and STRING ownership controls through my C creation path."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests import test_numeric_builtin_selection as numeric

ROOT = numeric.ROOT


class SourceCheckerComponent(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command

    def test_actual_source_checker_native_component(self):
        compiler = ROOT/'bin/nanoc_c'
        self.assertTrue(compiler.is_file(), 'I require my actual C seed.')
        source = ROOT/'src_nano/typecheck_driver.nano'
        inputs = [compiler, source, ROOT/'src_nano/typecheck.nano', ROOT/'src_nano/parser.nano',
                  ROOT/'src_nano/compiler/lexer.nano']
        before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in inputs}
        output = ROOT/'.test_output/source-checker-component'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='actual-native-', dir=output))
        (work/'inputs-before.json').write_text(json.dumps(before, indent=2)+'\n')
        product = work/'checker'
        product_before = None
        try:
            self.command(work, 'source-build', [compiler, source, '--verbose', '--keep-c', '-o', product], build=True)
            self.assertTrue(product.is_file())
            with product.open('rb') as stream:
                magic = stream.read(4)
            if sys.platform.startswith('linux'):
                self.assertEqual(magic, b'\x7fELF', 'I require an actual native ELF product.')
            elif sys.platform == 'darwin':
                self.assertIn(magic, (bytes.fromhex('cffaedfe'), bytes.fromhex('feedfacf'),
                    bytes.fromhex('cafebabe'), bytes.fromhex('bebafeca'),
                    bytes.fromhex('cafebabf'), bytes.fromhex('bfbafeca')),
                    'I require an actual native Mach-O product.')
            product_before = hashlib.sha256(product.read_bytes()).hexdigest()
            result = self.command(work, 'native-run', [product])
            self.assertIn('I passed my checker entry assertions.\n', result.stdout)
        finally:
            after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in inputs}
            product_after = hashlib.sha256(product.read_bytes()).hexdigest() if product.is_file() else None
            (work/'inputs.json').write_text(json.dumps({'creation_path': 'C-seed-to-native',
                'native_lifecycle_admitted': False, 'fresh_stages_qualified': False,
                'platform': sys.platform,
                'product_magic_hex': product.read_bytes()[:4].hex() if product.is_file() else None,
                'before': before, 'after': after, 'product_before_run': product_before,
                'product_after_run': product_after}, indent=2)+'\n')
            self.assertEqual(before, after)
            if product_before is not None:
                self.assertEqual(product_before, product_after)

    def test_string_self_assignment_language_contract(self):
        compiler = ROOT/'bin/nanoc_c'
        output = ROOT/'.test_output/source-checker-component'
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='language-string-self-', dir=output))
        source = work/'main.nano'
        source.write_text('''fn keep(value:string)->string{
 let mut held:string=value
 set held held
 set held (str_concat held "!")
 return held
}
shadow keep{assert (== (keep "call") "call!")}
fn main()->int{
 let mut lone:string=(str_concat "ret" "ained")
 set lone lone
 assert (== lone "retained")
 let mut alias:string=lone
 set lone "changed"
 assert (== alias "retained")
 set alias alias
 assert (== alias "retained")
 assert (== (keep alias) "retained!")
 assert (== alias "retained")
 return 0
}
shadow main{assert (== (main) 0)}
''')
        product = work/'native'
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [compiler, source]}
        product_before = None
        try:
            self.command(work, 'source-build', [compiler, source, '--keep-c', '-o', product], build=True)
            magic = product.read_bytes()[:4]
            if sys.platform.startswith('linux'):
                self.assertEqual(magic, b'\x7fELF')
            elif sys.platform == 'darwin':
                self.assertIn(magic, (bytes.fromhex('cffaedfe'), bytes.fromhex('feedfacf'),
                    bytes.fromhex('cafebabe'), bytes.fromhex('bebafeca'),
                    bytes.fromhex('cafebabf'), bytes.fromhex('bfbafeca')))
            product_before = hashlib.sha256(product.read_bytes()).hexdigest()
            result = self.command(work, 'native-run', [product])
            self.assertEqual(result.stdout, '')
        finally:
            after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [compiler, source]}
            product_after = hashlib.sha256(product.read_bytes()).hexdigest() if product.is_file() else None
            (work/'inputs.json').write_text(json.dumps({'scope': 'C interpreter shadows and generated native STRING self-assignment',
                'before': before, 'after': after,
                'product_before_run': product_before, 'product_after_run': product_after}, indent=2)+'\n')
            self.assertEqual(before, after)
            if product_before is not None:
                self.assertEqual(product_before, product_after)
