"""I retain manifest ownership and scalar declarations in canonical products."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT/'bin/nanoc_stage2')).resolve()


class DeclaredScalarSource(unittest.TestCase):
    def test_mixed_scalar_manifest_through_native_and_vm(self):
        with tempfile.TemporaryDirectory(prefix='declared_scalar_', dir=ROOT/'modules') as tmp:
            module = Path(tmp)
            (module/'module.json').write_text(json.dumps({'name':module.name,'c_sources':['provider.c']}))
            (module/'provider.c').write_text('''#include <stdint.h>
#include <string.h>
int64_t scalar_mix(int64_t a, double b, uint8_t c, uint8_t d, const char *text) {
    return a == -7 && b == 2.5 && c == 1 && d == 250 && !strcmp(text, "payload") ? 37 : -1;
}
const char *scalar_alias(const char *text) { return text; }
''')
            (module/'api.nano').write_text('''module ScalarFixture
extern fn scalar_mix(a: int, b: float, c: bool, d: u8, text: string) -> int
extern fn scalar_alias(text: string) -> string
pub fn value() -> int { unsafe { return (scalar_mix -7 2.5 true 250 "payload") } }
shadow value { assert (== (value) 37) }
pub fn alias(text: string) -> string { unsafe { return (scalar_alias text) } }
shadow alias { assert (== (alias "retained") "retained") }
''')
            program = module/'main.nano'
            program.write_text('module "'+str(module/'api.nano')+'" as fixture\n'
                'fn main() -> int { assert (== (fixture.value) 37) '
                'assert (== (fixture.alias "retained") "retained") return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            for suffix, flags in [('native',[]),('nvm',['--emit-nvm'])]:
                with self.subTest(product=suffix):
                    output = module/('product.'+suffix)
                    result = subprocess.run([COMPILER,program,*flags,'-o',output],cwd=ROOT,
                        capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    command = [output] if suffix == 'native' else [ROOT/'bin/nano_vm',output]
                    result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    if suffix == 'nvm':
                        result = subprocess.run([ROOT/'bin/nanoisa','dump',output],capture_output=True,text=True,timeout=30)
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertIn('declared_scalar_artifact',result.stdout)
                        self.assertIn('int int float bool u8 string',result.stdout)

    def test_missing_manifest_preserves_prior_output(self):
        with tempfile.TemporaryDirectory(prefix='scalar_no_manifest_') as tmp:
            directory = Path(tmp)
            source, output = directory/'main.nano', directory/'program'
            source.write_text('extern fn uncatalogued_scalar() -> int\n'
                'fn main() -> int { unsafe { return (uncatalogued_scalar) } }\n'
                'shadow main { assert true }\n')
            output.write_text('prior output')
            result = subprocess.run([COMPILER,source,'-o',output],cwd=ROOT,
                capture_output=True,text=True,timeout=120)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('owning module artifact',result.stdout+result.stderr)
            self.assertEqual(output.read_text(),'prior output')

    def test_unsupported_declared_shape_preserves_prior_output(self):
        for declaration, invocation in [
            ('extern fn unsupported(value: array<int>) -> int', '(unsupported [1])'),
            ('extern fn unsupported() -> array<int>', '(array_length (unsupported))'),
        ]:
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory(prefix='scalar_shape_') as tmp:
                directory = Path(tmp)
                source, output = directory/'main.nano', directory/'program'
                source.write_text(declaration+'\nfn main() -> int { unsafe { return '+invocation+' } }\n'
                                  'shadow main { assert true }\n')
                output.write_text('prior output')
                result = subprocess.run([COMPILER,source,'-o',output],cwd=ROOT,
                    capture_output=True,text=True,timeout=120)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('scalar artifact',result.stdout+result.stderr)
                self.assertEqual(output.read_text(),'prior output')
