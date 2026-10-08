"""I retain nested field shapes when helpers mutate shared record arrays."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2'))
TRANSLATOR = Path(os.environ.get('NANOLANG_TEST_NVM2C', ROOT / 'bin/nvm2c'))


class MutableRecordArrays(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, work):
        for name, producer in [('seed', ROOT / 'bin/nano_virt'), ('selfhost', COMPILER)]:
            with self.subTest(producer=name):
                module, c, binary = [work / (name + suffix) for suffix in ('.nvm', '.c', '.exe')]
                self.checked([producer, source, '--emit-nvm', '-o', module],
                             env={**os.environ, 'NANOLANG_ROOT': str(ROOT)})
                expected = self.checked([ROOT / 'bin/nano_vm', module]).stdout
                self.checked([TRANSLATOR, module, '-o', c])
                cc = os.environ.get('NANO_NATIVE_TEST_CC') or shutil.which('cc')
                self.assertTrue(cc)
                self.checked([cc, '-std=c11', '-O0', '-Wall', '-Wextra', '-Werror',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              c, ROOT / 'bin/nano_aot_runtime.o', '-lm',
                              *(['-Wl,--export-dynamic', '-ldl'] if sys.platform.startswith('linux') else []),
                              '-o', binary])
                actual = self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
                self.assertEqual(actual.stdout, expected)

    def test_nested_fields_after_direct_forwarded_and_indirect_mutation(self):
        source = """struct Location {file:string, line:int}
struct Diagnostic {location:Location}
struct Output {file:string, line:int}
fn project(d:Diagnostic)->Output {
 return Output {file:d.location.file, line:d.location.line}
}
shadow project {
 let o:Output = (project Diagnostic {location:Location {file:"abc",line:42}})
 assert (== o.file "abc")
 assert (== o.line 42)
}
fn add(a:array<Diagnostic>, d:Diagnostic)->void { (array_push a d) }
shadow add {
 let a:array<Diagnostic> = []
 (add a Diagnostic {location:Location {file:"abc",line:42}})
 assert (== (array_length a) 1)
}
fn forward(a:array<Diagnostic>, d:Diagnostic)->void { (add a d) }
shadow forward {
 let a:array<Diagnostic> = []
 (forward a Diagnostic {location:Location {file:"abc",line:42}})
 assert (== (array_length a) 1)
}
fn main()->int {
 let a:array<Diagnostic> = []
 let alias:array<Diagnostic> = a
 let d:Diagnostic = Diagnostic {location:Location {file:"retained",line:42}}
 MUTATION
 assert (== (array_length a) 1)
 let o:Output = (project (at a 0))
 assert (== o.file "retained")
 assert (== o.line 42)
 return 0
}
shadow main { assert (== (main) 0) }
"""
        for mutation in ['(array_push alias d)', '(add alias d)', '(forward alias d)',
                         'let callback:fn(array<Diagnostic>, Diagnostic)->void = forward\n (callback alias d)']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='nano-mutable-records-') as tmp:
                work = Path(tmp)
                path = work / 'input.nano'
                path.write_text(source.replace('MUTATION', mutation))
                self.execute(path, work)

    def test_existing_populated_diagnostic_formatter(self):
        with tempfile.TemporaryDirectory(prefix='nano-diagnostic-format-') as tmp:
            self.execute(ROOT / 'tests/test_error_messages.nano', Path(tmp))


if __name__ == '__main__':
    unittest.main()
