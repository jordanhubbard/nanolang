"""I retain synchronous admission controls; host callbacks are modeled explicitly."""
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from tests import test_portable_read_adapters as retained

ROOT = Path(__file__).resolve().parents[1]

class OrdinaryAdmission(unittest.TestCase):
    dump = retained.PortableReadAdapters.dump
    retain = retained.PortableReadAdapters.retain
    products = retained.PortableReadAdapters.products
    command = retained.PortableReadAdapters.command

    def test_real_vm_admission_lifetime(self):
        supplied = os.environ.get('ORDINARY_ADMISSION_ARTIFACTS')
        self.artifacts = Path(supplied).resolve() if supplied else Path(tempfile.mkdtemp(prefix='nano-admission-'))
        if supplied:
            self.artifacts.mkdir(parents=True, exist_ok=False)
        self.store = self.artifacts / 'objects'
        self.work = self.artifacts / 'products'
        self.store.mkdir()
        self.work.mkdir()
        self.index = 0
        self.env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='', NANO_VM_TRACE='0')
        cc = shlex.split(os.environ.get('ORDINARY_ADMISSION_CC', 'cc'))
        self.assertTrue(cc)
        found = shutil.which(cc[0])
        self.assertIsNotNone(found)
        cc[0] = str(Path(found).resolve())
        flags = shlex.split(os.environ.get('ORDINARY_ADMISSION_CFLAGS', ''))
        objects = shlex.split(os.environ['ORDINARY_ADMISSION_OBJECTS'])
        self.assertFalse(any(Path(p).name in ('vm.o', 'service_bindings_module.o') for p in objects))
        libraries = shlex.split(os.environ['ORDINARY_ADMISSION_LDFLAGS'])
        inputs = [Path(cc[0]), *(ROOT / p for p in objects),
                  ROOT / 'src/nanovm/vm.c', ROOT / 'src/nanoisa/service_bindings_module.c',
                  ROOT / 'src/nanoisa/service_classification_private.h',
                  ROOT / 'tests/nanovm/test_ordinary_admission.c',
                  ROOT / 'tests/nanoisa/owned_fixture.h', Path(__file__), Path(retained.__file__)]
        inventory = lambda: {str(p.resolve()): self.retain(p) for p in inputs}
        before = inventory()
        self.dump('inputs-before.json', before)
        try:
            original = (ROOT / 'src/nanoisa/service_bindings_module.c').read_text()
            entry = 'bool nvm_service_execution_pending(const NvmModule *m) {'
            self.assertEqual(original.count(entry), 1)
            instrumented = self.work / 'service_bindings_module_observed.c'
            instrumented.write_text(original.replace(entry, entry +
                '\n    extern void ordinary_observe_service(const NvmModule *);\n'
                '    ordinary_observe_service(m);', 1))
            self.dump('predicate-observer.json', {
                'original': self.retain(ROOT / 'src/nanoisa/service_bindings_module.c'),
                'instrumented': self.retain(instrumented),
                'scope': 'One observation before unchanged real predicate body; all provider callers.'})
            for threaded in (False, True):
                name = 'threaded' if threaded else 'switch'
                common = [*cc, *flags, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                          '-D_GNU_SOURCE', '-UNDEBUG', '-Isrc', '-Isrc/nanoisa',
                          '-iquote', str(ROOT / 'src/nanoisa'),
                          '-DNANO_COMPUTED_GOTO' if threaded else '-DNANO_NO_COMPUTED_GOTO']
                source = 'tests/nanovm/test_ordinary_admission.c'
                macros = self.command([*common, '-dM', '-E', source])
                self.assertIn('#define NANO_COMPUTED_GOTO 1' if threaded else '#define NANO_NO_COMPUTED_GOTO 1', macros)
                if not threaded:
                    self.assertNotIn('#define NANO_COMPUTED_GOTO ', macros)
                exe = self.work / name
                self.command([*common, source, instrumented, *objects, *libraries, '-o', exe])
                self.assertIn('ordinary admission checks passed', self.command([exe]))
        finally:
            after = inventory()
            self.dump('inputs-after.json', after)
            self.assertEqual(before, after)

if __name__ == '__main__':
    unittest.main()
