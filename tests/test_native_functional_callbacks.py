"""I execute functional callable arrays through NanoISA and native C AOT."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_native_callables as harness

ROOT = harness.ROOT
TRANSLATOR = Path(os.environ.get('NANO_DEAD_CALLABLE_TRANSLATOR', ROOT/'bin/nvm2c'))


class NativeFunctionalCallbacks(unittest.TestCase):
    def test_retained_source_operations(self):
        for evidence, count in (('callable-array-operations-20261010', 9),
                                ('generic-array-callbacks-20261010', 7)):
            sources = sorted((ROOT/'docs/evidence'/evidence).glob('test_*.nano'))
            self.assertEqual(len(sources), count)
            for source in sources:
                with self.subTest(source=source.name), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp)
                    module, c = work/'input.nvm', work/'input.c'
                    runner = harness.NativeCallables()
                    runner.checked([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module])
                    runner.checked([ROOT/'bin/nano_vm', module])
                    runner.checked([TRANSLATOR, module, '-o', c])
                    runner.sanitized(c, work/'native')

    def test_incompatible_tagged_parameter_preserves_output(self):
        text = '''.entry main
.function leaf 0 0 0 int 1
PUSH_I64 7
RET
.end
.function retain 1 1 0 function 1
LOAD_LOCAL 0
RET
.end
.function main 0 0 0 int 1
FUNCREF leaf
CALL retain
POP
PUSH_I64 7
ARR_LITERAL 1 1
PUSH_I64 0
ARR_GET
CALL retain
POP
PUSH_I64 0
RET
.end
'''
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            assembly, module, output = work/'input.nasm', work/'input.nvm', work/'output.c'
            assembly.write_text(text)
            harness.NativeCallables().checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
            output.write_bytes(b'prior accepted output')
            result = subprocess.run([TRANSLATOR, module, '-o', output], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(output.read_bytes(), b'prior accepted output')
