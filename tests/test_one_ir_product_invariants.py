"""I enforce two release invariants: no legacy transpiler product source, and native NanoISA products without a VM."""
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# I match symbols of the NanoVM interpreter, not the portable runtime or generated code.
VM_SYMBOL = re.compile(r'(^|[\s_])(nano_vm|vm_core_execute|vm_execute|vm_run|vm_init)\b')


class OneIrProductInvariants(unittest.TestCase):
    def test_transpiler_nano_is_absent_from_the_product_compiler(self):
        self.assertEqual(list((ROOT / 'src_nano').rglob('transpiler.nano')), [])
        for source in (ROOT / 'src_nano').rglob('*.nano'):
            if 'generated' in source.parts:
                continue
            self.assertNotIn('transpiler.nano', source.read_text(errors='replace'), str(source))

    def test_nvm2c_native_product_does_not_link_the_vm(self):
        tools = {name: ROOT / 'bin' / name for name in ('nano_virt', 'nvm2c', 'nano_aot_runtime.o')}
        missing = [name for name, path in tools.items() if not path.exists()]
        if missing:
            self.skipTest('I require built tools: ' + ', '.join(missing))
        compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', os.environ.get('CC', 'cc')))
        with tempfile.TemporaryDirectory(prefix='nano-one-ir-product-') as directory:
            work = Path(directory)
            source = work / 'hello.nano'
            source.write_text('fn main() -> int { (println "I run without a VM.") return 0 }\n'
                              'shadow main { assert true }\n')
            module, c_source, binary = work / 'hello.nvm', work / 'hello.c', work / 'hello'
            commands = [
                [tools['nano_virt'], source, '--emit-nvm', '-o', module],
                [tools['nvm2c'], module, '-o', c_source],
                [*compiler, '-std=c11', '-O1', c_source, tools['nano_aot_runtime.o'], '-lm',
                 *(['-ldl', '-Wl,--export-dynamic'] if os.uname().sysname == 'Linux' else []),
                 *shlex.split(os.environ.get('LDFLAGS', '')), '-o', binary],
            ]
            for command in commands:
                result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                        text=True, timeout=300)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn('I run without a VM.', ran.stdout)
            symbols = subprocess.run(['nm', str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(symbols.returncode, 0, symbols.stderr)
            leaked = [line for line in symbols.stdout.splitlines() if VM_SYMBOL.search(line)]
            self.assertEqual(leaked, [], 'I found NanoVM symbols in a native product.')
            self.assertNotIn('nanovm', c_source.read_text(errors='replace').lower().replace('nanovm_', ''),
                             'I found a NanoVM include in generated C.')


if __name__ == '__main__':
    unittest.main()
