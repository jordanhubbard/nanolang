"""I retain canonical guard behavior through NanoISA and native translation."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_canonical_match_guards as fixtures

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('NANOLANG_MATCH_BIN', ROOT / 'bin'))


class NanoISAMatchGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nanoisa-match-guards-'))
        cls.serial = 0
        cls.driver = Path(os.environ.get('NANOLANG_MATCH_DRIVER', cls.work / 'driver'))
        if 'NANOLANG_MATCH_DRIVER' not in os.environ:
            source = cls.work / 'driver.nano'
            source.write_text((ROOT / 'tests/nanoisa/fixtures/program_closure_driver.nano.txt').read_text())
            result = subprocess.run([str(BIN / 'nanoc_c'), str(source), '-o', str(cls.driver)],
                                    cwd=ROOT, text=True, capture_output=True, timeout=600)
            (cls.work / 'driver-build.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f'I could not build my match driver; see {cls.work}')
        print('I retain NanoISA match evidence at', cls.work, flush=True)

    def command(self, *args, expected=0):
        type(self).serial += 1
        result = subprocess.run(list(map(str, args)), cwd=ROOT, text=True,
                                capture_output=True, timeout=600)
        (self.work / f'command-{self.serial}.log').write_text(
            repr(list(map(str, args))) + '\n' + result.stdout + result.stderr)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def qualify(self, name, text):
        source = self.work / (name + '.nano')
        source.write_text(text)
        for producer in ('seed', 'selfhost'):
            with self.subTest(producer=producer):
                module = self.work / (name + '-' + producer + '.nvm')
                if producer == 'seed':
                    self.command(BIN / 'nano_virt', source, '--emit-nvm', '-o', module)
                else:
                    assembly = module.with_suffix('.nasm')
                    assembly.write_text(self.command(self.driver, source, 'program').stdout)
                    self.command(BIN / 'nanoisa', 'asm', assembly, '-o', module)
                self.command(BIN / 'nano_vm', '--verify-only', module)
                vm = self.command(BIN / 'nano_vm', module)
                native_c, binary = module.with_suffix('.c'), module.with_suffix('.native')
                self.command(BIN / 'nvm2c', module, '-o', native_c)
                self.command(*shlex.split(os.environ.get('CC', 'clang')), '-std=c11',
                             '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                             '-fno-sanitize-recover=all', native_c, '-lm', '-o', binary)
                self.assertEqual(self.command(binary).stdout, vm.stdout)

    def test_named_union_guard_effects(self):
        self.qualify('named-union', fixtures.TRACE)

    def test_integer_wildcard_effects_and_exits(self):
        self.qualify('integer-wildcard', fixtures.WILDCARDS)

    def test_block_values_and_union_wildcards(self):
        self.qualify('block-values', fixtures.TRACE.split('fn selected()', 1)[0] + r'''fn choose(n: int) -> int {
 let local: int = 40
 let result: int = match n {
  1 => { if (> n 0) { return 7 } let local = 8 (+ local 1) }
  _ => { let local = (+ n 2) let next = (+ local 3) next }
 }
 assert (== local 40)
 return result
}
shadow choose { assert (== (choose 1) 7) assert (== (choose 2) 7) }
fn main() -> int {
 set trace 0
 let value: int = match (fresh) {
  _ if (gate 2 false) => 99
  Some(payload) if (gate 3 true) => { let local = payload.number (+ local 1) }
  _ => 0
 }
 assert (== value 8) assert (== trace 123)
 assert (== (choose 1) 7) assert (== (choose 2) 7)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_invalid_guards_domains_and_coverage(self):
        cases = {
            'guard': 'match 1 { 1 if 7 => 2 _ => 0 }',
            'coverage': 'match 1 { 1 if true => 2 }',
            'unreachable': 'match 1 { _ if true => 2 1 => 0 }',
            'domain': 'match "text" { _ => 1 }',
            'union-coverage': 'match Choice.Some { number: 1 } { Some(p) if (> p.number 0) => p.number None(n) => 0 }',
        }
        for name, value in cases.items():
            with self.subTest(case=name):
                source = self.work / (name + '.nano')
                source.write_text(fixtures.PREFIX + 'fn main() -> int { return ' + value + ' } shadow main { assert true }')
                output = source.with_suffix('.nvm')
                output.write_bytes(b'previous-output')
                self.command(BIN / 'nano_virt', source, '--emit-nvm', '-o', output, expected=1)
                self.assertEqual(output.read_bytes(), b'previous-output')
                self.assertEqual(self.command(self.driver, source, 'program', expected=1).stdout, '')
