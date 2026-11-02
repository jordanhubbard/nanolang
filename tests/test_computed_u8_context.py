"""I narrow checked computed integers once and retain exact byte tags."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get('NANO_U8_TOOLS', ROOT / 'bin'))
DRIVERS = [Path(p) for p in os.environ.get('NANO_U8_COMPILERS', os.pathsep.join(
    str(ROOT / 'bin' / p) for p in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'))).split(os.pathsep) if p]

SOURCE = '''let mut calls: int = 0
fn tick()->int {set calls (+ calls 1) return 257}
shadow tick {assert (== (tick) 257)}
fn narrow(value:int)->u8{return value}
shadow narrow {assert (== (cast_int (narrow 257)) 1)}
fn direct()->u8{return (tick)}
shadow direct {assert (== (cast_int (direct)) 1)}
fn indirect(f:fn()->int)->u8{return (f)}
shadow indirect {assert (== (cast_int (indirect tick)) 1)}
fn identity(value:T)->T{return value}
shadow identity {assert (== (identity 257) 257)}
fn specialized()->u8{return (identity 257)}
shadow specialized {assert (== (cast_int (specialized)) 1)}
fn take(value:u8)->int{return (cast_int value)}
shadow take {assert (== (take 255) 255)}
fn negative()->u8{return (- 0 1)}
shadow negative {assert (== (cast_int (negative)) 255)}
fn addition()->u8{return (+ 127 1)}
shadow addition {assert (== (cast_int (addition)) 128)}
fn main()->int {
 set calls 0
 let first:u8=(tick)
 assert (== calls 1)
 assert (== (cast_int first) 1)
 let mut changed:u8=0
 set changed (tick)
 assert (== calls 2)
 assert (== (cast_int changed) 1)
 assert (== (take (tick)) 1)
 assert (== calls 3)
 assert (== (cast_int (direct)) 1)
 assert (== calls 4)
 assert (== (cast_int (indirect tick)) 1)
 assert (== calls 5)
 let mut bytes:array<u8> = [(- 0 1), (+ 127 1)]
 assert (== (cast_int (at bytes 0)) 255)
 assert (== (cast_int (at bytes 1)) 128)
 set bytes (array_push bytes (tick))
 assert (== calls 6)
 assert (== (cast_int (at bytes 2)) 1)
 (array_set bytes 0 (tick))
 assert (== calls 7)
 assert (== (cast_int (at bytes 0)) 1)
 assert (== (cast_int (specialized)) 1)
 assert (== (cast_int (negative)) 255)
 assert (== (cast_int (addition)) 128)
 assert (== (cast_int (narrow 256)) 0)
 assert (== (cast_int (narrow (- 0 256))) 0)
 assert (== (cast_int (narrow 9223372036854775807)) 255)
 assert (== (cast_int (narrow (- (- 0 9223372036854775807) 1))) 0)
 return 0
}
shadow main {assert (== (main) 0)}
'''


class ComputedU8Context(unittest.TestCase):
    def checked(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_contextual_conversion_execution_and_exact_tags(self):
        with tempfile.TemporaryDirectory(prefix='nano-computed-u8-') as tmp:
            work = Path(tmp); source = work / 'conversion.nano'
            source.write_text(SOURCE)
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    module = work / 'program.nvm'
                    self.checked([driver, source, '--emit-nvm', '-o', module])
                    assembly = self.checked([TOOLS / 'nanoisa', 'dump', module])
                    self.assertIn('CAST_U8', assembly)
                    for function in ('narrow', 'direct', 'indirect', 'specialized', 'negative', 'addition'):
                        self.assertRegex(assembly, rf'(?m)^\.function {function} \d+ \d+ \d+ u8 1$')
                    self.assertRegex(assembly, r'(?m)^\.parameters \d+ u8$')
                    self.assertNotIn('TAIL_CALL tick', assembly)
                    self.checked([TOOLS / 'nano_vm', '--verify-only', module])
                    self.checked([TOOLS / 'nano_vm', module])
                    native_source = work / 'program.c'; native = work / 'program'
                    self.checked([TOOLS / 'nvm2c', module, '-o', native_source])
                    self.checked(shlex.split(os.environ.get('CC', 'cc')) + [
                        '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                        native_source, '-lm', '-o', native])
                    self.checked([native])

    def test_literal_and_definite_wrong_type_refusals_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-computed-u8-refusal-') as tmp:
            work = Path(tmp); source = work / 'wrong.nano'; output = work / 'prior.nvm'
            for expression in ('256', '-1', 'true', '1.5', '"wrong"'):
                source.write_text(f'fn octet()->u8{{return {expression}}}\n'
                                  'shadow octet {assert true}\n'
                                  'fn main()->int{return (cast_int (octet))}\n'
                                  'shadow main {assert true}\n')
                for driver in DRIVERS:
                    with self.subTest(driver=driver, expression=expression):
                        output.write_bytes(b'previous')
                        result = subprocess.run([str(driver), str(source), '--emit-nvm', '-o', str(output)],
                                                cwd=ROOT, capture_output=True, text=True, timeout=180)
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__':
    unittest.main()
