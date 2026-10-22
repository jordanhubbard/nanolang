"""I retain checked match sites without making advisory claims execution authority."""
import ast
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(p) for p in os.environ.get('NANOLANG_MATCH_COMPILERS', str(ROOT/'bin/nano_virt')).split(os.pathsep) if p]


class MatchCoverageMetadata(unittest.TestCase):
    def run_checked(self, argv, expected=0):
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout

    def test_checked_sites_and_advisory_transport(self):
        source = '''module "other.nano" as other
union Choice { Yes {value:int}, No {} }
fn choose(c:Choice)->int { return match c { Yes(v) if false => 99 Yes(v) => v.value No(n) => 0 } }
fn select(n:int)->int { return match n { 1 if true => 1 _ => 2 } }
fn main()->int { assert (== (choose Choice.Yes {value:7}) 7) assert (== (select 9) 2) assert (== (other.pick 4) 8) return 0 }
'''
        with tempfile.TemporaryDirectory(prefix='nano-match-facts-') as directory:
            work = Path(directory)
            src = work/'case.nano'; src.write_text(source)
            (work/'other.nano').write_text('pub fn pick(n:int)->int { return match n { _ if true => 8 } }\n')
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    module = work/'case.nvm'
                    self.run_checked([driver, src, '--emit-nvm', '-o', module])
                    canonical = self.run_checked([ROOT/'bin/nanoisa', 'dump', module])
                    lines = canonical.splitlines()
                    strings = [ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
                    functions = [shlex.split(line)[1] for line in lines if line.startswith('.function ')]
                    entries = [line for line in lines if line.startswith('.metadata ') and strings[int(line.split()[1])] == b'nano.match.v1']
                    self.assertEqual(len(entries), 3, canonical)
                    facts = []
                    for entry in entries:
                        data = strings[int(entry.split()[2])]
                        function, line, column, coverage, coordinates, reserved, length = struct.unpack('<IIIBBHI', data[:20])
                        self.assertLess(function, len(functions)); self.assertGreater(line, 0); self.assertGreater(column, 0)
                        self.assertEqual(reserved, 0); self.assertEqual(length, len(data)-20)
                        self.assertIn(coordinates, (1, 2))
                        facts.append((functions[function], coverage, data[20:]))
                    self.assertEqual(sorted(f[1] for f in facts), [1, 1, 2], facts)
                    self.assertEqual(sum(f[2].endswith(b'other.nano') for f in facts), 1, facts)
                    assembly = work/'cycle.nasm'; rebuilt = work/'cycle.nvm'
                    assembly.write_text(canonical)
                    self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
                    self.assertEqual(module.read_bytes(), rebuilt.read_bytes())
                    self.run_checked([ROOT/'bin/nano_vm', module])
                    native = work/'case.c'; binary = work/'case'
                    self.run_checked([ROOT/'bin/nvm2c', module, '-o', native]); original_c = native.read_bytes()
                    self.run_checked(shlex.split(os.environ.get('CC', 'cc')) + ['-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', str(native), '-lm', '-o', str(binary)])
                    self.run_checked([binary])
                    # I reuse pooled values safely while forging a well-formed UNKNOWN claim.
                    forged = lines.copy(); entry = entries[0]; value = bytearray(strings[int(entry.split()[2])]); value[12] = 0
                    value = bytes(value)
                    if value in strings: replacement = strings.index(value)
                    else:
                        replacement = len(strings)
                        first_metadata = next(i for i, text in enumerate(forged) if text.startswith('.metadata '))
                        forged.insert(first_metadata, '.string "' + ''.join('\\x%02x' % b for b in value) + '"')
                    forged[forged.index(entry)] = f'.metadata {entry.split()[1]} {replacement}'
                    variants = [[line for line in lines if line not in entries], forged, lines + [entries[0]]]
                    for changed in variants:
                        assembly.write_text('\n'.join(changed)+'\n')
                        self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
                        self.run_checked([ROOT/'bin/nano_vm', rebuilt])
                        self.run_checked([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.assertEqual(original_c, native.read_bytes())

    def test_generic_templates_remain_unknown(self):
        with tempfile.TemporaryDirectory(prefix='nano-match-generic-') as directory:
            work = Path(directory); src = work/'generic.nano'; output = work/'generic.nvm'
            src.write_text('fn tally(value:T)->int {return match 1 {_ => 7}}\n'
                           'fn main()->int {assert (== (tally true) 7) assert (== (tally 9) 7) return 0}\n')
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    self.run_checked([driver, src, '--emit-nvm', '-o', output])
                    lines = self.run_checked([ROOT/'bin/nanoisa', 'dump', output]).splitlines()
                    strings = [ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
                    values = [strings[int(line.split()[2])] for line in lines if line.startswith('.metadata ')
                              and strings[int(line.split()[1])] == b'nano.match.v1']
                    self.assertGreaterEqual(len(values), 2)
                    self.assertTrue(all(value[12] == 0 for value in values), values)
                    self.run_checked([ROOT/'bin/nano_vm', output])
                    native = work/'generic.c'; binary = work/'generic'
                    self.run_checked([ROOT/'bin/nvm2c', output, '-o', native])
                    self.run_checked(shlex.split(os.environ.get('CC', 'cc')) + ['-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', str(native), '-lm', '-o', str(binary)])
                    self.run_checked([binary])

    def test_refusal_preserves_output(self):
        sources = [
            'fn main()->int { return match 1 { 1 => 2 } }',
            'fn main()->int { return match 1 { _ if false => 2 } }',
            'fn main()->int { return match 1 { _ => 2 1 => 3 } }',
            'union Choice { Yes {}, No {} } fn main()->int { return match Choice.Yes {} { Yes(y) => 2 } }',
        ]
        with tempfile.TemporaryDirectory(prefix='nano-match-refuse-') as directory:
            work = Path(directory); src = work/'bad.nano'; output = work/'keep.nvm'
            for driver in DRIVERS:
                for source in sources:
                    with self.subTest(driver=driver, source=source):
                        src.write_text(source); output.write_bytes(b'previous product')
                        p = subprocess.run([str(driver), str(src), '--emit-nvm', '-o', str(output)], cwd=ROOT, text=True, capture_output=True, timeout=180)
                        self.assertNotEqual(p.returncode, 0, p.stdout+p.stderr)
                        self.assertEqual(output.read_bytes(), b'previous product')


if __name__ == '__main__':
    unittest.main()
