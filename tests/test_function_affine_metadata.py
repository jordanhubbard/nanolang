"""I retain affine checker completion without granting runtime authority."""
import ast
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(p) for p in os.environ.get('NANOLANG_AFFINE_COMPILERS',
    str(ROOT / 'bin/nano_virt')).split(os.pathsep) if p]


class FunctionAffineMetadata(unittest.TestCase):
    def run_checked(self, argv, expected=0):
        p = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True,
                           text=True, timeout=180)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p.stdout

    def exercise(self, source, expected, inspect, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-affine-') as directory:
            work = Path(directory)
            src = work / 'case.nano'
            src.write_text(source)
            for name, contents in (files or {}).items():
                (work / name).write_text(contents)
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    module = work / 'case.nvm'
                    self.run_checked([driver, src, '--emit-nvm', '-o', module])
                    canonical = self.run_checked([ROOT/'bin/nanoisa', 'dump', module])
                    lines = canonical.splitlines()
                    strings = [ast.literal_eval(line[len('.string '):]).encode('latin1')
                               for line in lines if line.startswith('.string ')]
                    functions = [shlex.split(line)[1] for line in lines if line.startswith('.function ')]
                    keys = {i for i, value in enumerate(strings) if value == b'nano.affine.v1'}
                    entries = [line for line in lines if line.startswith('.metadata ')
                               and int(line.split()[1]) in keys]
                    self.assertTrue(entries)
                    facts = {}
                    for entry in entries:
                        value = strings[int(entry.split()[2])]
                        self.assertEqual(len(value), 8)
                        index, declared = struct.unpack('<IB3x', value)
                        self.assertEqual(value[5:], b'\0\0\0')
                        self.assertIn(declared, (0, 1))
                        self.assertLess(index, len(functions))
                        self.assertNotIn(functions[index], facts)
                        facts[functions[index]] = declared
                    self.assertEqual(set(facts), set(functions))
                    inspect(facts)
                    assembly = work/'cycle.nasm'; rebuilt = work/'cycle.nvm'
                    assembly.write_text(canonical)
                    self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
                    self.assertEqual(module.read_bytes(), rebuilt.read_bytes())
                    stdout = self.run_checked([ROOT/'bin/nano_vm', module], expected)
                    native = work/'case.c'; binary = work/'case'
                    self.run_checked([ROOT/'bin/nvm2c', module, '-o', native])
                    original_c = native.read_bytes()
                    self.run_checked(shlex.split(os.environ.get('CC', 'cc')) + [
                        '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', str(native), '-lm', '-o', str(binary)])
                    self.assertEqual(self.run_checked([binary], expected), stdout)
                    # Stripping, ambiguity, unknown versions and a forged declaration
                    # cannot authorize execution or alter emitted native code.
                    stripped = [line for line in lines if line not in entries]
                    forged = lines.copy()
                    forged_entry = next((entry for entry in entries if strings[int(entry.split()[2])][4] == 0), entries[0])
                    value_index = int(forged_entry.split()[2])
                    value = bytearray(strings[value_index]); value[4] ^= 1
                    # I preserve string indices when my forged value deduplicates
                    # against another advisory record for the same function.
                    value = bytes(value)
                    if value in strings:
                        replacement = strings.index(value)
                    else:
                        replacement = len(strings)
                        first_metadata = next(i for i, line in enumerate(forged)
                                              if line.startswith('.metadata '))
                        forged.insert(first_metadata, '.string "' + ''.join(
                            f'\\x{b:02x}' for b in value) + '"')
                    forged[forged.index(forged_entry)] = (
                        f'.metadata {forged_entry.split()[1]} {replacement}')
                    alternatives = [stripped, lines + [entries[0]], forged,
                        [line.replace('nano.affine.v1', 'nano.affine.v2') if line == '.string "nano.affine.v1"' else line for line in lines]]
                    for alternate in alternatives:
                        assembly.write_text('\n'.join(alternate) + '\n')
                        self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
                        self.assertEqual(self.run_checked([ROOT/'bin/nano_vm', rebuilt], expected), stdout)
                        self.run_checked([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.assertEqual(original_c, native.read_bytes())

    def test_called_resource_routes(self):
        for name in ('source_borrow_shared.nano', 'source_borrow_exclusive.nano', 'source_borrow_order.nano', 'source_consuming_leaf.nano'):
            with self.subTest(fixture=name):
                self.exercise((ROOT/'tests/nanoisa/fixtures'/name).read_text(), 0,
                              lambda facts: self.assertTrue(all(value == 1 for value in facts.values()), facts))

    def test_no_resource_pass_is_unknown(self):
        self.exercise('fn id(x:T)->T{return x} fn main()->int{assert (id true) assert (== (id 7) 7) return 0}',
                      0, lambda facts: self.assertTrue(all(value == 0 for value in facts.values()), facts))

    def test_checked_generic_instances_stay_unknown(self):
        def inspect(facts):
            self.assertEqual(facts['main'], 1)
            self.assertEqual(len(facts), 3)
            self.assertTrue(all(value == 0 for name, value in facts.items() if name != 'main'), facts)
        self.exercise('resource struct Box {value:int} fn id(x:T)->T{return x} '
                      'fn main()->int{assert (id true) assert (== (id 7) 7) return 0}', 0, inspect)

    def test_raw_parser_checker_and_external_reset(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-checker-') as directory:
            work = Path(directory); source = work/'checker.nano'; binary = work/'checker'
            source.write_text((ROOT/'tests/nanoisa/fixtures/affine_checker_facts.nano.txt').read_text())
            compiler = Path(os.environ.get('NANOLANG_AFFINE_NATIVE_COMPILER', ROOT/'bin/nanoc_c'))
            result = subprocess.run([str(compiler), str(source), '-o', str(binary)], cwd=ROOT,
                text=True, capture_output=True, timeout=600,
                env={**os.environ, 'NANO_SHADOW_TIMEOUT_SECONDS': '300'})
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.run_checked([binary])

    def test_resource_refusals_preserve_output(self):
        prelude = 'resource struct Box { value:int } fn consume(x:Box)->int{let Box{value}=x return value} fn mix(a:&Box,b:&mut Box)->int{return a.value} '
        bodies = [
            'let x:Box=Box{value:7} return x.value',
            'let mut x:Box=Box{value:7} let n:int=(mix &x &mut x) return (consume x)',
            'let x:Box=Box{value:7} let a:int=(consume x) return (consume x)',
            'let x:Box=Box{value:7} if true {let a:int=(consume x)} return (consume x)',
        ]
        with tempfile.TemporaryDirectory(prefix='nano-affine-refusal-') as directory:
            src=Path(directory)/'bad.nano'; output=Path(directory)/'bad.nvm'
            for body in bodies:
                src.write_text(prelude+'fn main()->int{'+body+'}')
                for driver in DRIVERS:
                    with self.subTest(body=body, driver=driver):
                        output.write_bytes(b'prior artifact')
                        result=subprocess.run([str(driver), str(src), '--emit-nvm', '-o', str(output)],
                            cwd=ROOT, text=True, capture_output=True, timeout=180)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior artifact')


if __name__ == '__main__':
    unittest.main()
