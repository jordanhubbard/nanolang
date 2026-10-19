"""I retain inferred conservative effects without granting it runtime authority."""
import ast
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(p) for p in os.environ.get('NANOLANG_EFFECTS_COMPILERS',
    str(ROOT / 'bin/nano_virt')).split(os.pathsep) if p]


class FunctionEffectsMetadata(unittest.TestCase):
    def run_checked(self, argv, expected=0):
        p = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True,
                           text=True, timeout=180)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p.stdout

    def exercise(self, source, expected, inspect, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-effects-') as directory:
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
                    keys = {i for i, value in enumerate(strings) if value == b'nano.effects.v1'}
                    entries = [line for line in lines if line.startswith('.metadata ')
                               and int(line.split()[1]) in keys]
                    self.assertTrue(entries)
                    facts = {}
                    for entry in entries:
                        value = strings[int(entry.split()[2])]
                        self.assertEqual(len(value), 8)
                        index, declared = struct.unpack('<IB3x', value)
                        self.assertEqual(value[5:], b'\0\0\0')
                        self.assertIn(declared, range(64))
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
                        '-std=c11', '-Wall', '-Wextra', '-Werror', str(native), '-lm', '-o', str(binary)])
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
                        [line.replace('nano.effects.v1', 'nano.effects.v2') if line == '.string "nano.effects.v1"' else line for line in lines]]
                    for alternate in alternatives:
                        assembly.write_text('\n'.join(alternate) + '\n')
                        self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
                        self.assertEqual(self.run_checked([ROOT/'bin/nano_vm', rebuilt], expected), stdout)
                        self.run_checked([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.assertEqual(original_c, native.read_bytes())

    def test_closed_calls_io_mutation_and_unknown(self):
        def inspect(facts):
            self.assertEqual(facts['plain'], 0)
            self.assertEqual(facts['chain'], 0)
            self.assertEqual(facts['noisy'], 1)
            self.assertTrue(facts['changed'] & 4)
            self.assertTrue(facts['loop'] & 8)
            self.assertEqual(facts['recur'], 0)
            self.assertTrue(facts['main'] & 1)
            self.assertTrue(facts['main'] & 4)
            self.assertTrue(facts['main'] & 8)
        self.exercise("""fn plain(x:int)->int{return (+ x 1)}
shadow plain {assert (== (plain 1) 2)}
fn chain(x:int)->int{return (plain x)}
shadow chain {assert (== (chain 1) 2)}
fn noisy()->int{(println "observable") return (chain 1)}
shadow noisy {assert (== (noisy) 2)}
fn changed()->int{let mut x:int=1 set x 2 return x}
shadow changed {assert (== (changed) 2)}
fn loop(n:int)->int{while (< n 0) {return 1} return n}
shadow loop {assert (== (loop 2) 2)}
fn recur(n:int)->int{if (<= n 0) {return 2} return (recur (- n 1))}
shadow recur {assert (== (recur 1) 2)}
fn main()->int{assert (== (changed) 2) assert (== (loop 2) 2) assert (== (recur 2) 2) return (noisy)}
""", 2, inspect)

    def test_module_budget_is_explicit_unknown(self):
        source = ''.join(f'fn f{i}()->int{{return {i}}}\nshadow f{i} {{assert (== (f{i}) {i})}}\n'
                         for i in range(65))
        source += 'fn main()->int{return (f1)}\n'
        def inspect(facts):
            self.assertTrue(all(value == 8 for value in facts.values()), facts)
        self.exercise(source, 1, inspect)

    def test_body_budget_does_not_become_closed_empty(self):
        # I exhaust traversal work without exceeding the backend's local limit.
        source = 'fn large()->int{\n' + 'assert (== 1 1)\n' * 1500 + 'return 2}\n'
        source += 'shadow large {assert (== (large) 2)}\nfn main()->int{assert (== (large) 2) return 0}\n'
        def inspect(facts):
            self.assertTrue(facts['large'] & 8, facts)
        self.exercise(source, 0, inspect)

    def test_callback_is_unknown(self):
        def inspect(facts):
            self.assertEqual(facts['plain'], 0)
            self.assertTrue(facts['apply'] & 8)
        self.exercise("""fn plain(x:int)->int{return (+ x 1)}
shadow plain {assert (== (plain 1) 2)}
fn apply(f:fn(int)->int)->int{return (f 1)}
shadow apply {assert (== (apply plain) 2)}
fn main()->int{return (apply plain)}
""", 2, inspect)


if __name__ == '__main__':
    unittest.main()
