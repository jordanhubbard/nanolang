"""I retain declared purity without granting it runtime authority."""
import ast
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(p) for p in os.environ.get('NANOLANG_PURITY_COMPILERS',
    str(ROOT / 'bin/nano_virt')).split(os.pathsep) if p]


class FunctionPurityMetadata(unittest.TestCase):
    def run_checked(self, argv, expected=0):
        p = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True,
                           text=True, timeout=180)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p.stdout

    def exercise(self, source, expected, inspect, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-purity-') as directory:
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
                    keys = {i for i, value in enumerate(strings) if value == b'nano.purity.v1'}
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
                        '-std=c11', '-Wall', '-Wextra', '-Werror', str(native), '-lm', '-o', str(binary)])
                    self.assertEqual(self.run_checked([binary], expected), stdout)
                    # Stripping, ambiguity, unknown versions and a forged declaration
                    # cannot authorize execution or alter emitted native code.
                    stripped = [line for line in lines if line not in entries]
                    forged = lines.copy()
                    forged_entry = next((entry for entry in entries if strings[int(entry.split()[2])][4] == 0), entries[0])
                    value_index = int(forged_entry.split()[2])
                    string_line = [i for i, line in enumerate(lines) if line.startswith('.string ')][value_index]
                    value = bytearray(strings[value_index]); value[4] ^= 1
                    forged[string_line] = '.string "' + ''.join(f'\\x{b:02x}' for b in value) + '"'
                    alternatives = [stripped, lines + [entries[0]], forged,
                        [line.replace('nano.purity.v1', 'nano.purity.v2') if line == '.string "nano.purity.v1"' else line for line in lines]]
                    for alternate in alternatives:
                        assembly.write_text('\n'.join(alternate) + '\n')
                        self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', rebuilt])
                        self.assertEqual(self.run_checked([ROOT/'bin/nano_vm', rebuilt], expected), stdout)
                        self.run_checked([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.assertEqual(original_c, native.read_bytes())

    def test_declared_and_undeclared_purity(self):
        def inspect(facts):
            self.assertEqual(facts['twice'], 1)
            self.assertEqual(facts['plain'], 0)
            self.assertEqual(facts['main'], 0)
        self.exercise('''pure fn twice(x:int)->int{return (+ x x)}
shadow twice {assert (== (twice 3) 6)}
fn plain()->int{(println "observable") return 3}
shadow plain {assert (== (plain) 3)}
fn main()->int{return (twice (plain))}
''', 6, inspect)

    def test_generic_specialization(self):
        def inspect(facts):
            copies = [v for k, v in facts.items() if k != 'main']
            self.assertGreaterEqual(len(copies), 2)
            self.assertTrue(all(v == 0 for v in copies))
        self.exercise('''fn copy(value:T)->T{return value}
shadow copy {assert (== (copy 7) 7)}
fn main()->int {assert (copy true) return (copy 7)}
''', 7, inspect)

    def test_generic_pure_refusal_preserves_prior_artifact(self):
        # An unconstrained T may be a resource or mutable alias. The existing
        # purity checker refuses it; advisory metadata must not bypass that.
        with tempfile.TemporaryDirectory(prefix='nano-purity-refusal-') as directory:
            work = Path(directory); source = work/'case.nano'; output = work/'case.nvm'
            source.write_text('pure fn copy(value:T)->T{return value}\nfn main()->int{return (copy 7)}\n')
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    output.write_bytes(b'prior artifact')
                    self.run_checked([driver, source, '--emit-nvm', '-o', output], 1)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_imported_declarations(self):
        def inspect(facts):
            self.assertTrue(any('twice' in k and v == 1 for k, v in facts.items()))
            self.assertEqual(facts['main'], 0)
        self.exercise('''module "helper.nano" as helper
fn main()->int{return (helper.twice 4)}
''', 8, inspect, {'helper.nano': '''pub pure fn twice(x:int)->int{return (+ x x)}
shadow twice {assert (== (twice 4) 8)}
'''})

    def test_imported_same_name_declarations(self):
        def inspect(facts):
            self.assertEqual(sorted(v for k, v in facts.items() if k != 'main'), [0, 1])
        self.exercise('''module "left.nano" as left
module "right.nano" as right
fn main()->int{return (+ (left.value 4) (right.value 9))}
''', 17, inspect, {
            'left.nano': 'pub pure fn value(x:int)->int{return (+ x x)}\nshadow value {assert (== (value 4) 8)}\n',
            'right.nano': 'pub fn value(x:int)->int{(println "right") return x}\nshadow value {assert (== (value 9) 9)}\n'})

    def test_transitive_imported_same_name_declarations(self):
        def inspect(facts):
            self.assertEqual(sorted(v for k, v in facts.items() if k != 'main'), [0, 0, 1])
        self.exercise('''module "left.nano" as left
module "bridge.nano" as bridge
fn main()->int{return (+ (left.value 4) (bridge.call 9))}
''', 17, inspect, {
            'left.nano': 'pub pure fn value(x:int)->int{return (+ x x)}\nshadow value {assert (== (value 4) 8)}\n',
            'right.nano': 'pub fn value(x:int)->int{(println "right") return x}\nshadow value {assert (== (value 9) 9)}\n',
            'bridge.nano': 'module "right.nano" as right\npub fn call(x:int)->int{return (right.value x)}\nshadow call {assert (== (call 9) 9)}\n'})

    def test_anonymous_capture(self):
        def inspect(facts):
            self.assertGreaterEqual(len(facts), 3)
            self.assertTrue(all(v == 0 for v in facts.values()))
        self.exercise('''fn factory(x:int)->fn()->int {return fn()->int{return x}}
shadow factory {let f:fn()->int=(factory 9) assert (== (f) 9)}
fn main()->int{let f:fn()->int=(factory 9) return (f)}
''', 9, inspect)

    def test_nested_and_captured_functions(self):
        def inspect(facts):
            self.assertGreaterEqual(len(facts), 3)
            self.assertTrue(all(value == 0 for value in facts.values()))
        self.exercise('''fn factory(x:int)->fn()->int {
 fn inner()->int{return x}
 return inner
}
shadow factory {let f:fn()->int=(factory 9) assert (== (f) 9)}
fn main()->int{let f:fn()->int=(factory 9) return (f)}
''', 9, inspect)


if __name__ == '__main__':
    unittest.main()
