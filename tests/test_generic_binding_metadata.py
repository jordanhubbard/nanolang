"""I preserve generic provenance and bindings without executable authority."""
import ast
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(p) for p in os.environ.get('NANOLANG_GENERIC_METADATA_COMPILERS',
    str(ROOT/'bin/nano_virt')).split(os.pathsep) if p]
IDENTITY = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 7) 7)}\n'


class GenericBindingMetadata(unittest.TestCase):
    def run_checked(self, argv, expected=0):
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout

    def exercise(self, source, inspect, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-generic-metadata-') as directory:
            work = Path(directory); path = work/'case.nano'; path.write_text(source)
            for name, contents in (files or {}).items():
                (work/name).write_text(contents)
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    module = work/'case.nvm'; assembly = work/'cycle.nasm'; cycle = work/'cycle.nvm'
                    self.run_checked([driver, path, '--emit-nvm', '-o', module])
                    canonical = self.run_checked([ROOT/'bin/nanoisa', 'dump', module])
                    lines = canonical.splitlines()
                    string_lines = [i for i, line in enumerate(lines) if line.startswith('.string ')]
                    strings = [ast.literal_eval(lines[i][len('.string '):]).encode('latin1') for i in string_lines]
                    functions = [shlex.split(line)[1] for line in lines if line.startswith('.function ')]
                    key = strings.index(b'nano.generic.v1')
                    entries = [line for line in lines if line.startswith('.metadata ') and int(line.split()[1]) == key]
                    self.assertTrue(entries)
                    records = []; seen = set(); origins = {}
                    for entry in entries:
                        data = strings[int(entry.split()[2])]
                        index, *lengths = struct.unpack('<5I', data[:20])
                        self.assertLess(index, len(functions)); self.assertEqual(len(data), 20 + sum(lengths))
                        fields = []; offset = 20
                        for length in lengths:
                            fields.append(data[offset:offset+length].decode('utf8')); offset += length
                        name, source_path, variable, resolved_type = fields
                        self.assertTrue(name and variable and resolved_type)
                        self.assertNotIn((index, variable), seen); seen.add((index, variable))
                        self.assertEqual(origins.setdefault(index, (name, source_path)), (name, source_path))
                        self.assertNotEqual(functions[index], 'main')
                        records.append((index, name, Path(source_path).name, variable, resolved_type))
                    inspect(records)
                    assembly.write_text(canonical)
                    self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', cycle])
                    self.assertEqual(module.read_bytes(), cycle.read_bytes())
                    stdout = self.run_checked([ROOT/'bin/nano_vm', module])
                    native = work/'case.c'; binary = work/'case'
                    self.run_checked([ROOT/'bin/nvm2c', module, '-o', native]); original = native.read_bytes()
                    self.run_checked(shlex.split(os.environ.get('CC', 'cc')) + ['-std=c11', '-Wall', '-Wextra', '-Werror', str(native), '-lm', '-o', str(binary)])
                    self.assertEqual(self.run_checked([binary]), stdout)
                    forged = lines.copy(); value_index = int(entries[0].split()[2]); data = strings[value_index]
                    header = list(struct.unpack('<5I', data[:20])); old_type_length = header[4]
                    header[4] = len(b'not-a-type'); data = struct.pack('<5I', *header) + data[20:-old_type_length] + b'not-a-type'
                    forged[string_lines[value_index]] = '.string "' + ''.join(f'\\x{b:02x}' for b in data) + '"'
                    alternatives = [[line for line in lines if line not in entries], lines+[entries[0]],
                        lines+[f'.metadata {key} {key}'], forged,
                        [line.replace('nano.generic.v1', 'nano.generic.v2') if line == '.string "nano.generic.v1"' else line for line in lines]]
                    for altered in alternatives:
                        assembly.write_text('\n'.join(altered)+'\n')
                        self.run_checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', cycle])
                        self.assertEqual(self.run_checked([ROOT/'bin/nano_vm', cycle]), stdout)
                        self.run_checked([ROOT/'bin/nvm2c', cycle, '-o', native])
                        self.assertEqual(native.read_bytes(), original)

    def test_scalars_and_reused_instances(self):
        def inspect(records):
            self.assertEqual(sorted(r[4] for r in records), ['bool', 'float', 'int', 'string', 'u8'])
            self.assertTrue(all(r[1:4] == ('identity', 'case.nano', 'T') for r in records))
        self.exercise(IDENTITY+'''fn main()->int {
 assert (== (identity 7) 7) assert (== (identity 9) 9)
 assert (identity true) assert (== (identity 1.5) 1.5)
 assert (== (identity "kept") "kept")
 let octet:u8=255 assert (== (identity octet) 255)
 return 0
}''', inspect)

    def test_arrays_and_tuples(self):
        def inspect(records):
            self.assertEqual(sorted(r[4] for r in records), ['(int,bool)', 'array<array<int>>', 'array<string>'])
        self.exercise(IDENTITY+'''fn main()->int {
 let values:array<array<int>> = (identity [[7]])
 let words:array<string> = (identity ["kept"])
 let pair:(int,bool)=(identity (7,true))
 assert (== (at (at values 0) 0) 7) assert (== (at words 0) "kept")
 assert (== pair.0 7) assert pair.1 return 0
}''', inspect)

    def test_multiple_type_variables(self):
        def inspect(records):
            self.assertEqual({(r[3], r[4]) for r in records}, {('T', 'int'), ('U', 'string')})
            self.assertEqual(len({r[0] for r in records}), 1)
        self.exercise('''fn first(left:T,right:U)->T{return left}
shadow first {assert (== (first 7 "value") 7)}
fn main()->int {assert (== (first 7 "value") 7) return 0}
''', inspect)

    def test_contextual_callable_reuses_direct_instance(self):
        def inspect(records):
            self.assertEqual([(r[1], r[3], r[4]) for r in records], [('identity', 'T', 'int')])
        self.exercise(IDENTITY+'''fn main()->int {
 let callback:fn(int)->int=identity
 assert (== (callback 7) (identity 7)) return 0
}''', inspect)

    def test_nested_body_is_not_a_second_specialization(self):
        def inspect(records):
            self.assertEqual(sorted((r[1], r[3], r[4]) for r in records), [('keep', 'T', 'bool'), ('keep', 'T', 'int')])
        self.exercise('''fn keep(value:T)->T {
 let callback:fn()->int=fn()->int{return 7}
 assert (== (callback) 7) return value
}
shadow keep {assert (== (keep 7) 7)}
fn main()->int{assert (== (keep 7) 7) assert (keep true) return 0}
''', inspect)

    def test_specialized_returned_captures(self):
        def inspect(records):
            self.assertEqual({(r[1], r[3], r[4]) for r in records}, {
                ('capture', 'T', 'int'), ('capture', 'T', 'bool'), ('capture', 'T', 'array<int>')})
        self.exercise('''fn capture(value:T)->fn()->T{return fn()->T{return value}}
shadow capture {let f:fn()->int=(capture 7) assert (== (f) 7)}
fn main()->int {
 let number:fn()->int=(capture 7) let flag:fn()->bool=(capture true)
 let values:fn()->array<int> = (capture [9])
 assert (== (number) 7) assert (flag) assert (== (at (values) 0) 9) return 0
}
''', inspect)

    def test_imported_specialized_returned_captures(self):
        def inspect(records):
            self.assertEqual({(r[1], r[3], r[4]) for r in records}, {
                ('capture', 'T', 'int'), ('capture', 'T', 'bool')})
        self.exercise('''module "callbacks.nano" as callbacks
fn main()->int {
 let number:fn()->int=(callbacks.capture 7)
 let flag:fn()->bool=(callbacks.capture true)
 assert (== (number) 7) assert (flag) return 0
}
''', inspect, {'callbacks.nano': '''pub fn capture(value:T)->fn()->T{return fn()->T{return value}}
shadow capture {let f:fn()->int=(capture 7) assert (== (f) 7)}
'''})

    def test_imported_callable_result_mismatch_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-import-refusal-') as directory:
            work = Path(directory); source = work/'case.nano'; output = work/'case.nvm'
            (work/'callbacks.nano').write_text('pub fn capture(value:T)->fn()->T{return fn()->T{return value}}\n')
            source.write_text('module "callbacks.nano" as callbacks\nfn main()->int{let wrong:fn()->bool=(callbacks.capture 7) return 0}\n')
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    output.write_bytes(b'prior artifact')
                    self.run_checked([driver, source, '--emit-nvm', '-o', output], 1)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_specialized_nested_parameters(self):
        def inspect(records):
            self.assertEqual({r[4] for r in records}, {'int', 'bool'})
        self.exercise('''fn select(value:T)->fn(T)->T{return fn(other:T)->T{return other}}
shadow select {let f:fn(int)->int=(select 7) assert (== (f 9) 9)}
fn main()->int {
 let number:fn(int)->int=(select 7) let flag:fn(bool)->bool=(select false)
 assert (== (number 9) 9) assert (flag true) return 0
}
''', inspect)

    def test_specialized_grandparent_capture(self):
        def inspect(records):
            self.assertEqual({r[4] for r in records}, {'int', 'bool'})
        self.exercise('''fn capture(value:T)->fn()->fn()->T {
 return fn()->fn()->T{return fn()->T{return value}}
}
shadow capture {let outer:fn()->fn()->int=(capture 7) let inner:fn()->int=(outer) assert (== (inner) 7)}
fn main()->int {
 let first:fn()->fn()->int=(capture 7) let second:fn()->fn()->bool=(capture true)
 let number:fn()->int=(first) let flag:fn()->bool=(second)
 assert (== (number) 7) assert (flag) return 0
}
''', inspect)

    def test_specialized_named_recursive_capture(self):
        def inspect(records):
            self.assertEqual({r[4] for r in records}, {'int', 'bool'})
        self.exercise('''fn make(value:T,offset:int)->fn(int)->int {
 fn count(n:int)->int {if (== n 0){return offset} return (+ 1 (count (- n 1)))}
 return count
}
shadow make {let f:fn(int)->int=(make 7 3) assert (== (f 4) 7)}
fn main()->int {
 let first:fn(int)->int=(make 7 3) let second:fn(int)->int=(make true 9)
 assert (== (first 4) 7) assert (== (second 4) 13) return 0
}
''', inspect)

    def test_specialized_callable_result_mismatch_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-refusal-') as directory:
            work=Path(directory); source=work/'case.nano'; output=work/'case.nvm'
            source.write_text('fn capture(value:T)->fn()->T{return fn()->T{return value}}\nfn main()->int{let wrong:fn()->bool=(capture 7) return 0}\n')
            for driver in DRIVERS:
                with self.subTest(driver=driver):
                    output.write_bytes(b'prior artifact')
                    self.run_checked([driver, source, '--emit-nvm', '-o', output], 1)
                    self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_callable_binding(self):
        def inspect(records):
            self.assertEqual([r[4] for r in records], ['fn(int)->int'])
        self.exercise(IDENTITY+'''fn plus(value:int)->int{return (+ value 1)}
shadow plus {assert (== (plus 7) 8)}
fn main()->int{let callback:fn(int)->int=(identity plus) assert (== (callback 7) 8) return 0}
''', inspect)

    def test_nominal_and_generic_union_bindings(self):
        def inspect(records):
            # I preserve each frontend's resolved nominal identity, including
            # the C checker's explicit root-owner names.
            self.assertIn({r[4] for r in records}, [
                {'Item', 'Color', 'Box<int>'},
                {'Item', '__nano_root_enum__Color', '__nano_root_union__Box<int>'}])
        self.exercise(IDENTITY+'''struct Item {value:int}
enum Color {Red,Blue}
union Box<T> {Some {value:T},None {}}
fn main()->int {
 let item:Item=(identity Item {value:7}) assert (== item.value 7)
 let color:Color=(identity Color.Blue) assert (== color Color.Blue)
 let box:Box<int> = (identity Box<int>.Some {value:7})
 match box {Some(p)=>{assert (== p.value 7)} None(p)=>{assert false}}
 return 0
}''', inspect)

    def test_imported_nominal_types_remain_distinct(self):
        def inspect(records):
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r[1:4] == ('identity', 'case.nano', 'T') for r in records))
            self.assertEqual(len({r[4] for r in records}), 2)
        self.exercise(IDENTITY+'''module "left.nano" as left
module "right.nano" as right
fn main()->int {
 let first=(identity (left.make 7)) let second=(identity (right.make 9))
 assert (== first.value 7) assert (== second.value 9) return 0
}
''', inspect, {
            'left.nano': 'struct Item {value:int}\npub fn make(value:int)->Item{return Item {value:value}}\nshadow make {let item=(make 7) assert (== item.value 7)}\n',
            'right.nano': 'struct Item {value:int}\npub fn make(value:int)->Item{return Item {value:value}}\nshadow make {let item=(make 9) assert (== item.value 9)}\n'})

    def test_imported_declaration_origins(self):
        def inspect(records):
            self.assertEqual({r[1:] for r in records}, {('identity', 'left.nano', 'T', 'int'), ('identity', 'right ü.nano', 'T', 'int')})
            self.assertEqual(len({r[0] for r in records}), 2)
        self.exercise('''module "left.nano" as left
module "right ü.nano" as right
fn main()->int {assert (== (left.identity 7) 7) assert (== (right.identity 9) 9) return 0}
''', inspect, {'left.nano': 'pub '+IDENTITY, 'right ü.nano': 'pub '+IDENTITY})


if __name__ == '__main__':
    unittest.main()
