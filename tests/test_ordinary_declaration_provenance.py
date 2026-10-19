"""I qualify checked ordinary declarations without guessing nominal identity."""
from pathlib import Path
import os
import re
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OrdinaryDeclarationProvenance(unittest.TestCase):
    def command(self, *args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def check(self, source, modules=None, fields=None):
        for compiler in os.environ.get('NANO_ORDINARY_COMPILERS', 'nano_virt').split(','):
            with self.subTest(compiler=compiler):
                self.check_compiler(compiler, source, modules, fields)

    def check_compiler(self, compiler, source, modules, fields):
        with tempfile.TemporaryDirectory(prefix='nano-declaration-provenance-') as directory:
            work = Path(directory)
            for name, text in (modules or {}).items():
                (work/name).write_text(text)
            program, module = work/'main.nano', work/'main.nvm'
            program.write_text(source)
            self.command(ROOT/'bin'/compiler, program, '--emit-nvm', '-o', module)
            self.command(ROOT/'bin/nano_vm', '--verify-only', module)
            self.command(ROOT/'bin/nano_vm', module)
            dump = self.command(ROOT/'bin/nanoisa', 'dump', module)
            assembly, roundtrip = work/'roundtrip.nasm', work/'roundtrip.nvm'
            assembly.write_text(dump)
            self.command(ROOT/'bin/nanoisa', 'asm', assembly, '-o', roundtrip)
            self.command(ROOT/'bin/nano_vm', '--verify-only', roundtrip)
            self.assertEqual(self.command(ROOT/'bin/nanoisa', 'dump', roundtrip), dump)
            raw = lambda key: bytes.fromhex(''.join(re.findall(
                r'^\.' + key + r' "([0-9a-f]*)"$', dump, re.M)))
            layouts, authority = raw('layouts'), raw('ownership')
            if fields is None:
                self.assertEqual((layouts, authority), (b'', b''))
            else:
                self.assertEqual(struct.unpack_from('<I', layouts)[0], len(fields))
                self.assertEqual(struct.unpack_from('<II', authority), (1, len(fields)))
                self.assertEqual(authority[8:8+len(fields)], b'\x01'*len(fields))
                at, actual = 4, []
                for expected in fields:
                    kind, reserved, count, name = struct.unpack_from('<BBHI', layouts, at)
                    at += 8
                    self.assertEqual((kind, reserved, name), (0, 0, 0xffffffff))
                    row = []
                    for _ in range(count):
                        tag, child, label = struct.unpack_from('<B3xII', layouts, at)
                        at += 12
                        self.assertEqual(label, 0xffffffff)
                        row.append((tag, child))
                    actual.append(row)
                self.assertEqual(actual, fields)
                self.assertEqual(at, len(layouts))
                # Concrete specializations retain exact coarse function facts;
                # no advisory type spelling becomes an invented nominal index.
                cursor=8+((len(fields)+3)&~3)
                functions=struct.unpack_from('<I',authority,cursor)[0];cursor+=4
                self.assertEqual(functions,len(re.findall(r'^\.function ',dump,re.M)))
                for _ in range(functions):
                    locals_,parameters=struct.unpack_from('<HH',authority,cursor);cursor+=4
                    self.assertLessEqual(parameters,locals_)
                    for slot in range(locals_+1):
                        tag,reserved,nominal=struct.unpack_from('<B3sI',authority,cursor);cursor+=8
                        self.assertEqual((reserved,nominal),(b'\0'*3,0xffffffff))
                        self.assertIn(tag,range(0 if slot==0 else 1,9))
                self.assertEqual(cursor,len(authority))
            native, binary = work/'main.c', work/'native'
            self.command(ROOT/'bin/nvm2c', module, '-o', native)
            self.command(*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O1',
                         '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                         '-fno-sanitize-recover=all', native, '-lm', '-o', binary)
            self.command(binary)

    def test_distinct_equal_shape_and_forward_ordinals(self):
        self.check('''struct Root { first: First, second: Second }
struct First { value: int }
struct Second { value: int }
fn main() -> int {
 let root: Root = Root { first: First { value: 7 }, second: Second { value: 11 } }
 assert (== root.first.value 7) assert (== root.second.value 11) return 0
}
shadow main { assert (== (main) 0) }
''', fields=[[(8, 1), (8, 2)], [(1, 0xffffffff)], [(1, 0xffffffff)]])

    def test_repeated_direct_and_transitive_import_retains_authority(self):
        self.check('''module "leaf.nano" as leaf
module "relay.nano" as relay
fn main() -> int { assert (== (leaf.value) 7) assert (== (relay.value) 7) return 0 }
shadow main { assert (== (main) 0) }
''', {'leaf.nano': '''module Leaf
pub struct Entry { value: int }
pub fn value() -> int { let entry: Entry = Entry { value: 7 } return entry.value }
shadow value { assert (== (value) 7) }
''', 'relay.nano': '''module Relay
module "leaf.nano" as leaf
pub fn value() -> int { return (leaf.value) }
shadow value { assert (== (value) 7) }
'''}, fields=[[(1, 0xffffffff)]])

    def test_same_spelling_in_distinct_imports_retains_distinct_authority(self):
        modules = {}
        for name, value in [('Left', 7), ('Right', 11)]:
            modules[name.lower()+'.nano'] = f'''module {name}
pub struct Entry {{ value: int }}
pub fn value() -> int {{ let entry: Entry = Entry {{ value: {value} }} return entry.value }}
shadow value {{ assert (== (value) {value}) }}
'''
        self.check('''module "left.nano" as left
module "right.nano" as right
fn main() -> int { assert (== (left.value) 7) assert (== (right.value) 11) return 0 }
shadow main { assert (== (main) 0) }
''', modules, fields=[[(1, 0xffffffff)], [(1, 0xffffffff)]])

    def test_imported_field_identity_preserves_distinct_same_spelling(self):
        modules = {'left.nano': 'module Left\npub struct Entry { value: int }\n',
                   'right.nano': 'module Right\npub struct Entry { value: string }\n'}
        # I require dependency-first ordinals and distinct int/string records.
        expected = [[(1, 0xffffffff)], [(5, 0xffffffff)], [(8, 0), (8, 1)]]
        self.check('''module "left.nano" as left
module "right.nano" as right
struct Root { left: left.Entry, right: right.Entry }
fn main() -> int {
 let root: Root = Root { left: left.Entry { value: 7 }, right: right.Entry { value: "eleven" } }
 assert (== root.left.value 7) assert (== root.right.value "eleven") return 0
}
shadow main { assert (== (main) 0) }
''', modules, fields=expected)

    def test_nested_array_schema_retains_existing_unknown_boundary(self):
        self.check('''struct Entries { values: array<array<int>> }
fn main() -> int { let entries: Entries = Entries { values: [[7, 11]] }
 assert (== (at (at entries.values 0) 1) 11) return 0 }
shadow main { assert (== (main) 0) }
''')


    def test_concrete_generic_equal_shapes_keep_forward_ordinals(self):
        self.check('''struct Root { first: First, second: Second }
struct First { value: int }
struct Second { value: int }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn forward(value: T) -> T { let copy: T = (identity value) return (identity copy) }
shadow forward { assert (== (forward true) true) }
fn main() -> int {
 let first: First = (forward First { value: 7 })
 let second: Second = (forward Second { value: 11 })
 let root: Root = (identity Root { first: first, second: second })
 assert (== root.first.value 7) assert (== root.second.value 11) return 0
}
shadow main { assert (== (main) 0) }
''', fields=[[(8, 1), (8, 2)], [(1, 0xffffffff)], [(1, 0xffffffff)]])

    def test_imported_generic_instances_preserve_source_owners(self):
        modules={}
        for name,value in [('Left',7),('Right',11)]:
            chosen='first' if name=='Left' else 'second'
            arguments='7 97' if name=='Left' else '97 11'
            modules[name.lower()+'.nano']=f'''module {name}
pub struct Entry {{ value: int }}
pub fn select(first: T, second: T) -> T {{ return {chosen} }}
shadow select {{ assert (== (select {arguments}) {value}) }}
'''
        self.check('''module "left.nano" as left
module "right.nano" as right
struct Root { first: left.Entry, second: right.Entry }
fn main() -> int {
 let first: left.Entry = (left.select left.Entry { value: 7 } left.Entry { value: 97 })
 let second: right.Entry = (right.select right.Entry { value: 97 } right.Entry { value: 11 })
 let root: Root = Root { first: first, second: second }
 assert (== root.first.value 7) assert (== root.second.value 11) return 0
}
shadow main { assert (== (main) 0) }
''', modules, fields=[[(1, 0xffffffff)], [(1, 0xffffffff)], [(8, 0), (8, 1)]])

    def test_generic_repeated_variable_refuses_equal_shape_distinct_nominals(self):
        for compiler in os.environ.get('NANO_ORDINARY_COMPILERS','nano_virt').split(','):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-generic-authority-refusal-') as directory:
                work=Path(directory);source=work/'main.nano';output=work/'main.nvm'
                source.write_text('''struct First { value: int }
struct Second { value: int }
fn choose(first: T, second: T) -> T { return first }
shadow choose { assert (== (choose 1 2) 1) }
fn main() -> int {
 let first: First = (choose First { value: 7 } Second { value: 11 })
 return first.value
}
shadow main { assert true }
''')
                output.write_bytes(b'prior-output')
                result=subprocess.run([str(ROOT/'bin'/compiler),str(source),'--emit-nvm','-o',str(output)],
                                      cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('one concrete identity',result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior-output')


    def test_generic_repeated_variable_refuses_distinct_import_owners(self):
        for compiler in os.environ.get('NANO_ORDINARY_COMPILERS','nano_virt').split(','):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-generic-import-refusal-') as directory:
                work=Path(directory);source=work/'main.nano';output=work/'main.nvm'
                for name in ('Left','Right'):
                    (work/(name.lower()+'.nano')).write_text(f'module {name}\npub struct Entry {{ value: int }}\n')
                source.write_text('''module "left.nano" as left
module "right.nano" as right
fn choose(first: T, second: T) -> T { return first }
shadow choose { assert (== (choose 1 2) 1) }
fn main() -> int {
 let first: left.Entry = (choose left.Entry { value: 7 } right.Entry { value: 11 })
 return first.value
}
shadow main { assert true }
''')
                output.write_bytes(b'prior-output')
                result=subprocess.run([str(ROOT/'bin'/compiler),str(source),'--emit-nvm','-o',str(output)],
                                      cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('one concrete identity',result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(),b'prior-output')


if __name__ == '__main__':
    unittest.main()
