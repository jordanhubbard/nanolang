"""I order callback declarations and aggregate layouts without erasing signatures."""
import unittest
import subprocess
import tempfile
from pathlib import Path
from tests import test_native_global_types as harness

IDENTITY = 'fn identity(value:T)->T{return value}\nshadow identity {assert (== (identity 1) 1)}\n'


class NativeCallbackFields(unittest.TestCase):
    def execute(self, source):
        harness.NativeGlobalTypes().execute(IDENTITY + source)

    def test_forward_record_signature(self):
        self.execute('''struct Holder {callback:fn(Box)->Box}
struct Box {value:int}
fn main()->int {let holder=Holder {callback:identity}
let callback:fn(Box)->Box=holder.callback
let result=(callback Box {value:7}) assert (== result.value 7) return 0}''')

    def test_self_record_signature(self):
        self.execute('''struct Node {value:int, callback:fn(Node)->int}
fn read(node:Node)->int{return node.value}
shadow read {assert (== (read Node {value:7,callback:read}) 7)}
fn main()->int {let node=Node {value:9,callback:read}
let callback:fn(Node)->int=node.callback assert (== (callback node) 9) return 0}''')

    def test_tuple_signature_with_record(self):
        self.execute('''struct Holder {callback:fn((Box,bool))->(Box,bool)}
struct Box {value:int}
fn main()->int {let holder=Holder {callback:identity}
let callback:fn((Box,bool))->(Box,bool)=holder.callback
let result=(callback (Box {value:7},true))
assert (== result.0.value 7) assert result.1 return 0}''')

    def test_nested_callable_signature(self):
        self.execute('''struct Holder {callback:fn(fn(int)->int)->fn(int)->int}
fn main()->int {let holder=Holder {callback:identity}
let callback:fn(fn(int)->int)->fn(int)->int=holder.callback
let number:fn(int)->int=(callback identity)
assert (== (number 7) 7) return 0}''')

    def test_tuple_field_layout_dependency(self):
        self.execute('''struct Holder {pair:(Box,fn(int)->int)}
struct Box {value:int}
fn main()->int {let holder=Holder {pair:(Box {value:7},identity)}
let callback:fn(int)->int=holder.pair.1
assert (== (callback holder.pair.0.value) 7) return 0}''')

    def test_incompatible_fields_preserve_output(self):
        cases = [
            ('callback:fn(fn(int)->int)->fn(bool)->bool', 'callback:identity'),
            ('pair:(int,fn(int)->bool)', 'pair:(7,identity)'),
        ]
        for field, initializer in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                source, binary = Path(tmp) / 'main.nano', Path(tmp) / 'main'
                source.write_text(IDENTITY + 'struct Holder {' + field + '}\n'
                    + 'fn main()->int {let holder=Holder {' + initializer + '} return 0}\n'
                    + 'shadow main {assert true}\n')
                binary.write_bytes(b'prior executable')
                result = subprocess.run(list(map(str, [harness.COMPILER, source, '-o', binary])),
                    cwd=harness.ROOT, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(binary.read_bytes(), b'prior executable')
