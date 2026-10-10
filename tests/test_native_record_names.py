"""I keep ordinary record names distinct from my external runtime typedefs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeRecordNames(unittest.TestCase):
    def command(self, args):
        result = subprocess.run(list(map(str,args)), cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result

    def execute(self, source, destination):
        compiler = Path(os.environ.get('NANO_RECORD_COMPILER', ROOT/'bin/nanoc_c'))
        self.command([compiler,source,'-o',destination])
        self.command([destination])

    def test_original_nested_record_array_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.execute(ROOT/'tests/nanoisa/fixtures/record_arrays.nano',Path(tmp)/'original')

    def test_runtime_name_families_and_ordinary_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('NSType','Token','ASTPayload','CompilerPayload','List_Payload','MyPhaseOutput'):
                with self.subTest(name=name):
                    source=root/(name+'.nano')
                    source.write_text(f'''struct {name} {{ value: int }}
struct Neighbor_{name} {{ value: int }}
fn identity(value: {name}) -> {name} {{ return value }}
shadow identity {{ assert (== (identity {name} {{ value: 7 }}).value 7) }}
fn main() -> int {{
 let values: array<{name}> = [(identity {name} {{ value: 7 }})]
 let ordinary: Neighbor_{name} = Neighbor_{name} {{ value: 9 }}
 assert (== (at values 0).value 7)
 assert (== ordinary.value 9)
 let replacement: {name} = {name} {{ value: 11 }}
 (array_set values 0 replacement)
 assert (== (array_get values 0).value 11)
 let removed: {name} = (array_pop values)
 assert (== removed.value 11)
 assert (== (array_length values) 0)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
''')
                    self.execute(source,root/'native')

    def test_generic_record_arguments_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'generic.nano'
            source.write_text('struct NSType { value: int }\n'
                'fn identity(value: T) -> T { return value }\n'
                'shadow identity { assert (== (identity 1) 1) }\n'
                'fn make() -> NSType { return NSType { value: 39 } }\n'
                'shadow make { assert (== (make).value 39) }\n'
                'fn main() -> int { let value: NSType = (identity NSType { value: 37 }) '
                'assert (== value.value 37) '
                'let local: NSType = (identity value) assert (== local.value 37) '
                'let computed: NSType = (identity (make)) assert (== computed.value 39) return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            self.execute(source,root/'native')

    def test_record_array_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'callbacks.nano'
            source.write_text("""struct NSType { value: int }
fn keep(value: NSType) -> bool { return (> value.value 0) }
shadow keep { assert (keep NSType { value: 1 }) }
fn change(value: NSType) -> NSType { return NSType { value: (+ value.value 1) } }
shadow change { assert (== (change NSType { value: 1 }).value 2) }
fn sum(left: NSType, right: NSType) -> NSType { return NSType { value: (+ left.value right.value) } }
shadow sum { assert (== (sum NSType { value: 1 } NSType { value: 2 }).value 3) }
fn exercise(values: array<NSType>) -> int {
 let kept: array<NSType> = (filter values keep)
 let changed: array<NSType> = (map kept change)
 let result: NSType = (reduce changed NSType { value: 1 } sum)
 return result.value
}
shadow exercise { let values: array<NSType> = [] assert (== (exercise values) 1) }
fn main() -> int {
 let values: array<NSType> = [NSType { value: 7 }, NSType { value: -2 }]
 assert (== (exercise values) 9)
 let empty: array<NSType> = []
 assert (== (exercise empty) 1)
 let dynamic: array<NSType> = (array_push empty NSType { value: 7 })
 assert (== (exercise dynamic) 9)
 assert (== (at values 0).value 7)
 assert (== (at dynamic 0).value 7)
 return 0
}
shadow main { assert (== (main) 0) }
""")
            self.execute(source,root/'native')

    def test_imported_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'records.nano').write_text('pub struct NSType { value: int }\n'
                'pub fn value() -> NSType { return NSType { value: 42 } }\n'
                'shadow value { assert (== (value).value 42) }\n')
            source=root/'main.nano'
            source.write_text('from "records.nano" import NSType, value\n'
                'fn main() -> int { let result: NSType = (value) assert (== result.value 42) return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            self.execute(source,root/'native')

    def test_external_schema_type_keeps_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'external.nano'
            source.write_text('from "src_nano/generated/compiler_contracts.nano" import NSType\n'
                'fn main() -> int {\n'
                'let value: NSType = NSType { kind: 1, name: "kept", element_type_kind: 0, element_type_name: "" }\n'
                'assert (== value.kind 1) assert (== value.name "kept") return 0 }\n'
                'shadow main { assert (== (main) 0) }\n')
            self.execute(source,root/'native')


if __name__ == '__main__':
    unittest.main()
