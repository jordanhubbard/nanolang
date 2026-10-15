"""I retain named enum types and value shadowing in native C emission."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANO_ENUM_NATIVE_COMPILER', ROOT/'bin/nanoc_c'))


class NativeEnumIdentity(unittest.TestCase):
    def execute(self, source):
        with tempfile.TemporaryDirectory(prefix='nano-native-enum-') as tmp:
            work = Path(tmp)
            path, binary = work/'main.nano', work/'main'
            path.write_text(source)
            for args in ([COMPILER, path, '-o', binary], [binary]):
                result = subprocess.run(list(map(str, args)), cwd=ROOT,
                                        text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_existing_enum_fixture(self):
        self.execute((ROOT/'tests/nl_types_enum.nano').read_text())

    def test_generic_enum_result_and_local_shadow(self):
        self.execute('''enum Shade {VALUE=7}
struct Box {VALUE:int}
fn identity(value:T)->T{return value}
shadow identity {assert (== (identity 1) 1)}
fn main()->int {
    let value:Shade = (identity Shade.VALUE)
    assert (== value 7)
    let Shade:Box = Box {VALUE:8}
    assert (== (identity Shade.VALUE) 8)
    return 0
}
shadow main {assert (== (main) 0)}
''')


if __name__ == '__main__':
    unittest.main()
