"""I retain each native parameter's declared record identity during emission."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / 'bin/nanoc_c'


class ParameterNominalMetadata(unittest.TestCase):
    def compile(self, source, directory):
        path, binary = directory / 'probe.nano', directory / 'probe'
        path.write_text(source)
        result = subprocess.run([COMPILER, path, '-o', binary], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        return result, binary

    def test_same_parameter_name_keeps_distinct_record_metadata(self):
        records = 'struct Branch { then_body: int }\nstruct Check { condition: int }\n'
        first = '''fn first(node: Branch) -> int { return (+ node.then_body 1) }
shadow first { assert (== (first Branch { then_body: 4 }) 5) }
'''
        second = '''fn second(node: Check) -> int { return (+ node.condition 1) }
shadow second { assert (== (second Check { condition: 2 }) 3) }
'''
        main = '''fn main() -> int {
 assert (== (first Branch { then_body: 4 }) 5)
 assert (== (second Check { condition: 2 }) 3)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        for functions in (first + second, second + first):
            with self.subTest(functions=functions), tempfile.TemporaryDirectory() as tmp:
                result, binary = self.compile(records + functions + main, Path(tmp))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('E004', result.stderr)
                executed = subprocess.run([binary], capture_output=True, timeout=30)
                self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_same_field_name_keeps_declared_string_arithmetic(self):
        source = '''struct Text { value: string }
struct Count { value: int }
fn text(node: Text) -> string { return (+ node.value "!") }
shadow text { assert (== (text Text { value: "yes" }) "yes!") }
fn count(node: Count) -> int { return (+ node.value 1) }
shadow count { assert (== (count Count { value: 4 }) 5) }
fn main() -> int {
 assert (== (text Text { value: "yes" }) "yes!")
 assert (== (count Count { value: 4 }) 5)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory() as tmp:
            result, binary = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('E004', result.stderr)
            executed = subprocess.run([binary], capture_output=True, timeout=30)
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_actual_wrong_record_field_remains_rejected(self):
        source = '''struct Branch { then_body: int }
struct Check { condition: int }
fn wrong(node: Check) -> int { return (+ node.then_body 1) }
shadow wrong { assert true }
fn main() -> int { return 0 }
shadow main { assert true }
'''
        with tempfile.TemporaryDirectory() as tmp:
            result, binary = self.compile(source, Path(tmp))
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('E004', result.stderr)
            self.assertFalse(binary.exists())


if __name__ == '__main__':
    unittest.main()
