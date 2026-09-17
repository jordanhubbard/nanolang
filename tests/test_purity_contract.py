"""I require the same closed-purity decisions from both frontend implementations."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = [ROOT / "bin/nanoc_c", Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2"))]

ACCEPT = {
    "recursive": '''pure fn sum(n: int) -> int { if (<= n 0) { return 0 } return (+ n (sum (- n 1))) }
fn main() -> int { assert (== (sum 4) 10) return 0 }''',
    "transitive_unannotated": '''fn helper(x: int) -> int { return (* x 2) }
pure fn value(x: int) -> int { return (helper x) }
fn main() -> int { assert (== (value 21) 42) return 0 }''',
    "ordinary_record_allocation": '''struct Pair { value: int }
pure fn value(x: int) -> Pair { return Pair { value: (+ x 1) } }
pure fn read(p: Pair) -> int { return p.value }
fn main() -> int { let p: Pair = (value 41) assert (== (read p) 42) return 0 }''',
    "private_gc": '''pure fn value(x: string) -> string { return (str_concat x "!") }
fn main() -> int { assert (== (value "yes") "yes!") return 0 }''',
    "array_and_lexical_shadow": 'let mut state: int = 9\npure fn value(x: int) -> array<int> { let state: int = x return [state, (+ state 1)] }\nfn main() -> int { let result: array<int> = (value 41) assert (== (at result 1) 42) return 0 }',
    "immutable_global": '''let answer: int = 42
pure fn value() -> int { return answer }
fn main() -> int { assert (== (value) 42) return 0 }''',
}
REJECT = {
    "mutable_global": '''let mut state: int = 1
pure fn value() -> int { return state }''',
    "transitive_io": '''fn leaf() -> int { (println "effect") return 1 }
fn middle() -> int { return (leaf) }
pure fn value() -> int { return (middle) }''',
    "recursive_effect_cycle": '''fn left(n: int) -> int { if (> n 0) { return (right (- n 1)) } return 0 }
fn right(n: int) -> int { (println "effect") return (left n) }
pure fn value() -> int { return (left 2) }''',
    "record_field_effect": '''struct Pair { value: int }
fn leaf() -> int { (println "effect") return 1 }
pure fn value() -> Pair { return Pair { value: (leaf) } }''',
    'immutable_array_global': 'let values: array<int> = [1]\npure fn value() -> int { return (at values 0) }',
    'array_parameter': 'pure fn value(values: array<int>) -> int { return (at values 0) }',
    'wrapped_array_parameter': 'struct Box { values: array<int> }\npure fn value(box: Box) -> int { return (at box.values 0) }',
    'wrapped_array_global': 'struct Box { values: array<int> }\nlet box: Box = Box { values: [1] }\npure fn value() -> int { return (at box.values 0) }',
    'map_parameter': 'pure fn value(values: HashMap<string,int>) -> int { return (map_length values) }',
    'immutable_map_global': 'fn empty() -> HashMap<string,int> { return (map_new) }\nlet values: HashMap<string,int> = (empty)\npure fn value() -> int { return (map_length values) }',
    "mutation": '''pure fn value() -> int { let mut x: int = 0 set x 1 return x }''',
    "unsafe": '''pure fn value() -> int { unsafe { (println "effect") } return 0 }''',
    "annotated_extern": '''pure extern fn external_value() -> int
pure fn value() -> int { unsafe { return (external_value) } }''',
    "callback": '''fn callback(x: int) -> int { return x }
pure fn value(callback: fn(int) -> int) -> int { return (callback 2) }''',
    "higher_order_intrinsic": '''fn callback(x: int) -> int { (println "effect") return x }
pure fn value() -> array<int> { return (array_map [1, 2] callback) }''',
    "resource_return": 'resource struct Handle { id: int }\npure fn value() -> Handle { return Handle { id: 1 } }',
    "resource_transfer": 'resource struct Handle { id: int }\npure fn value(h: Handle) -> Handle { return h }',
    "resource_wrapped": 'resource struct Handle { id: int }\nstruct Box { value: Handle }\npure fn value(h: Box) -> Box { return h }',
    "loop": '''pure fn value() -> int { while false { return 1 } return 0 }''',
}

class PurityContract(unittest.TestCase):
    def command(self, args):
        return subprocess.run([str(x) for x in args], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)

    def test_shared_frontend_contract(self):
        with tempfile.TemporaryDirectory(prefix="nano-purity-") as temporary:
            root = Path(temporary)
            for accepted, cases in [(True, ACCEPT), (False, REJECT)]:
                for name, source in cases.items():
                    path = root / f"{name}.nano"
                    path.write_text(source + ("" if accepted else "\nfn main() -> int { return 0 }") + "\n")
                    for compiler in COMPILERS:
                        with self.subTest(case=name, compiler=compiler.name):
                            output = root / f"{name}-{compiler.name}"
                            if not accepted:
                                output.write_bytes(b"prior-output")
                            result = self.command([compiler, path, "-o", output])
                            if accepted:
                                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                                ran = self.command([output])
                                self.assertEqual(ran.returncode, 0, ran.stdout)
                            else:
                                self.assertNotEqual(result.returncode, 0, result.stdout)
                                self.assertIn("closed empty effect summary", result.stdout)
                                self.assertEqual(output.read_bytes(), b"prior-output")

    def test_bound_transitive_import(self):
        with tempfile.TemporaryDirectory(prefix="nano-purity-import-") as temporary:
            root = Path(temporary)
            module = root / "helper.nano"
            module.write_text('module Helper\npub fn helper(x: int) -> int { return (+ x 1) }\nshadow helper { assert (== (helper 1) 2) }\n')
            source = root / "main.nano"
            source.write_text(f'module "{module}" as dependency\npure fn value(x: int) -> int {{ return (dependency.helper x) }}\nfn main() -> int {{ assert (== (value 41) 42) return 0 }}\n')
            for compiler in COMPILERS:
                with self.subTest(compiler=compiler.name):
                    result = self.command([compiler, source, "-o", root / compiler.name])
                    self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                    ran = self.command([root / compiler.name])
                    self.assertEqual(ran.returncode, 0, ran.stdout)
            module.write_text(module.read_text().replace('return (+ x 1)', '(println "effect") return (+ x 1)'))
            for compiler in COMPILERS:
                with self.subTest(compiler=compiler.name, effect=True):
                    result = self.command([compiler, source, "-o", root / compiler.name])
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("closed empty effect summary", result.stdout)

    def test_generated_intrinsic_contract(self):
        result = self.command(["python3", ROOT / "scripts/gen_purity_intrinsics.py", "--check"])
        self.assertEqual(result.returncode, 0, result.stdout)

if __name__ == "__main__":
    unittest.main()
