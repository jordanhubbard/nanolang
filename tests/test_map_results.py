"""I preserve map transform result storage, including empty arrays."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from itertools import product

ROOT = Path(__file__).resolve().parents[1]


class MapResults(unittest.TestCase):
    def test_scalar_results(self):
        inputs = (("int", "7"), ("float", "2.5"), ("bool", "true"), ("string", '"input"'))
        outputs = (("int", "42"), ("float", "1.5"), ("bool", "true"), ("string", '"mapped"'))
        for (input_type, input_value), (result_type, value) in product(inputs, outputs):
            for compiler, vm in (("nanoc_c", False), ("nano_virt", True)):
                with self.subTest(input=input_type, output=result_type, compiler=compiler), tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / "map.nano"
                    source.write_text(f'''fn transform(x: {input_type}) -> {result_type} {{ return {value} }}
shadow transform {{ assert (== (transform {input_value}) {value}) }}
fn choose() -> fn({input_type}) -> {result_type} {{ return transform }}
shadow choose {{ assert (== ((choose) {input_value}) {value}) }}
fn main() -> int {{
    let source: array<{input_type}> = [{input_value}, {input_value}]
    let result: array<{result_type}> = (map source transform)
    assert (== (array_length result) 2)
    assert (== (at result 0) {value})
    assert (== (at result 1) {value})
    assert (== (at source 0) {input_value})
    let callback: fn({input_type}) -> {result_type} = transform
    let variable: array<{result_type}> = (map source callback)
    assert (== (at variable 0) {value})
    let returned: array<{result_type}> = (map source (choose))
    assert (== (at returned 0) {value})
    let empty: array<{input_type}> = []
    let mapped: array<{result_type}> = (map empty callback)
    assert (== (array_length mapped) 0)
    let filled: array<{result_type}> = (array_push mapped {value})
    assert (== (at filled 0) {value})
    assert (== (array_length empty) 0)
    return 0
}}
shadow main {{ assert (== (main) 0) }}
''')
                    output = Path(tmp) / ("map.nvm" if vm else "map")
                    command = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                    if vm: command.append("--emit-nvm")
                    built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60,
                                           env=dict(os.environ, TMPDIR=tmp))
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    command = [str(ROOT / "bin/nano_vm"), str(output)] if vm else [str(output)]
                    ran = subprocess.run(command, capture_output=True, timeout=20)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)


if __name__ == "__main__":
    unittest.main()
