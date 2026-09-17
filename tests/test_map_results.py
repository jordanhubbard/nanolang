"""I preserve map transform result storage, including empty arrays."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from itertools import product

ROOT = Path(__file__).resolve().parents[1]


class MapResults(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("NANOLANG_MAP_SELFHOST"),
                         "I run this Stage2 trace in test-selfhost-map-results")
    def test_selfhost_map_evaluation_order(self):
        with tempfile.TemporaryDirectory(prefix="nano-map-order-") as tmp:
            source = Path(tmp) / "order.nano"
            source.write_text('''fn input() -> array<int> {
    (println "source")
    return [1, 2]
}
shadow input { assert (== (array_length (input)) 2) }
fn transform(n: int) -> float {
    (println "transform")
    return 1.5
}
shadow transform { assert (== (transform 1) 1.5) }
fn choose() -> fn(int) -> float {
    (println "choose")
    return transform
}
shadow choose { assert (== ((choose) 1) 1.5) }
fn main() -> int {
    let mapped: array<float> = (map (input) (choose))
    assert (== (array_length mapped) 2)
    assert (== (at mapped 0) 1.5)
    assert (== (at mapped 1) 1.5)
    return 0
}
shadow main { assert (== (main) 0) }
''')
            artifact = Path(tmp) / "order"
            built = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                    "-o", str(artifact)], cwd=ROOT,
                                   capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([str(artifact)], capture_output=True, timeout=20)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertEqual(ran.stdout, b"source\nchoose\ntransform\ntransform\n")

    def test_scalar_results(self):
        inputs = (("int", "7"), ("float", "2.5"), ("bool", "true"), ("string", '"input"'))
        outputs = (("int", "42"), ("float", "1.5"), ("bool", "true"), ("string", '"mapped"'))
        for (input_type, input_value), (result_type, value) in product(inputs, outputs):
            backends = (("nanoc_stage2", False),) if os.environ.get("NANOLANG_MAP_SELFHOST") else (
                ("nanoc_c", False), ("nano_virt", True))
            for compiler, vm in backends:
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
