"""I preserve exact tuple values across my canonical VM and native routes."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
from pathlib import Path
import os
import platform
import subprocess
import tempfile
import unittest

from tests.native_toolchain import native_cc


ROOT = Path(__file__).resolve().parents[1]


class CanonicalTupleValues(unittest.TestCase):
    def checked(self, *args, env=None):
        result = subprocess.run(
            [str(value) for value in args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_both_producers_preserve_tuple_calls_results_and_projection(self):
        fixture = ROOT / "tests/nanoisa/fixtures/tuple_values.nano"
        expected = "first\nsecond\nthird\nvalue\n12\n"
        functions = (
            "first_value",
            "second_value",
            "third_value",
            "make_tuple",
            "relay_tuple",
            "forward_tuple",
            "tuple_score",
            "main",
        )
        with tempfile.TemporaryDirectory(prefix="nano-tuple-values-") as temporary:
            work = Path(temporary)
            seed = work / "seed.nvm"
            assembly = work / "source.nasm"
            emitted = work / "source.nvm"
            self.checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            comparison = self.checked(
                ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly, *functions
            )
            self.assertIn("0 failed", comparison.stdout)
            text = assembly.read_text()
            first = text.index("CALL first_value")
            second = text.index("CALL second_value", first)
            third = text.index("CALL third_value", second)
            packed = text.index("AGG_PACK 2 0 0 3", third)
            self.assertLess(first, second)
            self.assertLess(second, third)
            self.assertLess(third, packed)
            self.assertIn("AGG_GET 0", text)
            self.assertIn("AGG_GET 1", text)
            self.assertIn("AGG_GET 2", text)
            self.checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)

            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    self.checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.assertEqual(self.checked(ROOT / "bin/nano_vm", module).stdout, expected)
                    generated = work / f"{module.stem}.c"
                    native = work / f"{module.stem}.native"
                    self.checked(ROOT / "bin/nvm2c", module, "-o", generated)
                    generated_text = generated.read_text()
                    self.assertIn(".kind != 2", generated_text)
                    self.checked(
                        *native_cc(),
                        "-std=c11",
                        "-O1",
                        "-g",
                        "-fno-omit-frame-pointer",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-fsanitize=address,undefined",
                        "-fno-sanitize-recover=all",
                        generated,
                        "-o",
                        native,
                    )
                    runtime_env = os.environ.copy()
                    runtime_env["ASAN_OPTIONS"] = (
                        "detect_leaks=0" if platform.system() == "Darwin" else asan_options()
                    )
                    self.assertEqual(self.checked(native, env=runtime_env).stdout, expected)

    def test_invalid_tuple_shapes_refuse_without_replacing_output(self):
        cases = {
            "arity": (
                'fn pair() -> (int,string) { return (1, "one", true) }\n',
                "tuple arity",
            ),
            "element": (
                "fn pair() -> (int,string) { return (1, true) }\n",
                "tuple element 1",
            ),
            "result": (
                "fn numbers() -> (int,int) { return (1, 2) } "
                "fn text() -> (string,int) { return (numbers) }\n",
                "declared tuple return type",
            ),
            "projection": (
                "fn main() -> int { let pair: (int,string) = (1, \"one\") return pair.2 }\n",
                "tuple projection",
            ),
        }
        with tempfile.TemporaryDirectory(prefix="nano-tuple-refusal-") as temporary:
            work = Path(temporary)
            for name, (program, diagnostic) in cases.items():
                with self.subTest(case=name):
                    source = work / f"{name}.nano"
                    output = work / f"{name}.nasm"
                    source.write_text(program)
                    output.write_text("retained output")
                    result = subprocess.run(
                        [ROOT / "bin/nanoisa_emit", source, "-o", output],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(diagnostic, result.stdout + result.stderr)
                    self.assertEqual(output.read_text(), "retained output")


if __name__ == "__main__":
    unittest.main()
