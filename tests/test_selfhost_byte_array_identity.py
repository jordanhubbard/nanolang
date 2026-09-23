"""I keep byte-producing builtins aligned with parsed array<u8> annotations."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get("NANO_BYTE_IDENTITY_ROOT",
                           Path(__file__).resolve().parents[1]))

DRIVER = r'''import "src_nano/typecheck.nano"
import "src_nano/parser.nano"
import "src_nano/compiler/lexer.nano"
import "src_nano/compiler/module_bindings.nano"
from "src_nano/compiler/diagnostics.nano" import diag_list_new
extern fn get_argc() -> int
extern fn get_argv(index: int) -> string
extern fn file_read(path: string) -> string
fn main() -> int {
 if (!= (get_argc) 2) { return 0 }
 let path: string = (get_argv 1)
 let diagnostics: List<CompilerDiagnostic> = (diag_list_new)
 let tokens: List<LexerToken> = (tokenize_string (file_read path) path diagnostics)
 let parsed: Parser = (parse_program tokens (list_LexerToken_length tokens) path)
 if (parser_has_error parsed) { (println "PARSE") return 2 }
 (mb_reset [])
 let checked: TypecheckPhaseOutput = (typecheck_phase_with_shadows parsed path [] 0)
 if checked.had_error { (println "TYPECHECK") return 1 }
 (println "CHECKED")
 return 0
}
shadow main { assert (== (main) 0) }
'''


class SelfhostByteArrayIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nanolang-byte-identity-"))
        print("I retain byte identity artifacts in " + str(cls.work), flush=True)
        driver_source = cls.work / "checker.nano"
        driver_source.write_text(DRIVER)
        cls.drivers = []
        for producer in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            output = cls.work / ("checker-" + producer)
            result = subprocess.run(
                [str(ROOT / "bin" / producer), str(driver_source), "-o", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=900)
            (cls.work / (producer + ".setup.log")).write_text(
                result.stdout + result.stderr)
            if result.returncode:
                raise AssertionError(producer + ": " + result.stdout + result.stderr)
            cls.drivers.append(output)

    def check(self, label, source, expected):
        path = self.work / (label + ".nano")
        path.write_text(source)
        for driver in self.drivers:
            with self.subTest(case=label, driver=driver.name):
                result = subprocess.run([str(driver), str(path)], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, expected,
                                 result.stdout + result.stderr)
                self.assertEqual(result.stdout.strip(),
                                 "CHECKED" if expected == 0 else "TYPECHECK")

    def seed_check(self, label, source, expected):
        path = self.work / ("seed-" + label + ".nano")
        output = self.work / ("seed-" + label)
        path.write_text(source)
        output.write_text("previous")
        result = subprocess.run(
            [str(ROOT / "bin" / "nanoc_c"), str(path), "-o", str(output)],
            cwd=ROOT, capture_output=True, text=True,
            timeout=180)
        self.assertEqual(result.returncode == 0, expected == 0,
                         result.stdout + result.stderr)
        if expected:
            self.assertEqual(output.read_text(), "previous")
        return result

    def test_c_seed_keeps_byte_results_and_resolves_local_bindings_first(self):
        self.seed_check("accepted", '''fn main() -> int {
 let from_string: array<u8> = (bytes_from_string "hi")
 let from_file: array<u8> = (file_read_bytes "/dev/null")
 return (+ (array_length from_string) (array_length from_file))
}
shadow main { assert true }
''', 0)
        self.seed_check("wrong", '''fn main() -> int {
 let wrong: array<int> = (bytes_from_string "hi")
 return (array_length wrong)
}
shadow main { assert true }
''', 1)
        self.seed_check("local", '''fn local_bytes(value: int) -> array<int> { return [value] }
shadow local_bytes { assert (== (at (local_bytes 41) 0) 41) }
fn main() -> int {
 let bytes_from_string: fn(int) -> array<int> = local_bytes
 let values: array<int> = (bytes_from_string 41)
 return (at values 0)
}
shadow main { assert (== (main) 41) }
''', 0)

    def test_c_seed_builtin_redefinition_is_a_checked_refusal(self):
        result = self.seed_check("reserved", '''fn bytes_from_string(value: int) -> array<int> {
 return [value]
}
shadow bytes_from_string { assert (== (at (bytes_from_string 41) 0) 41) }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
''', 1)
        self.assertGreater(result.returncode, 0)
        self.assertIn("Cannot redefine built-in function 'bytes_from_string'",
                      result.stderr)

    def test_byte_results_keep_u8_identity(self):
        self.check("accepted", '''fn main() -> int {
 let from_string: array<u8> = (bytes_from_string "hi")
 let from_file: array<u8> = (file_read_bytes "/dev/null")
 return (+ (array_length from_string) (array_length from_file))
}
shadow main { assert true }
''', 0)
        for builtin, call in (("string", '(bytes_from_string "hi")'),
                              ("file", '(file_read_bytes "/dev/null")')):
            self.check("reject-" + builtin, f'''fn main() -> int {{
 let wrong: array<int> = {call}
 return (array_length wrong)
}}
shadow main {{ assert true }}
''', 1)


if __name__ == "__main__":
    unittest.main()
