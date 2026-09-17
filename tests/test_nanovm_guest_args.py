"""I preserve compiler arguments at the standalone VM boundary."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GuestArguments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="nano-guest-args-")
        cls.directory = Path(cls.temp.name)
        cls.module = cls.directory / "guest.nvm"
        source = cls.directory / "guest.nano"
        source.write_text('''extern fn get_argc() -> int
extern fn get_argv(index: int) -> string
fn main() -> int {
    unsafe {
        (println (get_argc))
        let mut i: int = 0
        while (< i (get_argc)) {
            (println (+ "[" (+ (get_argv i) "]")))
            set i (+ i 1)
        }
    }
    return 0
}
''')
        subprocess.run([ROOT / "bin/nano_virt", source, "-o", cls.module],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_vm(self, *args):
        return subprocess.run([ROOT / "bin/nano_vm", *map(str, args)],
                              capture_output=True, text=True, timeout=20)

    def test_compiler_flags_and_empty_argument(self):
        args = ["source.nano", "-o", "output.nvm", "--help", "", "two words"]
        result = self.run_vm(self.module, "--", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         ["7", f"[{self.module}]", *[f"[{arg}]" for arg in args]])

    def test_no_guest_arguments(self):
        for delimiter in ([], ["--"]):
            with self.subTest(delimiter=delimiter):
                result = self.run_vm(self.module, *delimiter)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["1", f"[{self.module}]"])

    def test_vm_options_are_not_guest_arguments(self):
        profile = self.directory / "profile.json"
        result = self.run_vm("--profile-isa", profile, self.module, "--", "-o")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["2", f"[{self.module}]", "[-o]"])
        self.assertTrue(profile.exists())

    def test_bytecode_emitter_driver(self):
        emitter = self.directory / "emitter.nvm"
        assembly = self.directory / "result.nasm"
        result_module = self.directory / "result.nvm"
        subprocess.run([ROOT / "bin/nano_virt", ROOT / "src_nano/nanoisa_emit.nano",
                        "-o", emitter], cwd=ROOT, check=True, capture_output=True,
                       text=True, timeout=60)
        result = self.run_vm(emitter, "--", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage: nanoisa_emit", result.stdout)
        result = self.run_vm(emitter, "--", ROOT / "tests/nanoisa/fixtures/cut_a_add.nano",
                             "-o", assembly)
        self.assertEqual(result.returncode, 0, result.stderr)
        subprocess.run([ROOT / "bin/nanoisa", "asm", assembly, "-o", result_module],
                       check=True, capture_output=True, text=True, timeout=20)
        result = self.run_vm(result_module)
        self.assertEqual(result.returncode, 42, result.stderr)

    def test_ambiguous_or_unsupported_invocations(self):
        for args in [("--", "guest"), (self.module, "second.nvm"),
                     ("--daemon", self.module, "--", "guest"),
                     ("--verify-only", self.module, "--", "guest")]:
            with self.subTest(args=args):
                result = self.run_vm(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("I ", result.stderr)
