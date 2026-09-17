"""I publish deterministic verified modules through my C-seed-hosted driver."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "bin/nanoisa_emit"
SOURCE = '''fn add(a: int, b: int) -> int { return (+ a b) }
shadow add { assert (== (add 2 3) 5) }
fn main() -> int {
 assert (== (add 2 3) 5)
 (println "module-ok")
 return 0
}
shadow main { assert true }
'''


class NanoisaEmitDriver(unittest.TestCase):
    def run_command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, timeout=90)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_repeatable_module_executes_in_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-") as tmp:
            directory = Path(tmp)
            source, first, second = (directory / name for name in ("source.nano", "a.nvm", "b.nvm"))
            source.write_text(SOURCE)
            for output in (first, second):
                self.run_command([DRIVER, source, "--emit-nvm", "-o", output])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[:4], b"NVM\x02")
            self.assertEqual(self.run_command([ROOT / "bin/nano_vm", first]).stdout, b"module-ok\n")
            generated, binary = directory / "module.c", directory / "native"
            self.run_command([ROOT / "bin/nvm2c", first, "-o", generated])
            self.assertNotIn("nano_vm", generated.read_text())
            self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", generated, "-lm", "-o", binary])
            self.assertEqual(self.run_command([binary]).stdout, b"module-ok\n")
            self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_assembly_mode_stays_equivalent(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-text-") as tmp:
            directory = Path(tmp)
            source, assembly, module, direct = (directory / name for name in
                                                ("source.nano", "source.nasm", "from-text.nvm", "direct.nvm"))
            source.write_text(SOURCE)
            stdout = self.run_command([DRIVER, source]).stdout
            self.run_command([DRIVER, source, "-o", assembly])
            self.assertEqual(assembly.read_bytes(), stdout)
            self.run_command([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
            self.run_command([DRIVER, source, "--emit-nvm", "-o", direct])
            self.assertEqual(module.read_bytes(), direct.read_bytes())

    def test_rejection_preserves_previous_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-driver-errors-") as tmp:
            directory = Path(tmp)
            source, output = directory / "bad.nano", directory / "prior.nvm"
            for body in ("fn broken(", "fn main() -> float { return 1.5 }\nshadow main { assert true }\n"):
                source.write_text(body)
                output.write_bytes(b"previous bytes")
                self.run_command([DRIVER, source, "--emit-nvm", "-o", output], expected=1)
                self.assertEqual(output.read_bytes(), b"previous bytes")
            source.write_text(SOURCE)
            self.run_command([DRIVER, source, "--emit-nvm"], expected=2)
            blocked = directory / "blocked"
            blocked.mkdir()
            self.run_command([DRIVER, source, "--emit-nvm", "-o", blocked], expected=1)
            self.assertEqual(output.read_bytes(), b"previous bytes")
            self.assertEqual(list(directory.glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
