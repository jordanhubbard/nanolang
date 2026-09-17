"""I preserve canonical frontend checks before explicit NanoISA publication."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()


class CanonicalNvmOutput(unittest.TestCase):
    def run_command(self, args, expected=0):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, timeout=120)
        self.assertEqual(result.returncode, expected, (result.stdout + result.stderr)[-6000:])
        return result

    def sources(self, directory, bad_dependency=False):
        dependency = directory / "values.nano"
        dependency.write_text('module Values\npub fn value() -> int { return 37 }\n'
                              + ('shadow value { assert false }\n' if bad_dependency else
                                 'shadow value { assert (== (value) 37) }\n'))
        source = directory / "main.nano"
        source.write_text(f'module "{dependency}" as values\n'
                          'fn main() -> int { assert (== (values.value) 37) (println "canonical-nvm") return 0 }\n'
                          'shadow main { assert (== (values.value) 37) }\n')
        return source, dependency

    def test_native_string_prefix_builtin(self):
        with tempfile.TemporaryDirectory(prefix="canonical-prefix-") as tmp:
            directory = Path(tmp)
            source, output = directory / "prefix.nano", directory / "prefix"
            source.write_text('''fn main() -> int {
 assert (str_starts_with "alpha" "alp")
 assert (str_starts_with "alpha" "alpha")
 assert (str_starts_with "alpha" "")
 assert (str_starts_with "" "")
 assert (not (str_starts_with "al" "alpha"))
 assert (not (str_starts_with "alpha" "lp"))
 assert (not (str_starts_with "" "a"))
 return 0
}
shadow main { assert (== (main) 0) }
''')
            self.run_command([COMPILER, source, "-o", output])
            self.run_command([output])

    def test_bound_imports_repeatable_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix="canonical-nvm-") as tmp:
            directory = Path(tmp)
            source, dependency = self.sources(directory)
            first, second = directory / "one.nvm", directory / "two.nvm"
            for output in (first, second):
                self.run_command([COMPILER, source, "--emit-nvm", "-o", output])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[:4], b"NVM\x02")
            self.assertEqual(self.run_command([ROOT / "bin/nano_vm", first]).stdout, b"canonical-nvm\n")
            c_file, executable = directory / "out.c", directory / "native"
            self.run_command([ROOT / "bin/nvm2c", first, "-o", c_file])
            self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", c_file, "-lm", "-o", executable])
            self.assertEqual(self.run_command([executable]).stdout, b"canonical-nvm\n")

    def test_dependency_and_root_shadow_failures_preserve_output(self):
        with tempfile.TemporaryDirectory(prefix="canonical-nvm-shadows-") as tmp:
            directory = Path(tmp)
            source, dependency = self.sources(directory, bad_dependency=True)
            output = directory / "prior.nvm"
            output.write_bytes(b"prior")
            rejected = self.run_command([COMPILER, source, "--emit-nvm", "-o", output], 1)
            self.assertIn(b"after failed shadows", rejected.stdout + rejected.stderr)
            self.assertEqual(output.read_bytes(), b"prior")
            self.sources(directory)
            source.write_text(source.read_text().replace('shadow main { assert (== (values.value) 37) }',
                                                        'shadow main { assert false }'))
            rejected = self.run_command([COMPILER, source, "--emit-nvm", "-o", output], 1)
            self.assertIn(b"after failed shadows", rejected.stdout + rejected.stderr)
            self.assertEqual(output.read_bytes(), b"prior")

    def test_source_aliases_and_rejections_preserve_files(self):
        with tempfile.TemporaryDirectory(prefix="canonical-nvm-guards-") as tmp:
            directory = Path(tmp)
            source, dependency = self.sources(directory)
            for original in (source, dependency):
                before = original.read_bytes()
                symlink = directory / (original.stem + "-link")
                symlink.symlink_to(original)
                hardlink = directory / (original.stem + "-hard")
                os.link(original, hardlink)
                for output in (original, symlink, hardlink):
                    self.run_command([COMPILER, source, "--emit-nvm", "-o", output], 1)
                    self.assertEqual(original.read_bytes(), before)
            output = directory / "prior.nvm"
            for body in ('fn main() -> int { return "wrong" }\nshadow main { assert true }\n',
                         'fn main() -> int { let value: float = 1.25 (println value) return 0 }\nshadow main { assert true }\n'):
                source.write_text(body)
                output.write_bytes(b"prior")
                self.run_command([COMPILER, source, "--emit-nvm", "-o", output], 1)
                self.assertEqual(output.read_bytes(), b"prior")
            self.assertFalse(list(directory.glob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
