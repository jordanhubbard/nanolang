"""I preserve canonical frontend checks before explicit NanoISA publication."""
import os
import json
import sys
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

    def test_artifact_paths_preserve_library_identity_and_execution(self):
        with tempfile.TemporaryDirectory(prefix="canonical-artifact-") as tmp:
            directory = Path(tmp)
            source, output = directory / "paths.nano", directory / "paths.nvm"
            source.write_text('module "modules/std/fs.nano" as fs\n'
                              'fn main() -> int { '
                              'assert (== (fs.basename "/tmp/one.txt") "one.txt") '
                              'assert (== (fs.dirname "/tmp/one.txt") "/tmp") '
                              'assert (== (fs.join "/tmp" "one.txt") "/tmp/one.txt") '
                              'assert (== (fs.normalize "/tmp/./one.txt") "/tmp/one.txt") '
                              'return 0 }\nshadow main { assert (== (main) 0) }\n')
            self.run_command([COMPILER, source, "--emit-nvm", "-o", output])
            first = output.read_bytes()
            self.run_command([COMPILER, source, "--emit-nvm", "-o", output])
            self.assertEqual(first, output.read_bytes())
            self.run_command([ROOT / "bin/nano_vm", "--verify-only", output])
            self.run_command([ROOT / "bin/nano_vm", output])
            c_file, executable = directory / "out.c", directory / "native"
            self.run_command([ROOT / "bin/nvm2c", output, "-o", c_file])
            self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", c_file,
                              ROOT / "bin/nano_aot_runtime.o", "-lm",
                              *(["-Wl,--export-dynamic", "-ldl"] if sys.platform.startswith("linux") else []),
                              "-o", executable])
            self.run_command([executable])

    def test_assembly_publication_status_and_error_run_in_both_backends(self):
        with tempfile.TemporaryDirectory(prefix="canonical-publisher-") as tmp:
            directory = Path(tmp)
            source, output = directory / "publisher.nano", directory / "publisher.nvm"
            target = directory / "published.nvm"
            assembly = '.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
            source.write_text('module "modules/nanoisa/nanoisa.nano" as isa\n'
                              'fn main() -> int { assert (== (isa.assemble_text_save ' + json.dumps(assembly) + ' ' + json.dumps(str(target)) + ') 0) '
                              'assert (!= (isa.assemble_text_save "bad assembly" ' + json.dumps(str(target)) + ') 0) '
                              'assert (> (str_length (isa.last_error)) 0) return 0 }\nshadow main { assert true }\n')
            self.run_command([COMPILER, source, "--emit-nvm", "-o", output])
            self.run_command([ROOT / "bin/nano_vm", output])
            published = target.read_bytes()
            self.assertEqual(published[:4], b"NVM\x02")
            self.run_command([ROOT / "bin/nano_vm", "--verify-only", target])
            self.run_command([ROOT / "bin/nano_vm", target])
            native_c, binary = directory / "publisher.c", directory / "publisher"
            self.run_command([ROOT / "bin/nvm2c", output, "-o", native_c])
            self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", native_c,
                              ROOT / "bin/nano_aot_runtime.o", "-lm",
                              *(["-Wl,--export-dynamic", "-ldl"] if sys.platform.startswith("linux") else []),
                              "-o", binary])
            self.run_command([binary])
            self.assertEqual(published, target.read_bytes())

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
            other = directory / "other.nano"
            other.write_text("module Other\nfn base() -> int { return 12 }\n"
                             "shadow base { assert (== (base) 12) }\n"
                             "pub fn value() -> int { return (base) }\n"
                             "shadow value { assert (== (value) 12) }\n")
            source.write_text(f'module "{other}" as other\n' + source.read_text().replace(
                '(println "canonical-nvm")', 'assert (== (other.value) 12) (println "canonical-nvm")'))
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

    def test_unreachable_helper_still_runs_dependency_shadows(self):
        with tempfile.TemporaryDirectory(prefix="canonical-program-") as tmp:
            directory = Path(tmp)
            dependency = directory / "unused.nano"
            dependency.write_text('module Unused\npub fn unused() -> float { return 1.25 }\n'
                                  'shadow unused { assert (> (unused) 1.0) }\n')
            source = directory / "main.nano"
            source.write_text(f'module "{dependency}" as unused\n'
                              'fn main() -> int { (println "program-closure") return 0 }\n'
                              'shadow main { assert true }\n')
            output = directory / "program.nvm"
            self.run_command([COMPILER, source, "--emit-nvm", "-o", output])
            accepted = output.read_bytes()
            self.assertEqual(accepted[:4], b"NVM\x02")
            self.assertEqual(self.run_command([ROOT / "bin/nano_vm", output]).stdout,
                             b"program-closure\n")
            c_file, executable = directory / "out.c", directory / "native"
            self.run_command([ROOT / "bin/nvm2c", output, "-o", c_file])
            self.run_command(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", c_file, "-lm", "-o", executable])
            self.assertEqual(self.run_command([executable]).stdout, b"program-closure\n")
            # I retain full dependency validation even when its code is unreachable.
            dependency.write_text(dependency.read_text().replace('assert (> (unused) 1.0)', 'assert false'))
            rejected = self.run_command([COMPILER, source, "--emit-nvm", "-o", output], 1)
            self.assertIn(b"after failed shadows", rejected.stdout + rejected.stderr)
            self.assertEqual(output.read_bytes(), accepted)

    def test_reachable_refusal_reports_precise_boundary(self):
        with tempfile.TemporaryDirectory(prefix="canonical-program-route-") as tmp:
            directory = Path(tmp)
            source, output = directory / "main.nano", directory / "main.nvm"
            output.write_bytes(b"prior")
            source.write_text('fn required() -> float { return 1.5 }\n'
                              'fn main() -> int { (required) return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            rejected = self.run_command([COMPILER, source, "--emit-nvm", "-o", output], 1)
            self.assertIn(b"I cannot lower this checked program: unsupported result type float",
                          rejected.stdout + rejected.stderr)
            self.assertEqual(output.read_bytes(), b"prior")

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
