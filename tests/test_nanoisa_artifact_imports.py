"""I retain artifact namespace, ABI and source owner during NanoISA lowering."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="nano-artifact-emitter-")
        cls.directory = Path(cls.temporary.name)
        source = cls.directory / "driver.nano"
        source.write_text((ROOT / "tests/nanoisa/fixtures/artifact_import_driver.nano.txt").read_text())
        cls.driver = cls.directory / "driver"
        result = subprocess.run([ROOT / "bin/nanoc_c", source, "-o", cls.driver], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def command(self, *args, expected=0):
        environment = os.environ.copy()
        environment["NANO_AS_CAPTURE_HELPER"] = str(ROOT / "bin/nano_as_capture.so")
        result = subprocess.run([str(x) for x in args], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def module(self, directory, name, result):
        path = directory / name
        path.mkdir()
        source = path / "api.nano"
        source.write_text('extern fn path_basename(path: string) -> string\n')
        (path / "module.json").write_text(json.dumps({"name": name, "c_sources": ["api.c"]}))
        (path / "api.c").write_text('const char *path_basename(const char *value) { (void)value; return "' + result + '"; }\n')
        return source

    def test_duplicate_symbols_keep_owner_and_raw_api_clears_bindings(self):
        with tempfile.TemporaryDirectory(prefix="nano-artifact-owners-") as tmp:
            directory = Path(tmp)
            left = self.module(directory, "left", "left-result")
            right = self.module(directory, "right", "right-result")
            source = directory / "merged.nano"
            source.write_text(left.read_text() + right.read_text() +
                              'fn main() -> int { assert (== (left.base "a") "left-result") '
                              'assert (== (right.base "b") "right-result") '
                              'assert (== (selected "c") "right-result") return 0 }\n')
            assembly = self.command(self.driver, source, left, right, "raw").stdout
            self.assertEqual(assembly, self.command(self.driver, source, left, right, "program").stdout)
            self.assertEqual(assembly.count('"path_basename" string string'), 2)
            self.assertIn('.import_kind 0 artifact', assembly)
            self.assertIn('.import_kind 1 artifact', assembly)
            self.assertNotIn('.import ""', assembly)
            self.assertNotIn('/current/', assembly)
            text, module = directory / "out.nasm", directory / "out.nvm"
            text.write_text(assembly)
            self.command(ROOT / "bin/nanoisa", "asm", text, "-o", module)
            self.command(ROOT / "bin/nano_vm", "--verify-only", module)
            self.command(ROOT / "bin/nano_vm", module)
            c_source, binary = directory / "out.c", directory / "native"
            self.command(ROOT / "bin/nvm2c", module, "-o", c_source)
            self.command("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", c_source, "-ldl", "-o", binary)
            self.command(binary)

    def test_cseed_and_selfhost_preserve_exact_artifact_abi_and_code(self):
        with tempfile.TemporaryDirectory(prefix="nano-artifact-parity-") as tmp:
            directory = Path(tmp)
            dependency = self.module(directory, "provider", "retained")
            source, merged = directory / "main.nano", directory / "merged.nano"
            body = ('fn main() -> int { unsafe { assert (== (path_basename "input") "retained") } '
                    'return 0 }\n')
            source.write_text('import "' + str(dependency) + '"\n' + body + 'shadow main { assert true }\n')
            merged.write_text(dependency.read_text() + '\n' + body.replace('path_basename', 'left.base'))
            seed, assembly = directory / "seed.nvm", directory / "selfhost.nasm"
            self.command(ROOT / "bin/nano_virt", source, "--emit-nvm", "--strip-debug", "-o", seed)
            assembly.write_text(self.command(self.driver, merged, dependency, dependency, "program").stdout)
            compared = self.command(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly, "main", "--imports")
            self.assertIn("5 passed, 0 failed", compared.stdout)

    def test_integer_status_and_zero_argument_diagnostic_match_cseed(self):
        with tempfile.TemporaryDirectory(prefix="nano-assembly-contract-") as tmp:
            directory = Path(tmp)
            provider = directory / "provider"
            provider.mkdir()
            api = provider / "api.nano"
            declarations = ('extern fn nl_nanoisa_assemble_text_save(source: string, path: string) -> int\n'
                            'extern fn nl_nanoisa_last_error() -> string\n')
            api.write_text(declarations)
            (provider / "module.json").write_text(json.dumps({"name": "publisher", "c_sources": ["api.c"]}))
            (provider / "api.c").write_text(
                '#include <string.h>\n'
                'long long nl_nanoisa_assemble_text_save(const char *source, const char *path) { '
                'return !strcmp(source,"first") && !strcmp(path,"second") ? 37 : -1; }\n'
                'const char *nl_nanoisa_last_error(void) { return "fixture-error"; }\n')
            body = ('fn main() -> int { unsafe { '
                    'let status: int = (nl_nanoisa_assemble_text_save "first" "second") '
                    'assert (== status 37) assert (== (nl_nanoisa_last_error) "fixture-error") '
                    '} return 0 }\n')
            root, merged = directory / "main.nano", directory / "merged.nano"
            root.write_text('import "' + str(api) + '"\n' + body + 'shadow main { assert true }\n')
            merged.write_text(declarations + body.replace('nl_nanoisa_assemble_text_save', 'publish').replace('nl_nanoisa_last_error', 'diagnostic'))
            seed, text, module = directory / "seed.nvm", directory / "out.nasm", directory / "out.nvm"
            self.command(ROOT / "bin/nano_virt", root, "--emit-nvm", "--strip-debug", "-o", seed)
            assembly = self.command(self.driver, merged, api, api, "raw").stdout
            text.write_text(assembly)
            self.assertIn('"nl_nanoisa_assemble_text_save" int string string', assembly)
            self.assertIn('"nl_nanoisa_last_error" string\n', assembly)
            compared = self.command(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, text, "main", "--imports")
            self.assertIn("5 passed, 0 failed", compared.stdout)
            self.command(ROOT / "bin/nanoisa", "asm", text, "-o", module)
            self.command(ROOT / "bin/nano_vm", module)
            c_source, binary = directory / "out.c", directory / "native"
            self.command(ROOT / "bin/nvm2c", module, "-o", c_source)
            self.command("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", c_source, "-ldl", "-o", binary)
            self.command(binary)

    def test_actual_assembly_publication_and_diagnostic_vm_native(self):
        with tempfile.TemporaryDirectory(prefix="nano-assembly-publication-") as tmp:
            directory = Path(tmp)
            provider = ROOT / "modules/nanoisa/nanoisa.nano"
            target = directory / "published.nvm"
            valid = '.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
            merged = directory / "merged.nano"
            merged.write_text(
                'extern fn nl_nanoisa_assemble_text_save(source: string, path: string) -> int\n'
                'extern fn nl_nanoisa_last_error() -> string\n'
                'fn main() -> int { let status: int = (publish ' + json.dumps(valid) + ' ' + json.dumps(str(target)) + ') '
                'assert (== status 0) assert (!= (publish "invalid assembly" ' + json.dumps(str(target)) + ') 0) '
                'assert (> (str_length (diagnostic)) 0) return 0 }\n')
            assembly = self.command(self.driver, merged, provider, provider, "program").stdout
            text, module = directory / "out.nasm", directory / "out.nvm"
            text.write_text(assembly)
            self.command(ROOT / "bin/nanoisa", "asm", text, "-o", module)
            self.command(ROOT / "bin/nano_vm", "--verify-only", module)
            self.command(ROOT / "bin/nano_vm", module)
            first = target.read_bytes()
            self.assertEqual(first[:4], b"NVM\x02")
            self.command(ROOT / "bin/nano_vm", "--verify-only", target)
            self.command(ROOT / "bin/nano_vm", target)
            c_source, binary = directory / "out.c", directory / "native"
            self.command(ROOT / "bin/nvm2c", module, "-o", c_source)
            self.command("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", c_source,
                         ROOT / "bin/nano_aot_runtime.o", "-lm",
                         *(["-Wl,--export-dynamic", "-ldl"] if sys.platform.startswith("linux") else []),
                         "-o", binary)
            self.command(binary)
            self.assertEqual(first, target.read_bytes())

    def test_publication_contract_refuses_wrong_result_and_zero_arg_shapes(self):
        with tempfile.TemporaryDirectory(prefix="nano-assembly-refusal-") as tmp:
            directory = Path(tmp)
            provider = ROOT / "modules/nanoisa/nanoisa.nano"
            merged = directory / "merged.nano"
            declarations = [
                ('extern fn nl_nanoisa_assemble_text_save(s: string, p: string) -> string', '(publish "x" "y")'),
                ('extern fn nl_nanoisa_assemble_text_save(s: string) -> int', '(publish "x")'),
                ('extern fn nl_nanoisa_assemble_text_save(s: int, p: string) -> int', '(publish 1 "y")'),
                ('extern fn nl_nanoisa_last_error() -> int', '(diagnostic)'),
                ('extern fn nl_nanoisa_last_error(p: string) -> string', '(diagnostic "x")'),
                ('extern fn nl_nanoisa_last_error() -> string', '(diagnostic "x")'),
            ]
            for declaration, call in declarations:
                with self.subTest(declaration=declaration, call=call):
                    # My diagnostic declaration belongs to the second source owner.
                    prefix = '\n' + declaration if 'last_error' in declaration else declaration + '\n'
                    merged.write_text(prefix + '\nfn main() -> int { ' + call + ' return 0 }\n')
                    self.assertEqual(self.command(self.driver, merged, provider, provider, "program", expected=1).stdout, "")

    def test_bad_artifact_signatures_operands_and_missing_owner_refuse(self):
        with tempfile.TemporaryDirectory(prefix="nano-artifact-refusal-") as tmp:
            directory = Path(tmp)
            left = self.module(directory, "left", "left")
            right = self.module(directory, "right", "right")
            source = directory / "merged.nano"
            declarations = [
                'extern fn path_basename(path: int) -> string',
                'extern fn path_basename(path: string) -> int',
                'extern fn path_basename() -> string',
                'extern fn path_basename(a: string, b: string) -> string',
            ]
            for declaration in declarations:
                with self.subTest(declaration=declaration):
                    source.write_text(declaration + '\n' + right.read_text() +
                                      'fn main() -> int { (left.base "x") return 0 }\n')
                    result = self.command(self.driver, source, left, right, "program", expected=1)
                    self.assertEqual(result.stdout, "")
            for call in ('(left.base 12)', '(left.base)', '(left.base "x" "y")', '(path_basename "unbound")'):
                with self.subTest(call=call):
                    source.write_text(left.read_text() + right.read_text() +
                                      'fn main() -> int { ' + call + ' return 0 }\n')
                    self.assertEqual(self.command(self.driver, source, left, right, "program", expected=1).stdout, "")
            source.write_text(left.read_text() + right.read_text() +
                              'fn main() -> int { (left.base "x") return 0 }\n')
            self.assertEqual(self.command(self.driver, source, directory / "missing.nano", right,
                                          "program", expected=1).stdout, "")


if __name__ == "__main__":
    unittest.main()
