"""I require the real Json artifact contract through both compiler routes."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''from "modules/std/json/json.nano" import parse, free, get, get_index, as_string, as_int, as_float, as_bool, stringify, new_object, new_array, new_string, new_int, new_bool, new_null, object_set, json_array_push, keys, array_size, is_null, is_bool, is_number, is_string, is_array, is_object, object_has
fn empty_handle()->Json { return 0 }
shadow empty_handle { assert (== (empty_handle) 0) }
fn main()->int {
 assert (== (empty_handle) 0)
 let obj:Json = (new_object)
 assert (is_object obj)
 assert (== (array_length (keys obj)) 0)
 let text:Json = (new_string "owned-child")
 assert (== text text)
 assert (!= text 0)
 assert (is_string text)
 assert (object_set obj "key" text)
 assert (object_has obj "key")
 assert (not (object_has obj "missing"))
 assert (== (get obj "missing") 0)
 (free text)
 let child:Json = (get obj "key")
 assert (!= child obj)
 let key_names:array<string> = (keys obj)
 let encoded:string = (stringify obj)
 (free obj)
 assert (== (as_string child) "owned-child")
 assert (== (at key_names 0) "key")
 (free child)
 let restored:Json = (parse encoded)
 let extracted:Json = (get restored "key")
 (free restored)
 assert (== (as_string extracted) "owned-child")
 (free extracted)
 let arr:Json = (new_array)
 assert (is_array arr)
 let value:Json = (new_int 42)
 assert (is_number value)
 assert (> (as_float value) 41.0)
 assert (json_array_push arr value)
 (free value)
 assert (== (array_size arr) 1)
 let copied:Json = (get_index arr 0)
 (free arr)
 assert (== (as_int copied) 42)
 (free copied)
 let flag:Json = (new_bool true)
 assert (is_bool flag)
 assert (as_bool flag)
 (free flag)
 let nil:Json = (new_null)
 assert (is_null nil)
 (free nil)
 assert (== (parse "{") 0)
 assert (== (stringify 0) "null")
 (free 0)
 return 0
}
shadow main { assert true }
'''

class JsonArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-json-artifacts-"))
        print("I retain Json artifact evidence at", cls.work, flush=True)
        override = os.environ.get("NANOLANG_JSON_COMPILER")
        cls.compilers = [ROOT / "bin/nano_virt"] + (
            [Path(override)] if override else [ROOT / "bin/nanoc_stage1", ROOT / "bin/nanoc_stage2"])
        cls.sequence = 0

    @classmethod
    def command(cls, *args, success=True, cwd=None):
        cls.sequence += 1
        result = subprocess.run(list(map(str, args)), cwd=cwd or ROOT, capture_output=True,
                                text=True, timeout=180)
        (cls.work / f"command-{cls.sequence}.log").write_text(
            repr(list(map(str,args))) + "\n" + result.stdout + result.stderr)
        if (result.returncode == 0) != success:
            raise AssertionError(f"{args}: {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    def native(self, module):
        native_c, binary = module.with_suffix(".c"), module.with_suffix(".native")
        self.command(ROOT / "bin/nvm2c", module, "-o", native_c)
        cc = shlex.split(os.environ.get("CC", "clang"))
        self.command(*cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                     "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                     native_c, ROOT / "bin/nano_aot_runtime.o",
                     *(["-Wl,--export-dynamic"] if sys.platform.startswith("linux") else []),
                     "-ldl", "-lm", "-o", binary)
        return binary

    def test_real_artifact_ownership_vm_and_native(self):
        source = self.work / "ownership.nano"
        source.write_text(SOURCE)
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):
                module = self.work / (compiler.name + ".nvm")
                self.command(compiler, source, "--emit-nvm", "-o", module)
                self.command(ROOT / "bin/nano_vm", "--verify-only", module)
                self.command(ROOT / "bin/nano_vm", module)
                self.command(self.native(module))

    def test_actual_schema_generator_outputs(self):
        paths = ["src_nano/generated/compiler_schema.nano", "src_nano/generated/compiler_ast.nano",
                 "src_nano/generated/compiler_contracts.nano", "src/generated/compiler_schema.h"]
        baseline = None
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):
                module = self.work / ("schema-" + compiler.name + ".nvm")
                self.command(compiler, ROOT / "scripts/gen_compiler_schema.nano", "--emit-nvm", "-o", module)
                self.command(ROOT / "bin/nano_vm", "--verify-only", module)
                binary = self.native(module)
                for route, command in (("vm", [ROOT / "bin/nano_vm", module]), ("native", [binary])):
                    directory = self.work / ("schema-" + compiler.name + "-" + route)
                    for folder in ("schema", "src_nano/generated", "src/generated"):
                        (directory / folder).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / "schema/compiler_schema.json", directory / "schema/compiler_schema.json")
                    self.command(*command, cwd=directory)
                    outputs = {path: (directory / path).read_bytes() for path in paths}
                    self.assertTrue(all(outputs.values()))
                    if baseline is None:
                        baseline = outputs
                    else:
                        self.assertEqual(outputs, baseline)

    def test_nonzero_opaque_returns_are_rejected(self):
        source = self.work / "bad-return.nano"
        source.write_text('opaque type Handle\nfn invalid()->Handle { return 7 }\n'
                          'shadow invalid { assert true }\n'
                          'fn main()->int { let value:Handle = (invalid) return 0 }\n'
                          'shadow main { assert true }\n')
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):
                output = self.work / "bad-return.nvm"
                output.write_bytes(b"prior output")
                self.command(compiler, source, "--emit-nvm", "-o", output, success=False)
                self.assertEqual(output.read_bytes(), b"prior output")
        for integer in (0, 7):
            with self.subTest(raw_return=integer):
                assembly, module = self.work / "return.nasm", self.work / "return.nvm"
                assembly.write_text('.entry 0\n.function main 0 0 0 int 1\n'
                                    'CALL 1\nPOP\nPUSH_I64 0\nRET\n.end\n'
                                    '.function handle 0 0 0 opaque 1\n'
                                    f'PUSH_I64 {integer}\nRET\n.end\n')
                self.command(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
                self.command(ROOT / "bin/nano_vm", module, success=integer == 0)
                self.command(self.native(module), success=integer == 0)

    def test_native_wrong_import_signatures_preserve_output(self):
        for signature in ('"nl_json_free" void int', '"nl_json_get" opaque opaque int',
                          '"nl_json_new_int" int int', '"nl_json_object_set" int opaque string'):
            with self.subTest(signature=signature):
                assembly, module = self.work / "wrong-import.nasm", self.work / "wrong-import.nvm"
                assembly.write_text('.import "/missing/json.so" ' + signature +
                                    '\n.import_kind 0 artifact\n.entry 0\n.function main 0 0 0 int 1\n'
                                    'PUSH_I64 0\nRET\n.end\n')
                self.command(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
                output = module.with_suffix(".c")
                output.write_bytes(b"prior native output")
                self.command(ROOT / "bin/nvm2c", module, "-o", output, success=False)
                self.assertEqual(output.read_bytes(), b"prior native output")

    def test_nonzero_integer_cannot_reach_opaque_string_provider(self):
        source, module = self.work / "provider.nano", self.work / "provider.nvm"
        source.write_text('from "modules/std/json/json.nano" import as_string\n'
                          'fn main()->int { assert (== (as_string 0) "") return 0 }\n'
                          'shadow main { assert true }\n')
        self.command(ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", module)
        assembly = self.command(ROOT / "bin/nanoisa", "dump", module).stdout
        providers = [shlex.split(line)[1] for line in assembly.splitlines()
                     if line.startswith(".import ") and '"nl_json_as_string"' in line]
        self.assertEqual(len(providers), 1)
        for integer in (0, 7):
            with self.subTest(integer=integer):
                text, raw = self.work / "opaque-value.nasm", self.work / "opaque-value.nvm"
                text.write_text('.import "' + providers[0] + '" "nl_json_as_string" string opaque\n'
                                '.import_kind 0 artifact\n.entry 0\n.function main 0 0 0 int 1\n'
                                f'PUSH_I64 {integer}\nCALL_EXTERN 0\nPOP\nPUSH_I64 0\nRET\n.end\n')
                self.command(ROOT / "bin/nanoisa", "asm", text, "-o", raw)
                self.command(ROOT / "bin/nano_vm", "--verify-only", raw)
                vm = self.command(ROOT / "bin/nano_vm", raw, success=integer == 0)
                native = self.command(self.native(raw), success=integer == 0)
                if integer:
                    self.assertIn("opaque value or zero null", vm.stderr)
                    self.assertIn("nvalue_require_opaque", native.stderr)

    def test_wrong_artifact_abi_preserves_output(self):
        # I exercise my canonical self-hosted boundary directly. C-seed's
        # general FFI accepts user declarations and is a different contract.
        for declaration, call in (
            ("extern fn nl_json_free(value:int)->void", "(nl_json_free 7)"),
            ("extern fn nl_json_get(value:Json,key:int)->Json", "(nl_json_get 0 7)"),
            ("extern fn nl_json_new_int(value:int)->int", "(nl_json_new_int 7)"),
            ("extern fn nl_json_free(value:Json,extra:int)->void", "(nl_json_free 0 1)"),
        ):
            for compiler in self.compilers[1:]:
                with self.subTest(declaration=declaration, compiler=str(compiler)):
                    source = self.work / "wrong-abi.nano"
                    output = self.work / "wrong-abi.nvm"
                    source.write_text("opaque type Json\n" + declaration +
                                      "\nfn main()->int { unsafe { " + call +
                                      " } return 0 }\nshadow main { assert true }\n")
                    output.write_bytes(b"prior artifact")
                    self.sequence += 1
                    command = list(map(str, [compiler, source, "--emit-nvm", "-o", output]))
                    result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                            text=True, timeout=180)
                    (self.work / f"refusal-{self.sequence}.log").write_text(
                        repr(command) + "\n" + result.stdout + result.stderr)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_bytes(), b"prior artifact")
