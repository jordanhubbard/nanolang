"""I require the real Json artifact contract through both compiler routes."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''from "modules/std/json/json.nano" import parse, free, get, get_index, as_string, as_int, stringify, new_object, new_array, new_string, new_int, object_set, json_array_push, keys, array_size
fn main()->int {
 let obj:Json = (new_object)
 let text:Json = (new_string "owned-child")
 assert (object_set obj "key" text)
 (free text)
 let child:Json = (get obj "key")
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
 let value:Json = (new_int 42)
 assert (json_array_push arr value)
 (free value)
 assert (== (array_size arr) 1)
 let copied:Json = (get_index arr 0)
 (free arr)
 assert (== (as_int copied) 42)
 (free copied)
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
    def command(cls, *args):
        cls.sequence += 1
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180)
        (cls.work / f"command-{cls.sequence}.log").write_text(
            repr(list(map(str,args))) + "\n" + result.stdout + result.stderr)
        if result.returncode:
            raise AssertionError(f"{args}: {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    def test_real_artifact_ownership_vm_and_native(self):
        source = self.work / "ownership.nano"
        source.write_text(SOURCE)
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):
                module = self.work / (compiler.name + ".nvm")
                self.command(compiler, source, "--emit-nvm", "-o", module)
                self.command(ROOT / "bin/nano_vm", "--verify-only", module)
                self.command(ROOT / "bin/nano_vm", module)
                native_c, binary = module.with_suffix(".c"), module.with_suffix(".native")
                self.command(ROOT / "bin/nvm2c", module, "-o", native_c)
                cc = shlex.split(os.environ.get("CC", "clang"))
                self.command(*cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                             "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                             native_c, "-ldl", "-lm", "-o", binary)
                self.command(binary)

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
