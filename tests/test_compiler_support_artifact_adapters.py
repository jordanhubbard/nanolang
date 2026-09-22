"""I call the real compiler-support artifact through canonical typed imports."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import test_file_cyclic

ROOT = Path(__file__).resolve().parents[1]


class CompilerSupportArtifactAdapters(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    def setUp(self):
        self.artifacts = Path(tempfile.mkdtemp(prefix="nano-compiler-support-adapters-"))
        print("I retain compiler-support adapter products at", self.artifacts, flush=True)
        self.serial = 0
        self.compiler = shlex.split(os.environ.get("NANO_NATIVE_TEST_CC") or
                                    os.environ.get("CC") or "cc")
        self.flags = shlex.split(os.environ.get("NANO_ARTIFACT_CFLAGS", ""))
        self.links = shlex.split(os.environ.get("NANO_ARTIFACT_LDFLAGS", ""))

    def run_checked(self, args):
        self.serial += 1
        return self.command(f"{self.serial:03d}", list(map(str, args)))

    def native(self, module):
        source, output = self.artifacts / "program.c", self.artifacts / "program"
        self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
        self.run_checked([*self.compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                          *self.flags, source, ROOT / "bin/nano_aot_runtime.o", "-lm",
                          *(["-Wl,--export-dynamic", "-ldl"]
                            if sys.platform.startswith("linux") else []),
                          *self.links, "-o", output])
        return self.run_checked([output])

    def test_real_provider_root_and_abi(self):
        driver_source = self.artifacts / "driver.nano"
        template = (ROOT / "tests/nanoisa/fixtures/artifact_import_driver.nano.txt").read_text()
        marker = '    (mb_add_target 2 "selected" "path_basename" 1)'
        self.assertEqual(template.count(marker), 1)
        driver_source.write_text(template.replace(marker, marker +
            '\n    (mb_add_target 2 "compiler_abi" "nlc_native_array_abi" 0)' +
            '\n    (mb_add_target 2 "compiler_root" "nlc_runtime_root" 1)'))
        driver = self.artifacts / "driver"
        merged = self.artifacts / "merged.nano"
        merged.write_text(
            'extern fn nlc_native_array_abi() -> int\n'
            'extern fn nlc_runtime_root() -> string\n'
            'fn main() -> int { let first: string = (compiler_root) '
            'let second: string = (compiler_root) '
            f'assert (== first {json.dumps(str(ROOT))}) '
            'assert (== first second) (println (compiler_abi)) return 0 }\n')
        provider_source = ROOT / "modules/compiler_support/compiler_support.nano"
        with patch.dict(os.environ, {"NANOLANG_SDK_ROOT": str(ROOT),
                                    "NANO_AS_CAPTURE_HELPER": str(ROOT / "bin/nano_as_capture.so")}):
            self.run_checked([ROOT / "bin/nanoc_c", driver_source, "-o", driver])
            assembly = self.run_checked([driver, merged, provider_source, provider_source, "program"])
            self.assertIn(b'"nlc_native_array_abi" int\n', assembly)
            self.assertIn(b'"nlc_runtime_root" string\n', assembly)
            self.assertEqual(assembly.count(b".import_kind"), 2)
            imports = [shlex.split(line.decode()) for line in assembly.splitlines()
                       if line.startswith(b".import ")]
            self.assertEqual({entry[2]: entry[3:] for entry in imports},
                             {"nlc_native_array_abi": ["int"], "nlc_runtime_root": ["string"]})
            libraries = {Path(entry[1]) for entry in imports}
            before = {str(library): hashlib.sha256(library.read_bytes()).hexdigest()
                      for library in libraries}
            oracle_source, oracle = self.artifacts / "oracle.c", self.artifacts / "oracle"
            oracle_source.write_text('#include <assert.h>\n#include <dlfcn.h>\n'
                '#include <inttypes.h>\n#include <stdio.h>\n'
                'int main(int argc, char **argv) { assert(argc == 2); '
                'void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL); assert(library); '
                'int64_t (*abi)(void) = (int64_t (*)(void))dlsym(library, "nlc_native_array_abi"); '
                'const char *(*root)(void) = (const char *(*)(void))dlsym(library, "nlc_runtime_root"); '
                'assert(abi && root && dlsym(library, "nlc_module_artifact")); '
                'assert(!dlsym(library, "module_builder_verbose")); '
                'assert(!dlsym(library, "module_build")); '
                'assert(!dlsym(library, "nano_native_sdk_prepare")); '
                'printf("%" PRId64 "\\n%s\\n", abi(), root()); '
                'dlclose(library); return 0; }\n')
            self.run_checked([*self.compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                              *self.flags, oracle_source, ROOT / "bin/nano_aot_runtime.o", "-lm",
                              *(["-Wl,--export-dynamic", "-ldl"]
                                if sys.platform.startswith("linux") else []),
                              *self.links, "-o", oracle])
            oracle_results = {}
            for library in sorted(libraries):
                abi, actual_root = self.run_checked([oracle, library]).split(b"\n", 1)
                self.assertEqual(actual_root, str(ROOT).encode() + b"\n")
                oracle_results[str(library)] = abi.decode()
            self.assertEqual(len(set(oracle_results.values())), 1)
            abi_library = next(entry[1] for entry in imports if entry[2] == "nlc_native_array_abi")
            expected = oracle_results[abi_library].encode() + b"\n"
            asm, module = self.artifacts / "input.nasm", self.artifacts / "input.nvm"
            asm.write_bytes(assembly)
            self.run_checked([ROOT / "bin/nanoisa", "asm", asm, "-o", module])
            self.assertEqual(self.run_checked([ROOT / "bin/nano_vm", module]), expected)
            self.assertEqual(self.native(module), expected)
            self.assertEqual({str(library): hashlib.sha256(library.read_bytes()).hexdigest()
                              for library in libraries}, before)
            (self.artifacts / "provider.json").write_text(json.dumps(
                {"imports": imports, "sha256": before, "oracle_abi": oracle_results,
                 "declaration_owners": {"nlc_native_array_abi": 0, "nlc_runtime_root": 1},
                 "owner_sources": {"0": str(provider_source), "1": str(provider_source)}}, indent=2))

    def test_mutable_borrowed_root_is_snapshotted(self):
        source, library = self.artifacts / "provider.c", self.artifacts / "provider.so"
        source.write_text('#include <stdint.h>\n#include <stdio.h>\n'
            'int64_t nlc_native_array_abi(void) { return 47; }\n'
            'const char *nlc_runtime_root(void) { static char text[32]; static int calls; '
            'snprintf(text, sizeof text, "root-%d", ++calls); return text; }\n')
        self.run_checked([*self.compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                          *self.flags, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                          "-fPIC", source, *self.links, "-o", library])
        asm, module = self.artifacts / "input.nasm", self.artifacts / "input.nvm"
        asm.write_text(f'.import {json.dumps(str(library))} "nlc_native_array_abi" int\n'
            '.import_kind 0 artifact\n'
            f'.import {json.dumps(str(library))} "nlc_runtime_root" string\n'
            '.import_kind 1 artifact\n.string first "root-1"\n.entry main\n'
            '.function main 0 1 0 int 1\nCALL_EXTERN 0\nPUSH_I64 47\nI64_EQ\nASSERT\n'
            'CALL_EXTERN 1\nSTORE_LOCAL 0\nCALL_EXTERN 1\nPOP\n'
            'LOAD_LOCAL 0\nPUSH_STR first\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        self.run_checked([ROOT / "bin/nanoisa", "asm", asm, "-o", module])
        self.run_checked([ROOT / "bin/nano_vm", module])
        self.native(module)


if __name__ == "__main__":
    unittest.main()
