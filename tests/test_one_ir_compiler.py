"""I require a native compiler built from bytecode to compile a real program."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HOST_RUNTIME = [ROOT / "bin/nano_aot_runtime.o", "-lm",
                *(["-Wl,--export-dynamic", "-ldl"] if sys.platform.startswith("linux") else [])]


class OneIrCompiler(unittest.TestCase):
    def test_incoming_main_record_shapes_execute(self):
        fixtures = {
            "uncalled_record_parameter": (
                ".entry 1\n.function uncalled 1 1 0 struct 1\n"
                "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 1\nAGG_PACK 0 0 0 2\nRET\n.end\n"
                ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
            ),
            "array_in_large_record": (
                ".entry 0\n.function main 0 1 0 int 1\n"
                "PUSH_I64 7\nARR_LITERAL 1 1\n" + "PUSH_I64 0\n" * 10 +
                "AGG_PACK 0 0 0 11\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_GET 0\n"
                "ARR_LEN\nPUSH_I64 1\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n"
            ),
            "initializer_call_root": (
                ".entry main\n.function initialize 0 0 0 void 0\n"
                "PUSH_I64 42\nSTORE_GLOBAL 0\nRET\n.end\n"
                ".function __init__ 0 0 0 void 0\nCALL initialize\nRET\n.end\n"
                ".function main 0 0 0 int 1\nLOAD_GLOBAL 0\nPUSH_I64 42\nEQ\n"
                "ASSERT\nPUSH_I64 0\nRET\n.end\n"
            ),
            "uncalled_wrapper_chain": (
                ".entry main\n.function outer 1 1 0 struct 1\nLOAD_LOCAL 0\nTAIL_CALL inner\n.end\n"
                ".function inner 1 1 0 struct 1\nLOAD_LOCAL 0\nCALL unresolved\nRET\n.end\n"
                ".function unresolved 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_GET 0\n"
                "PUSH_I64 1\nAGG_PACK 0 0 0 2\nRET\n.end\n"
                ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
            ),
        }
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for name, assembly_text in fixtures.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="nano-main-shapes-") as tmp:
                work = Path(tmp)
                assembly, module, source, binary = (work / item for item in
                                                    ("input.nasm", "input.nvm", "input.c", "program"))
                assembly.write_text(assembly_text)
                self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                self.run_checked([binary])

    def run_checked(self, args, timeout=180):
        env = dict(os.environ, NANO_MODULE_PATH=str(ROOT / "modules"))
        process = subprocess.Popen([str(arg) for arg in args], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=10)
            self.fail(f"I exceeded {timeout}s running {args[0]}")
        self.assertEqual(process.returncode, 0,
                         f"I failed {args[0]}\n" + (stdout + stderr).decode(errors="replace")[-6000:])
        return stdout

    def test_declared_empty_array_returns_reach_native(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for element_type, value in (("int", "42"), ("string", '"answer"'),
                                    ("Point", "Point { x: 42 }")):
            with self.subTest(element_type=element_type), tempfile.TemporaryDirectory(prefix="nano-empty-return-") as tmp:
                work = Path(tmp)
                source, module, native_c, binary = (work / name for name in ("input.nano", "input.nvm", "input.c", "input"))
                source.write_text(
                    "struct Point { x: int }\n"
                    f"fn make(empty: bool) -> array<{element_type}> {{\n"
                    "  if empty { return [] }\n"
                    f"  return [{value}]\n}}\n"
                    "shadow make { assert (== (array_length (make true)) 0) }\n"
                    "fn main() -> int {\n"
                    "  assert (== (array_length (make true)) 0)\n"
                    "  assert (== (array_length (make false)) 1)\n"
                    "  return 0\n}\n"
                    "shadow main { assert (== (main) 0) }\n"
                )
                self.run_checked([ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", module])
                self.run_checked([ROOT / "bin/nvm2c", module, "-o", native_c])
                self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", native_c, "-o", binary])
                self.run_checked([binary])

    def test_compiler_bytecode_to_native_to_program(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        with tempfile.TemporaryDirectory(prefix="nano-one-ir-compiler-") as tmp:
            work = Path(tmp)
            module, source, compiler = work / "compiler.nvm", work / "compiler.c", work / "compiler"
            self.run_checked([ROOT / "bin/nano_virt", ROOT / "src_nano/nanoc_v06.nano",
                              "--emit-nvm", "--strip-debug", "-o", module], timeout=600)
            self.assertGreater(module.stat().st_size, 0)
            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source], timeout=240)
            self.assertNotIn("nano_vm", source.read_text())
            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O0",
                              source, "-o", compiler, *HOST_RUNTIME], timeout=240)
            help_output = self.run_checked([compiler, "--help"], timeout=10)
            self.assertIn(b"Compiler", help_output)
            hello = work / "hello"
            self.run_checked([compiler, ROOT / "examples/language/nl_hello.nano", "-o", hello])
            self.assertEqual(self.run_checked([hello], timeout=10), b"Hello from NanoLang!\n")

    def test_real_std_artifact_uses_host_runtime(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        with tempfile.TemporaryDirectory(prefix="nano-aot-host-runtime-") as tmp:
            work = Path(tmp)
            directory = work / "files"
            directory.mkdir()
            (directory / "one").write_text("payload")
            library = work / ("std.dylib" if sys.platform == "darwin" else "std.so")
            shared_flags = ["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
            self.run_checked([cc, "-std=c11", "-fPIC", *shared_flags,
                              ROOT / "modules/std/fs.c", ROOT / "modules/std/process.c", "-o", library])
            assembly, module, source = (work / name for name in ("input.nasm", "input.nvm", "input.c"))
            assembly.write_text(
                f'.import "{library}" "path_canonical" string string\n'
                f'.import "{library}" "fs_walkdir" array string\n'
                '.import_kind 0 artifact\n.import_kind 1 artifact\n'
                f'.string root "{directory}"\n.entry main\n'
                '.function main 0 1 0 int 1\n'
                'PUSH_STR root\nCALL_EXTERN 0\nCALL_EXTERN 1\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\n'
                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nSTR_LEN\nRET\n.end\n')
            self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
            missing = work / "missing-runtime"
            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source,
                              "-o", missing, *(["-ldl"] if sys.platform.startswith("linux") else [])])
            failed = subprocess.run([missing], capture_output=True, timeout=10)
            self.assertLess(failed.returncode, 0, "I require the foreign artifact's host ABI")
            generated = source.read_text().replace("int main(", "int generated_main(")
            source.write_text(generated + '\n#include "runtime/gc.h"\n'
                              'int main(void) { gc_init(); size_t before = gc_get_stats().num_objects; '
                              'int result = generated_main(0, NULL); '
                              f'if (result != {len(str((directory / "one").resolve()))} || gc_get_stats().num_objects != before) return 1; '
                              'gc_shutdown(); return 0; }\n')
            binary = work / "with-runtime"
            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", ROOT / "src",
                              source, *HOST_RUNTIME, "-o", binary])
            self.run_checked([binary])

    def test_projected_optional_arguments(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        header, functions = fixture.split(".function main", 1)
        producers = ".function choose" + functions.split(".function choose", 1)[1]
        for reverse in (False, True):
            for tail in (False, True):
                for incompatible in (False, True):
                    with self.subTest(reverse=reverse, tail=tail, incompatible=incompatible), tempfile.TemporaryDirectory(prefix="nano-projected-arg-") as tmp:
                        work = Path(tmp)
                        assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                        main = (
                            ".function main 0 0 0 int 1\n"
                            "PUSH_STR text\nCALL absent\nBOOL_NOT\nASSERT\n"
                            "LOAD_GLOBAL 0\nCALL absent\nASSERT\n"
                            "CALL present\nCALL project\nBOOL_NOT\nASSERT\n"
                            "PUSH_BOOL 1\nCALL choose\nCALL project\nBOOL_NOT\nASSERT\n"
                            "CALL missing\nCALL project\nASSERT\nPUSH_I64 0\nRET\n.end\n"
                        )
                        helpers = (
                            ".function absent 1 1 0 bool 1\nLOAD_LOCAL 0\nTYPE_CHECK 0\nRET\n.end\n"
                            ".function project 1 1 0 bool 1\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n"
                            + ("TAIL_CALL absent\n" if tail else "CALL absent\nRET\n") + ".end\n"
                        )
                        text = header + main + helpers + producers
                        if incompatible:
                            text = text.replace(".function present 0 0 0 struct 1\n  PUSH_STR text",
                                                ".function present 0 0 0 struct 1\n  PUSH_I64 42")
                        if reverse:
                            prefix, *blocks = text.split(".function ")
                            text = prefix + "".join(".function " + block for block in reversed(blocks))
                        assembly.write_text(text)
                        self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                        if incompatible:
                            result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(b"shape", result.stderr)
                        else:
                            self.run_checked([ROOT / "bin/nano_vm", module])
                            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                            self.run_checked([binary])

    def test_record_local_storage_joins(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        header, functions = fixture.split(".function main", 1)
        producers = ".function choose" + functions.split(".function choose", 1)[1]
        for reverse in (False, True):
            for branch in (None, False, True):
                for incompatible in (False, True):
                    with self.subTest(reverse=reverse, branch=branch, incompatible=incompatible), tempfile.TemporaryDirectory(prefix="nano-local-join-") as tmp:
                        work = Path(tmp)
                        assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                        present = "CALL present"
                        missing = "PUSH_BOOL 0\nCALL choose"
                        if incompatible:
                            present = "PUSH_I64 42\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1"
                        first, second = (missing, present) if reverse else (present, missing)
                        body = first + "\nSTORE_LOCAL 0\n"
                        if branch is not None:
                            body += f"PUSH_BOOL {int(branch)}\nJMP_FALSE done\n"
                        body += second + "\nSTORE_LOCAL 0\ndone:\n"
                        body += "LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\n"
                        ends_present = reverse if branch is not False else not reverse
                        body += "PUSH_STR text\nEQ\nASSERT\n" if ends_present else "TYPE_CHECK 0\nASSERT\n"
                        # A destination join must not rewrite the producer's string.
                        body += "CALL present\nAGG_GET 0\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n"
                        main = ".function main 0 1 0 int 1\n" + body + ".end\n"
                        assembly.write_text(header + (producers + main if reverse else main + producers))
                        self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                        if incompatible:
                            result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(b"shape", result.stderr)
                        else:
                            self.run_checked([ROOT / "bin/nano_vm", module])
                            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                            self.run_checked([binary])

    def test_branch_record_fields_survive_later_assignments(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for taken in (False, True):
                with self.subTest(reverse=reverse, taken=taken), tempfile.TemporaryDirectory(prefix="nano-branch-record-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = (
                        ".function main 0 1 0 int 1\nCALL present\nSTORE_LOCAL 0\n"
                        f"PUSH_BOOL {int(taken)}\nJMP_FALSE present_branch\n"
                        "CALL missing\nSTORE_LOCAL 0\nJMP joined\npresent_branch:\n"
                        "CALL present\nSTORE_LOCAL 0\njoined:\nLOAD_LOCAL 0\nCALL absent\n"
                        f"PUSH_BOOL {int(taken)}\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n"
                    )
                    helpers = (
                        ".function present 0 0 0 struct 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nRET\n.end\n"
                        ".function missing 0 0 0 struct 1\nLOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n"
                        ".function absent 1 1 0 bool 1\nLOAD_LOCAL 0\nAGG_GET 0\nTYPE_CHECK 0\nRET\n.end\n"
                    )
                    assembly.write_text('.string text "present"\n.entry main\n' + (helpers + main if reverse else main + helpers))
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nano_vm", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])

    def test_projected_record_arguments(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        header, functions = fixture.split(".function main", 1)
        producers = ".function choose" + functions.split(".function choose", 1)[1]
        for reverse in (False, True):
            for tail in (False, True):
                with self.subTest(reverse=reverse, tail=tail), tempfile.TemporaryDirectory(prefix="nano-projected-record-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = (
                        ".function main 0 0 0 int 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nCALL absent\nBOOL_NOT\nASSERT\n"
                        "PUSH_BOOL 1\nCALL choose\nCALL project\nBOOL_NOT\nASSERT\n"
                        "PUSH_BOOL 0\nCALL choose\nCALL project\nASSERT\nPUSH_I64 0\nRET\n.end\n"
                    )
                    helpers = (
                        ".function absent 1 1 0 bool 1\nLOAD_LOCAL 0\nAGG_GET 0\nTYPE_CHECK 0\nRET\n.end\n"
                        ".function project 1 1 0 bool 1\nLOAD_LOCAL 0\nAGG_GET 0\n"
                        + ("TAIL_CALL absent\n" if tail else "CALL absent\nRET\n") + ".end\n"
                    )
                    text = header + main + helpers + producers
                    if reverse:
                        prefix, *blocks = text.split(".function ")
                        text = prefix + "".join(".function " + block for block in reversed(blocks))
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nano_vm", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])

                    # A missing value still carries its map's exact payload
                    # type; inferred string storage must not admit integers.
                    incompatible = text.replace("LOAD_GLOBAL 0", "HM_NEW 5 1\nPUSH_STR text\nHM_GET")
                    assembly.write_text(incompatible)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b"shape", result.stderr)

    def test_declared_parameter_resolves_unused_packed_field(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for tag in ("int", "bool", "string", "struct"):
            for reverse in (False, True):
                with self.subTest(tag=tag, reverse=reverse), tempfile.TemporaryDirectory(prefix="nano-declared-field-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                    helper = ".function pack 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n"
                    text = ".entry main\n" + (helper + main if reverse else main + helper)
                    text += f".parameters {0 if reverse else 1} {tag}\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    # Keep pack uncalled in NanoISA, then exercise its declared
                    # ABI from a C harness without supplying inference facts.
                    argument, check = {
                        "int": ("42", "r.k[0] == 0 && r.f[0] == 42"),
                        "bool": ("1", "r.k[0] == 9 && r.f[0] == 1"),
                        "string": ('"hello"', 'r.k[0] == 1 && strcmp(r.s[0], "hello") == 0'),
                        "struct": ("(nrec_t){0}", "r.k[0] == 4 && r.rec[0] && r.rec[0]->n == 0"),
                    }[tag]
                    generated = source.read_text().replace("int main(", "int generated_main(")
                    source.write_text(generated + f"\nint main(void) {{ nrec_t r = nl_pack({argument}); return !(r.n == 1 && {check}); }}\n")
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])
                    untyped = text.split(".parameters", 1)[0]
                    assembly.write_text(untyped)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    self.assertNotIn("nl_pack", source.read_text())
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])
                    # A required unresolved layout still fails; omission only
                    # applies when no VM execution root reaches the function.
                    assembly.write_text(untyped.replace(".entry main", ".entry pack"))
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b"cannot resolve AGG_PACK field 0", result.stderr)

    def test_declared_parameters_keep_observed_tags(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for tag in ("int", "bool", "string"):
            for reverse in (False, True):
                with self.subTest(tag=tag, reverse=reverse), tempfile.TemporaryDirectory(prefix="nano-declared-tagged-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = ".function main 0 0 0 int 1\nLOAD_GLOBAL 0\nCALL absent\nASSERT\nPUSH_I64 0\nRET\n.end\n"
                    helper = ".function absent 1 1 0 bool 1\nLOAD_LOCAL 0\nTYPE_CHECK 0\nRET\n.end\n"
                    text = ".entry main\n" + (helper + main if reverse else main + helper)
                    text += f".parameters {0 if reverse else 1} {tag}\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nano_vm", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])

    def test_nominal_constructor_scalar_evidence(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for case in ("int", "bool", "string", "type", "variant", "width", "conflict", "alias_conflict"):
                with self.subTest(reverse=reverse, case=case), tempfile.TemporaryDirectory(prefix="nano-nominal-field-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    value = "PUSH_BOOL 1" if case == "bool" else "PUSH_STR text" if case == "string" else "PUSH_I64 42"
                    seed_pack = "0 1 0 1" if case == "type" else "1 0 1 1" if case == "variant" else "0 0 0 2" if case == "width" else "0 0 0 1"
                    copy_pack = "1 0 0 1" if case == "variant" else "0 0 0 1"
                    main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                    seed = f".function seed 0 0 0 struct 1\n{value}\n" + ("PUSH_I64 7\n" if case == "width" else "") + f"AGG_PACK {seed_pack}\nRET\n.end\n"
                    copy = f".function copy 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_PACK {copy_pack}\nRET\n.end\n"
                    other = ".function other 0 0 0 struct 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nRET\n.end\n" if case == "conflict" else ""
                    if case == "alias_conflict":
                        copy = copy.replace("RET\n", "POP\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_PACK 0 1 0 1\nRET\n")
                        other = ".function other 0 0 0 struct 1\nPUSH_STR text\nAGG_PACK 0 1 0 1\nRET\n.end\n"
                    blocks = [main, seed, copy] + ([other] if other else [])
                    if reverse:
                        blocks.reverse()
                    text = '.types 2 0 1\n.string text "hello"\n.entry main\n' + "".join(blocks)
                    text += f".parameters {blocks.index(copy)} struct\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    if case not in ("int", "bool", "string"):
                        if case != "alias_conflict":
                            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                            self.assertNotIn("nl_copy", source.read_text())
                            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                            self.run_checked([binary])
                            assembly.write_text(text.replace(".entry main", ".entry copy"))
                            self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                        result = subprocess.run([ROOT / "bin/nvm2c", module, "-o", source], capture_output=True, timeout=30)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(b"conflicting nominal scalar field evidence" if case == "alias_conflict" else b"cannot resolve AGG_PACK field 0", result.stderr)
                        continue
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    check = "r.k[0] == 9 && r.f[0] == 1" if case == "bool" else 'r.k[0] == 1 && strcmp(r.s[0], "hello") == 0' if case == "string" else "r.k[0] == 0 && r.f[0] == 42"
                    generated = source.read_text().replace("int main(", "int generated_main(")
                    source.write_text(generated + f"\nint main(void) {{ nrec_t r = nl_copy(nl_seed()); return !(r.n == 1 && {check}); }}\n")
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])

    def test_projected_array_length_uses_runtime_storage_tag(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for case in ("int", "bool", "string", "record", "empty", "bad_tag", "bad_width", "null", "record_get", "record_set"):
                with self.subTest(reverse=reverse, case=case), tempfile.TemporaryDirectory(prefix="nano-projected-length-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                    helper = ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nARR_LEN\nRET\n.end\n"
                    text = ".entry main\n" + (helper + main if reverse else main + helper)
                    text += f".parameters {0 if reverse else 1} struct\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    storage, member, tag = ("nsarr_s", "sa", 5) if case == "string" else ("nrarr_s", "ra", 6) if case.startswith("record") else ("narr_s", "a", 10 if case == "bool" else 3)
                    initial = 0 if case == "empty" else 3
                    setup = f"{storage} a = {{.len = {initial}}}; nrec_t r = {{.n = 1}}; r.k[0] = {tag}; r.{member}[0] = &a;"
                    if case == "bad_tag":
                        setup += "r.k[0] = 0;"
                    elif case == "bad_width":
                        setup += "r.n = 0;"
                    elif case == "null":
                        setup += "r.a[0] = NULL;"
                    body = setup + f"if (nl_length(r) != {initial}) return 1; a.len = 5; return nl_length(r) != 5 || r.k[0] != {tag};"
                    if case in ("record_get", "record_set"):
                        operation = "nvalue_array_get((nmap_value){7, 6, (char *)&a}, 0)" if case == "record_get" else "nvalue_array_set((nmap_value){7, 6, (char *)&a}, 0, (nmap_value){1, 0, NULL})"
                        body = setup + f"if (nl_length(r) != {initial}) return 1; (void){operation}; return 0;"
                    source.write_text(source.read_text().replace("int main(", "int generated_main(") + f"\nint main(void) {{ {body} }}\n")
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    if case in ("bad_tag", "bad_width", "null", "record_get", "record_set"):
                        result = subprocess.run([binary], capture_output=True, timeout=10)
                        self.assertLess(result.returncode, 0, "I trap an invalid projected array")
                    else:
                        self.run_checked([binary])

    def test_nested_record_consumer_constrains_projection(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for case in ("int", "bool", "string", "outer_tag", "inner_tag", "null", "outer_bounds", "inner_bounds"):
                with self.subTest(reverse=reverse, case=case), tempfile.TemporaryDirectory(prefix="nano-nested-consumer-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    result_tag = case if case in ("int", "bool", "string") else "int"
                    main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                    helper = f".function read 1 1 0 {result_tag} 1\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nRET\n.end\n"
                    text = ".entry main\n" + (helper + main if reverse else main + helper)
                    text += f".parameters {0 if reverse else 1} struct\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    setup = "nrec_t inner = {.n = 1}, outer = {.n = 1}; outer.k[0] = 4; outer.rec[0] = &inner; inner.f[0] = 42;"
                    check = "nl_read(outer) == 42"
                    if case == "bool":
                        setup += "inner.k[0] = 9; inner.f[0] = 1;"
                        check = "nl_read(outer) == 1"
                    elif case == "string":
                        setup += 'inner.k[0] = 1; inner.s[0] = "hello";'
                        check = 'strcmp(nl_read(outer), "hello") == 0'
                    elif case == "outer_tag":
                        setup += "outer.k[0] = 0;"
                    elif case == "inner_tag":
                        setup += 'inner.k[0] = 1; inner.s[0] = "bad";'
                    elif case == "null":
                        setup += "outer.rec[0] = NULL;"
                    elif case == "outer_bounds":
                        setup += "outer.n = 0;"
                    elif case == "inner_bounds":
                        setup += "inner.n = 0;"
                    generated = source.read_text().replace("int main(", "int generated_main(")
                    source.write_text(generated + f"\nint main(void) {{ {setup} return !({check}); }}\n")
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    if case in ("int", "bool", "string"):
                        self.run_checked([binary])
                    else:
                        result = subprocess.run([binary], capture_output=True, timeout=10)
                        self.assertLess(result.returncode, 0, "I trap invalid nested record storage")

    def test_string_consumers_constrain_projected_local(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        operations = {
            "length": ("LOAD_LOCAL 1\nSTR_LEN", "int", "nl_use(r) == 5"),
            "concat_right": ("LOAD_LOCAL 1\nPUSH_STR bang\nSTR_CONCAT", "string", 'strcmp(nl_use(r), "hello!") == 0'),
            "concat_left": ("PUSH_STR bang\nLOAD_LOCAL 1\nSTR_CONCAT", "string", 'strcmp(nl_use(r), "!hello") == 0'),
            "substring": ("LOAD_LOCAL 1\nPUSH_I64 1\nPUSH_I64 3\nSTR_SUBSTR", "string", 'strcmp(nl_use(r), "ell") == 0'),
            "starts": ("LOAD_LOCAL 1\nPUSH_STR prefix\nSTR_STARTS_WITH", "bool", "nl_use(r) == 1"),
            "ends": ("LOAD_LOCAL 1\nPUSH_STR suffix\nSTR_ENDS_WITH", "bool", "nl_use(r) == 1"),
            "contains": ("LOAD_LOCAL 1\nPUSH_STR middle\nSTR_CONTAINS", "bool", "nl_use(r) == 1"),
            "char": ("LOAD_LOCAL 1\nPUSH_I64 0\nSTR_CHAR_AT", "int", "nl_use(r) == 104"),
        }
        for reverse in (False, True):
            for name, (operation, result_tag, check) in operations.items():
                with self.subTest(reverse=reverse, operation=name), tempfile.TemporaryDirectory(prefix="nano-string-consumer-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                    helper = f".function use 1 2 0 {result_tag} 1\nLOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\n{operation}\nRET\n.end\n"
                    text = '.string bang "!"\n.string prefix "he"\n.string suffix "lo"\n.string middle "ell"\n.entry main\n' + (helper + main if reverse else main + helper)
                    text += f".parameters {0 if reverse else 1} struct\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    generated = source.read_text().replace("int main(", "int generated_main(")
                    source.write_text(generated + f'\nint main(int argc, char **argv) {{ (void)argv; nrec_t r = {{.n = 1}}; r.k[0] = argc > 1 ? 0 : 1; r.s[0] = "hello"; return !({check}); }}\n')
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])
                    rejected = subprocess.run([binary, "bad-tag"], capture_output=True, timeout=10)
                    self.assertLess(rejected.returncode, 0, "I trap a non-string field before its consumer")

    def test_projected_array_reads_constrain_container(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for tag in ("int", "bool", "string", "record"):
                with self.subTest(reverse=reverse, tag=tag), tempfile.TemporaryDirectory(prefix="nano-array-consumer-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    result_tag = "int" if tag == "record" else tag
                    main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                    helper = f".function read 1 1 0 {result_tag} 1\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_GET\n" + ("AGG_GET 0\n" if tag == "record" else "") + "RET\n.end\n"
                    text = ".entry main\n" + (helper + main if reverse else main + helper)
                    text += f".parameters {0 if reverse else 1} struct\n"
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    if tag == "string":
                        setup = 'const char *data[] = {"hello"}; nsarr_s a = {.data = data, .len = 1}; r.k[0] = 5; r.sa[0] = &a;'
                        check = 'strcmp(nl_read(r), "hello") == 0'
                    elif tag == "record":
                        setup = "nrarr_s a = {.len = 1}; a.data[0].n = 1; a.data[0].f[0] = 42; r.k[0] = 6; r.ra[0] = &a;"
                        check = "nl_read(r) == 42"
                    else:
                        value = 1 if tag == "bool" else 42
                        setup = f"int64_t data[] = {{{value}}}; narr_s a = {{.data = data, .len = 1}}; r.k[0] = {10 if tag == 'bool' else 3}; r.a[0] = &a;"
                        check = f"nl_read(r) == {value}"
                    generated = source.read_text().replace("int main(", "int generated_main(")
                    wrong_storage = 3 if tag in ("bool", "string") else 10 if tag == "int" else 5
                    source.write_text(generated + f'\nint main(int argc, char **argv) {{ nrec_t r = {{.n = 1}}; {setup} if (argc > 1) {{ if (argv[1][0] == \'t\') r.k[0] = 0; else if (argv[1][0] == \'s\') r.k[0] = {wrong_storage}; else a.len = 0; }} return !({check}); }}\n')
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])
                    for invalid in ("tag", "storage", "bounds"):
                        result = subprocess.run([binary, invalid], capture_output=True, timeout=10)
                        self.assertLess(result.returncode, 0, "I retain projected array tag and bounds checks")

    def test_generic_array_parameters_keep_runtime_tags(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for tail in (False, True):
                for tag in ("int", "bool", "string"):
                    with self.subTest(reverse=reverse, tail=tail, tag=tag), tempfile.TemporaryDirectory(prefix="nano-generic-array-param-") as tmp:
                        work = Path(tmp)
                        assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                        element_tag = {"int": 1, "bool": 4, "string": 5}[tag]
                        blocks = [
                            ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",
                            f".function probe 1 2 0 bool 1\nLOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nTYPE_CHECK {element_tag}\nRET\n.end\n",
                            ".function forward 1 1 0 bool 1\nLOAD_LOCAL 0\n" + ("TAIL_CALL probe\n" if tail else "CALL probe\nRET\n") + ".end\n",
                            ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nARR_LEN\nRET\n.end\n",
                        ]
                        if reverse:
                            blocks.reverse()
                        text = ".entry main\n" + "".join(blocks)
                        text += "".join(f".parameters {i} array\n" for i, block in enumerate(blocks) if not block.startswith(".function main"))
                        assembly.write_text(text)
                        self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                        self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                        storage_kind = {"int": 3, "bool": 10, "string": 5}[tag]
                        setup = 'const char *data[] = {"one", "two"}; nsarr_s a = {.data = data, .len = 2};' if tag == "string" else "int64_t data[] = {1, 0}; narr_s a = {.data = data, .len = 2};"
                        body = setup + f"nmap_value value = {{7, {storage_kind}, (char *)&a}};"
                        body += "if (argc > 1) { if (argv[1][0] == 't') value.kind = 1; else if (argv[1][0] == 'n') value.text = NULL; else value.integer = 99; }"
                        body += f"if (nl_length(value) != 2 || !nl_forward(value)) return 1; a.len = 1; return nl_length(value) != 1 || value.integer != {storage_kind};"
                        generated = source.read_text().replace("int main(", "int generated_main(")
                        source.write_text(generated + f"\nint main(int argc, char **argv) {{ {body} }}\n")
                        self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                        self.run_checked([binary])
                        for invalid in ("tag", "null", "storage"):
                            rejected = subprocess.run([binary, invalid], capture_output=True, timeout=10)
                            self.assertLess(rejected.returncode, 0, "I trap an invalid generic array descriptor")

    def test_unconstrained_scalar_projection_preserves_tags(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            for local in (False, True):
                with self.subTest(reverse=reverse, local=local), tempfile.TemporaryDirectory(prefix="nano-projected-tag-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    main = '.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
                    helper = '.function probe 1 2 0 bool 1\nLOAD_LOCAL 0\nAGG_GET 0\n'
                    if local:
                        helper += 'STORE_LOCAL 1\nLOAD_LOCAL 1\n'
                    helper += 'PUSH_I64 24\nEQ\nRET\n.end\n'
                    assembly.write_text('.entry main\n' + (helper + main if reverse else main + helper) +
                                        f'.parameters {0 if reverse else 1} struct\n')
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    generated = source.read_text().replace("int main(", "int generated_main(")
                    source.write_text(generated + '''
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int which = atoi(argv[1]);
    nrec_t record = {.n = 1}; record.f[0] = 24; record.s[0] = "24";
    const uint8_t kinds[] = {0, 9, 1, 8, 8, 8, 8, 2, 4, 7, 8, 1};
    const uint8_t tags[] = {0, 0, 0, 1, 4, 5, 0, 0, 0, 0, 2, 0};
    record.k[0] = kinds[which]; record.vk[0] = tags[which];
    if (which == 1 || which == 4) record.f[0] = 1;
    if (which == 11) record.s[0] = NULL;
    return nl_probe(record) != (which == 0 || which == 3);
}
''')
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    for which in range(7):
                        self.run_checked([binary, str(which)])
                    for which in range(7, 12):
                        rejected = subprocess.run([binary, str(which)], capture_output=True, timeout=10)
                        self.assertLess(rejected.returncode, 0, "I reject unsupported projected storage")

    def test_record_temporaries_have_scoped_heap_storage(self):
        import resource
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        def bounded_stack():
            _, hard = resource.getrlimit(resource.RLIMIT_STACK)
            resource.setrlimit(resource.RLIMIT_STACK, (2 * 1024 * 1024, hard))
        for mode in ("ordinary", "self_tail", "cross_tail"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="nano-record-frame-") as tmp:
                work = Path(tmp)
                assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                recursive = "CALL walk\nRET\n" if mode == "ordinary" else "TAIL_CALL " + ("walk" if mode == "self_tail" else "forward") + "\n"
                assembly.write_text(
                    '.entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
                    '.function wide 0 0 0 struct 1\n' + 'PUSH_I64 0\n' * 75 + 'AGG_PACK 0 0 0 75\nRET\n.end\n'
                    '.function walk 2 2 0 struct 1\n' + 'LOAD_LOCAL 0\nPOP\n' * 300 +
                    'LOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_FALSE recurse\nLOAD_LOCAL 0\nRET\n'
                    'recurse:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\n' + recursive + '.end\n'
                    '.function forward 2 2 0 struct 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nTAIL_CALL walk\n.end\n'
                    '.parameters 2 struct int\n.parameters 3 struct int\n')
                self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                generated = source.read_text().replace("int main(", "int generated_main(")
                source.write_text('''#include <stdlib.h>
static size_t live, peak;
static int fail_allocation;
static void *tracked_calloc(size_t n, size_t size) {
    if (fail_allocation) return NULL;
    void *p = calloc(n, size); if (p) { ++live; if (live > peak) peak = live; } return p;
}
static void tracked_free(void *p) { if (p) { if (!live) abort(); --live; } free(p); }
#define calloc tracked_calloc
#define free tracked_free
''' + generated + '''
int main(int argc, char **argv) {
    (void)argv; fail_allocation = argc > 1;
    nrec_t input = {.n = 75}; input.f[0] = 24;
    for (int i = 0; i < 10; ++i) {
        nrec_t result = nl_walk(input, 12);
        if (result.n != 75 || result.f[0] != 24 || live) return 1;
    }
''' + ('if (peak != 1) return 2;\n' if mode == "self_tail" else 'if (peak < 13) return 2;\n') + 'return 0;\n}\n')
                self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O0", source, "-o", binary])
                result = subprocess.run([binary], capture_output=True, timeout=30, preexec_fn=bounded_stack)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                failed = subprocess.run([binary, "fail"], capture_output=True, timeout=10)
                self.assertLess(failed.returncode, 0, "I fail closed when a record frame cannot be allocated")

    def test_string_array_growth_owns_buffers_and_preserves_aliases(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        with tempfile.TemporaryDirectory(prefix="nano-string-growth-") as tmp:
            work = Path(tmp)
            assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
            assembly.write_text('.string text "seed"\n.entry main\n.function main 0 1 0 int 1\n'
                                'PUSH_STR text\nARR_LITERAL 5 1\nSTORE_LOCAL 0\n'
                                'LOAD_LOCAL 0\nPUSH_STR text\nARR_PUSH\nPOP\n'
                                'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nSTR_LEN\nRET\n.end\n')
            self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
            generated = source.read_text().replace("int main(", "int generated_main(")
            source.write_text('''#include <stdlib.h>
static size_t live, calls, fail_at;
static void *tracked_malloc(size_t n) { if (++calls == fail_at) return NULL; void *p = malloc(n); if (p) ++live; return p; }
static void *tracked_calloc(size_t n, size_t s) { if (++calls == fail_at) return NULL; void *p = calloc(n, s); if (p) ++live; return p; }
static void *tracked_realloc(void *p, size_t n) { if (++calls == fail_at) return NULL; int fresh = p == NULL; void *q = realloc(p, n); if (q && fresh) ++live; return q; }
static void tracked_free(void *p) { if (p) { if (!live) abort(); --live; } free(p); }
#define malloc tracked_malloc
#define calloc tracked_calloc
#define realloc tracked_realloc
#define free tracked_free
''' + generated + '''
int main(int argc, char **argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    if (mode >= 1 && mode <= 5) fail_at = (size_t)mode;
    nsarr_t a = nsarr_new(), alias = a;
    for (size_t i = 0; i < 70000; ++i) nsarr_push(a, i % 2 ? "odd" : "even");
    if (alias->len != 70000 || strcmp(nsarr_get(alias, 0), "even") || strcmp(nsarr_get(alias, 69999), "odd")) return 1;
    if (calls > 25) return 2;
    if (mode == 10 || mode == 11) fail_at = calls + (size_t)(mode - 9);
    const char *escaped = nsarr_copy_string("kept");
    nsarr_push(a, escaped); a->data[70000] = "replaced";
    if (strcmp(escaped, "kept")) return 3;
    const char **elements = malloc(70000 * sizeof *elements);
    if (!elements) return 4;
    for (size_t i = 0; i < 70000; ++i) elements[i] = "literal";
    nsarr_t literal = nsarr_lit(elements, 70000); free(elements);
    if (literal->len != 70000 || strcmp(nsarr_get(literal, 69999), "literal")) return 5;
    const char *borrowed_data[] = {"original"};
    nsarr_s borrowed = {.data = borrowed_data, .len = 1}; nsarr_t borrowed_alias = &borrowed;
    if (mode == 6 || mode == 12) fail_at = calls + (mode == 6 ? 1 : 2);
    nsarr_push(&borrowed, "added");
    if (borrowed_alias->len != 2 || strcmp(borrowed_data[0], "original") || strcmp(borrowed.data[1], "added")) return 6;
    if (mode == 7) { a->len = SIZE_MAX; nsarr_push(a, "overflow"); }
    if (mode == 8) nsarr_reserve(a, SIZE_MAX / sizeof *a->data + 1);
    if (mode == 9) { a->data = NULL; nsarr_push(a, "invalid"); }
    nsarr_release_owned(); if (live) return 7;
    nsarr_release_owned(); if (live) return 8;
    if (generated_main() != 4 || live) return 9;
    return 0;
}
''')
            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O0", source, "-o", binary])
            self.run_checked([binary])
            for mode in range(1, 13):
                failed = subprocess.run([binary, str(mode)], capture_output=True, timeout=10)
                self.assertLess(failed.returncode, 0, "I trap allocation, size and storage failures")

    def test_owned_strings_preserve_escaped_values_and_cleanup(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        with tempfile.TemporaryDirectory(prefix="nano-owned-strings-") as tmp:
            work = Path(tmp)
            assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
            assembly.write_text('.string text "x"\n.entry main\n.function main 0 0 0 int 1\n'
                                'PUSH_I64 42\nCAST_STRING\nPOP\nPUSH_STR text\nPUSH_STR text\nSTR_CONCAT\n'
                                'PUSH_I64 0\nPUSH_I64 1\nSTR_SUBSTR\nSTR_LEN\nRET\n.end\n')
            self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
            self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
            generated = source.read_text().replace("int main(", "int generated_main(")
            source.write_text('''#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static size_t live;
static int fail_allocation, fake_length;
static void *tracked_malloc(size_t n) { if (fail_allocation) return NULL; void *p = malloc(n); if (p) ++live; return p; }
static void tracked_free(void *p) { if (p) { if (!live) abort(); --live; } free(p); }
static size_t tracked_strlen(const char *s) { return fake_length && !strcmp(s, "left") ? SIZE_MAX : strlen(s); }
#define malloc tracked_malloc
#define free tracked_free
#define strlen tracked_strlen
''' + generated + '''
int main(int argc, char **argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    if (mode == 1) fail_allocation = 1;
    if (mode == 6) { fake_length = 1; (void)nstr_concat("left", "right"); }
    if (mode == 2) (void)nstr_allocate(SIZE_MAX);
    if (mode == 3) (void)nstr_allocate(SIZE_MAX - sizeof(nstr_owned));
    const char *escaped = nstr_concat("kept", "value");
    char *input = malloc(100001); if (!input) return 1;
    memset(input, 'a', 100000); input[100000] = 0;
    const char *large = nstr_concat(input, input); free(input);
    if (strlen(large) != 200000 || large[0] != 'a' || large[199999] != 'a') return 2;
    if (mode == 4) fail_allocation = 1;
    const char *slice = nstr_substr(large, -4, INT64_MAX);
    if (strlen(slice) != 200000 || strcmp(slice, large)) return 3;
    size_t before = live;
    if (strcmp(nstr_substr(large, INT64_MAX, INT64_MAX), "") ||
        strcmp(nstr_substr(large, 0, -1), "") || live != before) return 4;
    if (mode == 5) fail_allocation = 1;
    const char *minimum = nstr_from_i64(INT64_MIN), *maximum = nstr_from_i64(INT64_MAX);
    for (int i = 0; i < 70000; ++i) if (strcmp(nstr_from_i64(42), "42")) return 5;
    if (strcmp(escaped, "keptvalue") || strcmp(minimum, "-9223372036854775808") ||
        strcmp(maximum, "9223372036854775807")) return 6;
    nstr_release_owned(); if (live) return 7;
    nstr_release_owned(); if (live) return 8;
    if (generated_main() != 1 || live) return 9;
    return 0;
}
''')
            self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O0", source, "-o", binary])
            self.run_checked([binary])
            for mode in range(1, 7):
                failed = subprocess.run([binary, str(mode)], capture_output=True, timeout=10)
                self.assertLess(failed.returncode, 0, "I reject allocation and string-size overflow")

    def test_uncalled_functions_are_warning_clean_not_executed(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory(prefix="nano-uncalled-native-") as tmp:
                work = Path(tmp)
                assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                main = ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
                unused = ".function unused 0 0 0 int 1\nPUSH_BOOL 0\nASSERT\nPUSH_I64 42\nRET\n.end\n"
                assembly.write_text(".entry main\n" + (unused + main if reverse else main + unused))
                self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                self.run_checked([ROOT / "bin/nano_vm", module])
                self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                self.run_checked([binary])

    def test_nested_optional_returns_reach_native(self):
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "I require the host C compiler")
        fixture = (ROOT / "tests/nanoisa/fixtures/nested_optional_returns.nasm").read_text()
        for tail in (False, True):
            for reverse in (False, True):
                with self.subTest(tail=tail, reverse=reverse), tempfile.TemporaryDirectory(prefix="nano-nested-optional-") as tmp:
                    work = Path(tmp)
                    assembly, module, source, binary = (work / name for name in ("input.nasm", "input.nvm", "input.c", "input"))
                    text = fixture
                    if not tail:
                        text = text.replace("TAIL_CALL present", "CALL present\n  RET")
                        text = text.replace("TAIL_CALL missing", "CALL missing\n  RET")
                    if reverse:
                        header, *functions = text.split(".function ")
                        text = header + "".join(".function " + block for block in reversed(functions))
                    assembly.write_text(text)
                    self.run_checked([ROOT / "bin/nanoisa", "asm", assembly, "-o", module])
                    self.run_checked([ROOT / "bin/nano_vm", module])
                    self.run_checked([ROOT / "bin/nvm2c", module, "-o", source])
                    self.run_checked([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary])
                    self.run_checked([binary])


if __name__ == "__main__":
    unittest.main()
