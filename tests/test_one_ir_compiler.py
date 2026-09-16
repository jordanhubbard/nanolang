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


class OneIrCompiler(unittest.TestCase):
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
                              source, "-o", compiler,
                              *(["-ldl"] if sys.platform.startswith("linux") else [])], timeout=240)
            help_output = self.run_checked([compiler, "--help"], timeout=10)
            self.assertIn(b"Compiler", help_output)
            hello = work / "hello"
            self.run_checked([compiler, ROOT / "examples/language/nl_hello.nano", "-o", hello])
            self.assertEqual(self.run_checked([hello], timeout=10), b"Hello from NanoLang!\n")

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
                    assembly.write_text(text.split(".parameters", 1)[0])
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
