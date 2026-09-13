"""I compile retained translation units and keep failed replacements private."""

import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest

from tests import test_module_cache_publication as cache
from tests.characterize_source_snapshot import measure


class SourceSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([shutil.which("cc"), "--version"], capture_output=True, timeout=10)
        if result.returncode or not (b"clang version" in result.stdout or
                                      b"Free Software Foundation" in result.stdout):
            raise unittest.SkipTest("I exercise ordinary Clang and GCC C here")
        cls.clang = b"clang version" in result.stdout
        cls.snapshot_suffix = ".s" if cls.clang else ".i"

    def setUp(self):
        self.support = cache.ModuleCachePublication()
        self.support.probe = cache.ROOT / "obj/test_module_generation_probe"
        self.support.setUp()

    def answer(self, library):
        result = subprocess.run([sys.executable, "-c",
            "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
            "lib.nano_build_answer.restype=ctypes.c_int64; print(lib.nano_build_answer())",
            str(library)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(result.stdout)

    def test_restored_source_and_header_changes(self):
        observed = measure(shutil.which("cc"))
        self.assertEqual(len(observed["cases"]), 4)
        for case in observed["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record", "generation_reused"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 1 if self.clang else 3)
                if self.clang:
                    self.assertGreater(case["total_assembly_captures"], case["cold_assembly_captures"])

    def test_clang_assembler_cache_restored_inputs(self):
        if not self.clang: self.skipTest("I have not integrated GCC assembler-input capture")
        for case in measure(shutil.which("cc"), ("assembler",))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "generation_reused", "reuse_record", "retained_assembly"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 1)
                self.assertGreater(case["total_assembly_captures"], case["cold_assembly_captures"])

    def test_assembler_cache_nested_changes_and_recovery(self):
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory(prefix="nano-assembly-cache-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                binary, include = module / "payload with 'quotes'.bin", module / "nested include.s"
                binary.write_bytes(b"xx42yy")
                include_text = f'.macro payload\n.incbin "{binary}", 2, 2\n.endm\npayload\n'
                include.write_text(include_text)
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                assembly = '__asm__(' + json.dumps(f'.data\n.globl {symbol}\n{symbol}:\n.include "{include}"\n.text\n') + ');\n'
                body = 'extern const unsigned char snapshot_payload[];\nlong long nano_build_answer(void) {\nreturn (snapshot_payload[0] - 48) * 10 + snapshot_payload[1] - 48;\n}\n'
                metadata = {"name": "answer_native", "c_sources": ["answer.c"], "cflags": ["-O2 -g -std=c11 -Wall -Wextra -Werror"]}
                if shared:
                    (module / "private.c").write_text(assembly)
                    metadata["shared_c_sources"] = ["private.c"]
                (module / "answer.c").write_text(body if shared else assembly + body)
                (module / "module.json").write_text(json.dumps(metadata))
                def build(answer):
                    self.support.probe_path("build", module, env)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertTrue((generation / "source_hashes.json").is_file())
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), answer)
                    self.support.probe_path("build", module, env)
                    self.assertEqual(self.support.probe_path("directory", module, env), generation)
                    return generation
                first = build(42)
                binary.write_bytes(b"xx43yy")
                second = build(43)
                self.assertNotEqual(first, second)
                include.write_text('.ascii "44"\n')
                third = build(44)
                self.assertNotEqual(second, third)
                include.unlink()
                failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(self.support.probe_path("directory", module, env), third)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 44)
                include.write_text(include_text)
                self.assertNotEqual(build(43), third)

    def test_gcc_assembler_restoration_withholds_stale_reuse(self):
        if self.clang: self.skipTest("I exercise the GCC object-output check here")
        for case in measure(shutil.which("cc"), ("assembler",))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (43, 42, 42))
                self.assertFalse(case["reuse_record"])
                self.assertFalse(case["generation_reused"])
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "retained_translation_unit"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 4)

    def test_gcc_validation_cleanup_and_cold_failure(self):
        if self.clang: self.skipTest("I exercise the GCC private object checker here")
        with tempfile.TemporaryDirectory(prefix="nano-gcc-validation-test-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            scratch = directory / "validation with spaces"
            scratch.mkdir()
            env["TMPDIR"] = str(scratch)
            self.support.probe_path("build", module, env)
            generation = self.support.probe_path("directory", module, env)
            self.assertTrue((generation / "source_hashes.json").is_file())
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), generation)
            self.assertEqual(list(scratch.iterdir()), [])
            wrapper, calls = directory / "cc", directory / "calls"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, sys
if os.environ.get("NANO_TEST_FAIL_VALIDATION") and any("nano-gcc-check-" in arg for arg in sys.argv):
    sys.exit(30)
if "-c" in sys.argv:
    path = pathlib.Path({str(calls)!r})
    if not path.exists():
        path.write_text("failed once")
        sys.exit(29)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(self.support.probe_path("directory", module, env), generation)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
            self.assertEqual(list(scratch.iterdir()), [])
            self.support.probe_path("build", module, env)
            self.assertEqual(list(scratch.iterdir()), [])
            env["NANO_TEST_FAIL_VALIDATION"] = "1"
            (module / "answer.c").write_text("long long nano_build_answer(void) { return 43; }\n")
            self.support.probe_path("build", module, env)
            unchecked = self.support.probe_path("directory", module, env)
            self.assertFalse((unchecked / "source_hashes.json").exists())
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
            self.assertEqual(list(scratch.iterdir()), [])
            del env["NANO_TEST_FAIL_VALIDATION"]
            self.support.probe_path("build", module, env)
            recovered = self.support.probe_path("directory", module, env)
            self.assertTrue((recovered / "source_hashes.json").is_file())
            self.assertEqual(list(scratch.iterdir()), [])

    def test_configured_flags_preserve_retained_input_and_phases(self):
        active = "cflags_macos" if sys.platform == "darwin" else "cflags_linux"
        inactive = "cflags_linux" if sys.platform == "darwin" else "cflags_macos"
        placements = ["common", "platform", "inactive", "literal", "package"]
        if sys.platform == "darwin": placements.append("framework")
        for placement in placements:
            with self.subTest(placement=placement), tempfile.TemporaryDirectory(prefix="nano-retained-flags-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                include = directory / ("include with 'quotes'" if placement in ("literal", "package") else "include")
                include.mkdir()
                (include / "offset.h").write_text("#define OFFSET 2\n")
                declared = directory / "declared includes"
                declared.mkdir()
                (declared / "check.h").write_text("#ifdef REMOVED\n#error I expected REMOVED to be undefined\n#endif\n")
                flags = ["-O2", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror",
                         "-DANSWER=40", "-DREMOVED=1", "-UREMOVED", "-I" + str(include)]
                metadata = {"name": "answer_native", "c_sources": ["answer.c"],
                            "include_dirs": [str(declared)]}
                metadata[active if placement == "platform" else "cflags"] = flags
                if placement in ("literal", "package"):
                    fragment = "'-O2' -g -std=c11 -Wall -Wextra -Werror -D ANSWER=40 -DREMOVED=1 -U REMOVED -I " + shlex.quote(str(include))
                    fragment += " -D " + shlex.quote('TEXT="a b"')
                    (include / "offset.h").write_text('#define OFFSET (sizeof(TEXT) - 2)\n')
                    if placement == "literal":
                        metadata["cflags"] = [fragment]
                    else:
                        metadata["cflags"] = []
                        metadata["pkg_config"] = ["fixture"]
                        pkg = directory / "pkg-config"
                        pkg.write_text(f'#!{sys.executable}\nimport sys\nprint({fragment!r} if "--cflags" in sys.argv else "")\n')
                        pkg.chmod(0o700)
                        env["PKG_CONFIG"] = str(pkg)
                if placement == "inactive": metadata[inactive] = ["-not-a-supported-option"]
                if placement == "framework":
                    metadata["frameworks"] = ["CoreFoundation"]
                    metadata["pkg_config"] = ["CoreFoundation"]
                (module / "module.json").write_text(json.dumps(metadata))
                source = module / "answer.c"
                source.write_text('#include <offset.h>\n#include <check.h>\n'
                                  'long long nano_build_answer(void) { return ANSWER + OFFSET; }\n')
                wrapper, calls = directory / "cc", directory / "calls"
                wrapper.write_text(f'''#!{sys.executable}
import json, os, pathlib, subprocess, sys
if any(phase in sys.argv for phase in ("-c", "-E", "-S")):
    with open({str(calls)!r}, "a") as log: log.write(json.dumps(sys.argv[1:]) + "\\n")
if "-c" in sys.argv:
    source = pathlib.Path({str(source)!r})
    data, stamp = source.read_bytes(), source.stat()
    try:
        source.write_bytes(data.replace(b"ANSWER + OFFSET", b"ANSWER + OFFSET + 1"))
        result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    finally:
        source.write_bytes(data)
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)
                self.support.probe_path("build", module, env)
                generation = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertTrue((generation / "source_hashes.json").is_file())
                self.support.probe_path("build", module, env)
                self.assertEqual(self.support.probe_path("directory", module, env), generation)
                commands = [json.loads(line) for line in calls.read_text().splitlines()]
                compiled = [argv for argv in commands if "-c" in argv]
                self.assertEqual(len(compiled), 1 if self.clang else 3)
                self.assertTrue(any(arg.endswith(self.snapshot_suffix) for arg in compiled[0]))
                for argv in commands:
                    for flag in flags[:6]:
                        if "-c" in argv and self.clang: self.assertNotIn(flag, argv)
                        else: self.assertIn(flag, argv)
                    preprocessing = flags[6:] if placement not in ("literal", "package") else [
                        "-D", "ANSWER=40", "-DREMOVED=1", "-U", "REMOVED", "-I", str(include), 'TEXT="a b"']
                    for flag in preprocessing:
                        if "-E" in argv or "-S" in argv: self.assertIn(flag, argv)
                        else: self.assertNotIn(flag, argv)

    def test_literal_words_match_shell_arguments(self):
        fragments = ["", "  \t", "''", "a''b \"c d\"", "'-DNAME=a b'",
                     r"a\ b 'a'\''b'", r'"a\qb" "\$x" "\`x\`" "a\\b"',
                     "a\\\nb '\n'", "'$HOME' '`id`' '*.c'", "x" * 4095]
        script = "import json,sys; print(json.dumps(sys.argv[1:]))"
        for fragment in fragments:
            with self.subTest(fragment=fragment[:80]):
                expected = subprocess.run(["/bin/sh", "-c", shlex.quote(sys.executable) +
                    " -c " + shlex.quote(script) + " " + fragment], capture_output=True, timeout=10, check=True)
                actual = subprocess.run([str(self.support.probe), "flag-words", fragment],
                                        capture_output=True, timeout=10, check=True)
                self.assertEqual(json.loads(actual.stdout), json.loads(expected.stdout))
        for fragment in ("$HOME", '"$HOME"', "$(id)", "`id`", "x;y", "a|b", "a&&b",
                         "*.c", "?", "[ab]", "~", "#comment", "{a,b}", "x\ny", "'x", '"x',
                         "x\\", "x" * 4096):
            with self.subTest(rejected=fragment[:80]):
                actual = subprocess.run([str(self.support.probe), "flag-words", fragment],
                                        capture_output=True, timeout=10)
                self.assertNotEqual(actual.returncode, 0)
                self.assertEqual(actual.stdout, b"")

    def test_clang_retained_assembly_expands_external_inputs(self):
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, check=True).stdout
        if b"clang version" not in version:
            self.skipTest("I test the Clang assembly-output candidate here")
        for nested in (False, True):
            for flags in ([], ["-O2", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror"]):
                with self.subTest(nested=nested, flags=flags), tempfile.TemporaryDirectory(prefix="nano-assembly-trial-") as tmp:
                    directory = Path(tmp)
                    binary = directory / "payload with 'quotes'.bin"
                    binary.write_bytes(b"xx42yy")
                    include = directory / "nested include.s"
                    outer = directory / "outer include.s"
                    include.write_text(f'.macro payload\n.incbin "{binary}", 2, 2\n.endm\npayload\n')
                    outer.write_text(f'.include "{include}"\n')
                    symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                    directive = f'.include "{outer}"' if nested else f'.incbin "{binary}", 2, 2'
                    assembly = f'.data\n.globl {symbol}\n{symbol}:\n{directive}\n.text\n'
                    source = directory / "answer.c"
                    source.write_text('extern const unsigned char snapshot_payload[];\n'
                        '__asm__(' + json.dumps(assembly) + ');\n'
                        'long long nano_build_answer(void) {\n'
                        'return (snapshot_payload[0] - 48) * 10 + snapshot_payload[1] - 48;\n}\n')
                    shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
                    direct, retained, changed = (directory / name for name in ("direct.so", "retained.so", "changed.so"))
                    def compile_run(args):
                        result = subprocess.run([compiler] + args, capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    compile_run(["-fPIC", *flags, shared, str(source), "-o", str(direct)])
                    self.assertEqual(self.answer(direct), 42)
                    captured = directory / "captured.s"
                    compile_run(["-fPIC", *flags, "-S", str(source), "-o", str(captured)])
                    binary.write_bytes(b"xx43yy")
                    compile_run(["-fPIC", *flags, shared, str(source), "-o", str(changed)])
                    self.assertEqual(self.answer(changed), 43)
                    binary.unlink()
                    include.unlink()
                    outer.unlink()
                    failed = subprocess.run([compiler, "-fPIC", *flags, "-S", str(source), "-o", str(directory / "failed.s")],
                                            capture_output=True, timeout=20)
                    self.assertNotEqual(failed.returncode, 0)
                    source.unlink()
                    compile_run([shared, "-x", "assembler", str(captured), "-o", str(retained)])
                    self.assertEqual(self.answer(retained), 42)

    def test_gcc_retained_object_reproducibility_and_external_inputs(self):
        if self.clang: self.skipTest("I test the GCC compiler-output candidate here")
        compiler = shutil.which("cc")
        for nested in (False, True):
            for flags in ([], ["-O2", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror"]):
                with self.subTest(nested=nested, flags=flags), tempfile.TemporaryDirectory(prefix="nano-object-trial-") as tmp:
                    directory = Path(tmp)
                    binary, include = directory / "payload with 'quotes'.bin", directory / "nested include.s"
                    binary.write_bytes(b"xx42yy")
                    include.write_text(f'.macro payload\n.incbin "{binary}", 2, 2\n.endm\npayload\n')
                    source = directory / "answer.c"
                    directive = f'.include "{include}"' if nested else f'.incbin "{binary}", 2, 2'
                    assembly = f'.data\n.globl snapshot_payload\nsnapshot_payload:\n{directive}\n.text\n'
                    source.write_text('extern const unsigned char snapshot_payload[];\n'
                        '__asm__(' + json.dumps(assembly) + ');\n'
                        'long long nano_build_answer(void) {\nreturn (snapshot_payload[0] - 48) * 10 + snapshot_payload[1] - 48;\n}\n')
                    def capture(folder):
                        output = directory / folder
                        output.mkdir()
                        obj = output / (folder + ".o")
                        result = subprocess.run([compiler, "-fPIC", *flags, "-c", str(source), "-o", str(obj)],
                                                capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        return obj
                    original, stamp = binary.read_bytes(), binary.stat()
                    first, second = capture("first"), capture("second")
                    self.assertEqual(first.read_bytes(), second.read_bytes())
                    binary.write_bytes(b"xx43yy")
                    os.utime(binary, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    changed = capture("changed")
                    self.assertNotEqual(first.read_bytes(), changed.read_bytes())
                    binary.write_bytes(original)
                    os.utime(binary, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    self.assertEqual(first.read_bytes(), capture("restored").read_bytes())
                    binary.unlink()
                    include.unlink()
                    failed = subprocess.run([compiler, "-fPIC", *flags, "-c", str(source), "-o", str(directory / "failed.o")],
                                            capture_output=True, timeout=20)
                    self.assertNotEqual(failed.returncode, 0)
                    source.unlink()
                    for obj, expected in ((first, 42), (changed, 43)):
                        library = obj.with_suffix(".so")
                        result = subprocess.run([compiler, "-shared", str(obj), "-o", str(library)],
                                                capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(self.answer(library), expected)

    def test_multiple_and_shared_only_sources(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-multiple-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            (module / "module.json").write_text(json.dumps({"name": "answer_native",
                "c_sources": ["answer.c", "extra.c"], "shared_c_sources": ["private.c"]}))
            (module / "answer.c").write_text("long long extra(void); long long private_value(void);\n"
                "long long nano_build_answer(void) { return 20 + extra() + private_value(); }\n")
            (module / "extra.c").write_text("long long extra(void) { return 10; }\n")
            (module / "private.c").write_text("long long private_value(void) { return 12; }\n")
            wrapper, calls = directory / "cc", directory / "calls"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
if "-c" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("C\\n")
    paths = list(pathlib.Path({str(module)!r}).glob("*.c"))
    original = [(p, p.read_bytes(), p.stat()) for p in paths]
    try:
        for p, data, stamp in original: p.write_bytes(data.replace(b"return 1", b"return 2"))
        result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    finally:
        for p, data, stamp in original:
            p.write_bytes(data)
            os.utime(p, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            self.support.probe_path("build", module, env)
            generation = self.support.probe_path("directory", module, env)
            self.assertEqual(len(list(generation.glob("__snapshot_*" + self.snapshot_suffix))), 3)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), generation)
            self.assertEqual(len(calls.read_text().splitlines()), 3 if self.clang else 9)
            self.assertTrue((generation / "source_hashes.json").is_file())

    def test_supported_scalar_flag_spellings(self):
        flags = ["-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og",
                 "-g", "-g0", "-g1", "-g2", "-g3", "-fPIC", "-fpic",
                 "-std=c89", "-std=c90", "-std=c99", "-std=c11", "-std=c17", "-std=c18",
                 "-std=gnu89", "-std=gnu90", "-std=gnu99", "-std=gnu11", "-std=gnu17", "-std=gnu18",
                 "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                 "-Wno-unused-parameter", "-Wno-unused-variable", "-Wno-unused-function"]
        with tempfile.TemporaryDirectory(prefix="nano-retained-scalar-flags-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            for flag in flags:
                with self.subTest(flag=flag):
                    (module / "module.json").write_text(json.dumps({"name": "answer_native",
                        "c_sources": ["answer.c"], "cflags": [flag]}))
                    self.support.probe_path("build", module, env)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertTrue((generation / ("__snapshot_0_0" + self.snapshot_suffix)).is_file())
                    self.assertTrue((generation / "source_hashes.json").is_file())
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_unknown_fragments_keep_original_compilation(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-unknown-flags-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            response = directory / "flags.rsp"
            response.write_text("-O2\n")
            for flags in (["-fno-builtin"], ["-O${NANO_TEST_LEVEL:-2}"],
                          ["-D", "NAME=42"], ["@" + str(response)]):
                with self.subTest(flags=flags):
                    (module / "module.json").write_text(json.dumps({"name": "answer_native",
                        "c_sources": ["answer.c"], "cflags": flags}))
                    self.support.probe_path("build", module, env)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertEqual(list(generation.glob("__snapshot_*")), [])
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_configured_warning_errors_preserve_generation(self):
        cases = [
            (["-Wall", "-Wextra", "-Werror"], "int nano_build_answer(void) { int unused; return 42; }\n"),
            (["-Werror", "-DANSWER=41"], "#define ANSWER 42\nint nano_build_answer(void) { return ANSWER; }\n"),
            (["-std=c89", "-Wpedantic", "-Werror"], "// I require a newer comment form.\nint nano_build_answer(void) { return 42; }\n"),
        ]
        for flags, body in cases:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory(prefix="nano-retained-warning-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                self.support.probe_path("build", module, env)
                generation = self.support.probe_path("directory", module, env)
                original = self.support.snapshot(generation)
                source = module / "answer.c"
                source.write_text(body)
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c"], "cflags": flags}))
                direct = subprocess.run([shutil.which("cc"), *flags, "-c", str(source),
                    "-o", str(directory / "direct.o")], capture_output=True, timeout=10)
                self.assertNotEqual(direct.returncode, 0, direct.stderr)
                result = subprocess.run([str(self.support.probe), "build", str(module)],
                    env=env, capture_output=True, timeout=20)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn(str(source).encode(), result.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), generation)
                self.assertEqual(self.support.snapshot(generation), original)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_compile_diagnostics_preserve_previous_generation(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-failure-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            self.support.probe_path("build", module, env)
            generation = self.support.probe_path("directory", module, env)
            library = self.support.probe_path("library", module, env)
            original = self.support.snapshot(generation)
            source = module / "answer.c"
            source.write_text("long long nano_build_answer(void) { return missing_value; }\n")
            result = subprocess.run([str(self.support.probe), "build", str(module)],
                                    env=env, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn((str(source) + ":1:").encode(), result.stderr)
            self.assertEqual(self.support.probe_path("directory", module, env), generation)
            self.assertEqual(self.support.snapshot(generation), original)
            self.assertEqual(self.answer(library), 42)
            source.write_text("long long nano_build_answer(void) { return 44; }\n")
            self.support.probe_path("build", module, env)
            recovered = self.support.probe_path("directory", module, env)
            self.assertNotEqual(recovered, generation)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 44)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_restored_edit_during_capture_cannot_authorize_reuse(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-capture-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            source, wrapper, marker = module / "answer.c", directory / "cc", directory / "captured"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
marker = pathlib.Path({str(marker)!r})
if ("-E" in sys.argv or "-S" in sys.argv) and not marker.exists():
    source = pathlib.Path({str(source)!r})
    data, stamp = source.read_bytes(), source.stat()
    try:
        source.write_bytes(data.replace(b"42", b"43"))
        result = subprocess.run([{shutil.which('cc')!r}] + sys.argv[1:])
    finally:
        source.write_bytes(data)
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    marker.touch()
    sys.exit(result.returncode)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            self.support.probe_path("build", module, env)
            first = self.support.probe_path("directory", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
            self.assertFalse((first / "source_hashes.json").exists())
            self.support.probe_path("build", module, env)
            recovered = self.support.probe_path("directory", module, env)
            self.assertNotEqual(recovered, first)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_gcc_implicit_pch_appearance_and_removal(self):
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, timeout=10)
        if b"Free Software Foundation" not in version.stdout:
            self.skipTest("I need GCC's implicit precompiled-header selection")
        with tempfile.TemporaryDirectory(prefix="nano-retained-gcc-pch-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            source, header = module / "answer.c", module / "answer.h"
            pch = module / "answer.h.gch"
            source.write_text('#include "answer.h"\nlong long nano_build_answer(void) { return ANSWER; }\n')
            header.write_text("#define ANSWER 43\n")
            stamp = header.stat()
            self.support.probe_path("build", module, env)
            plain = self.support.probe_path("directory", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
            self.assertTrue((plain / "source_hashes.json").is_file())
            header.write_text("#define ANSWER 42\n")
            flags = [compiler, "-fPIC", "-D_POSIX_C_SOURCE=200809L"]
            result = subprocess.run(flags + ["-x", "c-header", str(header), "-o", str(pch)],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            header.write_text("#define ANSWER 43\n")
            os.utime(header, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            ordinary = subprocess.run(flags + ["-E", str(source)], capture_output=True, timeout=10)
            aware = subprocess.run(flags + ["-E", "-fpch-preprocess", str(source)], capture_output=True, timeout=10)
            self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
            self.assertEqual(aware.returncode, 0, aware.stderr)
            self.assertIn(b"return 43", ordinary.stdout)
            self.assertIn(b"#pragma GCC pch_preprocess", aware.stdout)
            fresh = directory / "fresh.so"
            result = subprocess.run(flags + ["-shared", str(source), "-o", str(fresh)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.answer(fresh), 42)
            for _ in range(2):
                self.support.probe_path("build", module, env)
                generation = self.support.probe_path("directory", module, env)
                self.assertNotEqual(generation, plain)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertFalse((generation / "source_hashes.json").exists())
            pch.unlink()
            self.support.probe_path("build", module, env)
            recovered = self.support.probe_path("directory", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
            self.assertTrue((recovered / "source_hashes.json").is_file())
            self.support.probe_path("build", module, env)
            self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_gcc_pch_marker_across_read_boundaries(self):
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, timeout=10)
        if b"Free Software Foundation" not in version.stdout:
            self.skipTest("I need GCC's PCH preprocessing directive")
        with tempfile.TemporaryDirectory(prefix="nano-retained-pch-split-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            source, header = module / "answer.c", module / "answer.h"
            source.write_text('#include "answer.h"\nlong long nano_build_answer(void) { return ANSWER; }\n')
            header.write_text("#define ANSWER 42\n")
            result = subprocess.run([compiler, "-fPIC", "-D_POSIX_C_SOURCE=200809L",
                "-x", "c-header", str(header), "-o", str(module / "answer.h.gch")],
                capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            header.write_text("#define ANSWER 43\n")
            wrapper, calls = directory / "cc", directory / "calls"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
if "-E" in sys.argv:
    result = subprocess.run([{compiler!r}] + sys.argv[1:], capture_output=True)
    marker = b"#pragma GCC pch_preprocess"
    if result.returncode == 0 and marker in result.stdout:
        split = int(os.environ["NANO_TEST_PCH_SPLIT"])
        padding = (4096 - split - result.stdout.index(marker)) % 4096
        sys.stdout.buffer.write(b"\\n" * padding + result.stdout)
    else:
        sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
if "-c" in sys.argv:
    pathlib.Path({str(calls)!r}).write_text("snapshot" if any(a.endswith(".i") for a in sys.argv) else "original")
os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            for split in range(1, len(b"#pragma GCC pch_preprocess")):
                with self.subTest(split=split):
                    env["NANO_TEST_PCH_SPLIT"] = str(split)
                    self.support.probe_path("build", module, env)
                    self.assertEqual(calls.read_text(), "original")
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertFalse((generation / "source_hashes.json").exists())


if __name__ == "__main__":
    unittest.main()
