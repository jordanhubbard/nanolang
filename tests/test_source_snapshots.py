"""I compile retained translation units and keep failed replacements private."""

import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from tests import test_module_cache_publication as cache
from tests.characterize_source_snapshot import measure, require_consistent


class SourceSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([shutil.which("cc"), "--version"], capture_output=True, timeout=10)
        if result.returncode or not (b"clang version" in result.stdout or
                                      b"Free Software Foundation" in result.stdout):
            raise unittest.SkipTest("I exercise ordinary Clang and GCC C here")
        cls.clang = b"clang version" in result.stdout
        cls.snapshot_suffix = ".s" if cls.clang else ".i"
        cls.read_replay = False
        if not cls.clang and sys.platform == "linux":
            query = subprocess.run([shutil.which("cc"), "-print-prog-name=as"], capture_output=True, timeout=10)
            assembler = shutil.which(query.stdout.decode().strip()) if query.returncode == 0 else None
            if assembler:
                version = subprocess.run([assembler, "--version"], capture_output=True, timeout=10)
                cls.read_replay = version.returncode == 0 and subprocess.run(
                    [str(cache.ROOT / "obj/test_module_generation_probe"), "assembler-version", version.stdout.decode()],
                    capture_output=True, timeout=10).returncode == 0

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

    def test_supported_assembler_version_line(self):
        if not sys.platform.startswith("linux"): self.skipTest("I select GNU assembler replay only on Linux")
        accepted = ["GNU assembler (GNU Binutils for Debian) 2.40\nCopyright text\n",
                    "GNU assembler (GNU Binutils for Ubuntu) 2.42\nCopyright text\n"]
        rejected = ["GNU assembler (GNU Binutils) 2.41\n", "GNU assembler (GNU Binutils) 2.43\n",
                    "GNU assembler (GNU Binutils) 2.42.1\n", "GNU assembler (GNU Binutils) 2.420\n",
                    "GNU assembler (GNU Binutils) 2.42", "GNU ld (GNU Binutils) 2.42\n",
                    "GNU assembler (GNU Binutils) 9.99\nPrevious version 2.42\n",
                    "unrelated banner\nGNU assembler (GNU Binutils) 2.42\n"]
        for banner in accepted + rejected:
            with self.subTest(banner=banner):
                result = subprocess.run([str(self.support.probe), "assembler-version", banner], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0 if banner in accepted else 1, result.stderr)

    def test_consistency_gate_checks_cold_and_warm_answers(self):
        for cold, warm, fresh in ((42, 42, 42), (43, 42, 42), (42, 43, 42), (43, 43, 42)):
            with self.subTest(cold=cold, warm=warm, fresh=fresh):
                result = {"cases": [{"cold_answer": cold, "warm_answer": warm, "fresh_answer": fresh}]}
                if cold == warm == fresh:
                    require_consistent(result)
                else:
                    with self.assertRaises(SystemExit): require_consistent(result)

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
        if not self.clang: self.skipTest("I exercise GCC literal capture separately")
        for case in measure(shutil.which("cc"), ("assembler", "assembler-macro"))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "generation_reused", "reuse_record", "retained_assembly"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 1)
                self.assertGreater(case["total_assembly_captures"], case["cold_assembly_captures"])

    def test_clang_external_assembler_cache_restored_inputs(self):
        if not self.clang: self.skipTest("I need Clang's external assembler selector")
        for case in measure(shutil.which("cc"), ("assembler-external", "assembler-external-debug"))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "generation_reused",
                            "reuse_record", "retained_translation_unit", "retained_assembly"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["retained_assembler_files"], 1)
                self.assertEqual(case["total_object_compilations"], 3)
                self.assertEqual(case["external_assembly_compilations"], 3)

    def test_assembler_cache_nested_changes_and_recovery(self):
        self.assembler_cache_nested_changes_and_recovery(False)

    def test_external_assembler_cache_nested_changes_and_recovery(self):
        if not self.clang: self.skipTest("I need Clang's external assembler selector")
        self.assembler_cache_nested_changes_and_recovery(True)

    def test_apple_external_macro_inputs_use_selected_backend_capture(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I exercise the selected Apple external assembler backend")
        for case in measure(shutil.which("cc"), ("assembler-external-macro", "assembler-external-macro-debug"))["cases"]:
            with self.subTest(case=case):
                require_consistent({"cases": [case]})
                self.assertEqual(case["cold_answer"], 42)
                for key in ("reuse_record", "generation_reused", "retained_assembly", "retained_translation_unit",
                            "bytes_restored", "mtime_preserved", "size_preserved"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["retained_assembler_files"], 0)
                self.assertEqual(case["total_object_compilations"], 3)
                self.assertEqual(case["external_assembly_compilations"], 3)

    def test_apple_external_macro_nested_changes_and_recovery(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I exercise the selected Apple external assembler backend")
        self.assembler_cache_nested_changes_and_recovery(True, True)

    def assembler_cache_nested_changes_and_recovery(self, external, macro=False):
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory(prefix="nano-assembly-cache-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                binary, include = module / "payload with 'quotes'.bin", module / "nested include.s"
                binary.write_bytes(b"xx42yy")
                include_text = f'.macro payload\n.incbin "{binary}", 2, 2\n.endm\npayload\n'
                if macro:
                    include_text = '.macro payload file\n.incbin "\\file", 2, 2\n.endm\n' + f'payload "{binary}"\n'
                include.write_text(include_text)
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                assembly = '__asm__(' + json.dumps(f'.data\n.globl {symbol}\n{symbol}:\n.include "{include}"\n.text\n') + ');\n'
                body = 'extern const unsigned char snapshot_payload[];\nlong long nano_build_answer(void) {\nreturn (snapshot_payload[0] - 48) * 10 + snapshot_payload[1] - 48;\n}\n'
                metadata = {"name": "answer_native", "c_sources": ["answer.c"], "cflags": ["-O2 -g -std=c11 -Wall -Wextra -Werror"]}
                if external:
                    metadata["cflags"].append("-fno-integrated-as")
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
                if macro:
                    self.assertTrue((first / "__expanded_0_0.s").is_file())
                    originals = [module / "answer.c", binary, include]
                    if shared: originals.append(module / "private.c")
                    saved = {path: path.read_bytes() for path in originals}
                    try:
                        for path in originals: path.unlink()
                        replay = directory / "deleted-originals.so"
                        snapshots = [first / "__snapshot_0_0.s"]
                        if shared: snapshots.append(first / "__snapshot_1_0.s")
                        result = subprocess.run([shutil.which("cc"), "-fno-integrated-as", "-dynamiclib",
                            *map(str, snapshots), "-o", str(replay)], capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(self.answer(replay), 42)
                    finally:
                        for path, contents in saved.items(): path.write_bytes(contents)
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

    def test_gcc_assembler_restoration_uses_captured_inputs(self):
        if self.clang: self.skipTest("I exercise GCC literal assembler capture here")
        for case in measure(shutil.which("cc"), ("assembler", "assembler-nested", "assembler-include"))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "retained_translation_unit",
                            "retained_assembly", "reuse_record", "generation_reused"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 3)
                self.assertEqual(case["total_assembly_captures"], 3)

    def test_gcc_assembler_macro_argument_keeps_validation_fallback(self):
        if self.clang: self.skipTest("I exercise the GCC object-output fallback here")
        for case in measure(shutil.which("cc"), ("assembler-fallback",))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (43, 42, 42))
                self.assertFalse(case["reuse_record"])
                self.assertFalse(case["generation_reused"])
                self.assertFalse(case["retained_assembly"])
                self.assertEqual(case["retained_assembler_files"], 0)
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "retained_translation_unit"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 4)

    def test_gcc_macro_argument_replays_captured_reads(self):
        if not self.read_replay: self.skipTest("I need a supported Linux GNU assembler replay version")
        for case in measure(shutil.which("cc"), ("assembler-macro",))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "reuse_record",
                            "generation_reused", "retained_read_manifest"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 6)

    def test_gcc_replay_cleanup_failure_and_tool_selection(self):
        if not self.read_replay: self.skipTest("I need a supported Linux GNU assembler replay version")
        with tempfile.TemporaryDirectory(prefix="nano-replay-build-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            helper = directory / "helper with 'quotes'.so"
            shutil.copy2(cache.ROOT / "bin/nano_as_capture.so", helper)
            stub, ignored = directory / "ignored.c", directory / "ignored.so"
            stub.write_text("int nano_ignored_helper;\n")
            subprocess.run([shutil.which("cc"), "-shared", "-fPIC", str(stub), "-o", str(ignored)],
                           capture_output=True, check=True, timeout=10)
            env["NANO_AS_CAPTURE_HELPER"] = str(helper)
            scratch = directory / "private space"
            scratch.mkdir()
            env["TMPDIR"] = str(scratch)
            binary, wrapper = module / "payload.bin", directory / "cc"
            binary.write_bytes(b"42")
            inactive = module / "inactive.bin"
            assembly = '.data\n.globl snapshot_payload\nsnapshot_payload:\n.macro payload file\n.incbin "\\file"\n.endm\n'
            assembly += f'.if 0\n.incbin "{inactive}"\n.endif\npayload "{binary}"\n.text\n'
            (module / "payload.c").write_text('__asm__(' + json.dumps(assembly) + ');\n')
            (module / "answer.c").write_text('extern const unsigned char snapshot_payload[];\n'
                'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
            metadata = json.loads((module / "module.json").read_text())
            metadata["shared_c_sources"] = ["payload.c"]
            (module / "module.json").write_text(json.dumps(metadata))
            calls = directory / "calls"
            wrapper.write_text(f'''#!{sys.executable}
import json, os, pathlib, shutil, sys
assert not os.getenv("LD_PRELOAD"), "I leaked the helper into the compiler driver"
with open({str(calls)!r}, "a") as output: output.write(json.dumps(sys.argv[1:]) + "\\n")
if os.getenv("NANO_TEST_AS_REPLAY_FAIL") and os.getenv("NANO_AS_CAPTURE_PHASE") == "replay":
    stage = next(arg[2:] for arg in sys.argv if arg.startswith("-B"))
    if os.getenv("NANO_TEST_AS_REPLAY_FAIL") == "ignore":
        shutil.copy2({str(ignored)!r}, pathlib.Path(stage, "__as_helper.so"))
    else:
        pathlib.Path(stage, "__as_helper.so").unlink()
if os.getenv("NANO_TEST_AS_QUERY_FAIL") and "-print-prog-name=as" in sys.argv:
    print(os.environ["NANO_TEST_AS_QUERY_FAIL"])
    sys.exit(0)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            def build(answer):
                self.support.probe_path("build", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), answer)
                self.assertEqual(list(scratch.iterdir()), [])
                return self.support.probe_path("directory", module, env)
            first = build(42)
            self.assertTrue((first / "__as_read_0_0.manifest0").is_file())
            inactive.write_bytes(b"I am not consumed")
            self.assertEqual(build(42), first)
            for failure in ("remove", "ignore"):
                env["NANO_TEST_AS_REPLAY_FAIL"] = failure
                failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0, failed.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), first)
                self.assertEqual(list(scratch.iterdir()), [])
            del env["NANO_TEST_AS_REPLAY_FAIL"]
            binary.write_bytes(b"43")
            changed = build(43)
            self.assertNotEqual(first, changed)
            env["NANO_TEST_AS_QUERY_FAIL"] = "/missing/assembler"
            fallback = build(43)
            self.assertFalse((fallback / "__as_read_0_0.manifest0").exists())
            fake_as = directory / "as-wrapper"
            fake_as.write_text('#!/bin/sh\nexec ' + shlex.quote(shutil.which("as")) + ' "$@"\n')
            fake_as.chmod(0o700)
            env["NANO_TEST_AS_QUERY_FAIL"] = str(fake_as)
            fallback = build(43)
            self.assertFalse((fallback / "__as_read_0_0.manifest0").exists())
            del env["NANO_TEST_AS_QUERY_FAIL"]
            recovered = build(43)
            self.assertTrue((recovered / "__as_read_0_0.manifest0").is_file())
            self.assertTrue((recovered / "__as_read_1_0.manifest0").is_file())
            installed = directory / "installed bin"
            installed.mkdir()
            shutil.copy2(self.support.probe, installed / "probe")
            shutil.copy2(helper, installed / "nano_as_capture.so")
            self.support.probe = installed / "probe"
            del env["NANO_AS_CAPTURE_HELPER"]
            recovered = build(43)
            self.assertTrue((recovered / "__as_read_1_0.manifest0").is_file())
            binary.unlink()
            failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(self.support.probe_path("directory", module, env), recovered)
            self.assertEqual(list(scratch.iterdir()), [])

    def test_assembler_octal_data_capture_boundaries(self):
        comment = "; debug data" if sys.platform == "darwin" else "# debug data"
        accepted = [f'{directive} "\\000\\042\\134\\202\\377|" {comment}\n'
                    for directive in (".ascii", ".asciz", ".string")]
        accepted += ['\t.ascii "\\064\\062"\n', '.ascii "\\000" // debug data\n']
        rejected = ['.ascii "\\file"\n', '.ascii "\\0"\n', '.ascii "\\00"\n',
                    '.ascii "\\400"\n', '.ascii "\\128"\n', '.ascii "\\x42"\n',
                    '.ascii "\\064\n', '.ascii "\\\n', '.ascii "\\"\n',
                    '.ascii "\\064", "more"\n', 'label: .ascii "\\064"\n',
                    '.ascii_suffix "\\064"\n', '.incbin "\\064"\n',
                    '.ascii "\\064" .incbin "missing"\n',
                    '.ascii "\\064"; .incbin "missing"\n',
                    '.macro read file\n.incbin "\\file"\n.endm\n',
                    '.mri 1\n.ascii "\\064"\n', '.altmacro\n.ascii "\\064"\n']
        if sys.platform != "darwin": rejected += ['.ascii "\\064"; nop\n']
        for contents in accepted + rejected:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory(prefix="nano-octal-capture-") as tmp:
                directory = Path(tmp)
                source, retained = directory / "source.s", directory / "retained.s"
                source.write_text(contents)
                result = subprocess.run([str(self.support.probe), "capture-assembly", str(source), str(retained)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0 if contents in accepted else 1, result.stderr)
                self.assertEqual(result.stderr, b"")
                if result.returncode == 0: self.assertEqual(retained.read_bytes(), source.read_bytes())

    def test_assembler_octal_data_replay_bytes(self):
        for directive in (".ascii", ".asciz", ".string"):
            with self.subTest(directive=directive), tempfile.TemporaryDirectory(prefix="nano-octal-replay-") as tmp:
                directory = Path(tmp)
                source, retained = directory / "source.s", directory / "retained.s"
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                octets = "".join(f"\\{value:03o}" for value in range(256))
                data = f'{directive} "{octets}"\n'
                if directive == ".ascii":
                    data = '.macro data_bytes file\n' + data + '.endm\ndata_bytes unused\n'
                source.write_text(f'.data\n.globl {symbol}\n{symbol}:\n' + data)
                flags = ["-fno-integrated-as"] if self.clang else []
                def assemble(path, name):
                    library = directory / name
                    result = subprocess.run([shutil.which("cc"), *flags, "-fPIC",
                        "-dynamiclib" if sys.platform == "darwin" else "-shared", str(path), "-o", str(library)],
                        capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    result = subprocess.run([sys.executable, "-c",
                        "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                        "print(bytes((ctypes.c_ubyte * 256).in_dll(lib, 'snapshot_payload')).hex())", str(library)],
                        capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.decode().strip(), bytes(range(256)).hex())
                assemble(source, "direct.so")
                result = subprocess.run([str(self.support.probe), "capture-assembly", str(source), str(retained)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                source.unlink()
                assemble(retained, "retained.so")

    def test_literal_assembler_capture_boundaries(self):
        for spelling in ("literal", "empty", "semicolon", "label", "macro", "altmacro", "mri",
                         "missing", "fifo", "cycle", "nul", "oversize"):
            with self.subTest(spelling=spelling), tempfile.TemporaryDirectory(prefix="nano-assembler-boundary-") as tmp:
                directory = Path(tmp)
                private = directory / "capture"
                private.mkdir()
                source, binary = directory / "input.s", directory / "payload.bin"
                binary.write_bytes(b"\x00\x01\xff42")
                literal = f'.incbin "{binary}", (1+1), (3-1)\n'
                contents = literal
                if spelling == "empty": binary.write_bytes(b"")
                elif spelling == "semicolon": contents = literal.rstrip() + ";" + literal
                elif spelling == "label": contents = "label: " + literal
                elif spelling == "macro": contents = '.macro read file\n.incbin "\\file"\n.endm\n'
                elif spelling == "altmacro": contents = ".altmacro\n" + literal
                elif spelling == "mri": contents = ".mri 1\n" + literal
                elif spelling == "missing": binary.unlink()
                elif spelling == "fifo":
                    binary.unlink()
                    os.mkfifo(binary)
                elif spelling == "cycle": contents = f'.include "{source}"\n'
                elif spelling == "nul": contents = literal + "\x00"
                elif spelling == "oversize":
                    with binary.open("wb") as output: output.truncate(17 * 1024 * 1024)
                source.write_text(contents)
                result = subprocess.run([str(self.support.probe), "capture-assembly", str(source),
                    str(private / "input.s")], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0 if spelling in ("literal", "empty") else 1, result.stderr)
                self.assertEqual(result.stderr, b"")
                if result.returncode == 0:
                    copies = list(private.glob("*.bin"))
                    self.assertEqual(len(copies), 1)
                    self.assertEqual(copies[0].read_bytes(), binary.read_bytes())
                    retained = (private / "input.s").read_text()
                    self.assertIn(str(copies[0]), retained)
                    self.assertIn(", (1+1), (3-1)", retained)
                    self.assertNotIn(str(binary), retained)

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
                self.assertTrue(any(arg.endswith(".s") for arg in compiled[0]))
                for argv in commands:
                    for flag in flags[:6]:
                        if "-c" in argv: self.assertNotIn(flag, argv)
                        else: self.assertIn(flag, argv)
                    preprocessing = flags[6:] if placement not in ("literal", "package") else [
                        "-D", "ANSWER=40", "-DREMOVED=1", "-U", "REMOVED", "-I", str(include), 'TEXT="a b"']
                    for flag in preprocessing:
                        if "-E" in argv or ("-S" in argv and self.clang): self.assertIn(flag, argv)
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
        for spelling in ("literal", "nested", "macro-argument"):
            for flags in ([], ["-O2", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror"]):
                with self.subTest(spelling=spelling, flags=flags), tempfile.TemporaryDirectory(prefix="nano-assembly-trial-") as tmp:
                    directory = Path(tmp)
                    binary = directory / "payload with 'quotes'.bin"
                    binary.write_bytes(b"xx42yy")
                    include = directory / "nested include.s"
                    outer = directory / "outer include.s"
                    include.write_text(f'.macro payload\n.incbin "{binary}", 2, 2\n.endm\npayload\n')
                    outer.write_text(f'.include "{include}"\n')
                    symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                    if spelling == "macro-argument":
                        include.write_text('.macro payload file\n.incbin "\\file", 2, 2\n.endm\n'
                                           f'.if 0\n.incbin "{directory / "missing.bin"}"\n.endif\n'
                                           f'payload "{binary}"\n')
                    directive = f'.include "{outer}"' if spelling != "literal" else f'.incbin "{binary}", 2, 2'
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

    def test_clang_external_assembler_requires_read_capture(self):
        if not self.clang:
            self.skipTest("I characterize Clang's external assembler boundary")
        compiler = shutil.which("cc")
        with tempfile.TemporaryDirectory(prefix="nano-external-as-trial-") as tmp:
            directory = Path(tmp)
            payload = directory / "payload.bin"
            payload.write_bytes(b"42")
            source = directory / "answer.c"
            symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
            assembly = f'.data\n.globl {symbol}\n{symbol}:\n.incbin "{payload}"\n.text\n'
            source.write_text('__asm__(' + json.dumps(assembly) + ');\n'
                              'extern const unsigned char snapshot_payload[];\n'
                              'long long nano_build_answer(void) { return '
                              '(snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
            captured = directory / "captured.s"
            capture = subprocess.run([compiler, "-fno-integrated-as", "-fPIC", "-S", str(source),
                                      "-o", str(captured)], capture_output=True, timeout=20)
            self.assertEqual(capture.returncode, 0, capture.stderr)
            self.assertIn(str(payload), captured.read_text())
            private = directory / "private"
            private.mkdir()
            frozen = private / "captured.s"
            copied = subprocess.run([str(self.support.probe), "capture-assembly", str(captured), str(frozen)],
                                    capture_output=True, timeout=10)
            self.assertEqual(copied.returncode, 0, copied.stderr)
            shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
            for value in (42, 43):
                payload.write_bytes(str(value).encode())
                library = directory / f"answer{value}.so"
                result = subprocess.run([compiler, "-fno-integrated-as", shared, "-x", "assembler",
                                         str(captured), "-o", str(library)], capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.answer(library), value)
            payload.unlink()
            source.unlink()
            missing = subprocess.run([compiler, "-fno-integrated-as", shared, "-x", "assembler",
                                      str(captured), "-o", str(directory / "missing.so")],
                                     capture_output=True, timeout=20)
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse((directory / "missing.so").exists())
            replay = directory / "replayed.so"
            result = subprocess.run([compiler, "-fno-integrated-as", shared, "-x", "assembler",
                                     str(frozen), "-o", str(replay)], capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.answer(replay), 42)

    def test_apple_failed_capture_keeps_cold_consistency_gate_open(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I characterize failed Apple capture with restored inputs")
        for case in measure(shutil.which("cc"), ("assembler-external-macro-query-failure",))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (43, 42, 42))
                self.assertFalse(case["reuse_record"])
                self.assertFalse(case["generation_reused"])
                self.assertFalse(case["retained_assembly"])
                with self.assertRaises(SystemExit): require_consistent({"cases": [case]})

    def test_apple_external_query_failure_and_recovery(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I exercise selected Apple backend query failures")
        for failure in ("empty", "multiple", "truncated", "oversize", "error", "timeout"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="nano-as-query-failure-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                payload = module / "payload.bin"
                payload.write_bytes(b"42")
                assembly = '.data\n.globl _snapshot_payload\n_snapshot_payload:\n.macro payload file\n.incbin "\\file"\n.endm\n' + f'payload "{payload}"\n.text\n'
                (module / "answer.c").write_text('__asm__(' + json.dumps(assembly) + ');\n'
                    'extern const unsigned char snapshot_payload[];\n'
                    'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"],
                                                               "cflags": ["-fno-integrated-as"]}))
                wrapper = directory / "cc"
                banner = 'Apple clang version 21.0.0 (fixture)\n'
                wrapper.write_text(f'''#!{sys.executable}
import os, subprocess, sys, time
failure = os.getenv("NANO_QUERY_FAILURE")
if "-###" in sys.argv and failure:
    if failure == "timeout": time.sleep(60)
    if failure == "multiple": sys.stderr.write({banner!r} + ' "/missing/tool"\\n "/missing/other"\\n')
    if failure == "truncated": sys.stderr.write({banner!r} + ' "/unterminated')
    if failure == "oversize": sys.stderr.write("x" * 20000)
    sys.exit(1 if failure == "error" else 0)
os.execv({shutil.which("cc")!r}, [{shutil.which("cc")!r}] + sys.argv[1:])
''')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)
                env["NANO_QUERY_FAILURE"] = failure
                started = time.monotonic()
                self.support.probe_path("build", module, env, timeout=20)
                self.assertLess(time.monotonic() - started, 15)
                generation = self.support.probe_path("directory", module, env)
                self.assertFalse((generation / "source_hashes.json").exists())
                self.assertFalse(list(generation.glob("__expanded_*")))
                self.assertFalse(list(generation.glob("__snapshot_*.s")))
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                del env["NANO_QUERY_FAILURE"]
                self.support.probe_path("build", module, env, timeout=20)
                recovered = self.support.probe_path("directory", module, env)
                self.assertTrue((recovered / "source_hashes.json").is_file())
                self.assertTrue((recovered / "__expanded_0_0.s").is_file())
                self.support.probe_path("build", module, env, timeout=20)
                self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_apple_external_assembler_report_boundaries(self):
        if sys.platform != "darwin": self.skipTest("I decode the Apple driver report here")
        banner = "Apple clang version 21.0.0 (fixture)\nTarget: arm64-apple-darwin\nThread model: posix\nInstalledDir: /fixture\n"
        command = ' "/fixture/compiler with space" "-cc1as" "-o" "object with \'quotes\'.o"\n'
        accepted = banner + command
        accepted_reports = [accepted, banner + "clang: warning: argument unused during compilation: '-fPIC' [-Wunused-command-line-argument]\n" + command]
        reports = accepted_reports + ["", banner, command, accepted + command, accepted + "unexpected command\n",
                   accepted.replace("21.0.0", "22.0.0"), banner + ' "/unterminated\n',
                   accepted.replace("21.0.0", "21.0.01"),
                   banner + ' "/fixture/tool" "$(touch forbidden)"\n',
                   banner + ' "/fixture/tool"; touch forbidden\n',
                   banner + ' "/fixture/tool" "' + "x" * 4096 + '"\n',
                   banner + ' "/fixture/tool"' + ' "x"' * 252 + '\n']
        for report in reports:
            with self.subTest(report=report[:90]):
                result = subprocess.run([str(self.support.probe), "assembler-report", report], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0 if report in accepted_reports else 1, result.stderr)
                self.assertEqual(result.stderr, b"")
                if result.returncode == 0:
                    self.assertEqual(json.loads(result.stdout), ["/fixture/compiler with space", "-cc1as", "-o", "object with 'quotes'.o"])

    def test_apple_selected_external_backend_expands_macro_reads(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I characterize the selected Apple external assembler backend")
        compiler = shutil.which("cc")
        for spelling in ("literal", "nested", "macro"):
            for flags in ([], ["-O2", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror"]):
                with self.subTest(spelling=spelling, flags=flags), tempfile.TemporaryDirectory(prefix="nano-selected-as-") as tmp:
                    directory = Path(tmp)
                    payload, inner, outer = (directory / name for name in ("payload with 'quotes'.bin", "inner.s", "outer.s"))
                    payload_bytes = b"xx42yy" + bytes(range(256))
                    payload.write_bytes(payload_bytes)
                    literal = f'.incbin "{payload}", 2, {len(payload_bytes) - 2}\n'
                    inner.write_text(literal if spelling != "macro" else
                        '.macro read_payload file\n.incbin "\\file", 2, ' + str(len(payload_bytes) - 2) + '\n.endm\n' +
                        f'read_payload "{payload}"\n.if 0\n.incbin "missing.bin"\n.endif\n')
                    outer.write_text(f'.include "{inner}"\n')
                    directive = literal if spelling == "literal" else f'.include "{outer}"\n'
                    source, raw, captured = (directory / name for name in ("answer.c", "raw.s", "captured.s"))
                    source.write_text('__asm__(' + json.dumps('.data\n.globl _snapshot_payload\n_snapshot_payload:\n' +
                        directive + '.p2align 3\n1:\n.quad 1b\n.text\n') + ');\nextern const unsigned char snapshot_payload[];\n'
                        'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                    def run(args):
                        result = subprocess.run(args, capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        return result
                    def query(args):
                        result = run([*args, "-###"])
                        commands = [shlex.split(line) for line in result.stderr.decode().splitlines()
                                    if line.lstrip().startswith('"')]
                        self.assertEqual(len(commands), 1, result.stderr)
                        return commands[0]
                    run([compiler, "-fno-integrated-as", "-fPIC", *flags, "-S", str(source), "-o", str(raw)])
                    direct_object = directory / "direct.o"
                    external = query([compiler, "-fno-integrated-as", "-fPIC", "-c", "-x", "assembler",
                                      str(raw), "-o", str(direct_object)])
                    self.assertNotIn("-cc1as", external)
                    backend = query(external)
                    self.assertEqual(backend[1], "-cc1as")
                    self.assertEqual(backend.count("-filetype"), 1)
                    self.assertEqual(backend.count("-o"), 1)
                    self.assertEqual(backend[backend.index("-filetype") + 1], "obj")
                    run(external)
                    expanded = backend.copy()
                    expanded[expanded.index("-filetype") + 1] = "asm"
                    expanded[expanded.index("-o") + 1] = str(captured)
                    self.assertEqual(sum(a != b for a, b in zip(backend, expanded)), 2)
                    expanded.append("-msave-temp-labels")
                    run(expanded)
                    for path in (payload, inner, outer): self.assertNotIn(str(path), captured.read_text())
                    def answer(obj, library):
                        run([compiler, "-dynamiclib", str(obj), "-o", str(library)])
                        return self.answer(library)
                    self.assertEqual(answer(direct_object, directory / "direct.so"), 42)
                    payload.write_bytes(payload_bytes.replace(b"42", b"43", 1))
                    changed_object = directory / "changed.o"
                    changed = external.copy()
                    changed[changed.index("-o") + 1] = str(changed_object)
                    run(changed)
                    self.assertEqual(answer(changed_object, directory / "changed.so"), 43)
                    for path in (payload, inner, outer, source, raw): path.unlink()
                    replay_object = directory / "replay.o"
                    replay = external.copy()
                    replay[replay.index("-o") + 1] = str(replay_object)
                    replay[replay.index(str(raw))] = str(captured)
                    run(replay)
                    self.assertEqual(answer(replay_object, directory / "replay.so"), 42)
                    self.assertEqual(direct_object.read_bytes(), replay_object.read_bytes())

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
            for flags in (["-fno-builtin"], ["-O${NANO_TEST_LEVEL:-2}"],
                          ["-D", "NAME=42"]):
                with self.subTest(flags=flags):
                    (module / "module.json").write_text(json.dumps({"name": "answer_native",
                        "c_sources": ["answer.c"], "cflags": flags}))
                    self.support.probe_path("build", module, env)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertEqual(list(generation.glob("__snapshot_*")), [])
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_response_file_restored_arguments(self):
        observed = measure(shutil.which("cc"), ("response",))
        require_consistent(observed)
        for case in observed["cases"]:
            with self.subTest(case=case):
                self.assertEqual(case["cold_answer"], 42)
                for field in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record", "generation_reused"):
                    self.assertTrue(case[field], field)
                self.assertTrue(case["retained_translation_unit"] or case["retained_assembly"])

    def test_link_driver_response_arguments_are_retained(self):
        observed = measure(shutil.which("cc"), ("link-response", "link-response-platform", "link-response-pkg"))
        require_consistent(observed)
        self.assertEqual(len(observed["cases"]), 6)
        for case in observed["cases"]:
            with self.subTest(case=case):
                self.assertEqual(case["fresh_answer"], 42)
                for field in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record", "generation_reused"):
                    self.assertTrue(case[field], field)

    def test_large_response_arguments_are_retained(self):
        observed = measure(shutil.which("cc"), ("response-large",))
        require_consistent(observed)
        self.assertEqual({case["cache"] for case in observed["cases"]}, {"local", "shared"})
        self.assertEqual(len(observed["cases"]), 2)
        for case in observed["cases"]:
            with self.subTest(case=case):
                self.assertEqual(case["input"], "response-large")
                self.assertEqual(case["input_bytes"], 10212)
                self.assertEqual(case["fresh_answer"], 42)
                for field in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record", "generation_reused"):
                    self.assertTrue(case[field], field)

    def test_many_returned_compile_flags_preserve_count_and_order(self):
        platform = "cflags_macos" if sys.platform == "darwin" else "cflags_linux"
        for origin in ("cflags", platform, "include_dirs", "compiled"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-many-flags-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                values = [f"-DNANO_FLAG_{i}=1" for i in range(1300)]
                if origin == "include_dirs":
                    values = [f"/nano/include/{i}" for i in range(1300)]
                    values[-1] = "/nano/include/" + "x" * 300
                if origin == "compiled": values = [""] * 1300
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c"] if origin == "compiled" else [],
                    "cflags" if origin == "compiled" else origin: values}))
                result = subprocess.run([str(self.support.probe), "build-info", str(module)],
                                        env=env, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                returned = [line[len("compile:"):] for line in result.stdout.decode().splitlines()
                            if line.startswith("compile:")]
                if origin != "compiled":
                    decoded = []
                    for value in returned:
                        for word in shlex.split(value):
                            decoded.extend(shlex.split(Path(word[1:]).read_text()) if word.startswith("@") else [word])
                    returned = decoded
                self.assertEqual(returned, ["-I" + value for value in values] if origin == "include_dirs" else values)

    def test_aggregate_compiler_fragments_preserve_order_and_reuse(self):
        platform = "cflags_macos" if sys.platform == "darwin" else "cflags_linux"
        for origin in ("cflags", platform, "responses", "pkg_config"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-aggregate-flags-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                (module / "answer.c").write_text("long long nano_build_answer(void) { return ANSWER; }\n")
                values = ["-DNANO_PAD=1"] * 1300 + ["-DANSWER=42", "-UANSWER", "-DANSWER=43"]
                metadata = {"name": "answer_native", "c_sources": ["answer.c"]}
                if origin == "responses":
                    response = directory / "small.rsp"
                    response.write_text("-DNANO_PAD=1\n" * 30)
                    metadata["cflags"] = ["@" + str(response)] * 40 + values[-3:]
                elif origin == "pkg_config":
                    metadata["pkg_config"] = [f"fixture-{i}" for i in range(40)]
                    pkg = directory / "pkg-config"
                    pkg.write_text(f'#!{sys.executable}\nimport sys\n'
                        'if "--cflags" in sys.argv:\n'
                        ' index=int(sys.argv[-1].rsplit("-",1)[1])\n'
                        ' print("-DNANO_PAD=1 "*30 + ("-DANSWER=42 -UANSWER -DANSWER=43" if index == 39 else ""))\n')
                    pkg.chmod(0o700)
                    env["PKG_CONFIG"] = str(pkg)
                else: metadata[origin] = values
                (module / "module.json").write_text(json.dumps(metadata))
                original = (module / "module.json").read_bytes()
                self.support.probe_path("build", module, env, timeout=30 if origin == "pkg_config" else 10)
                generation = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                self.support.probe_path("build", module, env, timeout=30 if origin == "pkg_config" else 10)
                self.assertEqual(self.support.probe_path("directory", module, env), generation)
                self.assertTrue((generation / "source_hashes.json").is_file())
                self.assertEqual((module / "module.json").read_bytes(), original)

    def test_include_transport_preserves_search_order_and_lifetime(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled), tempfile.TemporaryDirectory(prefix="nano-include-transport-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                first = directory / "first 'quoted' $HOME"
                second = directory / 'second "quoted"'
                for path, answer in ((first, 42), (second, 43)):
                    path.mkdir()
                    (path / "answer.h").write_text(f"#define ANSWER {answer}\n")
                missing = [str(directory / f"missing-{i}") for i in range(200)]
                source = module / "answer.c"
                source.write_text("#include <answer.h>\nlong long nano_build_answer(void) { return ANSWER; }\n")
                later = directory / "later.c"
                later.write_text("#include <answer.h>\nANSWER\n")
                generations = []
                for paths, answer in (((first, second), 42), ((second, first), 43)):
                    metadata = {"name": "answer_native", "c_sources": ["answer.c"] if compiled else [],
                                "include_dirs": missing + list(map(str, paths))}
                    (module / "module.json").write_text(json.dumps(metadata))
                    original = (module / "module.json").read_bytes()
                    result = subprocess.run([str(self.support.probe), "build-info", str(module)],
                                            env=env, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    flags = " ".join(line[len("compile:"):] for line in result.stdout.decode().splitlines()
                                     if line.startswith("compile:"))
                    self.assertTrue(any(word.startswith("@") for word in shlex.split(flags)))
                    replay = subprocess.run(shlex.join([shutil.which("cc"), "-E", "-P", str(later)]) + " " + flags,
                                            shell=True, cwd=directory, env=env, capture_output=True, timeout=15)
                    self.assertEqual(replay.returncode, 0, replay.stderr)
                    self.assertEqual(replay.stdout.strip(), str(answer).encode())
                    if compiled:
                        generation = self.support.probe_path("directory", module, env)
                        generations.append(generation)
                        self.assertEqual(self.answer(self.support.probe_path("library", module, env)), answer)
                        self.support.probe_path("build", module, env)
                        self.assertEqual(self.support.probe_path("directory", module, env), generation)
                        self.assertTrue((generation / "source_hashes.json").is_file())
                    self.assertEqual((module / "module.json").read_bytes(), original)
                if compiled: self.assertNotEqual(*generations)

    def test_compile_flag_allocation_failures_are_atomic(self):
        for failure in (*map(str, range(6)), "overflow"):
            with self.subTest(failure=failure):
                result = subprocess.run([str(self.support.probe), "compile-flags-allocation", failure],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_coalesced_flag_allocation_failures_are_atomic(self):
        result = subprocess.run([str(self.support.probe), "coalesce-allocation", "all"],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_many_returned_link_flags_preserve_count_and_order(self):
        platform = "ldflags_macos" if sys.platform == "darwin" else "ldflags_linux"
        origins = ["system_libs", "ldflags", platform, "compiled"]
        if sys.platform == "darwin": origins.append("frameworks")
        for origin in origins:
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-many-links-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                values = [f"nano_{i}" for i in range(1300)]
                if origin == "system_libs": values[-1] = "nano_" + "x" * 300
                if origin in ("ldflags", platform): values = ["-L/" + value for value in values]
                if origin == "compiled": values = [""] * 1300
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c"] if origin == "compiled" else [],
                    "ldflags" if origin == "compiled" else origin: values}))
                result = subprocess.run([str(self.support.probe), "build-info", str(module)],
                                        env=env, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                returned = [line[len("link:"):] for line in result.stdout.decode().splitlines()
                            if line.startswith("link:")]
                if origin == "compiled":
                    self.assertTrue(returned[0].endswith("answer_native.o"))
                    returned = returned[1:]
                expected = (["-l" + value for value in values] if origin == "system_libs" else
                            [word for value in values for word in ("-framework", value)]
                            if origin == "frameworks" else values)
                self.assertEqual(returned, expected)

    def test_shared_link_preserves_framework_pairs(self):
        if sys.platform != "darwin": self.skipTest("I exercise Darwin framework pairs here")
        with tempfile.TemporaryDirectory(prefix="nano-framework-pairs-") as tmp:
            module, _, env = self.support.support.foreign_build_fixture(Path(tmp))
            (module / "module.json").write_text(json.dumps({"name": "answer_native",
                "c_sources": ["answer.c"], "frameworks": ["Foundation", "Security"]}))
            self.support.probe_path("build", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_link_flag_allocation_failures_are_atomic(self):
        result = subprocess.run([str(self.support.probe), "link-flags-allocation", "all"],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_link_does_not_drop_tail_flags_or_repeated_libraries(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-tail-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            self.support.probe_path("build", module, env)
            previous = self.support.probe_path("directory", module, env)
            metadata = {"name": "answer_native", "c_sources": ["answer.c"],
                        "system_libs": ["m", "c", "m"], "ldflags": [" "] * 1300 + ["-Wl,-nano-invalid-option"]}
            (module / "module.json").write_text(json.dumps(metadata))
            result = subprocess.run([str(self.support.probe), "shared-link-command", str(module)],
                                    env=env, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            words = []
            for word in shlex.split(result.stdout.decode()):
                words.extend(shlex.split(Path(word[1:]).read_text()) if word.startswith("@") else [word])
            self.assertEqual([word for word in words if word.startswith("-l")], ["-lm", "-lc", "-lm"])
            self.assertIn("-Wl,-nano-invalid-option", words)
            too_small = subprocess.run([str(self.support.probe), "shared-link-command", str(module), "64"],
                                       env=env, capture_output=True, timeout=15)
            self.assertNotEqual(too_small.returncode, 0)
            self.assertEqual(too_small.stdout, b"")
            rejected = subprocess.run([str(self.support.probe), "build", str(module)],
                                      env=env, capture_output=True, timeout=15)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"nano-invalid-option", rejected.stderr)
            self.assertEqual(self.support.probe_path("directory", module, env), previous)
            metadata.pop("ldflags")
            metadata.pop("system_libs")
            (module / "module.json").write_text(json.dumps(metadata))
            self.support.probe_path("build", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_response_words_match_the_real_compiler(self):
        with tempfile.TemporaryDirectory(prefix="nano-response-words-") as tmp:
            directory = Path(tmp)
            source, response = directory / "input.c", directory / "flags with spaces.rsp"
            source.write_text("VALUE\n")
            fragments = ["-DVALUE=42", "'-DVALUE=two words'", '"-DVALUE=cost$HOME"',
                         r"'-DVALUE=a\b'", r'"-DVALUE=a\qb"', r"-DVALUE=one\ two",
                         r'-DVALUE=\"literal\ text\"', "-DVALUE=42\r\n-U VALUE\n-D VALUE=43"]
            for fragment in fragments:
                with self.subTest(fragment=fragment):
                    response.write_text(fragment)
                    captured = subprocess.run([str(self.support.probe), "capture-response", shlex.quote("@" + str(response))],
                                              cwd=directory, capture_output=True, timeout=10)
                    self.assertEqual(captured.returncode, 0, captured.stderr)
                    arguments = shlex.split(captured.stdout.decode())
                    self.assertFalse(any(arg.startswith("@") for arg in arguments))
                    prefix = [shutil.which("cc"), "-E", "-P", str(source)]
                    native = subprocess.run(prefix + ["@" + str(response)], cwd=directory, capture_output=True, timeout=10)
                    replay = subprocess.run(" ".join(shlex.quote(arg) for arg in prefix) + " " + captured.stdout.decode(),
                                            shell=True, cwd=directory, capture_output=True, timeout=10)
                    self.assertEqual(native.returncode, 0, native.stderr)
                    self.assertEqual((replay.returncode, replay.stdout), (native.returncode, native.stdout), replay.stderr)
            nested = directory / "nested.rsp"
            nested.write_text("-DVALUE=42\n")
            sub = directory / "sub"
            sub.mkdir()
            (sub / "nested.rsp").write_text("-DVALUE=43\n")
            outer = sub / "outer.rsp"
            outer.write_text("@nested.rsp\n")
            captured = subprocess.run([str(self.support.probe), "capture-response", "@" + str(outer)],
                                      cwd=directory, capture_output=True, timeout=10)
            self.assertEqual(captured.returncode, 0, captured.stderr)
            native = subprocess.run([shutil.which("cc"), "-E", "-P", str(source), "@" + str(outer)],
                                    cwd=directory, capture_output=True, timeout=10)
            self.assertEqual(native.returncode, 0, native.stderr)
            self.assertEqual(native.stdout.strip(), b"42")
            self.assertEqual(shlex.split(captured.stdout.decode()), ["-DVALUE=42"])

    def test_response_transport_outlives_build_info_and_rejects_changed_sidecars(self):
        platform = "cflags_macos" if sys.platform == "darwin" else "cflags_linux"
        for origin in ("cflags", platform, "pkg_config"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-response-lifetime-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                response = directory / "arguments.rsp"
                contents = ('-O2\n' * 600 + '-DANSWER=42\n"-DVALUE=cost$HOME"\n' +
                            r'-DQUOTED=\"literal\ text\"' + '\n' + r"'-DBACK=a\\b'" + '\n')
                response.write_text(contents)
                metadata = {"name": "answer_native", "c_sources": []}
                fragment = "@" + str(response)
                if origin == "pkg_config":
                    metadata[origin] = ["response-fixture"]
                    pkg = directory / "pkg-config"
                    pkg.write_text(f'#!{sys.executable}\nimport sys\n'
                                   f'if "--cflags" in sys.argv: print({fragment!r})\n')
                    pkg.chmod(0o700)
                    env["PKG_CONFIG"] = str(pkg)
                else: metadata[origin] = [fragment]
                (module / "module.json").write_text(json.dumps(metadata))
                def build_info():
                    return subprocess.run([str(self.support.probe), "build-info", str(module)],
                                          env=env, cwd=directory, capture_output=True, timeout=15)
                result = build_info()
                self.assertEqual(result.returncode, 0, result.stderr)
                flags = " ".join(line[len("compile:"):] for line in result.stdout.decode().splitlines()
                                 if line.startswith("compile:"))
                sidecars = [Path(word[1:]) for word in shlex.split(flags) if word.startswith("@")]
                self.assertEqual(len(sidecars), 1)
                retained = sidecars[0]
                original = retained.read_bytes()
                source = directory / "later.c"
                source.write_text("ANSWER\nVALUE\nQUOTED\nBACK\n")
                prefix = [shutil.which("cc"), "-E", "-P", str(source)]
                direct = subprocess.run(prefix + [fragment], capture_output=True, timeout=15)
                self.assertEqual(direct.returncode, 0, direct.stderr)
                response.write_text("-DANSWER=43\n-DVALUE=changed\n")
                replay = subprocess.run(" ".join(map(shlex.quote, prefix)) + " " + flags,
                                        shell=True, capture_output=True, timeout=15)
                self.assertEqual((replay.returncode, replay.stdout), (0, direct.stdout), replay.stderr)
                response.write_text(contents)
                for failure in ("changed", "symlink", "fifo", "directory"):
                    with self.subTest(failure=failure):
                        retained.unlink()
                        if failure == "changed": retained.write_bytes(original.replace(b"42", b"43"))
                        elif failure == "symlink": retained.symlink_to(response)
                        elif failure == "fifo": os.mkfifo(retained)
                        else: retained.mkdir()
                        rejected = build_info()
                        self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
                        if failure == "directory": retained.rmdir()
                        else: retained.unlink()
                        repaired = build_info()
                        self.assertEqual(repaired.returncode, 0, repaired.stderr)
                        self.assertEqual(retained.read_bytes(), original)
                retained.unlink()
                with ThreadPoolExecutor(max_workers=6) as pool:
                    concurrent = list(pool.map(lambda _: build_info(), range(6)))
                for result in concurrent:
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(retained.read_bytes(), original)
                self.assertFalse([path for path in retained.parent.glob(".nano-args-*")
                                  if path.suffix != ".rsp"])

    def test_response_rebuild_errors_and_recovery(self):
        platform = "cflags_macos" if sys.platform == "darwin" else "cflags_linux"
        for origin in ("cflags", platform, "pkg_config"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-response-build-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                env["NANO_BUILD_CACHE"] = str(directory / "cache")
                source, outer, nested = module / "answer.c", directory / "outer.rsp", directory / "nested.rsp"
                source.write_text("long long nano_build_answer(void) { return ANSWER; }\n")
                outer.write_text(shlex.quote("@" + str(nested)) + "\n")
                padding = "-O2\n" * 600
                nested.write_text("-D ANSWER=42\n" + padding)
                fragment = shlex.quote("@" + str(outer))
                metadata = json.loads((module / "module.json").read_text())
                if origin == "pkg_config":
                    metadata["pkg_config"] = ["nano-response-fixture"]
                    tool = directory / "pkg-config"
                    tool.write_text(f'#!{sys.executable}\nimport sys\n'
                                    f'if "--cflags" in sys.argv: print({fragment!r})\n'
                                    'elif "--modversion" in sys.argv: print("1.0")\n')
                    tool.chmod(0o700)
                    env["PKG_CONFIG"] = str(tool)
                else: metadata[origin] = [fragment]
                (module / "module.json").write_text(json.dumps(metadata))
                original_metadata = (module / "module.json").read_bytes()
                def needs():
                    result = subprocess.run([str(self.support.probe), "needs-rebuild", str(module)],
                                            env=env, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return result.stdout.strip()
                self.support.probe_path("build", module, env)
                first = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertEqual(needs(), b"0")
                self.support.probe_path("build", module, env)
                self.assertEqual(self.support.probe_path("directory", module, env), first)
                stamp = nested.stat()
                nested.write_text("-D ANSWER=43\n" + padding)
                os.utime(nested, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assertEqual(needs(), b"1")
                self.support.probe_path("build", module, env)
                previous = self.support.probe_path("directory", module, env)
                self.assertNotEqual(previous, first)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                transports = [path for path in previous.parent.glob(".nano-args-*.rsp")
                              if b'ANSWER=43' in path.read_bytes()]
                self.assertEqual(len(transports), 1)
                transport = transports[0]
                retained_bytes = transport.read_bytes()
                transport.chmod(0o600)
                transport.write_bytes(retained_bytes.replace(b'ANSWER=43', b'ANSWER=44'))
                rejected = subprocess.run([str(self.support.probe), "build", str(module)],
                                          env=env, capture_output=True, timeout=15)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertEqual(self.support.probe_path("directory", module, env), previous)
                transport.unlink()
                self.support.probe_path("build", module, env)
                self.assertEqual(transport.read_bytes(), retained_bytes)
                self.assertEqual(self.support.probe_path("directory", module, env), previous)
                for failure in ("missing", "cycle", "fifo"):
                    with self.subTest(failure=failure):
                        nested.unlink()
                        if failure == "cycle": nested.write_text("@" + str(outer))
                        if failure == "fifo": os.mkfifo(nested)
                        self.assertEqual(needs(), b"1")
                        result = subprocess.run([str(self.support.probe), "build", str(module)],
                                                env=env, capture_output=True, timeout=10)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(self.support.probe_path("directory", module, env), previous)
                        self.assertFalse(list(previous.parent.glob(".nano-build-*")))
                        if failure == "fifo": nested.unlink()
                        nested.write_text("-D ANSWER=43\n" + padding)
                        self.support.probe_path("build", module, env)
                        self.assertEqual(self.support.probe_path("directory", module, env), previous)
                self.assertEqual((module / "module.json").read_bytes(), original_metadata)

    def test_response_unsupported_fragments_keep_the_original_path(self):
        with tempfile.TemporaryDirectory(prefix="nano-response-boundary-") as tmp:
            directory = Path(tmp)
            response = directory / "flags.rsp"
            fragment = "@" + str(response)
            for data in (b'"-DANSWER=42', b"-DANSWER=42\\", b"-O2 " * 14000,
                         b" " * 65537, b"-DANSWER=42\x00-O3", b"--driver-mode=cl -DANSWER=42"):
                with self.subTest(size=len(data), prefix=data[:20]):
                    response.write_bytes(data)
                    result = subprocess.run([str(self.support.probe), "capture-response", fragment],
                                            capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.decode().strip(), fragment)
            response.unlink()
            shell_fragment = "-O${NANO_RESPONSE_LEVEL:-2} " + fragment
            result = subprocess.run([str(self.support.probe), "capture-response", shell_fragment],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.decode().strip(), shell_fragment)

    def test_response_driver_mode_declines_the_whole_argument_capture(self):
        for placement in ("metadata", "escaped-metadata", "package", "escaped-package",
                          "nested-package", "escaped-nested-package", "named-cl-driver"):
            with self.subTest(placement=placement), tempfile.TemporaryDirectory(prefix="nano-response-mode-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                response, mode = directory / "flags.rsp", directory / "mode.rsp"
                response.write_text("-DANSWER=42\n")
                mode.write_text("--driv\\er-mode=cl\n" if placement == "escaped-nested-package" else "--driver-mode=cl\n")
                metadata = {"name": "answer_native", "c_sources": [],
                            "cflags": ["@" + str(response)]}
                if placement == "metadata": metadata["cflags"].append("--driver-mode=cl")
                elif placement == "escaped-metadata": metadata["cflags"].append("--driv\\er-mode=cl")
                elif placement == "named-cl-driver":
                    driver = directory / "clang-cl"
                    compiler = shutil.which("cc")
                    driver.write_text(f'#!{sys.executable}\nimport os,sys\nos.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])\n')
                    driver.chmod(0o700)
                    env["NANO_CC"] = str(driver)
                else:
                    metadata["pkg_config"] = ["mode-fixture", "response-fixture"]
                    mode_flag = ("--driver-mode=cl" if placement == "package" else
                                 "--driv\\er-mode=cl" if placement == "escaped-package" else "@" + str(mode))
                    pkg = directory / "pkg-config"
                    pkg.write_text(f'#!{sys.executable}\nimport sys\n'
                        f'if "--cflags" in sys.argv: print({mode_flag!r} if "mode-fixture" in sys.argv else {("@" + str(response))!r})\n')
                    pkg.chmod(0o700)
                    env["PKG_CONFIG"] = str(pkg)
                (module / "module.json").write_text(json.dumps(metadata))
                result = subprocess.run([str(self.support.probe), "build-info", str(module)],
                                        env=env, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(("@" + str(response)).encode(), result.stdout)
                self.assertNotIn(b"-DANSWER=42", result.stdout)

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
            pch_generation = None
            for _ in range(2):
                self.support.probe_path("build", module, env)
                generation = self.support.probe_path("directory", module, env)
                self.assertNotEqual(generation, plain)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertTrue(list(generation.glob("__pch_*.gch")), list(generation.iterdir()))
                self.assertTrue((generation / "source_hashes.json").exists())
                if pch_generation is not None: self.assertEqual(generation, pch_generation)
                pch_generation = generation
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
    pathlib.Path({str(calls)!r}).write_text("assembly" if any(a.endswith(".s") for a in sys.argv) else
        "snapshot" if any(a.endswith(".i") for a in sys.argv) else "original")
os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            for split in range(1, len(b"#pragma GCC pch_preprocess")):
                with self.subTest(split=split):
                    env["NANO_TEST_PCH_SPLIT"] = str(split)
                    self.support.probe_path("build", module, env)
                    self.assertEqual(calls.read_text(), "assembly")
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertTrue((generation / "source_hashes.json").exists())
                    self.assertTrue(list(generation.glob("__pch_*.gch")))

    def test_gcc_retained_pch_relocation_trial(self):
        """I establish relocation viability, not production PCH capture."""
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, timeout=10)
        if b"Free Software Foundation" not in version.stdout:
            self.skipTest("I need GCC's PCH preprocessing directive")
        with tempfile.TemporaryDirectory(prefix="nano-pch-relocation-") as tmp:
            directory = Path(tmp)
            original, retained = directory / "original", directory / "private copy"
            original.mkdir()
            retained.mkdir()
            header, source = original / "answer.h", original / "answer.c"
            pch, copy = original / "answer.h.gch", retained / "frozen.gch"
            header.write_text("#define ANSWER 42\n")
            source.write_text('#include "answer.h"\nlong long nano_build_answer(void) { return ANSWER; }\n')
            flags = [compiler, "-fPIC", "-D_POSIX_C_SOURCE=200809L"]
            subprocess.run(flags + ["-x", "c-header", str(header), "-o", str(pch)],
                           capture_output=True, check=True, timeout=10)
            result = subprocess.run(flags + ["-E", "-fpch-preprocess", str(source)],
                                    capture_output=True, check=True, timeout=10)
            marker = f'#pragma GCC pch_preprocess "{pch}"'.encode()
            self.assertEqual(result.stdout.count(marker), 1)
            copy.write_bytes(pch.read_bytes())
            copy.chmod(0o400)
            frozen = retained / "input.i"
            frozen.write_bytes(result.stdout.replace(marker,
                f'#pragma GCC pch_preprocess "{copy}"'.encode()))
            library = retained / "answer.so"
            def compile_frozen():
                return subprocess.run(flags + ["-shared", "-x", "cpp-output", str(frozen),
                    "-o", str(library)], capture_output=True, timeout=10)
            # The original PCH now disagrees with the private retained copy.
            header.write_text("#define ANSWER 43\n")
            subprocess.run(flags + ["-x", "c-header", str(header), "-o", str(pch)],
                           capture_output=True, check=True, timeout=10)
            fresh = original / "fresh.so"
            subprocess.run(flags + ["-shared", str(source), "-o", str(fresh)],
                           capture_output=True, check=True, timeout=10)
            self.assertEqual(self.answer(fresh), 43)
            for remove_originals in (False, True):
                with self.subTest(remove_originals=remove_originals):
                    if remove_originals:
                        for path in (pch, header, source): path.unlink()
                    result = compile_frozen()
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.answer(library), 42)
            copy.unlink()
            self.assertNotEqual(compile_frozen().returncode, 0)

    def test_pch_rewriter_preserves_input_on_rejected_capture(self):
        with tempfile.TemporaryDirectory(prefix="nano-pch-rewriter-") as tmp:
            directory = Path(tmp)
            pch = directory / "source with spaces.gch"
            pch.write_bytes(b"retained binary\x00bytes")
            good = f'#pragma GCC pch_preprocess "{pch}"\n'.encode()
            variants = [b"int value;\n", b" " + good, good.rstrip() + b" extra\n",
                        good + b"\x00\n", good.replace(b"source with", b"source\\ with"),
                        f'#pragma GCC pch_preprocess "{directory / "missing"}"\n'.encode(),
                        f'#pragma GCC pch_preprocess "{directory}"\n'.encode()]
            for number, data in enumerate([good] + variants):
                with self.subTest(number=number):
                    stage = directory / str(number)
                    stage.mkdir()
                    snapshot = stage / "input.i"
                    snapshot.write_bytes(data)
                    result = subprocess.run([str(self.support.probe), "capture-pch", snapshot, stage],
                                            capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0 if number == 0 else 1, result.stderr)
                    self.assertFalse(Path(str(snapshot) + ".pch").exists())
                    if number:
                        self.assertEqual(snapshot.read_bytes(), data)
                    else:
                        copy = stage / "__pch_0_0_0.gch"
                        self.assertEqual(copy.read_bytes(), pch.read_bytes())
                        self.assertEqual(snapshot.read_text(), f'#pragma GCC pch_preprocess "{copy}"\n')
            stage = directory / "occupied"
            stage.mkdir()
            snapshot = stage / "input.i"
            snapshot.write_bytes(good)
            temporary = Path(str(snapshot) + ".pch")
            temporary.write_bytes(b"I already exist")
            result = subprocess.run([str(self.support.probe), "capture-pch", snapshot, stage],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(temporary.read_bytes(), b"I already exist")
            self.assertEqual(snapshot.read_bytes(), good)

    def test_gcc_pch_restoration_and_private_copy_failure(self):
        if self.clang: self.skipTest("I exercise GCC retained PCH inputs")
        compiler = shutil.which("cc")
        for shared, shared_source in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(shared=shared, shared_source=shared_source), tempfile.TemporaryDirectory(prefix="nano-pch-build-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                source, header = module / "answer.c", module / "answer.h"
                pch, replacement = module / "answer.h.gch", directory / "replacement.gch"
                source.write_text('#include "answer.h"\n__attribute__((visibility("default"))) '
                                  'long long nano_build_answer(void) { return ANSWER; }\n')
                if shared_source:
                    source.rename(module / "pch_answer.c")
                    source.write_text("long long nano_build_anchor(void) { return 0; }\n")
                    source = module / "pch_answer.c"
                    metadata = json.loads((module / "module.json").read_text())
                    metadata["shared_c_sources"] = [source.name]
                    (module / "module.json").write_text(json.dumps(metadata))
                flags = [compiler, "-fPIC", "-D_POSIX_C_SOURCE=200809L", "-x", "c-header"]
                if shared_source: flags.append("-fvisibility=hidden")
                for answer, output in ((43, replacement), (42, pch)):
                    header.write_text(f"#define ANSWER {answer}\n")
                    subprocess.run(flags + [str(header), "-o", str(output)],
                                   capture_output=True, check=True, timeout=10)
                wrapper, calls = directory / "cc", directory / "calls"
                wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, re, subprocess, sys
pch = pathlib.Path({str(pch)!r})
if os.getenv("NANO_TEST_PCH_BREAK") and "-S" in sys.argv:
    for arg in sys.argv[1:]:
        if arg.endswith(".i"):
            for path in re.findall(r'#pragma GCC pch_preprocess "([^"\\n]+)"', pathlib.Path(arg).read_text()):
                pathlib.Path(path).unlink(missing_ok=True)
if "-c" in sys.argv and any(arg.endswith(".s") for arg in sys.argv):
    original, stamp = pch.read_bytes(), pch.stat()
    pch.write_bytes(pathlib.Path({str(replacement)!r}).read_bytes())
    with open({str(calls)!r}, "a") as file: file.write("mutated\\n")
    try:
        result = subprocess.run([{compiler!r}] + sys.argv[1:])
    finally:
        pch.write_bytes(original)
        os.utime(pch, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    sys.exit(result.returncode)
os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])
''')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)
                original, stamp = pch.read_bytes(), pch.stat()
                self.support.probe_path("build", module, env)
                generation = self.support.probe_path("directory", module, env)
                self.assertTrue((generation / "source_hashes.json").exists())
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertTrue(calls.read_text())
                self.assertEqual(pch.read_bytes(), original)
                self.assertEqual(pch.stat().st_mtime_ns, stamp.st_mtime_ns)
                self.support.probe_path("build", module, env)
                self.assertEqual(self.support.probe_path("directory", module, env), generation)
                source.write_text(source.read_text() + "\n/* I force a replacement attempt. */\n")
                env["NANO_TEST_PCH_BREAK"] = "1"
                result = subprocess.run([str(self.support.probe), "build", str(module)],
                                        env=env, capture_output=True, timeout=10)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), generation)
                self.assertFalse(list(generation.parent.glob(".nano-build-*")))
                del env["NANO_TEST_PCH_BREAK"]
                self.support.probe_path("build", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                pch.write_bytes(replacement.read_bytes())
                self.support.probe_path("build", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)


if __name__ == "__main__":
    unittest.main()
