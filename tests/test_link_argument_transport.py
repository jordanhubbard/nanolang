"""I reject incomplete or inconsistent linker transport measurements."""

import unittest
import errno
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from tests.characterize_link_argument_transport import measure, require_consistent
from tests.characterize_source_snapshot import measure as measure_snapshot
from tests import test_bytecode_shadows as shadows
from tests.characterize_linker_response_grammar import (
    materialize, require_argument_equivalent, require_equivalent, require_retained_equivalent,
)


class LinkArgumentAcceptance(unittest.TestCase):
    def test_forwarded_invocation_allocation_is_atomic_and_retryable(self):
        with tempfile.TemporaryDirectory(prefix="nano-forwarded-allocation-") as tmp:
            directory = Path(tmp)
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
            compiler, pkg = directory / "cc", directory / "pkg-config"
            compiler.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then\n"
                "printf 'Free Software Foundation\\n'\nelse\nprintf 'GNU ld (fixture) 2.40\\n'\nfi\n")
            pkg.write_text("#!/bin/sh\nprintf '%s\\n' '-Wl,@link.rsp'\n")
            compiler.chmod(0o700)
            pkg.chmod(0o700)
            env.update(NANO_CC=str(compiler), PKG_CONFIG=str(pkg))
            platform = "macos" if sys.platform == "darwin" else "linux"
            metadata = {"name": "answer_native", "c_sources": [], "pkg_config": ["fixture"]}
            metadata.update({group: ["-Wl,@link.rsp"] for group in
                ("cflags", "ldflags", "cflags_" + platform, "ldflags_" + platform)})
            (module / "module.json").write_text(json.dumps(metadata))
            (directory / "link.rsp").write_text("-lm\n")
            result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                "forwarded-invocation-allocation", str(module)], cwd=directory, env=env,
                capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_forwarded_restored_inputs_preserve_results_and_reuse(self):
        result = measure_snapshot(shutil.which("cc"), (
            "link-response-forwarded", "link-response-forwarded-platform", "link-response-forwarded-pkg"))
        self.assertEqual(len(result["cases"]), 6)
        for case in result["cases"]:
            with self.subTest(input=case["input"], cache=case["cache"]):
                for field in ("cold_answer", "warm_answer", "fresh_answer"):
                    self.assertEqual(case[field], 42)
                for field in ("bytes_restored", "size_preserved", "mtime_preserved",
                              "reuse_record", "generation_reused"):
                    self.assertTrue(case[field], field)

    def test_forwarded_response_edits_invalidate_all_argument_groups(self):
        platform = "macos" if sys.platform == "darwin" else "linux"
        for origin in ("ldflags", "ldflags_" + platform, "cflags", "cflags_" + platform,
                       "pkg_cflags", "pkg_libs"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-forwarded-edit-") as tmp:
                directory = Path(tmp) / "comma, space"
                directory.mkdir()
                module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
                compiler = shutil.which("cc")
                env["NANO_CC"] = compiler

                def run(argv):
                    result = subprocess.run([str(word) for word in argv], cwd=directory, env=env,
                                            capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return result.stdout.decode().strip()

                for value in (42, 43):
                    member, obj = directory / "member.c", directory / "member.o"
                    member.write_text(f"long long selected(void) {{ return {value}; }}\n")
                    run([compiler, "-fPIC", "-c", member, "-o", obj])
                    run(["ar", "rcs", directory / f"selected{value}.a", obj])
                (module / "answer.c").write_text("extern long long selected(void);\n"
                    "long long nano_build_answer(void) { return selected(); }\n")
                response = directory / "link.rsp"
                response.write_text(json.dumps(str(directory / "selected42.a")) + "\n")
                stamp = response.stat()
                metadata = {"name": "answer_native", "c_sources": ["answer.c"]}
                if origin.startswith("pkg_"):
                    metadata["pkg_config"] = ["link-fixture"]
                    pkg = directory / "pkg-config"
                    selector = "--cflags" if origin == "pkg_cflags" else "--libs"
                    pkg.write_text(f"#!{sys.executable}\nimport sys\n"
                                   f"if {selector!r} in sys.argv: print('-Wl,@link.rsp')\n")
                    pkg.chmod(0o700)
                    env["PKG_CONFIG"] = str(pkg)
                else:
                    metadata[origin] = ["-Wl,@link.rsp"]
                (module / "module.json").write_text(json.dumps(metadata))

                def probe(mode):
                    return run([shadows.ROOT / "obj/test_module_generation_probe", mode, module])

                def answer():
                    return int(run([sys.executable, "-c", "import ctypes,sys; "
                        "lib=ctypes.CDLL(sys.argv[1]); lib.nano_build_answer.restype=ctypes.c_int64; "
                        "print(lib.nano_build_answer())", probe("library")]))

                probe("build")
                first = probe("directory")
                self.assertEqual(answer(), 42)
                response.write_text(json.dumps(str(directory / "selected43.a")) + "\n")
                os.utime(response, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assertEqual(response.stat().st_size, stamp.st_size)
                probe("build")
                second = probe("directory")
                self.assertNotEqual(first, second)
                self.assertEqual(answer(), 43)
                probe("build")
                self.assertEqual(probe("directory"), second)
                self.assertEqual(answer(), 43)

    def test_forwarded_source_less_flags_outlive_response(self):
        with tempfile.TemporaryDirectory(prefix="nano-forwarded-lifetime-") as tmp:
            directory = Path(tmp)
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
            env["NANO_CC"] = shutil.which("cc")
            response = directory / "link.rsp"
            response.write_text("-lm -lc -lm\n")
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": [],
                "ldflags": ["-Wl,@link.rsp"]}))
            result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                "build-info", str(module)], cwd=directory, env=env, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            words = [word for line in result.stdout.decode().splitlines() if line.startswith("link:")
                     for word in shlex.split(line[len("link:"):])]
            response.unlink()
            self.assertEqual(words, ["-Xlinker", "-lm", "-Xlinker", "-lc", "-Xlinker", "-lm"])
            later = subprocess.run([shutil.which("cc"), "-dynamiclib" if sys.platform == "darwin" else "-shared",
                "-fPIC", str(module / "answer.c"), "-o", str(directory / "later.so"), *words],
                cwd=directory, env=env, capture_output=True, timeout=30)
            self.assertEqual(later.returncode, 0, later.stderr)

    def test_forwarded_unadmitted_candidate_keeps_original_flags(self):
        with tempfile.TemporaryDirectory(prefix="nano-forwarded-decline-") as tmp:
            directory = Path(tmp)
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
            compiler, marker = directory / "cc", directory / "queried"
            compiler.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                "if '--version' in sys.argv: print('Free Software Foundation')\n"
                f"else: Path({str(marker)!r}).touch(); print('GNU ld (fixture) 2.40')\n")
            compiler.chmod(0o700)
            env["NANO_CC"] = str(compiler)
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": [],
                "cflags": ["-O2"], "ldflags": ["-Wl,@link.rsp"]}))
            for contents in ("-lm -Map forbidden\n", "@link.rsp\n"):
                with self.subTest(contents=contents):
                    (directory / "link.rsp").write_text(contents)
                    result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                        "build-info", str(module)], cwd=directory, env=env, capture_output=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(b"compile:-O2\n", result.stdout)
                    words = [word for line in result.stdout.decode().splitlines() if line.startswith("link:")
                             for word in shlex.split(line[len("link:"):])]
                    self.assertEqual(words, ["-Wl,@link.rsp"])
                    self.assertFalse(marker.exists())
                    self.assertFalse((directory / "forbidden").exists())

    @unittest.skipUnless(sys.platform == "darwin", "I test Apple repeated-root rejection on Darwin")
    def test_forwarded_apple_repeated_root_rejects_and_recovers(self):
        with tempfile.TemporaryDirectory(prefix="nano-forwarded-repeat-") as tmp:
            directory = Path(tmp)
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
            env["NANO_CC"] = shutil.which("cc")
            metadata = {"name": "answer_native", "c_sources": ["answer.c"],
                        "ldflags": ["-Wl,@link.rsp,@link.rsp"]}
            (directory / "link.rsp").write_text("-lm\n")
            (module / "module.json").write_text(json.dumps(metadata))

            def build():
                return subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                    "build", str(module)], cwd=directory, env=env, capture_output=True, timeout=30)

            result = build()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"I reject repeated or cyclic response identities", result.stderr)
            self.assertEqual(list((module / ".build").glob(".nano-gen-*")), [])
            metadata["ldflags"] = ["-Wl,@link.rsp"]
            (module / "module.json").write_text(json.dumps(metadata))
            retry = build()
            self.assertEqual(retry.returncode, 0, retry.stderr)

    def test_argument_gate_distinguishes_identity_rejection_from_capture_failure(self):
        success, failure = {"status": 0, "answer": 42}, {"status": 1, "answer": None}
        require_argument_equivalent({"cases": [
            {"native": success, "captured_arguments": success, "argument_capture_status": 0, "argument_capture_errno": None},
            {"native": failure, "captured_arguments": None, "argument_capture_status": 1, "argument_capture_errno": errno.ELOOP},
        ]})
        for native, candidate, status, error in ((success, None, 1, errno.ELOOP), (failure, None, 1, errno.ENOMEM),
                (success, failure, 0, None), (failure, success, 0, None), (success, success, 1, errno.ELOOP),
                (success, {"status": 0, "answer": 43}, 0, None), (failure, None, -9, errno.ELOOP)):
            with self.subTest(native=native, candidate=candidate, status=status, error=error), self.assertRaises(SystemExit):
                require_argument_equivalent({"cases": [{"native": native, "captured_arguments": candidate,
                    "argument_capture_status": status, "argument_capture_errno": error,
                    "argument_identity_rejection": True, "argument_equivalent": True}]})
        with self.assertRaises(SystemExit): require_argument_equivalent({"cases": []})

    def test_materialized_gate_does_not_turn_decline_into_linker_rejection(self):
        for native in ({"status": 0, "answer": 42}, {"status": 1, "answer": None}):
            with self.subTest(native=native), self.assertRaises(SystemExit):
                require_retained_equivalent({"cases": [{"native": native, "materialized": None,
                    "materialized_equivalent": True}]}, "materialized")

    def test_materialized_graph_preserves_order_and_identity(self):
        with tempfile.TemporaryDirectory(prefix="nano-materialized-order-") as tmp:
            directory = Path(tmp) / "comma, space"
            directory.mkdir()
            left, right = directory / "left.rsp", directory / "right.rsp"
            left.write_text("-Lfirst\n")
            right.write_text("-Lsecond -lselected\n")
            decode = shlex.split  # Only this literal fixture, not a linker grammar.
            expected = ["-Xlinker", "-Lfirst", "-Xlinker", "-Lsecond", "-Xlinker", "-lselected"]
            self.assertEqual(materialize([left, right], "gnu", decode), (expected, None))
            self.assertEqual(materialize([right, left], "apple", decode),
                (expected[2:] + expected[:2], None))
            alias = directory / "alias.rsp"
            alias.symlink_to(left)
            self.assertEqual(materialize([left, alias], "gnu", decode),
                (["-Xlinker", "-Lfirst"] * 2, None))
            self.assertEqual(materialize([left, alias], "apple", decode),
                (None, "repeated resolved response"))
            distinct = directory / "distinct.rsp"
            distinct.hardlink_to(left)
            self.assertEqual(materialize([left, distinct], "apple", decode),
                (["-Xlinker", "-Lfirst"] * 2, None))
            right.write_text(shlex.quote("@" + str(right)))
            self.assertEqual(materialize([left, right], "gnu", decode), (None, "cycle"))
            left.write_bytes(b'"with\r\nline.a"')
            self.assertEqual(materialize([left], "gnu", decode),
                (["-Xlinker", "with\r\nline.a"], None))

    def test_linker_word_probe_checks_exact_tokens_and_no_partial_result(self):
        probe = str(shadows.ROOT / "obj/test_module_generation_probe")
        for grammar in ("gnu", "apple"):
            for payload, expected in (("'with space.a' sel\"ected\".a", ["with space.a", "selected.a"]),
                    ("'' -lm", ["", "-lm"]), ("'$;x'", ["$;x"]),
                    ("-lm 'last", ["-lm", "last"]), ("-lm last\\", ["-lm", "last"]),
                    ("'@inner.rsp", ["@inner.rsp"]), ("@inner.rsp\\", ["@inner.rsp"]),
                    ("-lm\v-lc\f", ["-lm", "-lc"] if grammar == "gnu" else ["-lm\v-lc\f"]),
                    ("x" * 4095, ["x" * 4095])):
                with self.subTest(grammar=grammar, payload=payload[:30]):
                    result = subprocess.run([probe, "link-response-words", grammar, payload],
                        capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout), expected)
            for payload in ("x" * 4096, "-lm " + "x" * 4096):
                result = subprocess.run([probe, "link-response-words", grammar, payload],
                    capture_output=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
        result = subprocess.run([probe, "link-response-words", "unknown", "-lm"],
            capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")

    def test_retained_grammar_gate_checks_observed_results(self):
        success = {"status": 0, "answer": 42}
        failure = {"status": 1, "answer": None}
        require_retained_equivalent({"cases": [{"native": success, "retained": success},
                                               {"native": failure, "retained": failure}]})
        for native, retained in ((success, failure), (failure, success),
                                 (success, {"status": 0, "answer": None}),
                                 (success, {"status": 0, "answer": 43})):
            with self.subTest(native=native, retained=retained), self.assertRaises(SystemExit):
                require_retained_equivalent({"cases": [{"native": native, "retained": retained,
                                                        "retained_equivalent": True}]})
        with self.assertRaises(SystemExit): require_retained_equivalent({"cases": []})

    def test_linker_grammar_gate_checks_observed_results(self):
        result = {"driver_decoder_admitted": True, "native": {"status": 0, "answer": 42},
                  "candidate": {"status": 0, "answer": 42}, "equivalent": True}
        require_equivalent({"cases": [result]})
        for candidate in ({"status": 1, "answer": None}, {"status": 0, "answer": 43},
                          {"status": 0, "answer": None}):
            with self.subTest(candidate=candidate), self.assertRaises(SystemExit):
                require_equivalent({"cases": [{**result, "candidate": candidate}]})
        with self.assertRaises(SystemExit): require_equivalent({"cases": []})
        require_equivalent({"cases": [{"driver_decoder_admitted": False}]})

    def test_link_response_metadata_allocation_rollback(self):
        result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                                 "link-response-allocation", "all"], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_returned_link_response_flags_outlive_inputs(self):
        platform = "ldflags_macos" if sys.platform == "darwin" else "ldflags_linux"
        for origin in ("ldflags", platform, "pkg_config"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(prefix="nano-link-response-lifetime-") as tmp:
                directory = Path(tmp)
                module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
                response = directory / "link.rsp"
                response.write_text("-L/selected42\n-lm\n-lc\n-lm\n")
                metadata = {"name": "answer_native", "c_sources": []}
                if origin == "pkg_config":
                    metadata[origin] = ["link-fixture"]
                    pkg = directory / "pkg-config"
                    pkg.write_text(f"#!{sys.executable}\nimport sys\n"
                                   f"if '--libs' in sys.argv: print({'@' + str(response)!r})\n")
                    pkg.chmod(0o700)
                    env["PKG_CONFIG"] = str(pkg)
                else: metadata[origin] = ["@" + str(response)]
                (module / "module.json").write_text(json.dumps(metadata))

                def invoke():
                    return subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                        "build-info", str(module)], env=env, capture_output=True, timeout=15)

                result = invoke()
                self.assertEqual(result.returncode, 0, result.stderr)
                words = [word for line in result.stdout.decode().splitlines() if line.startswith("link:")
                         for word in shlex.split(line[len("link:"):])]
                response.unlink()
                self.assertEqual(words, ["-L/selected42", "-lm", "-lc", "-lm"])
                later = subprocess.run([shutil.which("cc"), "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-fPIC", str(module / "answer.c"), "-o", str(directory / "later.so"), *words],
                    cwd=directory, env=env, capture_output=True, timeout=15)
                self.assertEqual(later.returncode, 0, later.stderr)
                self.assertNotEqual(invoke().returncode, 0)
                response.write_text("@" + str(response) + "\n")
                self.assertNotEqual(invoke().returncode, 0)
                response.unlink()
                os.mkfifo(response)
                self.assertNotEqual(invoke().returncode, 0)
                response.unlink()
                response.write_text("-L/selected43\n")
                retry = invoke()
                self.assertEqual(retry.returncode, 0, retry.stderr)
                self.assertIn(b"selected43", retry.stdout)
                compiler_response = directory / "compiler.rsp"
                compiler_response.write_text("-DANSWER=42\n")
                metadata["cflags"] = ["@" + str(compiler_response)]
                (module / "module.json").write_text(json.dumps(metadata))
                for override in ("--driver-mode=cl\n", "--driver\\-mode=cl\n"):
                    response.write_text(override)
                    fallback = invoke()
                    self.assertEqual(fallback.returncode, 0, fallback.stderr)
                    self.assertIn(("compile:@" + str(compiler_response)).encode(), fallback.stdout)
                    self.assertIn(("link:@" + str(response)).encode(), fallback.stdout)

    def test_changed_link_sidecar_rejects_replacement_and_recovers(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-sidecar-") as tmp:
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(Path(tmp))
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"],
                "ldflags": ["-L/nano/missing"] * 1200 + ["-lm", "-lc", "-lm"]}))

            def invoke(mode):
                return subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"), mode, str(module)],
                                      env=env, capture_output=True, timeout=15)

            built = invoke("build")
            self.assertEqual(built.returncode, 0, built.stderr)
            generation = invoke("directory").stdout
            command = invoke("shared-link-command")
            self.assertEqual(command.returncode, 0, command.stderr)
            paths = [Path(word[1:]) for word in shlex.split(command.stdout.decode()) if word.startswith("@")]
            self.assertEqual(len(paths), 1)
            retained = paths[0]
            original = retained.read_bytes()
            retained.chmod(0o600)
            retained.write_bytes(original.replace(b"missing", b"changed"))
            retained.chmod(0o400)
            source = module / "answer.c"
            source.write_text(source.read_text().replace("42", "43"))
            rejected = invoke("build")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(invoke("directory").stdout, generation)
            retained.unlink()
            repaired = invoke("build")
            self.assertEqual(repaired.returncode, 0, repaired.stderr)
            self.assertNotEqual(invoke("directory").stdout, generation)
            self.assertEqual(retained.read_bytes(), original)

    def test_actual_link_groups_preserve_results_and_reuse(self):
        require_consistent(measure(shutil.which("cc")))

    def test_fragment_allocation_failure_allows_retry(self):
        result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                                 "link-fragment-allocation", "all"], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_indirect_linker_arguments_are_not_hidden(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-indirect-") as tmp:
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(Path(tmp))
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": [],
                "ldflags": ["-L/nano/missing"] * 1200 + ["-Wl,@/nano/missing.rsp"]}))
            result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                                     "shared-link-command", str(module)], env=env,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            words = shlex.split(result.stdout.decode())
            self.assertIn("-Wl,@/nano/missing.rsp", words)
            self.assertFalse(any(word.startswith("@") for word in words))

    def test_every_acceptance_field_is_required(self):
        statuses = ("direct_status", "build_status", "returned_status", "later_status")
        answers = ("direct_answer", "cold_answer", "warm_answer", "later_answer")
        invariants = ("generation_reused", "initial_generation_preserved", "failed_replacement_preserved",
                      "metadata_unchanged", "returned_arguments_equal")
        case = {**dict.fromkeys(statuses, 0), **dict.fromkeys(answers, 42), **dict.fromkeys(invariants, True)}
        require_consistent({"cases": [case]})
        for key in statuses + answers + invariants:
            with self.subTest(key=key):
                changed = dict(case)
                changed[key] = 1 if key in statuses else None if key in answers else False
                with self.assertRaises(SystemExit):
                    require_consistent({"cases": [case, changed]})

    def test_empty_measurement_is_not_acceptance(self):
        with self.assertRaises(SystemExit):
            require_consistent({"cases": []})
