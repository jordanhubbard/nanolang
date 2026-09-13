"""I reject incomplete or inconsistent linker transport measurements."""

import unittest
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from tests.characterize_link_argument_transport import measure, require_consistent
from tests import test_bytecode_shadows as shadows
from tests.characterize_linker_response_grammar import materialize, require_equivalent, require_retained_equivalent


class LinkArgumentAcceptance(unittest.TestCase):
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
