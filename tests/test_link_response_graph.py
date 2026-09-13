"""I check bounded, identity-preserving linker response graph retention."""

import json
import errno
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests import test_bytecode_shadows as shadows


class LinkResponseGraph(unittest.TestCase):
    def arguments(self, directory, sources, grammar="gnu", env=None, budget=None):
        mode = "capture-link-arguments" if budget is None else "capture-link-arguments-allocation"
        args = [str(shadows.ROOT / "obj/test_module_generation_probe"), mode, grammar]
        if budget is not None: args.append(str(budget))
        return subprocess.run([*args, *map(str, sources)], cwd=directory, env=env,
                              capture_output=True, timeout=10)

    def test_argument_capture_quotes_exact_words_without_sidecars(self):
        with tempfile.TemporaryDirectory(prefix="nano-args-quoted-") as tmp:
            directory = Path(tmp) / "comma, space"
            directory.mkdir()
            inner, outer = directory / "inner.rsp", directory / "outer.rsp"
            inner.write_bytes(b'''"with\r\nline.a" '' "$;literal" "single'quote"\n''')
            outer.write_text('-lm @inner.rsp -lc\v-lm\f')
            for grammar in ("gnu", "apple"):
                result = self.arguments(directory, [outer], grammar)
                self.assertEqual(result.returncode, 0, result.stderr)
                fragments = json.loads(result.stdout)
                self.assertEqual(len(fragments), 1)
                values = ["-lm", "with\r\nline.a", "", "$;literal", "single'quote"]
                values += ["-lc", "-lm"] if grammar == "gnu" else ["-lc\v-lm\f"]
                self.assertEqual(shlex.split(fragments[0]), [word for value in values for word in ("-Xlinker", value)])
                self.assertEqual(set(directory.iterdir()), {inner, outer})

    def test_argument_capture_freezes_shared_inputs_and_apple_rejection(self):
        for grammar in ("gnu", "apple"):
            for action in ("rewrite", "remove", "retarget"):
                with self.subTest(grammar=grammar, action=action), tempfile.TemporaryDirectory(prefix="nano-args-freeze-") as tmp:
                    directory = Path(tmp)
                    shared = directory / "shared.rsp"
                    shared.write_text("-lm\n")
                    (directory / "other.rsp").write_text("-lc\n")
                    if action == "retarget":
                        shared.rename(directory / "original.rsp")
                        shared.symlink_to("original.rsp")
                    for name in ("first.rsp", "second.rsp"):
                        (directory / name).write_text("@shared.rsp\n")
                    env = os.environ.copy()
                    env.update(NANO_TEST_RESPONSE_MUTATE=str(shared), NANO_TEST_RESPONSE_ON_READ="1",
                        NANO_TEST_RESPONSE_ACTION=action, NANO_TEST_RESPONSE_TARGET="other.rsp")
                    result = self.arguments(directory, ["first.rsp", "second.rsp", "shared.rsp"], grammar, env)
                    if grammar == "gnu":
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual([shlex.split(value) for value in json.loads(result.stdout)],
                            [["-Xlinker", "-lm"]] * 3)
                    else:
                        self.assertEqual(result.returncode, 1)
                        self.assertEqual(result.stdout, b"")
                        self.assertIn(f"errno={errno.ELOOP}".encode(), result.stderr)
                    if action == "remove": self.assertFalse(shared.exists())
                    else: self.assertEqual(shared.read_text(), "-lc\n")
                    retry = self.arguments(directory, ["first.rsp"], grammar)
                    if action == "remove": self.assertNotEqual(retry.returncode, 0)
                    else:
                        self.assertEqual(retry.returncode, 0, retry.stderr)
                        self.assertEqual(shlex.split(json.loads(retry.stdout)[0]), ["-Xlinker", "-lc"])

    def test_argument_capture_bounds_and_atomic_allocation_retry(self):
        with tempfile.TemporaryDirectory(prefix="nano-args-bounds-") as tmp:
            directory = Path(tmp)
            first, second = directory / "first.rsp", directory / "second.rsp"
            first.write_text("-lm " * 4368 + "-lmm")
            result = self.arguments(directory, [first])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)[0]), 65536)
            first.write_text("-lm " * 4370)
            self.assertNotEqual(self.arguments(directory, [first]).returncode, 0)
            first.write_text("-lm " * 2185)
            result = self.arguments(directory, [first, first])
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")
            self.assertIn(f"errno={errno.E2BIG}".encode(), result.stderr)
            first.write_text("-lm\n")
            second.write_text("-lc\n")
            for sources in ([first, "missing.rsp"], [first] * 65, [first, ""]):
                result = self.arguments(directory, sources)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
            result = self.arguments(directory, [first] * 64)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)), 64)
            for grammar in ("gnu", "apple"):
                outcomes = set()
                for budget in range(40):
                    result = self.arguments(directory, [first, second], grammar, budget=budget)
                    self.assertEqual(result.returncode, 0, (budget, result.stderr))
                    outcomes.add(result.stdout.strip())
                self.assertEqual(outcomes, {b"captured", b"failed"})
            first.write_bytes(b" " * 40000)
            second.write_bytes(b" " * 26000)
            result = self.arguments(directory, [first, second])
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"errno={errno.E2BIG}".encode(), result.stderr)
            first.write_text("")
            self.assertEqual(json.loads(self.arguments(directory, [first]).stdout), [""])
            for payload in (b"a\0b", b"x" * 4096):
                first.write_bytes(payload)
                self.assertNotEqual(self.arguments(directory, [first]).returncode, 0)
            first.unlink()
            os.mkfifo(first)
            self.assertNotEqual(self.arguments(directory, [first]).returncode, 0)
            first.unlink()
            first.mkdir()
            self.assertNotEqual(self.arguments(directory, [first]).returncode, 0)
            self.assertNotEqual(self.arguments(directory, [second], "unknown").returncode, 0)

    def batch(self, directory, sources, grammar="gnu", env=None, budget=None):
        mode = "capture-link-responses" if budget is None else "capture-link-responses-allocation"
        args = [str(shadows.ROOT / "obj/test_module_generation_probe"), mode, grammar, str(directory)]
        if budget is not None: args.append(str(budget))
        return subprocess.run([*args, *map(str, sources)], cwd=directory, env=env,
                              capture_output=True, timeout=10)

    def test_batch_freezes_shared_inputs_between_roots(self):
        for action in ("rewrite", "remove", "retarget"):
            for grammar in ("gnu", "apple"):
                with self.subTest(action=action, grammar=grammar), tempfile.TemporaryDirectory(prefix="nano-batch-freeze-") as tmp:
                    directory = Path(tmp)
                    shared = directory / "shared.rsp"
                    shared.write_text("-lm\n")
                    other = directory / "other.rsp"
                    other.write_text("-lc\n")
                    if action == "retarget":
                        shared.rename(directory / "original.rsp")
                        shared.symlink_to("original.rsp")
                    for name in ("first.rsp", "second.rsp"):
                        (directory / name).write_text("@shared.rsp\n")
                    env = os.environ.copy()
                    env.update(NANO_TEST_RESPONSE_MUTATE=str(shared), NANO_TEST_RESPONSE_ACTION=action,
                               NANO_TEST_RESPONSE_TARGET=str(other))
                    sources = ["first.rsp", "second.rsp", "first.rsp", "shared.rsp"]
                    result = self.batch(directory, sources, grammar, env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    paths = [Path(line) for line in result.stdout.decode().splitlines()]
                    self.assertEqual(len(paths), len(sources))
                    self.assertEqual(paths[0], paths[2])
                    self.assertNotEqual(paths[0], paths[1])
                    nested = [Path(shlex.split(path.read_text())[0][1:]) for path in paths[:2]]
                    self.assertEqual(nested, [paths[3], paths[3]])
                    self.assertEqual(paths[3].read_bytes(), b"-lm\n")
                    if action == "remove": self.assertFalse(shared.exists())
                    else: self.assertEqual(shared.read_bytes(), b"-lc\n")
                    retry = self.batch(directory, sources, grammar)
                    if action == "remove":
                        self.assertNotEqual(retry.returncode, 0)
                        self.assertEqual(retry.stdout, b"")
                    else:
                        self.assertEqual(retry.returncode, 0, retry.stderr)
                        replacement = Path(retry.stdout.decode().splitlines()[3])
                        self.assertEqual(replacement.read_bytes(), b"-lc\n")
                        self.assertNotEqual(replacement, paths[3])

    def test_batch_failure_is_not_a_partial_result_and_budgets_are_shared(self):
        with tempfile.TemporaryDirectory(prefix="nano-batch-bounds-") as tmp:
            directory = Path(tmp)
            (directory / "first.rsp").write_bytes(b"a" * 40000)
            (directory / "second.rsp").write_bytes(b"b" * 26000)
            for sources in (["first.rsp", "missing.rsp"], ["first.rsp", "second.rsp"],
                            ["first.rsp"] * 65, ["./" * 2048 + "first.rsp"], [""]):
                result = self.batch(directory, sources)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
            result = self.batch(directory, ["first.rsp"] * 64)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 64)
            self.assertEqual(len(set(result.stdout.splitlines())), 1)
            (directory / "missing.rsp").write_text("-lm\n")
            self.assertEqual(self.batch(directory, ["first.rsp", "missing.rsp"]).returncode, 0)

    def test_source_spelling_budget(self):
        with tempfile.TemporaryDirectory(prefix="nano-batch-aliases-") as tmp:
            directory = Path(tmp)
            (directory / "inner.rsp").write_text("-lm\n")
            for index in range(128):
                (directory / f"alias-{index}.rsp").symlink_to("inner.rsp")
            outer = directory / "outer.rsp"
            outer.write_text(" ".join(f"@alias-{index}.rsp" for index in range(128)))
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)
            outer.write_text(" ".join(f"@alias-{index}.rsp" for index in range(127)))
            self.captured_path(directory, outer)

    def test_batch_allocation_failure_retries_without_partial_paths(self):
        with tempfile.TemporaryDirectory(prefix="nano-batch-allocation-") as tmp:
            directory = Path(tmp)
            (directory / "shared.rsp").write_text("-lm\n")
            for name in ("first.rsp", "second.rsp"):
                (directory / name).write_text("@shared.rsp\n")
            outcomes = set()
            for budget in range(64):
                result = self.batch(directory, ["first.rsp", "second.rsp", "shared.rsp"], budget=budget)
                self.assertEqual(result.returncode, 0, (budget, result.stderr))
                outcomes.add(result.stdout.strip())
            self.assertEqual(outcomes, {b"failed", b"captured"})

    def test_native_multi_root_order_identity_and_lifetime(self):
        with tempfile.TemporaryDirectory(prefix="nano-batch-native-") as tmp:
            directory = Path(tmp)
            compiler = shutil.which("cc")
            grammar = "apple" if sys.platform == "darwin" else "gnu"

            def run(args):
                return subprocess.run(list(map(str, args)), cwd=directory, capture_output=True, timeout=15)

            for name, answer in (("a", 42), ("b", 43)):
                library = directory / name
                library.mkdir()
                source = library / "member.c"
                source.write_text(f"long long selected(void) {{ return {answer}; }}\n")
                for args in ([compiler, "-fPIC", "-c", source, "-o", library / "member.o"],
                             ["ar", "rcs", library / "libselected.a", library / "member.o"]):
                    result = run(args)
                    self.assertEqual(result.returncode, 0, result.stderr)
            source = directory / "main.c"
            source.write_text("extern long long selected(void); long long answer(void) { return selected(); }\n")

            def link(name, paths=(), arguments=None):
                output = directory / (name + ".so")
                options = arguments if arguments is not None else ["-Wl,@" + str(path) for path in paths]
                result = run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                              "-fPIC", source, "-o", output, *options])
                answer = None
                if result.returncode == 0:
                    loaded = run([sys.executable, "-c", "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                                  "lib.answer.restype=ctypes.c_int64; print(lib.answer())", output])
                    self.assertEqual(loaded.returncode, 0, loaded.stderr)
                    answer = int(loaded.stdout)
                return result.returncode, answer

            for names, expected in ((["a.rsp", "b.rsp"], (0, 42)), (["b.rsp", "a.rsp"], (0, 43)),
                                    (["same.rsp", "same.rsp"], (1, None) if sys.platform == "darwin" else (0, 42)),
                                    (["same.rsp", "distinct.rsp"], (0, 42))):
                with self.subTest(names=names):
                    (directory / "a.rsp").write_text("-La\n")
                    (directory / "b.rsp").write_text("-Lb -lselected\n")
                    for name in ("same.rsp", "distinct.rsp"):
                        (directory / name).write_text("-La -lselected\n")
                    originals = [directory / name for name in names]
                    self.assertEqual(link("native", originals), expected)
                    captured = self.batch(directory, originals, grammar)
                    self.assertEqual(captured.returncode, 0, captured.stderr)
                    paths = [Path(line) for line in captured.stdout.decode().splitlines()]
                    captured_arguments = self.arguments(directory, originals, grammar)
                    for path in set(originals): path.unlink()
                    self.assertEqual(link("retained", paths), expected)
                    if expected[0]:
                        self.assertEqual(captured_arguments.returncode, 1)
                        self.assertEqual(captured_arguments.stdout, b"")
                        self.assertIn(f"errno={errno.ELOOP}".encode(), captured_arguments.stderr)
                    else:
                        self.assertEqual(captured_arguments.returncode, 0, captured_arguments.stderr)
                        arguments = [word for fragment in json.loads(captured_arguments.stdout)
                                     for word in shlex.split(fragment)]
                        self.assertEqual(link("arguments", arguments=arguments), expected)

    def capture(self, directory, source, grammar="gnu", budget=None):
        mode = "capture-link-response" if budget is None else "capture-link-response-allocation"
        args = [str(shadows.ROOT / "obj/test_module_generation_probe"), mode, grammar,
                str(directory), str(source)]
        if budget is not None: args.append(str(budget))
        return subprocess.run(args, cwd=directory, capture_output=True, timeout=10)

    def captured_path(self, directory, source, grammar="gnu"):
        result = self.capture(directory, source, grammar)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = Path(result.stdout.decode().strip())
        self.assertTrue(path.is_file())
        return path

    def test_nested_quoted_paths_preserve_other_bytes_and_lifetime(self):
        for grammar in ("gnu", "apple"):
            with self.subTest(grammar=grammar), tempfile.TemporaryDirectory(prefix="nano-graph-quoted-") as tmp:
                directory = Path(tmp) / 'space "quote" \\backslash'
                directory.mkdir()
                inner = directory / "inner file.rsp"
                payload = b"'selected weird.a'\v-lm\f\n"
                inner.write_bytes(payload)
                outer = directory / "outer.rsp"
                prefix, suffix = b"\t-lm  ", b"\r\n  -lc\v-lm\f\n"
                outer.write_bytes(prefix + b'@"inner file.rsp"' + suffix)
                captured = self.captured_path(directory, outer, grammar)
                data = captured.read_bytes()
                self.assertTrue(data.startswith(prefix), data)
                self.assertTrue(data.endswith(suffix), data)
                token = data[len(prefix):-len(suffix)].decode()
                nested = Path(json.loads(token)[1:])
                self.assertEqual(nested.read_bytes(), payload)
                self.assertEqual(self.captured_path(directory, outer, grammar), captured)
                outer.unlink()
                inner.unlink()
                self.assertEqual(nested.read_bytes(), payload)
                self.assertEqual(captured.read_bytes(), data)

    def test_resolved_path_identity_not_content_or_inode(self):
        with tempfile.TemporaryDirectory(prefix="nano-graph-identity-") as tmp:
            directory = Path(tmp)
            inner = directory / "inner.rsp"
            inner.write_text("-lm\n")
            (directory / "alias.rsp").symlink_to("inner.rsp")
            (directory / "hard.rsp").hardlink_to(inner)
            (directory / "other.rsp").write_bytes(inner.read_bytes())
            outer = directory / "outer.rsp"
            outer.write_text("@inner.rsp @./inner.rsp @alias.rsp @hard.rsp @other.rsp\n")
            retained = self.captured_path(directory, outer)
            paths = [Path(word[1:]) for word in shlex.split(retained.read_text())]
            self.assertEqual(paths[0], paths[1])
            self.assertEqual(paths[0], paths[2])
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(all(path.read_bytes() == b"-lm\n" for path in paths))

    def test_grammar_controls_only_nested_token_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="nano-graph-grammar-") as tmp:
            directory = Path(tmp)
            (directory / "inner file.rsp").write_text("-lm\n")
            outer = directory / "outer.rsp"
            for reference in ('@"inner file.rsp"', "'@inner\\ file.rsp'", '@inner\\ file.rsp',
                              '@in"ner file".rsp', '\\@"inner file.rsp"'):
                with self.subTest(reference=reference):
                    outer.write_text("-lc\v" + reference + "\f-lm")
                    apple = self.captured_path(directory, outer, "apple")
                    self.assertEqual(apple.read_bytes(), outer.read_bytes())
                    gnu = self.captured_path(directory, outer, "gnu")
                    self.assertNotEqual(gnu.read_bytes(), outer.read_bytes())
                    self.assertTrue(gnu.read_bytes().startswith(b"-lc\v\"@"))
                    self.assertTrue(gnu.read_bytes().endswith(b'"\f-lm'))

    def test_node_budget_is_bounded(self):
        with tempfile.TemporaryDirectory(prefix="nano-graph-nodes-") as tmp:
            directory = Path(tmp)
            for index in range(64):
                (directory / f"node-{index}.rsp").write_text("-lm\n")
            outer = directory / "outer.rsp"
            outer.write_text(" ".join(f"@node-{index}.rsp" for index in range(64)))
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)
            outer.write_text(" ".join(f"@node-{index}.rsp" for index in range(63)))
            self.captured_path(directory, outer)

    def test_nested_end_of_input_tokens_are_captured(self):
        for grammar in ("gnu", "apple"):
            with tempfile.TemporaryDirectory(prefix="nano-graph-eof-") as tmp:
                directory = Path(tmp) / "comma, space"
                directory.mkdir()
                inner, outer = directory / "inner file.rsp", directory / "outer.rsp"
                inner.write_text("-lm\n")
                for token in ("'@inner file.rsp", '"@inner file.rsp', "@inner\\ file.rsp\\"):
                    with self.subTest(grammar=grammar, token=token):
                        outer.write_text("-lc " + token)
                        captured = self.captured_path(directory, outer, grammar)
                        words = shlex.split(captured.read_text())
                        self.assertEqual(words[0], "-lc")
                        self.assertEqual(len(words), 2)
                        self.assertTrue(words[1].startswith("@"))
                        self.assertEqual(Path(words[1][1:]).read_bytes(), b"-lm\n")

    def test_invalid_inputs_and_budgets_fail_then_recover(self):
        with tempfile.TemporaryDirectory(prefix="nano-graph-failure-") as tmp:
            directory = Path(tmp)
            outer = directory / "outer.rsp"
            for payload in (b"@missing.rsp", b"@outer.rsp", b"a\0b", b"x" * 65537,
                            b"'@missing.rsp", b"@outer.rsp\\"):
                with self.subTest(payload=payload[:30]):
                    outer.write_bytes(payload)
                    self.assertNotEqual(self.capture(directory, outer).returncode, 0)
                    outer.write_text("-lm\n")
                    self.captured_path(directory, outer)
            outer.unlink()
            os.mkfifo(outer)
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)
            outer.unlink()
            outer.mkdir()
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)
            outer.rmdir()
            outer.write_text("'unterminated\n")
            self.assertEqual(self.captured_path(directory, outer).read_bytes(), outer.read_bytes())
            self.assertNotEqual(self.capture(directory, outer, "unknown").returncode, 0)
            for index in range(17):
                (directory / f"depth-{index}.rsp").write_text(f"@depth-{index + 1}.rsp" if index < 16 else "-lm")
            self.assertNotEqual(self.capture(directory, directory / "depth-0.rsp").returncode, 0)
            (directory / "big-a.rsp").write_bytes(b"a" * 40000)
            (directory / "big-b.rsp").write_bytes(b"b" * 26000)
            outer.write_text("@big-a.rsp @big-b.rsp")
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)

    def test_changed_retained_child_rejects_and_recovers(self):
        with tempfile.TemporaryDirectory(prefix="nano-graph-tamper-") as tmp:
            directory = Path(tmp)
            (directory / "inner.rsp").write_text("-lm\n")
            outer = directory / "outer.rsp"
            outer.write_text("@inner.rsp\n")
            retained = self.captured_path(directory, outer)
            child = Path(shlex.split(retained.read_text())[0][1:])
            child.chmod(0o600)
            child.write_text("-lc\n")
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)
            child.unlink()
            self.assertEqual(self.captured_path(directory, outer), retained)
            self.assertEqual(child.read_text(), "-lm\n")
            child.unlink()
            child.symlink_to(directory / "inner.rsp")
            self.assertNotEqual(self.capture(directory, outer).returncode, 0)

    def test_allocation_failures_retry_in_same_process(self):
        with tempfile.TemporaryDirectory(prefix="nano-graph-allocation-") as tmp:
            directory = Path(tmp)
            (directory / "inner.rsp").write_text("-lm\n")
            outer = directory / "outer.rsp"
            outer.write_text("@inner.rsp @inner.rsp\n")
            outcomes = set()
            for budget in range(32):
                result = self.capture(directory, outer, budget=budget)
                self.assertEqual(result.returncode, 0, (budget, result.stderr))
                outcomes.add(result.stdout.strip())
            self.assertEqual(outcomes, {b"failed", b"captured"})


if __name__ == "__main__":
    unittest.main()
