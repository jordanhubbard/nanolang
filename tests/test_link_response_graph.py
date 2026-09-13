"""I check bounded, identity-preserving linker response graph retention."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from tests import test_bytecode_shadows as shadows


class LinkResponseGraph(unittest.TestCase):
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
