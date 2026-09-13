"""I reject incomplete or inconsistent linker transport measurements."""

import unittest
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from tests.characterize_link_argument_transport import measure, require_consistent
from tests import test_bytecode_shadows as shadows


class LinkArgumentAcceptance(unittest.TestCase):
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
