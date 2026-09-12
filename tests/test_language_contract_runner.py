"""I test the contract runner's failure boundaries independently of compilers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_language_contract as contract


class ContractRunnerTests(unittest.TestCase):
    def test_versioned_manifest(self):
        self.assertEqual(len(contract.load_cases(contract.ROOT / "tests/language-contract/v1.json")), 7)

    def test_bad_manifests(self):
        valid = {"schema": "nanolang.language-contract.v1", "cases": ["01_arithmetic"]}
        cases = [None, [], {}, {**valid, "schema": "v2"}, {**valid, "cases": []},
                 {**valid, "cases": ["../escape"]}, {**valid, "cases": [None]},
                 {**valid, "cases": ["01_arithmetic", "01_arithmetic"]},
                 {**valid, "cases": ["99_missing"]}, {**valid, "skip": True}]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "contract.json"
            for data in cases:
                with self.subTest(data=data):
                    manifest.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        contract.load_cases(manifest)

    def test_exact_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, "-c", "print('ok')"]
            contract.check_output(command, b"ok\n", Path(tmp))
            for expected in (b"ok", b"ok\n\n", b"wrong\n"):
                with self.subTest(expected=expected), self.assertRaises(RuntimeError):
                    contract.check_output(command, expected, Path(tmp))

    def test_matching_output_with_failed_exit_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "exit 7"):
            contract.check_output([sys.executable, "-c", "print('ok'); raise SystemExit(7)"],
                                  b"ok\n", Path(tmp))

    def test_missing_tool_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(OSError):
            contract.run([Path(tmp) / "absent"], Path(tmp))

    def test_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(subprocess.TimeoutExpired):
            contract.run([sys.executable, "-c", "import time; time.sleep(5)"], Path(tmp), timeout=0.05)

    def test_vm_and_aot_use_same_artifact(self):
        tools = {name: name for name in (*contract.BACKENDS, "frontend")}
        with tempfile.TemporaryDirectory() as tmp, patch.object(contract, "run", return_value=b"7\n") as run:
            results = contract.check_case("01_arithmetic", tools, ["cc"], Path(tmp))
            self.assertEqual(results, dict.fromkeys(contract.BACKENDS))
            commands = [call.args[0] for call in run.call_args_list]
            artifact = Path(tmp) / "program.nvm"
            self.assertIn(["vm", artifact], commands)
            self.assertIn(["aot", artifact, "-o", Path(tmp) / "program.c"], commands)
            self.assertEqual(sum(command[0] == "frontend" for command in commands), 1)

    def test_frontend_failure_fails_both_bytecode_rows(self):
        tools = {name: name for name in (*contract.BACKENDS, "frontend")}

        def execute(command, *args, **kwargs):
            if command[0] == "frontend":
                raise RuntimeError("I failed to emit bytecode")
            return b"7\n"

        with tempfile.TemporaryDirectory() as tmp, patch.object(contract, "run", side_effect=execute):
            results = contract.check_case("01_arithmetic", tools, ["cc"], Path(tmp))
            self.assertIsNone(results["c-seed"])
            self.assertIsNone(results["selfhost"])
            self.assertIn("failed", results["vm"])
            self.assertEqual(results["vm"], results["aot"])

    def test_aot_refusal_is_not_a_pass_or_vm_failure(self):
        tools = {name: name for name in (*contract.BACKENDS, "frontend")}

        def execute(command, *args, **kwargs):
            if command[0] == "aot":
                raise RuntimeError("I refuse this opcode")
            return b"7\n"

        with tempfile.TemporaryDirectory() as tmp, patch.object(contract, "run", side_effect=execute):
            results = contract.check_case("01_arithmetic", tools, ["cc"], Path(tmp))
            self.assertIsNone(results["vm"])
            self.assertIn("refuse", results["aot"])

    def test_main_reports_failure_and_removes_scratch(self):
        directories = []

        def check(case, tools, cc, directory):
            self.assertTrue(directory.is_dir())
            directories.append(directory)
            (directory / "artifact").write_bytes(b"scratch")
            return {**dict.fromkeys(contract.BACKENDS), "aot": "I refuse this opcode"}

        with patch.object(sys, "argv", ["runner"]), patch.object(contract, "load_cases", return_value=["01_arithmetic"]), \
                patch.object(contract, "check_case", side_effect=check), patch("builtins.print"):
            self.assertEqual(contract.main(), 1)
        self.assertTrue(directories)
        self.assertTrue(all(not directory.exists() for directory in directories))


if __name__ == "__main__":
    unittest.main()
