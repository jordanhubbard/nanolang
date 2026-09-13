"""I identify the selected linker through bounded literal-command queries."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from tests import test_bytecode_shadows as shadows


class LinkResponseQuery(unittest.TestCase):
    def query(self, args, directory):
        command = args if isinstance(args, str) else shlex.join(list(map(str, args)))
        result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                                 "link-response-grammar", command], cwd=directory,
                                capture_output=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(result.stdout)

    def tool(self, directory, body, name="tool"):
        path = directory / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o700)
        return path

    def test_native_linker_and_selection_override(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-native-") as tmp:
            directory = Path(tmp)
            compiler, linker = shutil.which("cc"), shutil.which("ld")
            obj, output = directory / "empty.o", directory / "unpublished.so"
            result = subprocess.run([compiler, "-x", "c", "-c", "/dev/null", "-o", str(obj)],
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = directory / "selected.jsonl"
            self.tool(directory, "import json,os,sys\n"
                      f"with open({str(log)!r}, 'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')\n"
                      f"os.execv({linker!r}, [{linker!r}, *sys.argv[1:]])\n", "ld")
            command = [compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                       "-B" + str(directory) + "/", str(obj), "-o", str(output)]
            observed = self.query(command, directory)
            expected = 2 if sys.platform == "darwin" else 1
            reports = []
            if observed != expected:
                for option in ("-Wl,--version", "-Wl,-version_details"):
                    result = subprocess.run([*command, option], capture_output=True, timeout=8)
                    reports.append((result.returncode, result.stdout, result.stderr))
            self.assertEqual(observed, expected, reports)
            recorded = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertTrue(any("--version" in args for args in recorded))
            if sys.platform == "darwin": self.assertTrue(any("-version_details" in args for args in recorded))
            # Apple's version-details request can complete the disposable link.
            # This API must never receive a published generation's output path.
            if sys.platform == "darwin": self.assertGreater(output.stat().st_size, 0)
            else: self.assertFalse(output.exists())
            output.unlink(missing_ok=True)
            alternate_log = directory / "alternate-called"
            alternate = "ld64.lld" if sys.platform == "darwin" else "ld.gold"
            self.tool(directory, "from pathlib import Path\n"
                      f"Path({str(alternate_log)!r}).touch()\nprint('I am an unsupported linker')\n", alternate)
            selector = "-fuse-ld=lld" if sys.platform == "darwin" else "-fuse-ld=gold"
            self.assertEqual(self.query([*command, selector], directory), 0)
            self.assertTrue(alternate_log.exists())
            self.assertFalse(output.exists())
            alternate_log.unlink()
            response = directory / "driver.rsp"
            response.write_text(selector + "\n")
            self.assertEqual(self.query([*command, "@" + str(response)], directory), 0)
            self.assertTrue(alternate_log.exists())
            self.assertFalse(output.exists())

    def test_literal_arguments_are_not_shell_evaluated(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-literal-") as tmp:
            directory = Path(tmp)
            log = directory / "args.json"
            tool = self.tool(directory, "import json,sys\n"
                             f"open({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                             "print('GNU ld (fixture) 2.40')\n", 'tool with "quotes"')
            arguments = ["space here", "comma,here", "$(touch forbidden)", "'quote'", "back\\slash"]
            self.assertEqual(self.query([tool, *arguments], directory), 1)
            self.assertEqual(json.loads(log.read_text()), [*arguments, "-Wl,--version"])
            self.assertFalse((directory / "forbidden").exists())
            log.unlink()
            self.assertEqual(self.query(shlex.quote(str(tool)) + " ; touch forbidden", directory), 0)
            self.assertFalse(log.exists())
            self.assertFalse((directory / "forbidden").exists())
            for args in ([tool, *(["x"] * 2048)], [tool, "x" * 4096]):
                self.assertEqual(self.query(args, directory), 0)
                self.assertFalse(log.exists())

    def test_failed_oversized_and_binary_reports_are_declined(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-reject-") as tmp:
            directory = Path(tmp)
            bodies = ("print('GNU ld (fixture) 2.40'); raise SystemExit(1)\n",
                      "print('GNU ld (fixture) 2.40' + 'x'*9000)\n",
                      "import os\nos.write(1,b'GNU ld (fixture) 2.40\\n\\0')\n",
                      "print('GNU gold (fixture) 2.40')\n",
                      "print('LLVM LLD 21.0.0')\n")
            for body in bodies:
                with self.subTest(body=body):
                    self.assertEqual(self.query([self.tool(directory, body)], directory), 0)
            self.assertEqual(self.query("", directory), 0)
            self.assertEqual(self.query("missing-nanolang-query-executable", directory), 0)
            self.assertEqual(self.query("x" * 65537, directory), 0)

    def test_apple_report_requires_tested_version_and_vendor(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-apple-") as tmp:
            directory = Path(tmp)
            supported = {"version": "1267", "architectures": ["arm64"],
                         "tapi": {"version_string": "Apple TAPI version 21.0.0"}}
            for report, expected in ((supported, 2), ({**supported, "version": "unknown"}, 0),
                                     ({**supported, "architectures": []}, 0),
                                     ({**supported, "tapi": {}}, 0)):
                with self.subTest(report=report):
                    tool = self.tool(directory, "import sys\n"
                        "if sys.argv[-1] == '-Wl,--version': raise SystemExit(1)\n"
                        f"print({json.dumps(report)!r})\n")
                    self.assertEqual(self.query([tool], directory), expected)

    def test_deadline_kills_descendants_holding_output_open(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-deadline-") as tmp:
            directory = Path(tmp)
            marker = directory / "late-side-effect"
            tool = self.tool(directory, "import os,time\n"
                "if os.fork() == 0:\n"
                "    time.sleep(6)\n"
                f"    open({str(marker)!r}, 'w').write('unexpected')\n"
                "    os._exit(0)\n"
                "print('GNU ld (fixture) 2.40', flush=True)\n")
            started = time.monotonic()
            self.assertEqual(self.query([tool], directory), 0)
            elapsed = time.monotonic() - started
            self.assertGreaterEqual(elapsed, 4.5)
            self.assertLess(elapsed, 7)
            time.sleep(1.5)
            self.assertFalse(marker.exists())

    def test_success_also_cleans_descendants_without_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-background-") as tmp:
            directory = Path(tmp)
            marker = directory / "late-side-effect"
            tool = self.tool(directory, "import os,time\n"
                "if os.fork() == 0:\n"
                "    os.close(1); os.close(2)\n"
                "    time.sleep(0.5)\n"
                f"    open({str(marker)!r}, 'w').write('unexpected')\n"
                "    os._exit(0)\n"
                "print('GNU ld (fixture) 2.40', flush=True)\n")
            self.assertEqual(self.query([tool], directory), 1)
            time.sleep(0.75)
            self.assertFalse(marker.exists())

    def test_argument_allocation_failure_retries_in_same_process(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-query-allocation-") as tmp:
            directory = Path(tmp)
            tool = self.tool(directory, "print('GNU ld (fixture) 2.40')\n")
            command = shlex.join([str(tool), "one", "two"])
            outcomes = set()
            for budget in range(8):
                result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                    "link-response-query-allocation", command, str(budget)], cwd=directory,
                    capture_output=True, timeout=8)
                self.assertEqual(result.returncode, 0, (budget, result.stderr))
                outcomes.add(int(result.stdout))
            self.assertEqual(outcomes, {0, 1})


if __name__ == "__main__":
    unittest.main()
