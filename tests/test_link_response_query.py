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

    def private_query(self, command, directory, env=None, unchecked=False):
        result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
            "unchecked-private-link-response-grammar" if unchecked else "private-link-response-grammar",
            command, str(directory)], cwd=directory,
            env=env, capture_output=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(result.stdout), result.stderr

    def test_private_primary_output_preserves_metadata_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="nano-private-link-") as tmp:
            directory = Path(tmp) / "comma, space"
            directory.mkdir()
            module = directory / "module"
            module.mkdir()
            compiler = shutil.which("cc")
            env = os.environ.copy()
            env["NANO_CC"] = compiler
            compiled = subprocess.run([compiler, "-x", "c", "-c", "/dev/null", "-o", "fixture.o"],
                                      cwd=directory, capture_output=True, timeout=15)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            protected = [directory / "fixture.so", directory / "published.so"]
            for path in protected: path.write_bytes(b"I retain this published artifact.\n")
            before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in protected]
            platform = "macos" if sys.platform == "darwin" else "linux"
            groups = ("cflags", "cflags_" + platform, "ldflags", "ldflags_" + platform,
                      "pkg_cflags", "pkg_libs")
            for group in groups:
                for option in ("-o published.so", "-Wl,-o,published.so", "@driver.rsp", "-Wl,@link.rsp"):
                    with self.subTest(group=group, option=option):
                        (directory / "driver.rsp").write_text("-o published.so\n")
                        (directory / "link.rsp").write_text("-o published.so\n")
                        metadata = {"name": "fixture", "c_sources": []}
                        if group.startswith("pkg_"):
                            metadata["pkg_config"] = ["query-fixture"]
                            mode = "--cflags" if group == "pkg_cflags" else "--libs"
                            pkg = self.tool(directory, "import sys\n"
                                f"if {mode!r} in sys.argv: print({option!r})\n", "pkg-config")
                            env["PKG_CONFIG"] = str(pkg)
                        else: metadata[group] = [option]
                        (module / "module.json").write_text(json.dumps(metadata))
                        recipe = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                            "shared-link-command", str(module)], cwd=directory, env=env,
                            capture_output=True, timeout=8)
                        self.assertEqual(recipe.returncode, 0, recipe.stderr)
                        # I isolate output-pin behavior here. These raw indirect
                        # fixtures intentionally do not meet public admission.
                        grammar, error = self.private_query(recipe.stdout.decode().strip(), directory, env, unchecked=True)
                        self.assertEqual(grammar, 2 if sys.platform == "darwin" else 1, error)
                        self.assertEqual([(path.read_bytes(), path.stat().st_mtime_ns) for path in protected], before)
                        self.assertEqual(list(directory.glob(".nano-link-query-*")), [])

    def test_public_query_rejects_unclassified_controls_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="nano-query-admission-") as tmp:
            directory = Path(tmp)
            marker = directory / "queried"
            tool = self.tool(directory, "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('queried')\nprint('GNU ld (fixture) 2.40')\n")
            prefix = shlex.quote(str(tool))
            rejected = ("@raw.rsp", "-Wl,@raw.rsp", "-Xlinker @raw.rsp", "--driver-mode=cl",
                "-Xclang -load", "-fplugin=plugin.so", "-specs=specs", "-save-temps", "-MD", "-MF deps",
                "-Xassembler @assembler.rsp", "-Wa,-a=listing", "-unknown", "-lto_library plugin.so",
                "-o", "-L", "-B", "-Xlinker", "-Wl,", "-Wl,-lm,", "-Wl,-L",
                "-Xlinker -L fixture.o", "-Xlinker -o -Xlinker --", "-o @output.rsp",
                "-Wl,-lm,-Map,map", "-Xlinker -L -Xlinker lib -Xlinker -plugin",
                " ; touch forbidden", "x" * 4096, " ".join(["x"] * 2048))
            controls = ("--", "@raw.rsp", "-Map", "-Map=map", "--Map=map", "-map", "-dependency_info",
                "--dependency-file=deps", "--out-implib=exports", "--reproduce=repro", "-object_path_lto",
                "-lto_library", "-plugin", "--plugin=plugin.so", "-filelist", "-T", "--script=script.ld",
                "--version-script=exports", "-exported_symbols_list", "--unknown-control")
            for suffix in (*rejected, *("-Wl," + flag for flag in controls),
                           *("-Xlinker " + shlex.quote(flag) for flag in controls)):
                with self.subTest(suffix=suffix):
                    self.assertEqual(self.private_query(prefix + " " + suffix, directory)[0], 0)
                    self.assertFalse(marker.exists())
                    self.assertEqual(list(directory.glob(".nano-link-query-*")), [])

    def test_public_query_admits_explicit_forms_and_preserves_selection(self):
        with tempfile.TemporaryDirectory(prefix="nano-query-forms-") as tmp:
            directory = Path(tmp)
            record = directory / "arguments.json"
            tool = self.tool(directory, "import json,sys\nfrom pathlib import Path\n"
                f"Path({str(record)!r}).write_text(json.dumps(sys.argv[1:]))\n"
                "print('GNU ld (fixture) 2.40')\n")
            cases = (["-shared", "fixture.o", "-fPIC", "-O2", "-D", "ANSWER=42", "-Iinclude"],
                ["-B", "tools", "-fuse-ld=custom", "--target=aarch64-linux-gnu", "--sysroot=/sdk"],
                ["-Wl,-L,lib,-l,selected,-o,published.so"],
                ["-Xlinker", "-L", "-Xlinker", "comma, path", "-Xlinker", "-l", "-Xlinker", "selected"],
                ["-o", "published.so", "-Llib", "-l", "selected", "-lm", "-lc", "-pthread"],
                ["-arch", "arm64", "-isysroot", "/sdk", "-framework", "Foundation", "-undefined", "dynamic_lookup"])
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    self.assertEqual(self.private_query(shlex.join([str(tool), *arguments]), directory)[0], 1)
                    observed = json.loads(record.read_text())
                    self.assertEqual(observed[:len(arguments)], arguments)
                    self.assertEqual(observed[-1], "-Wl,--version")
                    self.assertEqual(list(directory.glob(".nano-link-query-*")), [])

    def test_public_query_with_captured_linker_arguments_uses_no_original_response(self):
        with tempfile.TemporaryDirectory(prefix="nano-query-captured-") as tmp:
            directory = Path(tmp) / "comma, space"
            directory.mkdir()
            compiler = shutil.which("cc")
            result = subprocess.run([compiler, "-x", "c", "-c", "/dev/null", "-o", "fixture.o"],
                cwd=directory, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            response = directory / "link.rsp"
            response.write_text("-lm\n")
            grammar = "apple" if sys.platform == "darwin" else "gnu"
            captured = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                "capture-link-arguments", grammar, str(response)], cwd=directory, capture_output=True, timeout=10)
            self.assertEqual(captured.returncode, 0, captured.stderr)
            response.unlink()
            command = shlex.join([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared", "fixture.o"])
            command += json.loads(captured.stdout)[0]
            self.assertEqual(self.private_query(command, directory)[0], 2 if sys.platform == "darwin" else 1)
            self.assertEqual(set(path.name for path in directory.iterdir()), {"fixture.o"})

    def test_private_output_cleanup_and_visible_control_rejection(self):
        with tempfile.TemporaryDirectory(prefix="nano-private-query-cleanup-") as tmp:
            directory = Path(tmp)
            tool = self.tool(directory, "from pathlib import Path\nimport sys\n"
                "Path(sys.argv[sys.argv.index('-o')+1]).write_text('private')\n"
                "print('GNU ld (fixture) 2.40')\n")
            command = shlex.quote(str(tool))
            self.assertEqual(self.private_query(command, directory)[0], 1)
            self.assertEqual(list(directory.glob(".nano-link-query-*")), [])
            for suffix in (" --", " -Xlinker --", " -Wl,--", " -Wl,-lm,--,-lc", " ; false"):
                self.assertEqual(self.private_query(command + suffix, directory)[0], 0)
                self.assertEqual(list(directory.glob(".nano-link-query-*")), [])
            self.assertEqual(self.private_query("", directory)[0], 0)
            tool = self.tool(directory, "from pathlib import Path\nimport sys\n"
                "(Path(sys.argv[sys.argv.index('-o')+1]).parent/'nested').mkdir(exist_ok=True)\n"
                "print('GNU ld (fixture) 2.40')\n")
            grammar, error = self.private_query(shlex.quote(str(tool)), directory)
            self.assertEqual(grammar, 0)
            self.assertIn(b"retained private build files", error)
            self.assertEqual(len(list(directory.glob(".nano-link-query-*"))), 1)

    def test_private_query_allocation_failures_clean_up_and_retry(self):
        with tempfile.TemporaryDirectory(prefix="nano-private-query-allocation-") as tmp:
            directory = Path(tmp)
            tool = self.tool(directory, "from pathlib import Path\nimport sys\n"
                "Path(sys.argv[sys.argv.index('-o')+1]).write_text('private')\n"
                "print('GNU ld (fixture) 2.40')\n")
            outcomes = set()
            for budget in range(32):
                result = subprocess.run([str(shadows.ROOT / "obj/test_module_generation_probe"),
                    "private-link-response-allocation", shlex.quote(str(tool)), str(directory), str(budget)],
                    cwd=directory, capture_output=True, timeout=8)
                self.assertEqual(result.returncode, 0, (budget, result.stderr))
                outcomes.add(int(result.stdout))
                self.assertEqual(list(directory.glob(".nano-link-query-*")), [])
            self.assertEqual(outcomes, {0, 1})

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
            self.assertEqual(self.private_query(shlex.join(command), directory)[0], expected)
            self.assertFalse(output.exists())
            alternate_log = directory / "alternate-called"
            alternate = "ld64.lld" if sys.platform == "darwin" else "ld.gold"
            self.tool(directory, "from pathlib import Path\n"
                      f"Path({str(alternate_log)!r}).touch()\nprint('I am an unsupported linker')\n", alternate)
            selector = "-fuse-ld=lld" if sys.platform == "darwin" else "-fuse-ld=gold"
            self.assertEqual(self.query([*command, selector], directory), 0)
            self.assertTrue(alternate_log.exists())
            self.assertFalse(output.exists())
            alternate_log.unlink()
            self.assertEqual(self.private_query(shlex.join([*command, selector]), directory)[0], 0)
            self.assertTrue(alternate_log.exists())
            self.assertFalse(output.exists())
            alternate_log.unlink()
            response = directory / "driver.rsp"
            response.write_text(selector + "\n")
            self.assertEqual(self.private_query(shlex.join([*command, "@" + str(response)]), directory)[0], 0)
            self.assertFalse(alternate_log.exists())
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
