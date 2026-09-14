"""I compile retained translation units and keep failed replacements private."""

import json
import os
import re
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
        cls.gnu_read_replay = False
        if sys.platform == "linux":
            query = subprocess.run([shutil.which("cc"), "-print-prog-name=as"], capture_output=True, timeout=10)
            assembler = shutil.which(query.stdout.decode().strip()) if query.returncode == 0 else None
            if assembler:
                version = subprocess.run([assembler, "--version"], capture_output=True, timeout=10)
                cls.gnu_read_replay = version.returncode == 0 and subprocess.run(
                    [str(cache.ROOT / "obj/test_module_generation_probe"), "assembler-version", version.stdout.decode()],
                    capture_output=True, timeout=10).returncode == 0
                cls.read_replay = cls.gnu_read_replay and not cls.clang

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

    def test_capture_timeout_configuration(self):
        accepted = ((None, 30000), ("1", 1), ("00012", 12), ("300000", 300000))
        rejected = ("", "0", "000", "300001", "-1", "+1", " 1", "1 ", "1\n", "1.0",
                    "1e3", "１２", "9" * 1000)
        for value, expected in (*accepted, *((v, None) for v in rejected)):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("NANO_CAPTURE_TIMEOUT_MS", None)
                if value is not None: env["NANO_CAPTURE_TIMEOUT_MS"] = value
                result = subprocess.run([str(self.support.probe), "capture-budget"], env=env,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1 if expected is None else 0, result.stderr)
                if expected is None:
                    self.assertEqual(result.stdout, b"")
                    self.assertEqual(result.stderr, b"I require NANO_CAPTURE_TIMEOUT_MS to be decimal milliseconds in 1..300000.\n")
                else:
                    self.assertGreaterEqual(int(result.stdout), expected)
                    self.assertLess(int(result.stdout), expected + 100)
                    self.assertEqual(result.stderr, b"")

    def test_configured_capture_default_and_expiry(self):
        for budget, command, success in ((None, "sleep 6; printf evidence", True),
                                          ("100", "sleep 30; printf evidence", False)):
            with self.subTest(budget=budget):
                env = dict(os.environ)
                env.pop("NANO_CAPTURE_TIMEOUT_MS", None)
                if budget is not None: env["NANO_CAPTURE_TIMEOUT_MS"] = budget
                result = subprocess.run([str(self.support.probe), "capture-configured", command], env=env,
                                        capture_output=True, timeout=40 if success else 5)
                self.assertEqual(result.returncode, 0 if success else 1, result.stderr)
                self.assertEqual(result.stdout, b"evidence" if success else b"")

    def test_capture_configuration_clock_rejection(self):
        for fault in ("error", "deadline-overflow"):
            for traced in (False, True):
                with self.subTest(fault=fault, traced=traced):
                    env = dict(os.environ, NANO_TEST_CAPTURE_CLOCK=fault, NANO_CAPTURE_TIMEOUT_MS="30000")
                    env.pop("NANO_TRACE_BUILD", None)
                    if traced: env["NANO_TRACE_BUILD"] = "1"
                    result = subprocess.run([str(self.support.probe), "capture-configured", "printf spawned"],
                                            env=env, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(result.stdout, b"")
                    self.assertEqual(result.stderr, b"I cannot establish a capture deadline.\n")

    def test_slow_production_capture_and_timeout_independent_reuse(self):
        if not self.clang: self.skipTest("I delay Clang capture discovery here")
        with tempfile.TemporaryDirectory(prefix="nano-slow-capture-") as tmp:
            root = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(root)
            symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
            assembly = f'.macro emit number\n.byte \\number\n.endm\n.data\n.globl {symbol}\n{symbol}:\nemit 42\n'
            (module / "payload.s").write_text(assembly)
            (module / "answer.c").write_text('extern unsigned char snapshot_payload[];\n'
                                             'long long nano_build_answer(void) { return snapshot_payload[0]; }\n')
            (module / "module.json").write_text(json.dumps({"name": "answer_native",
                "c_sources": ["answer.c", "payload.s"]}))
            marker, wrapper = root / "delayed", root / "cc"
            compiler = shutil.which("cc")
            wrapper.write_text(f'#!{sys.executable}\nimport os,pathlib,sys,time\n'
                f'marker=pathlib.Path({str(marker)!r})\n'
                'if "-###" in sys.argv and any(pathlib.Path(x).name == "__assembly_0_1.s" for x in sys.argv[1:]) and not marker.exists():\n'
                '    marker.touch()\n    time.sleep(6)\n'
                f'os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])\n')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            env["NANO_TRACE_BUILD"] = "1"
            env.pop("NANO_CAPTURE_TIMEOUT_MS", None)
            first = None
            for budget in (None, None, "60000"):
                if marker.exists(): marker.unlink()
                if budget is not None: env["NANO_CAPTURE_TIMEOUT_MS"] = budget
                result = subprocess.run([str(self.support.probe), "build", str(module)], env=env,
                                        capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(marker.exists(), result.stderr)
                queries = re.findall(rb'phase=tool-query-ms expected=\d+ observed=(\d+) accepted=1', result.stderr)
                self.assertTrue(any(int(t) >= 6000 for t in queries), result.stderr)
                current = self.support.probe_path("directory", module, env)
                self.assertTrue((current / "source_hashes.json").is_file())
                libraries = list(current.glob("libanswer_native.*"))
                self.assertEqual(len(libraries), 1)
                self.assertEqual(self.answer(libraries[0]), 42)
                if first is None: first = current
                else:
                    self.assertEqual(current, first, result.stderr)
                    self.assertIn(b"phase=reuse expected=1 observed=1 accepted=1", result.stderr)

    def test_configured_capture_descendant_cleanup(self):
        for traced in (False, True):
            with self.subTest(traced=traced), tempfile.TemporaryDirectory(prefix="nano-capture-cleanup-") as tmp:
                marker = Path(tmp) / "descendant-survived"
                # I leave the descendant holding the output pipe after its parent exits.
                code = ("import os,pathlib,time; pid=os.fork(); "
                        "os._exit(0) if pid else None; "
                        "print('started',flush=True); time.sleep(2); "
                        f"pathlib.Path({str(marker)!r}).touch()")
                env = dict(os.environ, NANO_CAPTURE_TIMEOUT_MS="1000")
                env.pop("NANO_TRACE_BUILD", None)
                if traced: env["NANO_TRACE_BUILD"] = "1"
                result = subprocess.run([str(self.support.probe), "capture-configured",
                                         "exec " + shlex.join([sys.executable, "-c", code])],
                                        env=env, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, b"started\n")
                time.sleep(1.5)
                self.assertFalse(marker.exists(), "I left a descendant executing after capture expiry")

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

    def test_build_evidence_trace_reports_tool_failure_and_deadline(self):
        env = os.environ.copy()
        env["NANO_TRACE_BUILD"] = "1"
        for command, phase, output in (("printf evidence; exit 1", "tool-output", b"evidence"),
                                       ("sleep 30", "tool-deadline", b"")):
            with self.subTest(phase=phase):
                result = subprocess.run([str(self.support.probe), "capture-environment", command], env=env,
                                        capture_output=True, timeout=8)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, rb"phase=tool-run-ms expected=[0-9]+ observed=[0-9]+ accepted=0")
                self.assertEqual(result.stdout, output)
                self.assertIn(f"phase={phase} ".encode(), result.stderr)
                self.assertIn(b"accepted=0", result.stderr)
                self.assertNotIn(command.encode(), result.stderr)
                if phase == "tool-deadline":
                    for name in ("first-output", "eof", "reaped"):
                        self.assertIn(f"phase=tool-{name}-ms expected=1 observed=0 accepted=0".encode(),
                                      result.stderr)
        env.pop("NANO_TRACE_BUILD")
        quiet = subprocess.run([str(self.support.probe), "capture-environment", "printf evidence; exit 1"],
                               env=env, capture_output=True, timeout=8)
        self.assertNotEqual(quiet.returncode, 0)
        self.assertEqual(quiet.stdout, b"evidence")
        self.assertEqual(quiet.stderr, b"")

    def test_tool_supervisor_rejects_late_completion(self):
        for traced in (False, True):
            for fault in ("expired", "error"):
                with self.subTest(traced=traced, fault=fault):
                    env = os.environ.copy()
                    env.pop("NANO_TRACE_BUILD", None)
                    if traced: env["NANO_TRACE_BUILD"] = "1"
                    env["NANO_TEST_COMPLETION_CLOCK"] = fault
                    result = subprocess.run([str(self.support.probe), "capture-environment", "printf evidence"],
                                            env=env, capture_output=True, timeout=8)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(result.stdout, b"evidence")
                    if traced:
                        phase = "tool-deadline" if fault == "expired" else "tool-output"
                        self.assertIn(f"phase={phase} ".encode(), result.stderr)
                        self.assertRegex(result.stderr, rb"phase=tool-run-ms expected=\d+ observed=\d+ accepted=0")
                    else:
                        self.assertEqual(result.stderr, b"")

    def test_early_build_failure_evidence(self):
        phases = ("metadata-path", "metadata", "invocation", "cache-directory",
                  "cache-path", "lock-open", "lock-acquire", "stage-create")
        for shared in (False, True):
            for phase in phases:
                with self.subTest(shared=shared, phase=phase), tempfile.TemporaryDirectory(prefix="nano-early-build-") as tmp:
                    root = Path(tmp)
                    module, _, env = self.support.support.foreign_build_fixture(root)
                    if shared: env["NANO_BUILD_CACHE"] = str(root / "cache")
                    cache_dir = self.support.probe_path("root", module, env)
                    manifest = module / "module.json"
                    original = manifest.read_text()
                    target = module
                    if phase == "metadata-path": target = module / "missing"
                    elif phase == "metadata": manifest.unlink()
                    elif phase == "invocation":
                        meta = json.loads(original)
                        meta["cflags"] = ["@missing.rsp"]
                        manifest.write_text(json.dumps(meta))
                    elif phase == "lock-open":
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        (cache_dir / ".build.lock").symlink_to(root / "missing-lock")
                    else:
                        env["NANO_TEST_EARLY_FAILURE"] = phase
                        env["NANO_TEST_EARLY_CACHE"] = str(cache_dir if shared else cache_dir.resolve())
                    for traced in (False, True):
                        failed_env = dict(env)
                        failed_env.pop("NANO_TRACE_BUILD", None)
                        if traced: failed_env["NANO_TRACE_BUILD"] = "1"
                        result = subprocess.run([str(self.support.probe), "build", str(target)],
                                                env=failed_env, capture_output=True, timeout=20)
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertEqual(result.stdout, b"")
                        marker = f"phase=build-{phase} expected=1 observed=0 accepted=0".encode()
                        if traced: self.assertIn(marker, result.stderr)
                        else: self.assertNotIn(b"I checked build evidence:", result.stderr)
                        self.assertFalse((cache_dir / "current").exists())
                        if cache_dir.is_dir(): self.assertFalse(list(cache_dir.glob(".nano-build-*")))
                    manifest.write_text(original)
                    if phase == "lock-open": (cache_dir / ".build.lock").unlink()
                    env.pop("NANO_TEST_EARLY_FAILURE", None)
                    env.pop("NANO_TEST_EARLY_CACHE", None)
                    self.support.probe_path("build", module, env, timeout=20)
                    first = self.support.probe_path("directory", module, env)
                    self.support.probe_path("build", module, env, timeout=20)
                    self.assertEqual(self.support.probe_path("directory", module, env), first)

    def test_tool_supervisor_timing_milestones(self):
        env = os.environ.copy()
        env["NANO_TRACE_BUILD"] = "1"
        for delayed_exit in (False, True):
            command = ("printf evidence; exec 1>&- 2>&-; sleep 0.3" if delayed_exit else
                       "sleep 0.3; printf evidence")
            with self.subTest(delayed_exit=delayed_exit):
                result = subprocess.run([str(self.support.probe), "capture-environment", command],
                                        env=env, capture_output=True, timeout=8)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, b"evidence")
                times = {}
                for name in ("spawn", "first-output", "eof", "reaped"):
                    match = re.search(fr"phase=tool-{name}-ms expected=1 observed=([0-9]+) accepted=1".encode(),
                                      result.stderr)
                    self.assertIsNotNone(match, result.stderr)
                    times[name] = int(match[1])
                self.assertLessEqual(times["spawn"], times["first-output"])
                self.assertLessEqual(times["first-output"], times["eof"])
                if delayed_exit:
                    self.assertGreaterEqual(times["reaped"] - times["eof"], 150, result.stderr)
                else:
                    self.assertGreaterEqual(times["first-output"] - times["spawn"], 150, result.stderr)

    def test_build_evidence_trace_and_post_capture_validation_failure(self):
        self.build_evidence_trace_and_post_capture_validation_failure()

    def test_cache_record_rename_failure_diagnostics_and_recovery(self):
        for shared in (False, True):
            for traced in (False, True):
                with self.subTest(shared=shared, traced=traced), tempfile.TemporaryDirectory(prefix="nano-record-write-") as tmp:
                    root = Path(tmp)
                    module, _, env = self.support.support.foreign_build_fixture(root)
                    if shared: env["NANO_BUILD_CACHE"] = str(root / "cache")
                    env.pop("NANO_TRACE_BUILD", None)
                    if traced: env["NANO_TRACE_BUILD"] = "1"
                    source = module / "answer.c"
                    source.write_text('long long nano_build_answer(void) { return 42; }\n')
                    (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": [source.name]}))
                    def build(value):
                        self.support.probe_path("build", module, env, timeout=30)
                        generation = self.support.probe_path("directory", module, env)
                        self.assertEqual(self.answer(self.support.probe_path("library", module, env)), value)
                        self.assertFalse(list(generation.parent.glob(".nano-build-*")))
                        self.assertFalse(list(generation.glob("source_hashes.json.*")))
                        return generation
                    first = build(42)
                    self.assertTrue((first / "source_hashes.json").is_file())
                    saved = self.support.snapshot(first)
                    source.write_text('long long nano_build_answer(void) { return 43; }\n')
                    env["NANO_TEST_RECORD_RENAME_FAILURE"] = "1"
                    uncached = build(43)
                    self.assertNotEqual(uncached, first)
                    self.assertFalse((uncached / "source_hashes.json").exists())
                    self.assertEqual(self.support.snapshot(first), saved)
                    if traced:
                        self.assertIn(b"phase=record-dependencies expected=1 observed=1 accepted=1", self.support.last_build_diagnostics)
                        self.assertIn(b"phase=record-write expected=1 observed=0 accepted=0", self.support.last_build_diagnostics)
                    else:
                        self.assertNotIn(b"phase=record-write", self.support.last_build_diagnostics)
                    env.pop("NANO_TEST_RECORD_RENAME_FAILURE")
                    recovered = build(43)
                    self.assertNotEqual(recovered, uncached)
                    self.assertTrue((recovered / "source_hashes.json").is_file())
                    if traced:
                        self.assertIn(b"phase=record-write expected=1 observed=1 accepted=1", self.support.last_build_diagnostics)
                    self.assertEqual(build(43), recovered, self.support.last_build_diagnostics)

    def test_raw_unit_post_link_validation_deadline(self):
        if not self.clang: self.skipTest("I need the selected Clang assembler query")
        for shared_unit in (False, True):
            with self.subTest(shared_unit=shared_unit):
                self.build_evidence_trace_and_post_capture_validation_failure(".s", shared_unit)

    def test_preprocessed_unit_post_link_validation_deadline(self):
        if not self.clang: self.skipTest("I need the selected Clang assembler query")
        for shared_unit in (False, True):
            with self.subTest(shared_unit=shared_unit):
                self.build_evidence_trace_and_post_capture_validation_failure(".S", shared_unit)

    def build_evidence_trace_and_post_capture_validation_failure(self, unit_suffix=None, shared_unit=False):
        modes = (False, True) if self.clang and (not unit_suffix or sys.platform == "darwin") else (False,)
        for external in modes:
            for shared in (False, True):
                with self.subTest(external=external, shared=shared), tempfile.TemporaryDirectory(prefix="nano-trace-evidence-") as tmp:
                    root = Path(tmp)
                    module, _, env = self.support.support.foreign_build_fixture(root)
                    env["NANO_CAPTURE_TIMEOUT_MS"] = "5000"
                    if shared: env["NANO_BUILD_CACHE"] = str(root / "cache")
                    env["NANO_AS_CAPTURE_HELPER"] = str(cache.ROOT / "bin/nano_as_capture.so")
                    source = module / "answer.c"
                    contents = 'long long nano_build_answer(void) { return 42; }\n'
                    metadata = {"name": "answer_native", "c_sources": ["answer.c"],
                                "cflags": ["-fno-integrated-as"] if external else []}
                    if unit_suffix:
                        source.write_text('extern unsigned char snapshot_payload[];\n'
                                          'long long nano_build_answer(void) { return snapshot_payload[0]; }\n')
                        source = module / ("payload" + unit_suffix)
                        symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                        # I require native expansion, not the literal-capture fast path.
                        contents = f'.macro emit number\n.byte \\number\n.endm\n.data\n.globl {symbol}\n{symbol}:\nemit 42\n'
                        metadata.setdefault("shared_c_sources" if shared_unit else "c_sources", []).append(source.name)
                        temporary = root / "temporary"
                        temporary.mkdir()
                        env["TMPDIR"] = str(temporary)
                    source.write_text(contents)
                    (module / "module.json").write_text(json.dumps(metadata))
                    marker, wrapper = root / "linked", root / "cc"
                    query_pid = root / "query-pid"
                    query_input = f"__assembly_{1 if shared_unit else 0}_{0 if shared_unit else 1}.s"
                    compiler = shutil.which("cc")
                    wrapper.write_text(f'#!{sys.executable}\nimport os,pathlib,sys,time\n'
                        f'marker=pathlib.Path({str(marker)!r})\n'
                        'if os.getenv("NANO_TEST_REJECT_VALIDATION"):\n'
                        f'    if marker.exists() and {bool(unit_suffix)!r} and "-###" in sys.argv and '
                        f'any(pathlib.Path(x).name == {query_input!r} for x in sys.argv[1:]):\n'
                        f'        pathlib.Path({str(query_pid)!r}).write_text(str(os.getpid()))\n'
                        '        time.sleep(30)\n'
                        f'    if marker.exists() and {not bool(unit_suffix)!r} and any(x in sys.argv for x in ("-E", "-S")): sys.exit(1)\n'
                        '    if "-###" not in sys.argv and "-o" in sys.argv and '
                        'pathlib.Path(sys.argv[sys.argv.index("-o")+1]).name in ("libanswer_native.dylib", "libanswer_native.so"):\n'
                        '        marker.touch()\n'
                        f'os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])\n')
                    wrapper.chmod(0o700)
                    env["NANO_CC"] = str(wrapper)
                    env.pop("NANO_TRACE_BUILD", None)
                    def build():
                        result = subprocess.run([str(self.support.probe), "build", str(module)], cwd=cache.ROOT,
                                                env=env, capture_output=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        return result
                    cold = build()
                    self.assertNotIn(b"I checked build evidence:", cold.stderr)
                    first = self.support.probe_path("directory", module, env)
                    self.assertTrue((first / "source_hashes.json").is_file())
                    env["NANO_TRACE_BUILD"] = "1"
                    warm = build()
                    self.assertIn(b"phase=reuse expected=1 observed=1 accepted=1", warm.stderr)
                    self.assertEqual(self.support.probe_path("directory", module, env), first)
                    self.assertNotIn(b"I checked build evidence:", warm.stdout)
                    self.assertNotIn(str(wrapper).encode(), warm.stderr)
                    source.write_text(contents.replace("42", "43"))
                    env["NANO_TEST_REJECT_VALIDATION"] = "1"
                    uncached = build()
                    self.assertTrue(marker.is_file())
                    if unit_suffix:
                        self.assertIn(b"phase=tool-deadline ", uncached.stderr)
                        self.assertRegex(uncached.stderr, rb"phase=tool-query-ms expected=[0-9]+ observed=[0-9]+ accepted=0")
                        self.assertTrue(query_pid.is_file(), uncached.stderr)
                        with self.assertRaises(ProcessLookupError): os.kill(int(query_pid.read_text()), 0)
                        self.assertFalse(list(temporary.glob("nano-gcc-check-*")))
                    self.assertRegex(uncached.stderr, rb'phase=publish-preprocessing expected=[1-9][0-9]* observed=0 accepted=0')
                    generation = self.support.probe_path("directory", module, env)
                    self.assertNotEqual(generation, first)
                    self.assertTrue(list(generation.glob("__snapshot_*")))
                    if unit_suffix:
                        self.assertTrue((generation / f"__native_unit_{1 if shared_unit else 0}_{0 if shared_unit else 1}.o").is_file())
                    self.assertFalse((generation / "source_hashes.json").exists())
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                    env.pop("NANO_TEST_REJECT_VALIDATION")
                    marker.unlink()
                    recovered = build()
                    self.assertIn(b"phase=reuse-record expected=1 observed=0 accepted=0", recovered.stderr)
                    second = self.support.probe_path("directory", module, env)
                    self.assertNotEqual(second, generation)
                    self.assertTrue((second / "source_hashes.json").is_file())
                    env.pop("NANO_TRACE_BUILD")
                    quiet = build()
                    self.assertNotIn(b"I checked build evidence:", quiet.stderr)
                    self.assertEqual(self.support.probe_path("directory", module, env), second)

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

    def test_assembler_translation_units_restored_inputs(self):
        kinds = (("assembler-external-unit", "assembler-external-preprocessed-unit") if self.clang
                 else ("assembler-unit", "assembler-preprocessed-unit"))
        for shared_unit in (False, True):
            for removed in (False, True):
                result = measure(shutil.which("cc"), kinds, shared_unit=shared_unit, remove_input=removed)
                require_consistent(result)
                for case in result["cases"]:
                    with self.subTest(shared_unit=shared_unit, removed=removed, case=case):
                        self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                        for key in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record",
                                    "generation_reused", "retained_assembly"):
                            self.assertTrue(case[key], key)

    def test_absolute_source_hash_and_overflow(self):
        with tempfile.TemporaryDirectory(prefix="nano-source-hash-") as tmp:
            root = Path(tmp)
            source = root / "source.c"
            source.write_text("first bytes")
            def digest(directory, name):
                return subprocess.run([str(self.support.probe), "source-hash", str(directory), str(name)],
                                      capture_output=True, timeout=10)
            relative = digest(root, source.name)
            absolute = digest(root / "not a directory", source)
            self.assertEqual(relative.returncode, 0, relative.stderr)
            self.assertEqual(absolute.returncode, 0, absolute.stderr)
            self.assertEqual(relative.stdout, absolute.stdout)
            source.write_text("changed bytes")
            self.assertNotEqual(digest(root, source).stdout, absolute.stdout)
            for directory, name in ((root, "x" * 4096), ("x" * 4096, "source.c")):
                failed = digest(directory, name)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(failed.stdout, b"0\n")

    def test_apple_lowercase_assembly_is_preprocessed(self):
        if sys.platform != "darwin" or not self.clang:
            self.skipTest("I exercise Apple Clang's lowercase assembler default")
        for shared_unit in (False, True):
            for removed in (False, True):
                result = measure(shutil.which("cc"), ("assembler-external-unit",),
                                 preprocess_raw=True, shared_unit=shared_unit, remove_input=removed)
                require_consistent(result)
                for case in result["cases"]:
                    with self.subTest(shared_unit=shared_unit, removed=removed, case=case):
                        self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                        self.assertTrue(case["generation_reused"])

    def test_assembler_filename_spelling_restored_inputs(self):
        kinds = ("assembler", "assembler-external") if self.clang else ("assembler",)
        names = ("space name.bin", "single'quote.bin", 'double"quote.bin', r"back\slash.bin", "naïve-λ.bin")
        for name, remove_input in ((name, remove_input) for name in names for remove_input in (False, True)):
            result = measure(shutil.which("cc"), kinds, payload_name=name, remove_input=remove_input)
            require_consistent(result)
            for case in result["cases"]:
                with self.subTest(name=name, remove_input=remove_input, case=case):
                    self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                    for key in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record", "generation_reused"):
                        self.assertTrue(case[key], key)
                    if case["input"] == "assembler-external":
                        self.assertEqual(case["external_assembly_compilations"], case["total_object_compilations"])

    def test_assembler_filename_spelling_and_cache_recovery(self):
        self.assembler_filename_spelling_and_cache_recovery()

    def test_alternate_assembler_cache_recovery(self):
        if not self.gnu_read_replay: self.skipTest("I need supported GNU assembler read replay")
        self.assembler_filename_spelling_and_cache_recovery(alternate=True)

    def test_assembler_translation_units_cache_recovery(self):
        for unit in ("s", "S"):
            self.assembler_filename_spelling_and_cache_recovery(unit=unit)

    def test_equals_path_cache_recovery(self):
        for unit in ("s", "S"):
            self.assembler_filename_spelling_and_cache_recovery(unit=unit, equals_paths=True)

    def assembler_filename_spelling_and_cache_recovery(self, alternate=False, unit=None, equals_paths=False):
        modes = (False, True) if self.clang else (False,)
        names = ("space name.bin", "single'quote.bin", 'double"quote.bin', r"back\slash.bin", "naïve-λ.bin")
        if alternate:
            modes = (True,) if self.clang else (False,)
            names = ("alternate payload.bin",)
        if unit:
            modes = (True,) if self.clang else (False,)
            names = ("unit payload.bin",)
        for external, shared, name in ((external, shared, name) for external in modes
                                      for shared in (False, True) for name in names):
            with self.subTest(external=external, shared=shared, name=name), tempfile.TemporaryDirectory(prefix="nano=assembler-path-" if equals_paths else "nano-assembler-path-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                env["NANO_AS_CAPTURE_HELPER"] = str(cache.ROOT / "bin/nano_as_capture.so")
                payload = module / name
                payload.write_bytes(b"42")
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                assembly = f'.data\n.globl {symbol}\n{symbol}:\n.incbin {json.dumps(str(payload), ensure_ascii=False)}\n.text\n'
                if alternate:
                    assembly = (f'.data\n.globl {symbol}\n{symbol}:\n.macro emit file\n'
                                '.incbin "\\file"\n.endm\nemit <' + str(payload) + '>\n.text\n')
                extra_sources = []
                if unit:
                    assembly = (f'.data\n.globl {symbol}\n{symbol}:\n.macro emit file\n'
                                '.incbin "\\file"\n.endm\nemit "' + str(payload) + '"\n.text\n')
                    if unit == "S":
                        assembly = '#define PAYLOAD ' + json.dumps(str(payload)) + '\n' + assembly.replace('emit "' + str(payload) + '"', 'emit PAYLOAD')
                    assembly_source = module / ("payload." + unit)
                    assembly_source.write_text(assembly)
                    extra_sources = [str(assembly_source)]
                source = module / "answer.c"
                source.write_text('extern const unsigned char snapshot_payload[];\n'
                    + ('' if unit else '__asm__(' + json.dumps(assembly) + ');\n') +
                    'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                flags = ["-fno-integrated-as"] if external else []
                if equals_paths: flags.append("-g")
                if alternate: flags.append("-Wa,--alternate")
                (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"] + extra_sources, "cflags": flags}))
                direct = directory / ("direct.dylib" if sys.platform == "darwin" else "direct.so")
                result = subprocess.run([shutil.which("cc"), "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-fPIC", *flags, str(source), *extra_sources, "-o", str(direct)], capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.answer(direct), 42)
                self.support.probe_path("build", module, env, timeout=20)
                first = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertTrue((first / "source_hashes.json").exists())
                self.support.probe_path("build", module, env, timeout=20)
                self.assertEqual(self.support.probe_path("directory", module, env), first)
                payload.write_bytes(b"43")
                self.support.probe_path("build", module, env, timeout=20)
                changed = self.support.probe_path("directory", module, env)
                self.assertNotEqual(changed, first)
                library = self.support.probe_path("library", module, env)
                self.assertEqual(self.answer(library), 43)
                saved = self.support.snapshot(changed)
                payload.unlink()
                failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0, failed.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), changed)
                self.assertEqual(self.support.snapshot(changed), saved)
                self.assertFalse(list(changed.parent.glob(".nano-build-*")))
                self.assertEqual(self.answer(library), 43)
                payload.write_bytes(b"44")
                self.support.probe_path("build", module, env, timeout=20)
                recovered = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 44)
                self.support.probe_path("build", module, env, timeout=20)
                self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_standalone_original_basename_and_flat_publication(self):
        from tests.characterize_assembler_debug import measure as measure_debug
        for alias in (False, True):
            for case in measure_debug(shutil.which("cc"), module_alias=alias)["cases"]:
                with self.subTest(alias=alias, suffix=case["suffix"], cache=case["cache"]):
                    self.assertEqual(case["published_unit_aliases"], [])
                    self.assertTrue(case["production"]["source_named"])
                    self.assertFalse(case["production"]["private_snapshot_named"])
                    self.assertTrue(case["generation_reused"])
                    self.assertEqual(case["production"], case["physical_native"])
                    self.assertTrue(case["physical_object_identical"], case["physical_debug_diff"])

    def test_unit_alias_copy_isolation_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="nano-unit-alias-") as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()
            paths = []
            for index in range(2):
                (stage / f"__snapshot_0_{index}.s").write_bytes(bytes([42 + index]))
                result = subprocess.run([str(self.support.probe), "unit-input", str(stage),
                                         f"/original/{index}/same name.s", str(index)], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                paths.append(Path(result.stdout.decode().strip()))
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual([p.read_bytes() for p in paths], [b"*", b"+"])
            self.assertTrue(all(p.name == "same name.s" and p.stat().st_mode & 0o777 == 0o400 for p in paths))
            result = subprocess.run([str(self.support.probe), "sync-generation", str(stage)], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            result = subprocess.run([str(self.support.probe), "remove-unit-aliases", str(stage)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(p.name for p in stage.iterdir()), ["__snapshot_0_0.s", "__snapshot_0_1.s"])
            nested = stage / "__unit_0_0" / "unexpected"
            nested.mkdir(parents=True)
            valuable = nested / "keep"
            valuable.write_bytes(b"I require explicit recursive cleanup.")
            result = subprocess.run([str(self.support.probe), "remove-unit-aliases", str(stage)], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(valuable.read_bytes(), b"I require explicit recursive cleanup.")
            result = subprocess.run([str(self.support.probe), "sync-generation", str(stage)], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)

    def test_native_unit_copy_rejects_substitutions(self):
        for kind in ("regular", "missing", "empty", "oversize", "source-shrink", "stage-link",
                     "source-link", "source-fifo", "source-directory", "output-file", "output-link",
                     "output-directory", "outside"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-native-copy-") as tmp:
                root = Path(tmp)
                stage = root / "stage"
                if kind == "stage-link":
                    actual = root / "actual"
                    actual.mkdir()
                    stage.symlink_to(actual, target_is_directory=True)
                else: stage.mkdir()
                valuable = root / "valuable"
                valuable.write_bytes(b"I stay unchanged.")
                source = stage / "__native_unit_0_0.o"
                output = stage / "output.o"
                if kind == "source-link": source.symlink_to(valuable)
                elif kind == "source-fifo": os.mkfifo(source)
                elif kind == "source-directory": source.mkdir()
                elif kind != "missing": source.write_bytes(b"" if kind == "empty" else bytes(range(256)) * 100)
                if kind == "oversize":
                    with source.open("r+b") as stream: stream.truncate(32 * 1024 * 1024 + 1)
                if kind == "output-file": output.write_bytes(b"existing")
                elif kind == "output-link": output.symlink_to(valuable)
                elif kind == "output-directory": output.mkdir()
                elif kind == "outside": output = stage / ".." / "valuable"
                env = os.environ.copy()
                if kind == "source-shrink":
                    env.update(NANO_TEST_RESPONSE_MUTATE=str(source), NANO_TEST_RESPONSE_ON_READ="1")
                result = subprocess.run([str(self.support.probe), "copy-native-unit", str(stage), str(output)],
                                        env=env, capture_output=True, timeout=5)
                self.assertEqual(result.returncode == 0, kind == "regular", result.stderr)
                self.assertEqual(valuable.read_bytes(), b"I stay unchanged.")
                if kind == "regular":
                    self.assertEqual(source.read_bytes(), output.read_bytes())
                    self.assertNotEqual(source.stat().st_ino, output.stat().st_ino)
                    self.assertEqual(output.stat().st_mode & 0o777, 0o400)
                elif kind == "output-file": self.assertEqual(output.read_bytes(), b"existing")
                elif kind not in ("output-link", "output-directory", "outside"):
                    self.assertFalse(output.exists())

    def test_native_unit_post_capture_reads(self):
        if sys.platform != "darwin": self.skipTest("I exercise selected Apple native unit transport")
        self.native_unit_post_capture_reads()

    def test_integrated_native_unit_post_capture_reads(self):
        if not shutil.which("clang"): self.skipTest("I require integrated Clang")
        self.native_unit_post_capture_reads(integrated=True)

    def test_equals_path_post_capture_reads(self):
        if not shutil.which("clang"): self.skipTest("I require selected Clang native capture")
        for integrated in ((False, True) if self.clang else (True,)):
            self.native_unit_post_capture_reads(integrated=integrated, equals_paths=True)

    def native_unit_post_capture_reads(self, integrated=False, equals_paths=False):
        compiler = shutil.which("clang" if integrated else "cc")
        for shared_unit in (False, True):
            for shared_cache in (False, True):
                for suffix in (".s", ".S"):
                    with self.subTest(shared_unit=shared_unit, shared_cache=shared_cache, suffix=suffix), tempfile.TemporaryDirectory(prefix="nano=native-timing-" if equals_paths else "nano-native-timing-") as tmp:
                        root = Path(tmp)
                        module, _, env = self.support.support.foreign_build_fixture(root)
                        if shared_cache: env["NANO_BUILD_CACHE"] = str(root / "cache")
                        payload = module / "payload.bin"
                        payload.write_bytes(b"*")
                        nested = module / "macro.s"
                        nested.write_text('.macro emit path\n.incbin "\\path"\n.endm\n')
                        source = module / ("payload" + suffix)
                        symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                        source.write_text(f'.include "{nested}"\n.data\n.globl {symbol}\n{symbol}:\nemit "{payload}"\n')
                        (module / "answer.c").write_text('extern unsigned char snapshot_payload[];\n'
                            'long long nano_build_answer(void) { return snapshot_payload[0]; }\n')
                        metadata = {"name": "answer_native", "c_sources": ["answer.c"],
                                    "cflags": ["-g"] + ([] if integrated else ["-fno-integrated-as"])}
                        metadata.setdefault("shared_c_sources" if shared_unit else "c_sources", []).append(str(source) if shared_unit else source.name)
                        (module / "module.json").write_text(json.dumps(metadata))
                        marker, restored = root / "mutated", root / "restored"
                        wrapper = root / "cc-wrapper"
                        native_name = f"__native_unit_{1 if shared_unit else 0}_{0 if shared_unit else 1}.o"
                        wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
marker, restored = pathlib.Path({str(marker)!r}), pathlib.Path({str(restored)!r})
payload, nested = pathlib.Path({str(payload)!r}), pathlib.Path({str(nested)!r})
if '-c' in sys.argv and '-###' not in sys.argv and os.getenv('NANO_AS_CAPTURE_PHASE') != 'capture':
    for arg in sys.argv[1:]:
        unit = pathlib.Path(arg)
        if unit.name == '__snapshot_0_0.s' and (unit.parent / {native_name!r}).is_file() and not marker.exists():
            payload.write_bytes(b'+')
            nested.rename(nested.with_suffix('.missing'))
            marker.write_text('I changed the binary and removed the macro after native capture.')
if any(flag in sys.argv for flag in ('-dynamiclib', '-shared')) and '-###' not in sys.argv and marker.exists() and not restored.exists():
    assert payload.read_bytes() == b'+' and not nested.exists()
    try:
        result = subprocess.run([{compiler!r}] + sys.argv[1:])
    finally:
        payload.write_bytes(b'*')
        nested.with_suffix('.missing').rename(nested)
        restored.write_text('I kept both changes through final linking.')
    sys.exit(result.returncode)
os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])
''')
                        wrapper.chmod(0o700)
                        env["NANO_CC"] = str(wrapper)
                        self.support.probe_path("build", module, env, timeout=30)
                        first = self.support.probe_path("directory", module, env)
                        self.assertTrue(marker.is_file())
                        self.assertTrue(restored.is_file())
                        self.assertEqual(payload.read_bytes(), b"*")
                        self.assertTrue(nested.is_file())
                        self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                        self.support.probe_path("build", module, env, timeout=30)
                        self.assertEqual(self.support.probe_path("directory", module, env), first)

    def test_native_single_source_output(self):
        if sys.platform != "darwin": self.skipTest("I exercise selected Apple native unit transport")
        for shared_cache in (False, True):
            for suffix in (".s", ".S"):
                with self.subTest(shared_cache=shared_cache, suffix=suffix), tempfile.TemporaryDirectory(prefix="nano-native-single-") as tmp:
                    root = Path(tmp)
                    module, _, env = self.support.support.foreign_build_fixture(root)
                    if shared_cache: env["NANO_BUILD_CACHE"] = str(root / "cache")
                    source = module / ("payload" + suffix)
                    source.write_text('.macro emit number\n.byte \\number\n.endm\n.data\n'
                                      '.globl _snapshot_payload\n_snapshot_payload:\nemit 42\n')
                    (module / "module.json").write_text(json.dumps({"name": "answer_native",
                        "c_sources": [source.name], "cflags": ["-g", "-g0", "-fno-integrated-as"]}))
                    self.support.probe_path("build", module, env, timeout=30)
                    first = self.support.probe_path("directory", module, env)
                    self.assertEqual((first / "answer_native.o").read_bytes(), (first / "__native_unit_0_0.o").read_bytes())
                    library = self.support.probe_path("library", module, env)
                    result = subprocess.run([sys.executable, "-c", 'import ctypes,sys; '
                        'print(ctypes.c_ubyte.in_dll(ctypes.CDLL(sys.argv[1]), "snapshot_payload").value)',
                        str(library)], capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), b"42")
                    self.support.probe_path("build", module, env, timeout=30)
                    self.assertEqual(self.support.probe_path("directory", module, env), first)

    def test_unit_aliases_do_not_follow_substituted_paths(self):
        for kind in ("source-link", "source-fifo", "source-directory", "alias-directory-link", "alias-file-link"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-unit-safe-") as tmp:
                root = Path(tmp)
                stage = root / "stage"
                stage.mkdir()
                external = root / "external"
                external.mkdir()
                valuable = external / "payload.s"
                valuable.write_bytes(b"I remain outside the build.")
                retained = stage / "__snapshot_0_0.s"
                alias = stage / "__unit_0_0"
                if kind == "source-link": retained.symlink_to(valuable)
                elif kind == "source-fifo": os.mkfifo(retained)
                elif kind == "source-directory": retained.mkdir()
                else:
                    retained.write_bytes(b"*")
                    if kind == "alias-directory-link": alias.symlink_to(external, target_is_directory=True)
                    else:
                        alias.mkdir()
                        (alias / "payload.s").symlink_to(valuable)
                result = subprocess.run([str(self.support.probe), "unit-input", str(stage),
                                         "/original/payload.s", "0"], capture_output=True, timeout=5)
                self.assertEqual(result.returncode == 0, kind == "alias-file-link", result.stderr)
                self.assertEqual(valuable.read_bytes(), b"I remain outside the build.")
                if kind == "alias-file-link": self.assertEqual((alias / "payload.s").read_bytes(), b"*")
                result = subprocess.run([str(self.support.probe), "remove-unit-aliases", str(stage)],
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(alias.exists())
                self.assertEqual(valuable.read_bytes(), b"I remain outside the build.")

    def test_unit_alias_failed_build_retains_published_generation(self):
        self.unit_alias_failed_build_retains_published_generation()

    def test_apple_native_unit_capture_failure_recovery(self):
        if sys.platform != "darwin": self.skipTest("I exercise the selected Apple native unit capture")
        self.unit_alias_failed_build_retains_published_generation(macro=True)

    def test_apple_native_unit_copy_failure_recovery(self):
        if sys.platform != "darwin": self.skipTest("I exercise selected Apple native unit transport")
        self.unit_alias_failed_build_retains_published_generation(macro=True, copy_failure=True)

    def test_integrated_native_unit_failure_recovery(self):
        if not shutil.which("clang"): self.skipTest("I require integrated Clang")
        for copy_failure, report_failure in ((False, False), (True, False), (False, True)):
            with self.subTest(copy_failure=copy_failure, report_failure=report_failure):
                self.unit_alias_failed_build_retains_published_generation(macro=True, copy_failure=copy_failure,
                                                                         integrated=True, report_failure=report_failure)

    def test_selected_tool_fifo_failure_recovery(self):
        if not shutil.which("clang"): self.skipTest("I require selected Clang reports")
        for integrated in ((True, False) if sys.platform == "darwin" else (True,)):
            with self.subTest(integrated=integrated):
                self.unit_alias_failed_build_retains_published_generation(
                    macro=True, integrated=integrated, report_failure=True, tool_fifo=True)

    def unit_alias_failed_build_retains_published_generation(self, macro=False, copy_failure=False, integrated=False, report_failure=False, tool_fifo=False):
        compiler = shutil.which("clang" if integrated else "cc")
        for shared_unit in (False, True):
            for shared_cache in (False, True):
                with self.subTest(shared_unit=shared_unit, shared_cache=shared_cache), tempfile.TemporaryDirectory(prefix="nano-unit-failure-") as tmp:
                    root = Path(tmp)
                    module, _, env = self.support.support.foreign_build_fixture(root)
                    env["NANO_TRACE_BUILD"] = "1"
                    if shared_cache: env["NANO_BUILD_CACHE"] = str(root / "cache")
                    symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                    source = module / "payload.s"
                    assembly = f'.data\n.globl {symbol}\n{symbol}:\n.byte 42\n'
                    if macro:
                        assembly = '.macro emit number\n.byte \\number\n.endm\n' + assembly.replace('.byte 42', 'emit 42')
                    source.write_text(assembly)
                    (module / "answer.c").write_text('extern unsigned char snapshot_payload[];\n'
                        'long long nano_build_answer(void) { return snapshot_payload[0]; }\n')
                    metadata = {"name": "answer_native", "c_sources": ["answer.c"],
                                "cflags": ["-g"] + (["-fno-integrated-as"] if self.clang and not integrated else [])}
                    if shared_unit: metadata["shared_c_sources"] = [str(source)]
                    else: metadata["c_sources"].append(source.name)
                    (module / "module.json").write_text(json.dumps(metadata))
                    wrapper = root / "cc-wrapper"
                    copy_marker = root / "copy-failed"
                    native_name = f"__native_unit_{1 if shared_unit else 0}_{0 if shared_unit else 1}.o"
                    failure_report = "I returned an unsupported assembler report"
                    if tool_fifo:
                        fifo = root / "selected-tool"
                        os.mkfifo(fifo)
                        banner = "Apple clang version 21.0.0 (fixture)" if sys.platform == "darwin" else "Debian clang version 14.0.6"
                        failure_report = banner + '\n "' + str(fifo) + '" "-cc1as"\n'
                    wrapper.write_text(f'#!{sys.executable}\nimport os,sys,pathlib\n'
                        f'if {report_failure!r} and os.getenv("NANO_TEST_UNIT_FAIL") and "-###" in sys.argv:\n'
                        f'    print({failure_report!r}, file=sys.stderr)\n    sys.exit(0)\n'
                        f'if {copy_failure!r} and os.getenv("NANO_TEST_UNIT_FAIL") and "-c" in sys.argv and '
                        '"-###" not in sys.argv and os.getenv("NANO_AS_CAPTURE_PHASE") != "capture":\n'
                        '    for arg in sys.argv[1:]:\n'
                        '        unit = pathlib.Path(arg)\n'
                        f'        native = unit.parent / {native_name!r}\n'
                        '        if unit.name == "__snapshot_0_0.s" and native.is_file():\n'
                        f'            native.unlink()\n            pathlib.Path({str(copy_marker)!r}).touch()\n'
                        f'if {not copy_failure!r} and os.environ.get("NANO_TEST_UNIT_FAIL") and "-c" in sys.argv and '
                        'any("/__unit_" in arg and arg.endswith("/payload.s") for arg in sys.argv):\n'
                        '    print("I failed unit assembly", file=sys.stderr)\n    sys.exit(1)\n'
                        f'os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])\n')
                    wrapper.chmod(0o700)
                    env["NANO_CC"] = str(wrapper)
                    self.support.probe_path("build", module, env, timeout=30)
                    first = self.support.probe_path("directory", module, env)
                    if macro:
                        self.assertTrue((first / f"__native_unit_{1 if shared_unit else 0}_{0 if shared_unit else 1}.o").is_file())
                    library = self.support.probe_path("library", module, env)
                    saved = library.read_bytes()
                    source.write_text(assembly.replace("42", "43"))
                    env["NANO_TEST_UNIT_FAIL"] = "1"
                    result = subprocess.run([str(self.support.probe), "build", str(module)], env=env,
                                            capture_output=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    if report_failure: self.assertIn(b"I could not retain", result.stderr)
                    elif copy_failure: self.assertTrue(copy_marker.is_file())
                    else: self.assertIn(b"I failed unit assembly", result.stderr)
                    if tool_fifo: self.assertIn(b"phase=tool-hash-input", result.stderr)
                    self.assertEqual(self.support.probe_path("directory", module, env), first)
                    self.assertEqual(library.read_bytes(), saved)
                    self.assertEqual(self.answer(library), 42)
                    self.assertEqual(list(first.parent.glob(".nano-build-*")), [])
                    env.pop("NANO_TEST_UNIT_FAIL")
                    self.support.probe_path("build", module, env, timeout=30)
                    second = self.support.probe_path("directory", module, env)
                    self.assertNotEqual(first, second)
                    self.assertTrue((second / "source_hashes.json").is_file(),
                                    self.support.last_build_diagnostics)
                    self.assertEqual(list(second.glob("__unit_*")), [])
                    unit_object = second / ("__shared_0.o" if shared_unit else "answer_native_1.o")
                    self.assertNotIn(b".nano-build-", unit_object.read_bytes())
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                    self.support.probe_path("build", module, env, timeout=30)
                    self.assertEqual(self.support.probe_path("directory", module, env), second,
                                     self.support.last_build_diagnostics)

    def test_standalone_assembler_debug_flag_phases(self):
        words = ["-g3", "-D", "VALUE=-g0", "-O2", "-g0", "-g", "-g1", "-g2",
                 "-Xassembler", "-I", "-Xassembler", "-g3"]
        debug = ["-g3", "-g0", "-g", "-g1", "-g2"]
        for phases, expected in ((16, debug), (4, words[-4:]), (20, debug + words[-4:]),
                                 (2, ["-g3", "-O2", "-g0", "-g", "-g1", "-g2"]), (7, words)):
            with self.subTest(phases=phases):
                result = subprocess.run([str(self.support.probe), "phase-flags", str(phases), shlex.join(words)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(shlex.split(result.stdout.decode()), expected)

    def test_standalone_assembler_debug_sections_and_precedence(self):
        from tests.characterize_assembler_debug import measure as measure_debug
        for options in (("-g",), ("-g", "-g0"), ("-g0", "-g")):
            for case in measure_debug(shutil.which("cc"), debug_options=options)["cases"]:
                with self.subTest(options=options, suffix=case["suffix"], cache=case["cache"]):
                    self.assertEqual(case["production"]["sections"], case["native"]["sections"])
                    self.assertEqual(case["production"]["compile_units"], case["native"]["compile_units"])
                    self.assertEqual(case["answer"], 42)
                    self.assertTrue(case["generation_reused"], case)
                    # I leave strict source provenance in the characterization:
                    # Expanded-source locations remain a separate requirement.

    def test_standalone_assembler_debug_macro_reads(self):
        from tests.characterize_assembler_debug import measure as measure_debug
        for case in measure_debug(shutil.which("cc"), macro_read=True)["cases"]:
            with self.subTest(suffix=case["suffix"], cache=case["cache"]):
                self.assertEqual(case["production"]["sections"], case["native"]["sections"])
                self.assertEqual(case["production"]["compile_units"], case["native"]["compile_units"])
                self.assertEqual(case["answer"], 42)
                self.assertTrue(case["generation_reused"], case)

    def test_apple_native_unit_capture_debug_identity(self):
        if sys.platform != "darwin": self.skipTest("I exercise the selected Apple external assembler capture")
        from tests.characterize_assembler_debug import measure as measure_debug
        for nested in (False, True):
            for case in measure_debug(shutil.which("cc"), macro_read=True, nested_read=nested, module_alias=True)["cases"]:
                with self.subTest(nested=nested, suffix=case["suffix"], cache=case["cache"]):
                    self.assertEqual(case["retained_native_object"], case["physical_native"])
                    self.assertTrue(case["retained_native_object_identical"])
                    self.assertEqual(case["production"], case["physical_native"])
                    self.assertTrue(case["physical_object_identical"])
                    self.assertEqual(case["answer"], 42)
                    self.assertTrue(case["generation_reused"], case)

    def test_integrated_native_unit_debug_identity(self):
        compiler = shutil.which("clang")
        if not compiler: self.skipTest("I require integrated Clang")
        from tests.characterize_assembler_debug import measure as measure_debug
        for macro, nested in ((False, False), (True, False), (True, True)):
            for case in measure_debug(compiler, macro_read=macro, nested_read=nested,
                                      module_alias=True, integrated=True)["cases"]:
                with self.subTest(macro=macro, nested=nested, suffix=case["suffix"], cache=case["cache"]):
                    self.assertTrue(case["retained_native_object_identical"])
                    self.assertEqual(case["production"], case["physical_native"])
                    self.assertTrue(case["physical_object_identical"])
                    self.assertEqual(case["answer"], 42)
                    self.assertTrue(case["generation_reused"], case)

    def test_integrated_units_preserve_c_assembler_search(self):
        compiler = shutil.which("clang")
        if not compiler: self.skipTest("I require integrated Clang")
        for shared_cache in (False, True):
            with self.subTest(shared_cache=shared_cache), tempfile.TemporaryDirectory(prefix="nano-integrated-sibling-") as tmp:
                root = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(root)
                env["NANO_CC"] = compiler
                if shared_cache: env["NANO_BUILD_CACHE"] = str(root / "cache")
                includes = root / "assembler search"
                includes.mkdir()
                (includes / "sibling.s").write_text('.byte 1\n')
                prefix = "_" if sys.platform == "darwin" else ""
                inline = f'.data\n.globl {prefix}sibling_payload\n{prefix}sibling_payload:\n.include "sibling.s"\n.text\n'
                csource = module / "answer.c"
                csource.write_text('extern unsigned char snapshot_payload[], sibling_payload[];\n'
                    '__asm__(' + json.dumps(inline) + ');\n'
                    'long long nano_build_answer(void) { return snapshot_payload[0] + sibling_payload[0]; }\n')
                unit = module / "payload.s"
                unit.write_text(f'.data\n.globl {prefix}snapshot_payload\n{prefix}snapshot_payload:\n.byte 42\n')
                flags = ["-g", "-Wa,-I," + str(includes)]
                (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": [csource.name, unit.name],
                    "cflags": [shlex.quote(flag) for flag in flags]}))
                control = root / ("control.dylib" if sys.platform == "darwin" else "control.so")
                result = subprocess.run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-fPIC", *flags, str(csource), str(unit), "-o", str(control)], capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.answer(control), 43)
                self.support.probe_path("build", module, env, timeout=30)
                first = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                self.support.probe_path("build", module, env, timeout=30)
                self.assertEqual(self.support.probe_path("directory", module, env), first)

    def test_assembler_requested_location_evidence(self):
        from tests.characterize_assembler_debug import requested_location
        for address in ("0", "0x0", "0x42"):
            row = f"logical source.c   137   {address}   x\n"
            self.assertTrue(requested_location(row, "linux"))
            self.assertFalse(requested_location(row.replace("137", "138"), "linux"))
            self.assertFalse(requested_location(row.replace("logical", "unrelated"), "linux"))
        self.assertFalse(requested_location("logical source.c 137 0garbage x\n", "linux"))
        self.assertTrue(requested_location("0x00000000 137 5 1 0 is_stmt\n", "darwin"))
        self.assertFalse(requested_location("0x00000000 138 5 1 0 is_stmt\n", "darwin"))
        self.assertFalse(requested_location("0x00000000 137 6 1 0 is_stmt\n", "darwin"))

    def test_equals_path_native_debug_identity(self):
        from tests.characterize_assembler_debug import measure as measure_debug
        compilers = [(shutil.which("cc"), False, ("-g",))]
        if shutil.which("clang"):
            compilers.append((shutil.which("clang"), True, ("-g",)))
            if sys.platform == "linux":
                # This selected driver does not give GNU as automatic -g data.
                compilers.append((shutil.which("clang"), False, ("-g0",)))
        for compiler, integrated, options in compilers:
            for layout in ("source", "cache", "all"):
                for macro, explicit in ((False, False), (True, False), (True, True)):
                    for case in measure_debug(compiler, integrated=integrated, debug_options=options,
                                              equals_paths=layout, macro_read=macro, nested_read=macro,
                                              instruction_macro=macro, explicit_locations=explicit,
                                              module_alias=True)["cases"]:
                        with self.subTest(compiler=compiler, integrated=integrated, layout=layout,
                                          macro=macro, explicit=explicit, suffix=case["suffix"], cache=case["cache"]):
                            self.assertTrue(case["physical_object_identical"], case["physical_debug_diff"])
                            self.assertEqual(case["production"], case["physical_native"])
                            self.assertEqual(case["published_unit_aliases"], [])
                            self.assertEqual(case["answer"], 42)
                            self.assertTrue(case["generation_reused"], case["warm_debug_diff"])
                            if explicit:
                                self.assertTrue(case["native_requested_location"])
                                self.assertTrue(case["production_requested_location"])

    def test_compiler_path_equals_is_not_an_assignment(self):
        with tempfile.TemporaryDirectory(prefix="nano-compiler-path-") as tmp:
            root = Path(tmp)
            wrapper = root / "cc=wrapper"
            wrapper.write_text("#!/bin/sh\nexit 99\n")
            wrapper.chmod(0o700)
            for spelling, accepted in ((str(wrapper), True), ("./cc=wrapper", True),
                                       ("cc=wrapper", False), ("NAME=value/cc", False),
                                       ("CC=cc ./cc=wrapper", False), (str(wrapper) + ";exit 0", False)):
                result = subprocess.run([str(self.support.probe), "compiler-path", spelling],
                                        cwd=root, capture_output=True, timeout=10)
                self.assertEqual(result.returncode == 0, accepted, result.stderr)
                if accepted: self.assertEqual(result.stdout.decode().strip(), str(wrapper.resolve()))

    def test_capture_environment_is_child_local(self):
        command = shlex.join([sys.executable, "-c", "import os; print(os.environ['NANO_AS_CAPTURE_PHASE']); print(os.environ['NANO_TEST_UNRELATED'])"])
        for inherited in (None, "replay"):
            env = os.environ.copy()
            env["NANO_TEST_UNRELATED"] = "preserved"
            env.pop("NANO_AS_CAPTURE_PHASE", None)
            if inherited is not None: env["NANO_AS_CAPTURE_PHASE"] = inherited
            result = subprocess.run([str(self.support.probe), "capture-environment", command],
                                    env=env, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, b"capture\npreserved\n")

    def test_assembler_descriptor_directory_boundaries(self):
        if sys.platform != "linux": self.skipTest("I use procfs names only for GNU descriptor transport")
        for kind in ("directory", "missing", "file", "fifo", "symlink", "failure", "overflow", "timeout"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-descriptor-") as tmp:
                parent = Path(tmp) / "alias=directory"
                if kind in ("directory", "failure", "overflow", "timeout"):
                    parent.mkdir()
                    (parent / "input").write_text("retained")
                elif kind == "file": parent.write_text("not a directory")
                elif kind == "fifo": os.mkfifo(parent)
                elif kind == "symlink": parent.symlink_to(Path(tmp), target_is_directory=True)
                code = ("import os,subprocess,sys; assert open(os.environ['NANO_TEST_READ_DIRECTORY']+'/input').read() == 'retained'; "
                        "assert not sys.stdin.read(); assert os.getcwd() == " + repr(str(cache.ROOT)) + "; "
                        "subprocess.run([sys.executable,'-c',\"import os; assert open(os.environ['NANO_TEST_READ_DIRECTORY']+'/input').read() == 'retained'\"],check=True,close_fds=True)")
                if kind == "failure": code = "raise SystemExit(1)"
                elif kind == "overflow": code = "print('x'*20000)"
                elif kind == "timeout": code = "import time; time.sleep(20)"
                result = subprocess.run([str(self.support.probe), "read-execute", str(parent),
                                         shlex.join([sys.executable, "-c", code])],
                                        cwd=cache.ROOT, env=dict(os.environ, NANO_CAPTURE_TIMEOUT_MS="1000"),
                                        capture_output=True, timeout=8)
                self.assertEqual(result.returncode == 0, kind == "directory", result.stderr)

    def test_linux_read_execute_default_accepts_slow_command(self):
        if sys.platform != "linux": self.skipTest("I exercise Linux captured-read execution")
        with tempfile.TemporaryDirectory(prefix="nano-slow-read-") as tmp:
            env = dict(os.environ)
            env.pop("NANO_CAPTURE_TIMEOUT_MS", None)
            code = "import time; time.sleep(6); print('retained')"
            result = subprocess.run([str(self.support.probe), "read-execute", tmp,
                                     shlex.join([sys.executable, "-c", code])],
                                    env=env, capture_output=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, b"retained\n")

    def test_assembler_instruction_and_location_provenance(self):
        self.assembler_instruction_and_location_provenance(shutil.which("cc"), integrated=False)

    def test_integrated_instruction_and_location_provenance(self):
        compiler = shutil.which("clang")
        if not compiler: self.skipTest("I require integrated Clang")
        self.assembler_instruction_and_location_provenance(compiler, integrated=True)

    def assembler_instruction_and_location_provenance(self, compiler, integrated):
        from tests.characterize_assembler_debug import measure as measure_debug
        for explicit in (False, True):
            for stem in ("payload", "code 'naïve-λ'", 'code "naïve-λ"'):
                for case in measure_debug(compiler, integrated=integrated, macro_read=True, nested_read=True,
                                          instruction_macro=True, explicit_locations=explicit, source_stem=stem,
                                          module_alias=True)["cases"]:
                    with self.subTest(integrated=integrated, explicit=explicit, stem=stem,
                                      suffix=case["suffix"], cache=case["cache"]):
                        self.assertTrue(case["physical_object_identical"], case["physical_debug_diff"])
                        self.assertEqual(case["production"], case["physical_native"])
                        self.assertEqual(case["answer"], 42)
                        self.assertTrue(case["generation_reused"], case["warm_debug_diff"])
                        if explicit:
                            self.assertTrue(case["native_requested_location"])
                            self.assertTrue(case["production_requested_location"])

    def test_assembler_include_flag_phases(self):
        assembler = ["-Wa,-I,first path,-Isecond", "-Xassembler", "-I", "-Xassembler", "third path,comma",
                     "-Xassembler", "-Ifourth", "-Wa,--alternate", "-Xassembler", "--alternate",
                     "-Wa,--alternate,-I,fifth,-Isixth,--alternate"]
        cflags = ["-O2", "-D", "VALUE=-Xassembler", "-I", "C headers"]
        expected = {1: cflags, 2: ["-O2"], 4: assembler, 7: cflags + assembler}
        for phases, words in expected.items():
            with self.subTest(phases=phases):
                result = subprocess.run([str(self.support.probe), "phase-flags", str(phases), shlex.join(cflags + assembler)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(shlex.split(result.stdout.decode()), words)
        for fragment in ("-Wa,", "-Wa,-I", "-Wa,-I,", "-Wa,-I,a,", "-Wa,-I,a,--MD,out.d",
                         "-Wa,--alternate,", "-Wa,--alternate,--MD,out.d", "-Wa,--alternateX",
                         "-Xassembler", "-Xassembler -I", "-Xassembler -I path",
                         "-Xassembler -I -Xassembler ''", "-Xassembler -o -Xassembler out.o"):
            with self.subTest(fragment=fragment):
                result = subprocess.run([str(self.support.probe), "phase-flags", "7", fragment], capture_output=True, timeout=10)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(result.stdout, b"")

    def test_alternate_assembler_restored_inputs(self):
        if not self.gnu_read_replay: self.skipTest("I need supported GNU assembler read replay")
        kinds = ("assembler-external-alternate",) if self.clang else ("assembler-alternate",)
        for split in (False, True):
            for removed in (False, True):
                result = measure(shutil.which("cc"), kinds, split_search=split, remove_input=removed)
                require_consistent(result)
                for case in result["cases"]:
                    with self.subTest(split=split, removed=removed, case=case):
                        self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                        for key in ("bytes_restored", "mtime_preserved", "size_preserved",
                                    "generation_reused", "reuse_record", "retained_read_manifest"):
                            self.assertTrue(case[key], key)

    def test_assembler_search_restored_inputs(self):
        kinds = ("assembler-search", "assembler-external-search") if self.clang else ("assembler-search",)
        for remove_input in (False, True):
            result = measure(shutil.which("cc"), kinds, remove_input=remove_input)
            require_consistent(result)
            for case in result["cases"]:
                with self.subTest(remove_input=remove_input, case=case):
                    self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                    self.assertTrue(case["generation_reused"])
                    self.assertTrue(case["reuse_record"])
                    self.assertTrue(case["retained_assembly"])
                    if case["input"] == "assembler-external-search":
                        self.assertEqual(case["external_assembly_compilations"], case["total_object_compilations"])

    def test_paired_fragment_normalization(self):
        for parts in (["-D", "VALUE=42", "-U", "OLD", "-I", "'C headers'"],
                      ["-Xassembler", "-I", "-Xassembler", "'a path,comma'"],
                      ["-Xassembler -I", "-Xassembler 'a path'"],
                      ["-Xlinker", "-rpath", "-Xlinker", "'/a path'"],
                      ["-O2", "-g"], ["-Xassembler", "-I", "'$SEARCH'"],
                      ["-Xassembler", "-I", "$SEARCH"]):
            with self.subTest(parts=parts):
                result = subprocess.run([str(self.support.probe), "coalesce-flags", *parts],
                    capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                captured = json.loads(result.stdout)
                self.assertEqual(shlex.split(" ".join(captured)), shlex.split(" ".join(parts)))
                if parts == ["-O2", "-g"] or "$SEARCH" in parts:
                    self.assertEqual(captured, parts)
                else:
                    self.assertEqual(captured[0], " ".join(parts))
                    self.assertTrue(all(part == "" for part in captured[1:]))
        result = subprocess.run([str(self.support.probe), "coalesce-allocation", "paired"],
            capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_split_assembler_search_restored_inputs(self):
        kinds = ("assembler-search", "assembler-external-search") if self.clang else ("assembler-search",)
        for removed in (False, True):
            observed = measure(shutil.which("cc"), kinds, split_search=True, remove_input=removed)
            require_consistent(observed)
            for case in observed["cases"]:
                with self.subTest(removed=removed, case=case):
                    for key in ("bytes_restored", "mtime_preserved", "size_preserved", "reuse_record", "generation_reused", "retained_assembly"):
                        self.assertTrue(case[key], key)

    def test_forwarded_include_operands_are_not_rebased(self):
        with tempfile.TemporaryDirectory(prefix="nano-operand-owner-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            include = directory / "assembler_fallback"
            include.mkdir()
            working = directory / "work"
            working.mkdir()
            for flags, expected in ((["-Iassembler_fallback"], ["-I" + str(include.resolve())]),
                                    (["-O${NANO_TEST_LEVEL:-2}", "-Iassembler_fallback"],
                                     ["-O${NANO_TEST_LEVEL:-2}", "-I" + str(include.resolve())]),
                                    (["-Xassembler", "-Iassembler_fallback"], ["-Xassembler", "-Iassembler_fallback"]),
                                    (["-Xlinker", "-Iassembler_fallback"], ["-Xlinker", "-Iassembler_fallback"]),
                                    (["-D", "-Iassembler_fallback"], ["-D", "-Iassembler_fallback"]),
                                    (["-I", shlex.quote(str(include))], ["-I", str(include)])):
                with self.subTest(flags=flags):
                    manifest = json.dumps({"name": "answer_native", "cflags": flags})
                    (module / "module.json").write_text(manifest)
                    result = subprocess.run([str(self.support.probe), "build-info", str(module)],
                        cwd=working, env=env, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    captured = [line[len("compile:"):] for line in result.stdout.decode().splitlines()
                                if line.startswith("compile:")]
                    self.assertEqual(shlex.split(" ".join(captured)), expected)
                    self.assertEqual((module / "module.json").read_text(), manifest)

    def test_assembler_search_order_phases_and_recovery(self):
        self.assembler_search_order_phases_and_recovery(("wa-paired", "wa-joined", "xassembler"))

    def test_split_assembler_search_order_phases_and_recovery(self):
        self.assembler_search_order_phases_and_recovery(("xassembler-split",))

    def test_alternate_assembler_search_phases_and_recovery(self):
        if not self.gnu_read_replay: self.skipTest("I need supported GNU assembler read replay")
        self.assembler_search_order_phases_and_recovery(("wa-paired", "xassembler-split"), alternate=True)

    def test_standalone_raw_assembler_search_recovery(self):
        self.assembler_search_order_phases_and_recovery(
            ("wa-paired", "wa-joined", "xassembler-split"), suffix=".s")

    def test_standalone_preprocessed_assembler_search_recovery(self):
        self.assembler_search_order_phases_and_recovery(
            ("wa-paired", "wa-joined", "xassembler-split"), suffix=".S")

    def test_shared_raw_assembler_search_recovery(self):
        self.assembler_search_order_phases_and_recovery(
            ("wa-paired", "wa-joined", "xassembler-split"), suffix=".s", shared_unit=True)

    def test_shared_preprocessed_assembler_search_recovery(self):
        self.assembler_search_order_phases_and_recovery(
            ("wa-paired", "wa-joined", "xassembler-split"), suffix=".S", shared_unit=True)

    def assembler_search_order_phases_and_recovery(self, styles, alternate=False,
                                                  suffix=None, shared_unit=False, cases=None):
        modes = (False, True) if self.clang else (False,)
        if alternate: modes = (True,) if self.clang else (False,)
        if cases is None: cases = ((style, placement, external, shared)
                for style in styles
                for placement in ("common", "platform", "package") for external in modes for shared in (False, True))
        for style, placement, external, shared in cases:
            with self.subTest(style=style, placement=placement, external=external, shared=shared), tempfile.TemporaryDirectory(prefix="nano-as-search-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                env["NANO_TRACE_BUILD"] = "1"
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                env["NANO_AS_CAPTURE_HELPER"] = str(cache.ROOT / "bin/nano_as_capture.so")
                early, late, c_headers = (module / name for name in ("early includes", "late includes", "C headers"))
                for folder in (early, late, c_headers): folder.mkdir()
                def selected(value):
                    if alternate:
                        return '.macro emit value\n.ascii "\\value"\n.endm\nemit <' + str(value) + '>\n'
                    return f'.ascii "{value}"\n'
                (late / "selected.s").write_text(selected(42))
                (late / "selection.h").write_text('#define ADJUST 100\n')
                (c_headers / "selection.h").write_text('#define ADJUST 0\n')
                asm_flags = [f"-Wa,-I,{early},-I,{late}"] if style == "wa-paired" else (
                    [f"-Wa,-I{early},-I{late}"] if style == "wa-joined" else
                    ["-Xassembler", "-I", "-Xassembler", str(early), "-Xassembler", "-I" + str(late)])
                if alternate:
                    if style.startswith("wa-"): asm_flags[0] = asm_flags[0].replace("-Wa,", "-Wa,--alternate,", 1)
                    else: asm_flags += ["-Xassembler", "--alternate"]
                cflags = ["-std=c11", "-Werror", "-I", str(c_headers)] + (["-fno-integrated-as"] if external else [])
                split = style == "xassembler-split"
                fragments = lambda words: [shlex.quote(word) for word in words] if split else [shlex.join(words)]
                metadata = {"name": "answer_native", "c_sources": ["answer.c"], "cflags": fragments(cflags)}
                if placement == "common": metadata["cflags"].extend(fragments(asm_flags))
                elif placement == "platform": metadata["cflags_macos" if sys.platform == "darwin" else "cflags_linux"] = fragments(asm_flags)
                else:
                    package_flags = fragments(asm_flags)
                    metadata["pkg_config"] = [f"assembler-search-fixture-{i}" for i in range(len(package_flags))]
                    pkg = directory / "pkg-config"
                    mapping = dict(zip(metadata["pkg_config"], package_flags))
                    pkg.write_text(f'#!{sys.executable}\nimport sys\nif "--cflags" in sys.argv: print({mapping!r}[sys.argv[-1]])\n')
                    pkg.chmod(0o700)
                    env["PKG_CONFIG"] = str(pkg)
                # I also exercise the guarded linker query with assembler operands.
                if style.startswith("xassembler") and placement == "common":
                    response = module / "link.rsp"
                    response.write_text("-lm\n")
                    metadata["ldflags"] = ["-Wl,@" + str(response)]
                    if split: metadata["cflags"].extend(["-Xlinker", "-lm"])
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                assembly = f'.data\n.globl {symbol}\n{symbol}:\n.include "selected.s"\n.text\n'
                source = module / "answer.c"
                source.write_text('#include "selection.h"\nextern const unsigned char snapshot_payload[];\n'
                    + ('' if suffix else '__asm__(' + json.dumps(assembly) + ');\n') +
                    'long long nano_build_answer(void) { return ADJUST + (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                sources = [str(source)]
                if suffix:
                    unit = module / ("payload" + suffix)
                    unit.write_text(assembly)
                    metadata.setdefault("shared_c_sources" if shared_unit else "c_sources", []).append(unit.name)
                    sources.append(str(unit))
                (module / "module.json").write_text(json.dumps(metadata))
                direct = directory / ("direct.dylib" if sys.platform == "darwin" else "direct.so")
                native_flags = asm_flags + cflags if placement == "package" else cflags + asm_flags
                result = subprocess.run([shutil.which("cc"), "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-fPIC", *native_flags, *sources, "-o", str(direct)], capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                baseline = self.answer(direct)
                self.assertEqual(baseline, 142 if self.clang and not external else 42)
                calls, wrapper = directory / "calls", directory / "cc"
                wrapper.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                    f'with open({str(calls)!r}, "a") as log: log.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                    f'os.execv({shutil.which("cc")!r}, [{shutil.which("cc")!r}] + sys.argv[1:])\n')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)
                def build(answer):
                    # I allow multiple 30-second unit scopes plus build work;
                    # this parent guard is not a production whole-build bound.
                    self.support.probe_path("build", module, env, timeout=240)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertTrue((generation / "source_hashes.json").exists(), self.support.last_build_diagnostics)
                    self.assertFalse(list(generation.parent.glob(".nano-link-source-*")))
                    self.assertFalse(list(generation.parent.glob(".nano-link-query-*")))
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), answer)
                    return generation
                first = build(baseline)
                self.assertEqual(build(baseline), first, self.support.last_build_diagnostics)
                commands = [json.loads(line) for line in calls.read_text().splitlines()]
                for argv in commands:
                    source_phase = ("-E" in argv and "-Xclang" not in argv) or ("-S" in argv and not (self.clang and not external))
                    if source_phase:
                        for flag in asm_flags:
                            if flag != "-I": self.assertNotIn(flag, argv)
                    if ("-dynamiclib" in argv or "-shared" in argv) and "-c" not in argv:
                        for flag in asm_flags:
                            if flag != "-I": self.assertNotIn(flag, argv)
                        self.assertIn("-Werror", argv)
                    if "assembler" in argv and "-c" in argv:
                        self.assertEqual([arg for arg in argv if arg in asm_flags], asm_flags)
                        self.assertNotIn("-std=c11", argv)
                (early / "selected.s").write_text(selected(43))
                changed = build(baseline + 1)
                self.assertNotEqual(first, changed)
                saved = self.support.snapshot(changed)
                (early / "selected.s").unlink()
                (late / "selected.s").unlink()
                failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0, failed.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), changed)
                self.assertEqual(self.support.snapshot(changed), saved)
                self.assertFalse(list(changed.parent.glob(".nano-build-*")))
                (late / "selected.s").write_text(selected(44))
                recovered = build(baseline + 2)
                self.assertEqual(build(baseline + 2), recovered, self.support.last_build_diagnostics)

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

    def test_gcc_incomplete_assembler_capture_refuses_cold_output(self):
        if self.clang: self.skipTest("I exercise GCC assembler capture failure here")
        for case in measure(shutil.which("cc"), ("assembler-fallback",))["cases"]:
            with self.subTest(case=case):
                self.assert_capture_refused(case)

    def assert_capture_refused(self, case):
        self.assertTrue(case.get("build_failed"), case)
        self.assertIn("I could not retain", case["diagnostic"])
        for key in ("total_object_compilations", "published_generations", "leaked_stages"):
            self.assertEqual(case[key], 0, key)
        self.assertFalse(case["current_exists"])
        self.assertFalse(case["mutation_started"])
        for key in ("bytes_restored", "size_preserved", "mtime_preserved"):
            self.assertTrue(case[key], key)
        with self.assertRaises(SystemExit): require_consistent({"cases": [case]})

    def test_ordinary_capture_failure_refuses_live_cold_output(self):
        for case in measure(shutil.which("cc"), ("capture-failure",))["cases"]:
            with self.subTest(case=case):
                self.assert_capture_refused(case)
                self.assertIn("I failed the requested capture phase", case["diagnostic"])

    def test_admitted_capture_phase_failure_and_recovery(self):
        phases = ("-S",) if self.clang else ("-E", "-S")
        for shared, phase in ((shared, phase) for shared in (False, True) for phase in phases):
            with self.subTest(shared=shared, phase=phase), tempfile.TemporaryDirectory(prefix="nano-capture-phase-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                scratch = directory / "temporary"
                scratch.mkdir()
                env["TMPDIR"] = str(scratch)
                wrapper, calls = directory / "cc", directory / "object-calls"
                wrapper.write_text(f'''#!{sys.executable}
import os, sys
if os.getenv("NANO_TEST_CAPTURE_FAIL") in sys.argv:
    print("I failed the requested capture phase", file=sys.stderr)
    sys.exit(29)
if "-c" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("C\\n")
os.execv({shutil.which("cc")!r}, [{shutil.which("cc")!r}] + sys.argv[1:])
''')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)
                env["NANO_TEST_CAPTURE_FAIL"] = phase
                root = self.support.probe_path("root", module, env)

                def refused():
                    result = subprocess.run([str(self.support.probe), "build", str(module)], env=env,
                                            capture_output=True, timeout=20)
                    self.assertNotEqual(result.returncode, 0, result.stderr)
                    self.assertIn(b"I failed the requested capture phase", result.stderr)
                    self.assertIn(b"I could not retain", result.stderr)
                    self.assertFalse(list(root.glob(".nano-build-*")))
                    self.assertEqual(list(scratch.iterdir()), [])

                refused()
                self.assertFalse(os.path.lexists(root / "current"))
                self.assertFalse(list(root.glob(".nano-gen-*")))
                self.assertFalse(calls.exists())
                del env["NANO_TEST_CAPTURE_FAIL"]
                self.support.probe_path("build", module, env)
                previous = self.support.probe_path("directory", module, env)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                self.assertTrue((previous / "source_hashes.json").exists())
                self.support.probe_path("build", module, env)
                self.assertEqual(self.support.probe_path("directory", module, env), previous)
                saved, before = self.support.snapshot(previous), calls.read_bytes()
                (module / "answer.c").write_text("long long nano_build_answer(void) { return 43; }\n")
                env["NANO_TEST_CAPTURE_FAIL"] = phase
                refused()
                self.assertEqual(self.support.probe_path("directory", module, env), previous)
                self.assertEqual(self.support.snapshot(previous), saved)
                self.assertEqual(calls.read_bytes(), before)
                del env["NANO_TEST_CAPTURE_FAIL"]
                self.support.probe_path("build", module, env)
                recovered = self.support.probe_path("directory", module, env)
                self.assertNotEqual(recovered, previous)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                self.assertTrue((recovered / "source_hashes.json").exists())
                self.support.probe_path("build", module, env)
                self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_gcc_macro_argument_replays_captured_reads(self):
        if not self.read_replay: self.skipTest("I need a supported Linux GNU assembler replay version")
        for case in measure(shutil.which("cc"), ("assembler-macro",))["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "mtime_preserved", "size_preserved", "reuse_record",
                            "generation_reused", "retained_read_manifest"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 6)

    def test_equals_path_restored_assembler_inputs(self):
        if not self.read_replay: self.skipTest("I require GNU captured-read replay")
        for shared_unit in (False, True):
            for removed in (False, True):
                result = measure(shutil.which("cc"), ("assembler-unit", "assembler-preprocessed-unit"),
                                 shared_unit=shared_unit, remove_input=removed, equals_paths=True)
                require_consistent(result)
                for case in result["cases"]:
                    self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                    self.assertTrue(case["retained_read_manifest"])
                    self.assertTrue(case["generation_reused"])

    def test_gcc_missing_capture_helper_preserves_generation_and_recovers(self):
        if not self.read_replay: self.skipTest("I need supported GNU assembler read capture")
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory(prefix="nano-helper-recovery-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                helper = directory / "capture.so"
                shutil.copy2(cache.ROOT / "bin/nano_as_capture.so", helper)
                env["NANO_AS_CAPTURE_HELPER"] = str(helper)
                payload = module / "payload.bin"
                payload.write_bytes(b"42")
                assembly = '.data\n.globl snapshot_payload\nsnapshot_payload:\n.macro read_payload file\n.incbin "\\file"\n.endm\n'
                assembly += f'read_payload "{payload}"\n.text\n'
                (module / "answer.c").write_text('extern const unsigned char snapshot_payload[];\n'
                    '__asm__(' + json.dumps(assembly) + ');\n'
                    'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                self.support.probe_path("build", module, env)
                previous = self.support.probe_path("directory", module, env)
                self.assertTrue((previous / "__as_read_0_0.manifest0").exists())
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
                saved = self.support.snapshot(previous)
                helper.unlink()
                payload.write_bytes(b"43")
                result = subprocess.run([str(self.support.probe), "build", str(module)], env=env,
                                        capture_output=True, timeout=20)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn(b"I could not retain", result.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), previous)
                self.assertEqual(self.support.snapshot(previous), saved)
                self.assertFalse(list(previous.parent.glob(".nano-build-*")))
                shutil.copy2(cache.ROOT / "bin/nano_as_capture.so", helper)
                self.support.probe_path("build", module, env)
                recovered = self.support.probe_path("directory", module, env)
                self.assertNotEqual(recovered, previous)
                self.assertTrue((recovered / "__as_read_0_0.manifest0").exists())
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                self.support.probe_path("build", module, env)
                self.assertEqual(self.support.probe_path("directory", module, env), recovered)

    def test_linux_clang_external_macros_replay_captured_reads(self):
        if not self.clang or sys.platform != "linux":
            self.skipTest("I exercise Clang with the Linux GNU assembler")
        compiler = shutil.which("cc")
        query = subprocess.run([compiler, "-print-prog-name=as"], capture_output=True, timeout=10, check=True)
        assembler = shutil.which(query.stdout.decode().strip())
        if not assembler: self.skipTest("I need the selected external assembler")
        version = subprocess.run([assembler, "--version"], capture_output=True, timeout=10, check=True)
        accepted = subprocess.run([str(self.support.probe), "assembler-version", version.stdout.decode()],
                                  capture_output=True, timeout=10)
        if accepted.returncode: self.skipTest("I need a tested GNU assembler replay version")
        result = measure(compiler, ("assembler-external-macro", "assembler-external-macro-debug"))
        require_consistent(result)
        for case in result["cases"]:
            with self.subTest(case=case):
                self.assertEqual((case["cold_answer"], case["warm_answer"], case["fresh_answer"]), (42, 42, 42))
                for key in ("bytes_restored", "size_preserved", "mtime_preserved", "reuse_record",
                            "generation_reused", "retained_read_manifest"):
                    self.assertTrue(case[key], key)
                self.assertEqual(case["total_object_compilations"], 6)
                self.assertEqual(case["external_assembly_compilations"], 6)

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
            fake_as = directory / "as-wrapper"
            fake_as.write_text('#!/bin/sh\nexec ' + shlex.quote(shutil.which("as")) + ' "$@"\n')
            fake_as.chmod(0o700)
            for tool in ("/missing/assembler", str(fake_as)):
                env["NANO_TEST_AS_QUERY_FAIL"] = tool
                previous = self.support.snapshot(changed)
                failed = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0, failed.stderr)
                self.assertIn(b"I could not retain", failed.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), changed)
                self.assertEqual(self.support.snapshot(changed), previous)
                self.assertEqual(list(scratch.iterdir()), [])
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

    def test_apple_failed_capture_refuses_uncaptured_cold_output(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I characterize failed Apple capture with restored inputs")
        for case in measure(shutil.which("cc"), ("assembler-external-macro-query-failure",))["cases"]:
            with self.subTest(case=case):
                self.assertTrue(case.get("build_failed"), case)
                self.assertIn("I could not retain external-assembler inputs", case["diagnostic"])
                for key in ("total_object_compilations", "published_generations", "leaked_stages"):
                    self.assertEqual(case[key], 0, key)
                self.assertFalse(case["current_exists"])
                self.assertFalse(case["mutation_started"])
                for key in ("bytes_restored", "size_preserved", "mtime_preserved"):
                    self.assertTrue(case[key], key)
                with self.assertRaises(SystemExit): require_consistent({"cases": [case]})

    def test_apple_external_query_failure_and_recovery(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I exercise selected Apple backend query failures")
        failures = ("empty", "multiple", "truncated", "oversize", "error", "timeout")
        for failure, shared in ((failure, shared) for failure in failures for shared in (False, True)):
            with self.subTest(failure=failure, shared=shared), tempfile.TemporaryDirectory(prefix="nano-as-query-failure-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                temporary = directory / "temporary"
                temporary.mkdir()
                env["TMPDIR"] = str(temporary)
                payload = module / "payload.bin"
                payload.write_bytes(b"42")
                assembly = '.data\n.globl _snapshot_payload\n_snapshot_payload:\n.macro payload file\n.incbin "\\file"\n.endm\n' + f'payload "{payload}"\n.text\n'
                (module / "answer.c").write_text('__asm__(' + json.dumps(assembly) + ');\n'
                    'extern const unsigned char snapshot_payload[];\n'
                    'long long nano_build_answer(void) { return (snapshot_payload[0]-48)*10 + snapshot_payload[1]-48; }\n')
                (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"],
                                                               "cflags": ["-fno-integrated-as"]}))
                wrapper = directory / "cc"
                calls = directory / "object-calls"
                banner = 'Apple clang version 21.0.0 (fixture)\n'
                wrapper.write_text(f'''#!{sys.executable}
import os, subprocess, sys, time
if "-c" in sys.argv and "-###" not in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("C\\n")
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
                # I bound injected faults separately from ordinary host latency.
                failed = subprocess.run([str(self.support.probe), "build", str(module)],
                                        env=dict(env, NANO_CAPTURE_TIMEOUT_MS="5000"),
                                        capture_output=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0, failed.stdout)
                self.assertIn(b"I could not retain external-assembler inputs", failed.stderr)
                self.assertLess(time.monotonic() - started, 15)
                root = self.support.probe_path("root", module, env)
                self.assertFalse(os.path.lexists(root / "current"))
                self.assertFalse(list(root.glob(".nano-build-*")))
                self.assertFalse(list(root.glob(".nano-gen-*")))
                self.assertFalse(calls.exists())
                self.assertFalse(list(temporary.glob("nano-gcc-check-*")))
                del env["NANO_QUERY_FAILURE"]
                self.support.probe_path("build", module, env, timeout=20)
                recovered = self.support.probe_path("directory", module, env)
                self.assertTrue((recovered / "source_hashes.json").is_file())
                self.assertTrue((recovered / "__expanded_0_0.s").is_file())
                self.support.probe_path("build", module, env, timeout=20)
                self.assertEqual(self.support.probe_path("directory", module, env), recovered)
                library = self.support.probe_path("library", module, env)
                previous_library = library.read_bytes()
                previous_record = (recovered / "source_hashes.json").read_bytes()
                previous_calls = calls.read_bytes()
                payload.write_bytes(b"43")
                env["NANO_QUERY_FAILURE"] = failure
                failed = subprocess.run([str(self.support.probe), "build", str(module)],
                                        env=dict(env, NANO_CAPTURE_TIMEOUT_MS="5000"),
                                        capture_output=True, timeout=25)
                self.assertNotEqual(failed.returncode, 0, failed.stdout)
                self.assertIn(b"I could not retain external-assembler inputs", failed.stderr)
                self.assertEqual(self.support.probe_path("directory", module, env), recovered)
                self.assertEqual(library.read_bytes(), previous_library)
                self.assertEqual((recovered / "source_hashes.json").read_bytes(), previous_record)
                self.assertEqual(calls.read_bytes(), previous_calls)
                self.assertFalse(list(root.glob(".nano-build-*")))
                self.assertFalse(list(temporary.glob("nano-gcc-check-*")))
                del env["NANO_QUERY_FAILURE"]
                self.support.probe_path("build", module, env, timeout=20)
                replacement = self.support.probe_path("directory", module, env)
                self.assertNotEqual(replacement, recovered)
                self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 43)
                self.support.probe_path("build", module, env, timeout=20)
                self.assertEqual(self.support.probe_path("directory", module, env), replacement)

    def test_external_unadmitted_flags_keep_the_original_path(self):
        if not self.clang: self.skipTest("I exercise Clang's external assembler selector")
        with tempfile.TemporaryDirectory(prefix="nano-external-unadmitted-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"],
                "cflags": ["-fno-integrated-as", "-fno-strict-aliasing"]}))
            wrapper, queried = directory / "cc", directory / "queried"
            wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, sys
if "-###" in sys.argv:
    pathlib.Path({str(queried)!r}).touch()
    sys.exit(1)
os.execv({shutil.which("cc")!r}, [{shutil.which("cc")!r}] + sys.argv[1:])
''')
            wrapper.chmod(0o700)
            env["NANO_CC"] = str(wrapper)
            self.support.probe_path("build", module, env)
            self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)
            generation = self.support.probe_path("directory", module, env)
            self.assertFalse(list(generation.glob("__snapshot_*")))
            self.assertFalse(queried.exists())

    def test_apple_external_capture_preserves_tool_diagnostics(self):
        if not self.clang or sys.platform != "darwin":
            self.skipTest("I preserve selected Apple capture diagnostics")
        for failure in ("c-error", "missing-assembly"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="nano-capture-diagnostic-") as tmp:
                directory = Path(tmp)
                module, _, env = self.support.support.foreign_build_fixture(directory)
                marker = "missing_capture_identifier" if failure == "c-error" else str(module / "missing_capture_include.s")
                contents = 'long long nano_build_answer(void) { return missing_capture_identifier; }\n' if failure == "c-error" else (
                    '__asm__(' + json.dumps(f'.include "{marker}"\n') + ');\nlong long nano_build_answer(void) { return 42; }\n')
                (module / "answer.c").write_text(contents)
                (module / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"],
                                                               "cflags": ["-fno-integrated-as"]}))
                result = subprocess.run([str(self.support.probe), "build", str(module)], env=env, capture_output=True, timeout=20)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(marker.encode(), result.stderr)
                self.assertIn(b"I could not retain external-assembler inputs", result.stderr)
                root = self.support.probe_path("root", module, env)
                self.assertFalse(os.path.lexists(root / "current"))
                self.assertFalse(list(root.glob(".nano-build-*")))

    def test_selected_assembler_tool_hash_bounds(self):
        seed = 14695981039346656037
        def fnv(data, value=seed):
            for byte in data:
                value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
            return value
        with tempfile.TemporaryDirectory(prefix="nano-tool-hash-") as tmp:
            root = Path(tmp)
            tool = root / "tool"
            contents = bytes(range(256)) * 33
            tool.write_bytes(contents)
            alias = root / "alias"
            alias.symlink_to(tool)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            fifo_alias = root / "fifo-alias"
            fifo_alias.symlink_to(fifo)
            def check(path, budget=5000):
                return subprocess.run([str(self.support.probe), "assembler-tool-hash", str(path), str(budget)],
                                      capture_output=True, timeout=2)
            for path in (tool, alias):
                with self.subTest(path=path):
                    result = check(path)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = fnv(str(path).encode() + b'\0')
                    expected = fnv(str(fnv(contents)).encode() + b'\0', expected)
                    self.assertEqual(int(result.stdout), expected)
            for path in (fifo, fifo_alias, root, root / "missing", Path('/dev/zero'), Path('relative-tool')):
                with self.subTest(path=path):
                    result = check(path)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(int(result.stdout), seed)
            self.assertEqual(check(tool, 0).returncode, 1)
            # A sparse regular input exercises expiry inside the read loop;
            # I do not allocate its logical length in memory or on disk.
            with tool.open('wb') as stream:
                stream.truncate(1024 * 1024 * 1024)
            result = subprocess.run([str(self.support.probe), "assembler-tool-hash", str(tool), "10"],
                                    env=dict(os.environ, NANO_TRACE_BUILD="1"), capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(int(result.stdout), seed)
            self.assertIn(b'phase=tool-hash-deadline', result.stderr)

    def test_selected_clang_assembler_report_boundaries(self):
        banner = "Apple clang version 21.0.0 (fixture)\nTarget: arm64-apple-darwin\nThread model: posix\nInstalledDir: /fixture\n"
        command = ' "/fixture/compiler with space" "-cc1as" "-o" "object with \'quotes\'.o"\n'
        accepted = banner + command
        accepted_reports = [accepted, banner + "clang: warning: argument unused during compilation: '-fPIC' [-Wunused-command-line-argument]\n" + command]
        debian = "Debian clang version 14.0.6\nTarget: aarch64-unknown-linux-gnu\nThread model: posix\nInstalledDir: /usr/bin\n (in-process)\n"
        accepted_reports.append(debian + command)
        reports = accepted_reports + ["", banner, command, accepted + command, accepted + "unexpected command\n",
                   debian.replace("14.0.6", "14.0.61") + command, debian.replace("14.0.6", "18.0.0") + command,
                   debian + command + command, debian.replace("(in-process)", "(unknown)") + command,
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
            # My probe lives in obj/, unlike installed drivers beside the helper.
            env["NANO_AS_CAPTURE_HELPER"] = str(cache.ROOT / "bin/nano_as_capture.so")
            for flag in flags:
                with self.subTest(flag=flag):
                    (module / "module.json").write_text(json.dumps({"name": "answer_native",
                        "c_sources": ["answer.c"], "cflags": [flag]}))
                    self.support.probe_path("build", module, env)
                    generation = self.support.probe_path("directory", module, env)
                    self.assertTrue((generation / ("__snapshot_0_0" + self.snapshot_suffix)).is_file())
                    self.assertTrue((generation / "source_hashes.json").is_file())
                    self.assertEqual(self.answer(self.support.probe_path("library", module, env)), 42)

    def test_split_c_preprocessor_flags_capture_and_reuse(self):
        with tempfile.TemporaryDirectory(prefix="nano-paired-c-flags-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            headers = directory / "C headers"
            headers.mkdir()
            header = headers / "config.h"
            header.write_text("#define OFFSET 0\n")
            (module / "answer.c").write_text('#include <config.h>\n#ifdef OLD\n#error OLD must be undefined\n#endif\n'
                'long long nano_build_answer(void) { return NAME + OFFSET; }\n')
            for joined in (False, True):
                with self.subTest(joined=joined):
                    header.write_text("#define OFFSET 0\n")
                    words = ["-D", "NAME=42", "-D", "OLD=1", "-U", "OLD", "-I", str(headers)]
                    flags = [shlex.join(words)] if joined else [shlex.quote(word) for word in words]
                    (module / "module.json").write_text(json.dumps({"name": "answer_native",
                        "c_sources": ["answer.c"], "cflags": flags}))
                    def build(expected):
                        self.support.probe_path("build", module, env)
                        generation = self.support.probe_path("directory", module, env)
                        self.assertTrue((generation / "source_hashes.json").is_file())
                        self.assertTrue(list(generation.glob("__snapshot_*")))
                        self.assertEqual(self.answer(self.support.probe_path("library", module, env)), expected)
                        return generation
                    first = build(42)
                    self.assertEqual(build(42), first)
                    header.write_text("#define OFFSET 1\n")
                    changed = build(43)
                    self.assertNotEqual(changed, first)
                    self.assertEqual(build(43), changed)

    def test_unknown_fragments_keep_original_compilation(self):
        with tempfile.TemporaryDirectory(prefix="nano-retained-unknown-flags-") as tmp:
            directory = Path(tmp)
            module, _, env = self.support.support.foreign_build_fixture(directory)
            for flags in (["-fno-builtin"], ["-O${NANO_TEST_LEVEL:-2}"]):
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
