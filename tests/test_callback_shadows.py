"""I execute dependency shadows through the retained foreign callback bridge."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

CONSTRUCTOR = r'''#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void *constructor_thread(void *unused) {
    (void)unused;
    const char *path = getenv("NANO_CONSTRUCTOR_TRACE");
    if (path) {
        FILE *file = fopen(path, "a");
        assert(file);
        assert(fprintf(file, "%ld %ld\n", (long)getpid(), (long)getppid()) > 0);
        assert(fclose(file) == 0);
    }
    return NULL;
}
__attribute__((constructor)) static void provider_constructor(void) {
    pthread_t thread;
    assert(pthread_create(&thread, NULL, constructor_thread, NULL) == 0);
    assert(pthread_join(thread, NULL) == 0);
}
'''


class CallbackShadows(unittest.TestCase):
    def compile_fixture(self, succeeds, execution, trace_constructor=False, declared_owner=False):
        with tempfile.TemporaryDirectory(prefix="nano-callback-shadows-") as directory:
            work = Path(directory)
            (work / "module.json").write_text(json.dumps({
                "name": "callback_shadow_fixture", "version": "1.0.0",
                "c_sources": ["fixture.c"], "cflags": ["-pthread"], "ldflags": ["-pthread"],
                "callback_adapters": {"invoke": {"symbol": "invoke_retained",
                    "abi": "retained_v1", "execution": execution}}}))
            header = ROOT / "src/runtime/nano_callback.h"
            native_source = f'#include "{header}"\n' + """
int64_t invoke(int64_t (*fn)(int64_t), int64_t value) { return fn(value); }
int64_t invoke_retained(NanoCallbackV1 *fn, int64_t value) {
    NanoCallbackValue arg = {.tag=NANO_CALLBACK_INT, .as.integer=value}, result;
    if (fn->invoke(fn, &arg, 1, &result) != NANO_CALLBACK_OK) return -1;
    return result.as.integer;
}
"""
            if trace_constructor:
                native_source = CONSTRUCTOR + native_source
            (work / "fixture.c").write_text(native_source)
            expected = 42 if succeeds else 99
            (work / "fixture.nano").write_text(("module ExplicitCallbacks\n" if declared_owner else "") + """extern fn invoke(callback: fn(int) -> int, value: int) -> int
fn increment(value: int) -> int { return (+ value 1) }
shadow increment { assert (== (increment 41) 42) }
fn callback_check() -> int { unsafe { return (invoke increment 41) } }
""" + f'shadow callback_check {{ assert (== (callback_check) {expected}) }}\n')
            source = work / "main.nano"
            source.write_text('module "fixture.nano" as Fixture\n'
                              'fn main() -> int { return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            output = work / "program"
            report = work / "shadows.json"
            environment = dict(os.environ)
            trace = work / "constructor.log"
            if trace_constructor:
                environment["NANO_CONSTRUCTOR_TRACE"] = str(trace)
            process = subprocess.Popen([str(ROOT / "bin/nanoc_c"), str(source),
                                        "-o", str(output), "--llm-shadow-json", str(report)],
                                       cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
            if trace_constructor:
                self.assertEqual(result.returncode, 0, stdout + stderr)
                entries = [list(map(int, line.split())) for line in trace.read_text().splitlines()]
                self.assertEqual(len(entries), 1, entries)
                child, parent = entries[0]
                self.assertNotEqual(child, process.pid)
                self.assertNotEqual(parent, process.pid, "I initialize inside the nested VM child")
            self.assertTrue(report.exists(), result.stdout + result.stderr)
            status = json.loads(report.read_text())
            self.assertEqual(status["success"], succeeds)
            self.assertTrue(status["completed"])
            self.assertEqual(status["backend"], "nano_vm")
            if succeeds:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run = subprocess.run([str(output)], capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("I failed a shadow", result.stderr)
                self.assertFalse(output.exists())

    def test_constructor_thread_in_final_callback_child(self):
        self.compile_fixture(True, "worker", trace_constructor=True)

    def test_constructor_thread_in_interpreter_child(self):
        with tempfile.TemporaryDirectory(prefix="nano-interpreter-constructor-") as directory:
            work = Path(directory)
            (work / "module.json").write_text(json.dumps({
                "name": "shadow_constructor_fixture", "version": "1.0.0",
                "c_sources": ["fixture.c"], "cflags": ["-pthread"], "ldflags": ["-pthread"]}))
            (work / "fixture.c").write_text(CONSTRUCTOR +
                "#include <stdint.h>\nint64_t constructor_value(void) { return 41; }\n")
            (work / "fixture.nano").write_text(
                'extern fn constructor_value() -> int\n'
                'fn check_provider() -> int { unsafe { return (constructor_value) } }\n'
                'shadow check_provider { assert (== (check_provider) 41) }\n')
            source, output, trace = work / "main.nano", work / "program", work / "constructor.log"
            source.write_text('module "fixture.nano" as Fixture\n'
                              'fn main() -> int { return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            environment = dict(os.environ, NANO_CONSTRUCTOR_TRACE=str(trace))
            process = subprocess.Popen([str(ROOT / "bin/nanoc_c"), str(source), "-o", str(output)],
                                       cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            self.assertEqual(process.returncode, 0, stdout + stderr)
            entries = [list(map(int, line.split())) for line in trace.read_text().splitlines()]
            self.assertEqual(len(entries), 1, entries)
            child, parent = entries[0]
            self.assertNotEqual(child, process.pid)
            self.assertEqual(parent, process.pid, "I initialize inside the interpreted shadow child")

    def test_declared_owner_replaces_retained_fallback(self):
        self.compile_fixture(True, "owner", declared_owner=True)

    def test_owner_callback_dependency(self):
        self.compile_fixture(True, "owner")

    def test_worker_callback_dependency(self):
        self.compile_fixture(True, "worker")

    def test_failed_dependency_prevents_publication(self):
        self.compile_fixture(False, "worker")


if __name__ == "__main__":
    unittest.main()
