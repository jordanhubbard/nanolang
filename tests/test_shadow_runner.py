"""I require native shadows to finish, not merely exit successfully."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ShadowRunner(unittest.TestCase):
    def test_completion_and_deadline(self):
        source = r'''
#include <stdlib.h>
#include <string.h>
#include "runtime/shadow_runner.h"
static const char *mode;
static int entry(void) {
    printf("child=%ld\n", (long)getpid());
    fflush(stdout);
    if (!strcmp(mode, "slow")) { sleep(2); return 0; }
    if (!strcmp(mode, "return")) return 0;
    if (!strcmp(mode, "failure")) return 7;
    if (!strcmp(mode, "exit")) exit(0);
    if (!strcmp(mode, "_exit")) _exit(0);
    if (!strcmp(mode, "exec")) { execl("/usr/bin/true", "true", (char *)0); return 8; }
    if (!strcmp(mode, "abort")) abort();
    if (!strcmp(mode, "hang")) for (;;) pause();
    return 9;
}
int main(int argc, char **argv) {
    if (argc != 2) return 10;
    mode = argv[1];
    if (!strcmp(mode, "budget")) {
        int seconds = nl_shadow_timeout_seconds(10);
        printf("%d\n", seconds);
        return seconds < 0;
    }
    return nl_run_shadow_entry(entry, 1);
}
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadow-runner-") as tmp:
            binary = Path(tmp) / "probe"
            built = subprocess.run(["cc", "-std=c99", "-D_POSIX_C_SOURCE=200809L",
                                    "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "src"),
                                    "-x", "c", "-", "-o", str(binary)], input=source,
                                   text=True, capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            clean_env = dict(os.environ)
            clean_env.pop("NANO_SHADOW_TIMEOUT_SECONDS", None)
            default = subprocess.run([str(binary), "budget"], capture_output=True, env=clean_env, timeout=5)
            self.assertEqual(default.stdout, b"10\n")
            for mode in ("return", "failure", "exit", "_exit", "exec", "abort", "hang"):
                with self.subTest(mode=mode):
                    run = subprocess.run([str(binary), mode], capture_output=True, env=clean_env, timeout=5)
                    self.assertEqual(run.returncode == 0, mode == "return", run.stderr)
                    if mode == "hang":
                        self.assertIn(b"after 1 seconds", run.stderr)
                        pid = int(run.stderr.split(b"child=")[1].splitlines()[0])
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
            for value in ("", "0", "-1", "+1", "1.5", " 2", "2x", "301", "999999999999999999"):
                with self.subTest(invalid_budget=value):
                    env = dict(clean_env, NANO_SHADOW_TIMEOUT_SECONDS=value)
                    run = subprocess.run([str(binary), "return"], capture_output=True, env=env, timeout=5)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(b"integer from 1 to 300", run.stderr)
                    self.assertNotIn(b"child=", run.stderr)
            for value in ("1", "60", "300"):
                env = dict(clean_env, NANO_SHADOW_TIMEOUT_SECONDS=value)
                run = subprocess.run([str(binary), "budget"], capture_output=True, env=env, timeout=5)
                self.assertEqual(run.returncode, 0)
                self.assertEqual(int(run.stdout), int(value))
            for value, expected in (("1", False), ("3", True)):
                env = dict(clean_env, NANO_SHADOW_TIMEOUT_SECONDS=value)
                run = subprocess.run([str(binary), "slow"], capture_output=True, env=env, timeout=5)
                self.assertEqual(run.returncode == 0, expected, run.stderr)
                pid = int(run.stderr.split(b"child=")[1].splitlines()[0])
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
