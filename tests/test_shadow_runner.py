"""I require native shadows to finish, not merely exit successfully."""
from pathlib import Path
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
            for mode in ("return", "failure", "exit", "_exit", "exec", "abort", "hang"):
                with self.subTest(mode=mode):
                    run = subprocess.run([str(binary), mode], capture_output=True, timeout=5)
                    self.assertEqual(run.returncode == 0, mode == "return", run.stderr)
                    if mode == "hang":
                        self.assertIn(b"after 1 seconds", run.stderr)


if __name__ == "__main__":
    unittest.main()
