"""I verify that my co-process release gate owns every process it stops."""

import errno
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_cop_lifecycle.sh"


class CopLifecycleGate(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="nanolang-cop-gate-")
        self.directory = Path(self.work.name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.probe = self.directory / "test_cop_lifecycle"
        self._tool("nano_virt", r'''#!/usr/bin/env python3
import pathlib, sys
try:
    output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
except (ValueError, IndexError):
    raise SystemExit(2)
output.write_bytes(b"NVM")
''')
        self._tool("nano_vm", r'''#!/usr/bin/env python3
import pathlib, sys
artifact = pathlib.Path(sys.argv[-1]).name
print("fake output for " + artifact)
''')
        self._tool("nano_vmd", r'''#!/usr/bin/env python3
import os, socket, time
path = os.environ["NANOVMD_SOCKET"]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(path)
server.listen(1)
while True:
    time.sleep(1)
''')
        self.probe.write_text("#!/bin/sh\necho owned-worker-probe\nexit 0\n")
        self.probe.chmod(0o755)
        self.sentinel_path = self.directory / "nano_cop-unrelated"
        self.sentinel_path.symlink_to("/bin/sleep")
        self.sentinel = subprocess.Popen([str(self.sentinel_path), "30"])

    def tearDown(self):
        if self.sentinel.poll() is None:
            self.sentinel.terminate()
            self.sentinel.wait(timeout=5)
        self.work.cleanup()

    def _tool(self, name, source):
        path = self.bin / name
        path.write_text(source)
        path.chmod(0o755)

    def _run(self, injection=None):
        pid_log = self.directory / ("pids-" + (injection or "normal"))
        env = os.environ.copy()
        env.update({
            "NANO_COP_LIFECYCLE_BIN_DIR": str(self.bin),
            "NANO_COP_LIFECYCLE_PROBE": str(self.probe),
            "NANO_COP_LIFECYCLE_PID_LOG": str(pid_log),
        })
        if injection:
            env["NANO_COP_LIFECYCLE_INJECT_FAIL"] = injection
        result = subprocess.run([str(SCRIPT)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=20)
        pids = []
        if pid_log.exists():
            pids = [int(line) for line in pid_log.read_text().splitlines() if line]
        time.sleep(0.05)
        for pid in pids:
            with self.subTest(injection=injection, owned_pid=pid):
                try:
                    os.kill(pid, 0)
                except OSError as error:
                    self.assertEqual(error.errno, errno.ESRCH)
                else:
                    self.fail("owned lifecycle child remains alive: " + str(pid))
        self.assertIsNone(self.sentinel.poll(), result.stdout + result.stderr)
        return result

    def test_normal_gate_preserves_unrelated_named_process(self):
        source = SCRIPT.read_text()
        self.assertNotIn("pgrep", source)
        self.assertNotIn("pkill", source)
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("13 passed, 0 failed", result.stdout)

    def test_compile_failure_is_not_masked(self):
        result = self._run("compile-ffi")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("I injected the compile-ffi failure", result.stderr)

    def test_client_failure_reaps_only_owned_processes(self):
        result = self._run("daemon-client-2")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("daemon client 2 succeeds", result.stdout)
        self.assertIn("expected=0 actual=97", result.stdout)


if __name__ == "__main__":
    unittest.main()
