"""I inject gate failures without launching or killing an ambient daemon."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.daemon_integration import compare

FAKE = r'''
import os, signal, socket, sys, time
from pathlib import Path
mode = os.environ['GATE_MODE']
role = Path(sys.argv[0]).name
sock = os.environ['NANOVMD_SOCKET']
assert os.environ['NANOVMD_NO_AUTOSTART'] == '1'
assert sock.startswith('/tmp/nano-vmd-')
record = Path(os.environ['GATE_RECORD'])
with record.open('a') as out:
    out.write(f'{os.getpid()} {sock}\n')
if role == 'compiler':
    if mode == 'compile_fail': sys.exit(7)
    if mode == 'compile_hang': time.sleep(30)
    if mode != 'missing': Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'nvm')
elif role == 'vm':
    daemon = '--daemon' in sys.argv
    if mode == 'both_fail' or (mode == 'daemon_fail' and daemon): sys.exit(3)
    if mode == 'vm_hang': time.sleep(30)
    if mode == 'daemon_hang' and daemon: time.sleep(30)
    if mode == 'death' and daemon:
        os.kill(int(Path(sock + '.pid').read_text()), signal.SIGKILL)
        time.sleep(.1)
    print('different' if mode == 'mismatch' and daemon else 'same')
elif role == 'daemon':
    if mode == 'startup_fail': sys.exit(2)
    if mode == 'ignore_term': signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == 'startup_hang': time.sleep(30)
    server = socket.socket(socket.AF_UNIX)
    server.bind(sock)
    server.listen()
    Path(sock + '.pid').write_text(str(os.getpid()))
    while True:
        conn, _ = server.accept()
        conn.recv(8)
        conn.sendall(bytes([1, 0x12 if mode == 'bad_pong' else 0x13, 0, 0, 0, 0, 0, 0]))
        conn.close()
'''


class DaemonGate(unittest.TestCase):
    def test_injected_failures_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="daemon-gate-") as directory:
            root = Path(directory)
            commands = []
            for role in ("compiler", "vm", "daemon"):
                command = root / role
                command.write_text(f"#!{sys.executable}\n" + FAKE)
                command.chmod(0o700)
                commands.append(command)
            record = root / "record"
            # I own this unrelated process too, but the runner must leave it alone.
            ambient = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            try:
                for mode in ("ok", "ignore_term", "compile_fail", "missing", "both_fail", "daemon_fail",
                             "mismatch", "death", "startup_fail", "startup_hang", "bad_pong",
                             "compile_hang", "vm_hang", "daemon_hang"):
                    with self.subTest(mode=mode):
                        record.write_text("")
                        output = io.StringIO()
                        with patch.dict(os.environ, GATE_MODE=mode, GATE_RECORD=str(record)), \
                             contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                            if mode in ("ok", "ignore_term"):
                                compare(*commands, [root / "test.nano"], .5, .5)
                            else:
                                with self.assertRaises(RuntimeError):
                                    compare(*commands, [root / "test.nano"], .5, .5)
                        for line in record.read_text().splitlines():
                            pid, sock = line.split(" ", 1)
                            self.assertFalse(Path(sock).parent.exists())
                            with self.assertRaises(ProcessLookupError):
                                os.kill(int(pid), 0)
                        self.assertIsNone(ambient.poll())
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    compare(*commands, [])
            finally:
                ambient.terminate()
                ambient.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
