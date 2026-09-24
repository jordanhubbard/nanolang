"""I check the preparatory grant, never a File dispatcher or host service."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileHostGrant(unittest.TestCase):
    def test_c99_consumers_share_c11_gate(self):
        cc = shlex.split(os.environ.get("NANO_FILE_HOST_GRANT_CC", "cc"))
        cppflags = shlex.split(os.environ.get("NANO_FILE_HOST_GRANT_CPPFLAGS", ""))
        flags = shlex.split(os.environ.get(
            "NANO_FILE_HOST_GRANT_CFLAGS", "-O1 -g -Wall -Wextra -Werror -std=c99"))
        ldflags = shlex.split(os.environ.get("NANO_FILE_HOST_GRANT_LDFLAGS", ""))
        root = Path(tempfile.mkdtemp(prefix="nano-file-host-grant-"))
        print(f"I retain grant artifacts at {root}", flush=True)
        commands = []
        env = dict(os.environ)
        env.pop("LSAN_OPTIONS", None)
        env["ASAN_OPTIONS"] = asan_options("halt_on_error=1")
        env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"

        def run(command, expected=0):
            index = len(commands)
            process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                output, _ = process.communicate(timeout=60)
                result = subprocess.CompletedProcess(command, process.returncode, output)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output, _ = process.communicate()
                (root / f"{index:02d}.log").write_text(output)
                commands.append({"command": command, "status": "timeout",
                                 "timeout_seconds": 60, "log": f"{index:02d}.log"})
                (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
                raise
            (root / f"{index:02d}.log").write_text(result.stdout)
            commands.append({"command": command, "status": result.returncode,
                             "log": f"{index:02d}.log"})
            (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            if expected == 0:
                self.assertEqual(result.returncode, 0, result.stdout)
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout)
            return result

        common = cc + cppflags + flags + ["-Wall", "-Wextra", "-Werror", "-UNDEBUG",
                                         "-Isrc/nanoisa", "-pthread"]
        peer = root / "peer.o"
        run(common + ["-std=c99", "-c", "tests/nanoisa/file_host_grant_peer.c",
                      "-o", str(peer)])
        for instrumented in (False, True):
            mode = "instrumented" if instrumented else "linked"
            provider = root / f"{mode}-provider.o"
            supplied = os.environ.get("NANO_FILE_HOST_GRANT_OBJECT")
            if not instrumented and supplied:
                provider = (ROOT / supplied).resolve()
                self.assertTrue(provider.is_file())
            else:
                source = ("tests/nanoisa/file_host_grant_instrumented.c" if instrumented
                          else "src/nanoisa/file_host_grant.c")
                run(common + ["-std=c11", "-c", source, "-o", str(provider)])
            main = root / f"{mode}-main.o"
            defines = ["-DFILE_HOST_INSTRUMENTED"] if instrumented else []
            run(common + ["-std=c99"] + defines + ["-c", "tests/nanoisa/test_file_host_grant.c",
                                                  "-o", str(main)])
            binary = root / mode
            run(cc + flags + [str(main), str(peer), str(provider), "-pthread"]
                + ldflags + ["-o", str(binary)])
            run([str(binary)])
        # The explicit recipe is required: the owning source must refuse C99.
        result = run(common + ["-std=c99", "-c", "src/nanoisa/file_host_grant.c",
                               "-o", str(root / "wrong-mode.o")], expected=1)
        self.assertIn("I require C11", result.stdout)


if __name__ == "__main__":
    unittest.main()
