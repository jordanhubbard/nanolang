"""I qualify non-admitting indirect File query facts; no File module executes."""
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
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FileIndirectTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-indirect_targets-"))
        print(f"I retain indirect target query artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_INDIRECT_TARGETS_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror"]
        cls.flags += shlex.split(os.environ.get("NANO_FILE_INDIRECT_TARGETS_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["FILE_INDIRECT_TARGETS_OBJECTS"])
        cls.ldflags = shlex.split(os.environ.get("FILE_INDIRECT_TARGETS_LDFLAGS", "-lm -lcrypto"))

    def command(self, name, args):
        (self.artifacts / f"{name}-command.txt").write_text(shlex.join(args) + "\n")
        env = dict(os.environ, ASAN_OPTIONS=asan_options("halt_on_error=1"),
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1", LSAN_OPTIONS="")
        status = {"timeout": False, "returncode": None, "leader_reaped": False,
                  "group_disappeared": None, "bound_seconds": 240, "errors": [],
                  "cleanup_signals": [], "launched": False}
        process = None
        with (self.artifacts / f"{name}-stdout.log").open("wb") as stdout, \
             (self.artifacts / f"{name}-stderr.log").open("wb") as stderr:
            try:
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                status["launched"] = True
                try:
                    process.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    status["timeout"] = True
            except Exception as error:
                status["errors"].append(f"launch/wait: {type(error).__name__}: {error}")
            finally:
                if process is not None:
                    def group_exists():
                        try:
                            os.killpg(process.pid, 0)
                            return True
                        except ProcessLookupError:
                            return False
                        except OSError as error:
                            status["errors"].append(f"group probe: {error}")
                            return True

                    # I inspect the group after normal completion too. Reaping
                    # its leader never proves that descendants disappeared.
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        if not group_exists():
                            break
                        try:
                            os.killpg(process.pid, sig)
                            status["cleanup_signals"].append(sig.name)
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            status["errors"].append(f"{sig.name}: {error}")
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            process.poll()  # Reap an exited leader independently.
                            if not group_exists():
                                break
                            time.sleep(0.05)
                    status["returncode"] = process.poll()
                    status["leader_reaped"] = process.returncode is not None
                    status["group_disappeared"] = not group_exists()
                (self.artifacts / f"{name}-status.json").write_text(
                    json.dumps(status, indent=2) + "\n")
        self.assertTrue(status["launched"], status)
        self.assertFalse(status["timeout"], status)
        self.assertFalse(status["errors"], status)
        self.assertTrue(status["leader_reaped"], status)
        self.assertTrue(status["group_disappeared"], status)
        self.assertEqual(status["returncode"], 0, status)
        return (self.artifacts / f"{name}-stdout.log").read_bytes()

    def qualify(self, name, instrument):
        executable = self.artifacts / name
        sources = ["tests/nanoisa/test_file_indirect_targets.c", "src/nanoisa/service_file_nominal.c", "src/nsi_file_plan.c"]
        if not instrument:
            sources += ["src/nanoisa/service_file_nominal_plan.c", "src/nanoisa/file_flow.c"]
        command = [*self.compiler, *self.flags, *(["-DFLOW_INSTRUMENT"] if instrument else []),
                   *sources, *self.objects, *self.ldflags, "-o", str(executable)]
        self.command(f"{name}-build", command)
        stdout = self.command(f"{name}-run", [str(executable)])
        self.assertIn(b"private indirect target checks; no pending module execution", stdout)
        print(stdout.decode(errors="replace").strip(), flush=True)

    def test_instrumented_indirect_targets_query(self):
        self.qualify("instrumented", True)

    def test_linked_indirect_targets_query(self):
        self.qualify("linked", False)


if __name__ == "__main__":
    unittest.main()
