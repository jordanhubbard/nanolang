"""I qualify private TCP carrier lifecycles; matched CODE dispatch remains separate."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class SocketRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-socket-runtime-"))
        print(f"I retain TCP runtime artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_SOCKET_RUNTIME_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror",
                     "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        cls.flags += shlex.split(os.environ.get("NANO_SOCKET_RUNTIME_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["SOCKET_RUNTIME_OBJECTS"])
        cls.ldflags = shlex.split(os.environ.get("SOCKET_RUNTIME_LDFLAGS", "-lm -lcrypto"))

    def qualify(self, name, instrument):
        exe = self.artifacts / name
        sources = ["tests/nanoisa/test_socket_runtime.c", "src/nanoisa/service_socket_nominal.c",
                   "src/nsi_file_plan.c", "src/nsi_socket_plan.c", "src/nanoisa/service_socket_nominal_plan.c",
                   "src/nanoisa/socket_flow.c", "src/nsi_cap.c", "src/nsi_socket_values.c"]
        runtime = self.artifacts / f"{name}-runtime.o"
        hooks = (["-include", "tests/nanoisa/socket_runtime_hooks.h", "-Dmalloc=socket_runtime_malloc",
                  "-Dcalloc=socket_runtime_calloc", "-Dfree=socket_runtime_free"] if instrument else [])
        compile_runtime = [*self.compiler, *self.flags, *hooks, "-c", "src/nanoisa/socket_runtime.c", "-o", str(runtime)]
        adapter = self.artifacts / f"{name}-adapter.o"
        compile_adapter = [*self.compiler, *self.flags, "-include", "tests/nanoisa/socket_runtime_hooks.h",
                           "-Dsocket=socket_runtime_socket", "-Dclose=socket_runtime_close",
                           "-c", "src/nsi_socket.c", "-o", str(adapter)]
        command = [*self.compiler, *self.flags, *(["-DSOCKET_RUNTIME_INSTRUMENT"] if instrument else []),
                   *sources, str(runtime), str(adapter), *self.objects, *self.ldflags, "-o", str(exe)]
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        for label, args in (("adapter-build", compile_adapter), ("runtime-build", compile_runtime), ("build", command), ("run", [str(exe)])):
            (self.artifacts / f"{name}-{label}-command.txt").write_text(shlex.join(args) + "\n")
            result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            (self.artifacts / f"{name}-{label}.log").write_text(result.stdout + result.stderr)
            (self.artifacts / f"{name}-{label}-status.txt").write_text(str(result.returncode) + "\n")
            self.assertEqual(result.returncode, 0, (args, result.stdout, result.stderr))
            if label == "run":
                self.assertIn("PASS", result.stdout)
                print(result.stdout.strip(), flush=True)

    def test_instrumented_carrier_allocation_recovery(self):
        self.qualify("instrumented", True)

    def test_linked_tcp_lifecycles_and_cleanup(self):
        self.qualify("linked", False)

if __name__ == "__main__":
    unittest.main()
