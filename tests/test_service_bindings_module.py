"""I qualify retained service data and checked refusal; I dispatch no service."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServiceModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-service-module-"))
        print(f"I retain service transport artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_SERVICE_MODULE_TEST_CC", "cc"))
        cls.flags = ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra",
                     "-Werror", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                     "-Isrc/nanoisa", "-Imodules/nanoisa"]
        cls.flags += shlex.split(os.environ.get("NANO_SERVICE_MODULE_TEST_CFLAGS", ""))
        cls.objects = shlex.split(os.environ["SERVICE_MODULE_OBJECTS"])
        cls.linkflags = shlex.split(os.environ.get("SERVICE_MODULE_LDFLAGS", ""))

    def command(self, name, command):
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        (self.artifacts / f"{name}-command.txt").write_text(shlex.join(command) + "\n")
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        (self.artifacts / f"{name}.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, (command, result.stdout, result.stderr))
        return result.stdout

    def qualify(self, instrumented):
        name = "allocation" if instrumented else "linked"
        objects = list(self.objects)
        if instrumented:
            for source in ("nvm_format", "nvm_v2_convert", "service_bindings_module"):
                old = next(p for p in objects if p.endswith(f"/nanoisa/{source}.o"))
                objects.remove(old)
                output = self.artifacts / f"{source}-instrumented.o"
                self.command(f"{source}-compile", [*self.compiler, *self.flags,
                    "-include", "tests/nanoisa/service_alloc_hooks.h", "-Dmalloc=service_test_malloc",
                    "-Dcalloc=service_test_calloc", "-Drealloc=service_test_realloc",
                    "-c", f"src/nanoisa/{source}.c", "-o", str(output)])
                objects.append(str(output))
        exe = self.artifacts / name
        defines = ["-DSERVICE_ALLOC_TEST"] if instrumented else []
        self.command(f"{name}-build", [*self.compiler, *self.flags, *defines,
            "tests/nanoisa/test_service_bindings_module.c", *objects, *self.linkflags, "-o", str(exe)])
        output = self.command(f"{name}-run", [str(exe), "tests/fixtures/nsi_file_plan.json"])
        self.assertIn("checks passed", output)
        print(output.strip(), flush=True)

    def test_linked_roundtrip_and_all_consumers(self):
        self.qualify(False)

    def test_attach_and_both_bridge_allocation_prefixes(self):
        self.qualify(True)
