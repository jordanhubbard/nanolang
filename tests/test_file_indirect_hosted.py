"""I qualify copied indirect hosted facts without runtime or service execution."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest

from tests import test_file_cyclic as cyclic_runner
from tests import test_file_hosted as hosted_providers


class FileIndirectHosted(unittest.TestCase):
    # I reuse the qualified file-backed, bounded TERM/KILL/leader/group runner.
    # Importing modules rather than TestCase names avoids duplicate discovery.
    command = cyclic_runner.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-indirect-hosted-"))
        print(f"I retain indirect hosted artifacts at {cls.artifacts}", flush=True)
        cls.compiler = shlex.split(os.environ.get("NANO_FILE_INDIRECT_HOSTED_CC", "cc"))
        cls.flags = shlex.split(os.environ.get("NANO_FILE_INDIRECT_HOSTED_CFLAGS", ""))
        cls.flags += ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror"]
        if os.environ.get("NANO_FILE_INDIRECT_HOSTED_SANITIZERS", "0") != "0":
            cls.flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        cls.objects = [p for p in shlex.split(os.environ["FILE_INDIRECT_HOSTED_OBJECTS"])
                       if Path(p).stem not in hosted_providers.PROVIDERS]
        cls.ldflags = shlex.split(os.environ.get("FILE_INDIRECT_HOSTED_LDFLAGS", "-lm -lcrypto"))

    def qualify(self, name, instrument):
        objects = []
        hooks = (["-include", "tests/nanoisa/file_hosted_alloc.h",
                  "-Dmalloc=file_test_malloc", "-Dcalloc=file_test_calloc",
                  "-Drealloc=file_test_realloc", "-Dfree=file_test_free"] if instrument else [])
        for provider in hosted_providers.PROVIDERS:
            if instrument and provider == "file_flow":
                continue  # Included under the same HOSTED hooks for internal controls.
            obj = self.artifacts / f"{name}-{provider}.o"
            self.command(f"{name}-{provider}-build", [*self.compiler, *self.flags, *hooks,
                         "-c", f"src/nanoisa/{provider}.c", "-o", str(obj)])
            objects.append(str(obj))
        exe = self.artifacts / name
        self.command(f"{name}-link", [*self.compiler, *self.flags,
                     *(["-DHOSTED_INSTRUMENT"] if instrument else []),
                     "tests/nanoisa/test_file_indirect_hosted.c", *objects, *self.objects,
                     *self.ldflags, "-o", str(exe)])
        stdout = self.command(f"{name}-run", [str(exe)])
        self.assertIn(b"private indirect hosted checks; no runtime or service execution", stdout)
        self.assertIn(b"private serialized File hosted-plan checks", stdout)
        print(stdout.decode(errors="replace").strip(), flush=True)

    def test_full_chain_allocations_and_all_variants(self):
        self.qualify("instrumented", True)

    def test_linked_copied_indirect_hosted_facts(self):
        self.qualify("linked", False)


if __name__ == "__main__":
    unittest.main()
