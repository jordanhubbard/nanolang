"""I prepare private union storage without exposing an executor."""
from tests.native_process import objects_before_archives
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic
from tests.test_record_array_vm import PROVIDERS

class UnionFieldPreparation(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-union-field-preparation-"))
        print(f"I retain union preparation artifacts at {cls.artifacts}", flush=True)
        cls.cc = shlex.split(os.environ.get("RECORD_ARRAY_VM_CC", "cc"))
        cls.flags = [*shlex.split(os.environ.get("RECORD_ARRAY_VM_CFLAGS", "")),
            "-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror",
            "-DNANO_UNION_FIELD_PRIVATE_PREPARATION", "-DNANO_NO_COMPUTED_GOTO", "-Isrc", "-Isrc/nanoisa"]
        names = {Path(source).stem for source in PROVIDERS}
        objects = [obj for obj in shlex.split(os.environ["RECORD_ARRAY_VM_OBJECTS"])
                   if Path(obj).stem not in names]
        # Only the known Make concatenation duplicate; retain other archive order.
        cls.objects = [obj for i, obj in enumerate(objects)
            if not (obj.endswith("/libnano_compiler_inputs.a") and obj in objects[i + 1:])]
        cls.objects = objects_before_archives(cls.objects)
        cls.ldflags = shlex.split(os.environ.get("RECORD_ARRAY_VM_LDFLAGS", "-lm -lcrypto -lffi -pthread"))
        cls.modules = shlex.split(os.environ.get("UNION_PREPARATION_MODULES", ""))
        if len(cls.modules) not in (0, 6):
            raise ValueError("I require zero or all six retained union original modules.")
        (cls.artifacts / "selection.json").write_text(json.dumps({
            "cc": cls.cc, "flags": cls.flags, "objects": cls.objects,
            "providers": PROVIDERS, "ldflags": cls.ldflags, "modules": cls.modules,
            "instrumentation": "Listed providers; common retained objects keep their recorded attribution."
        }, indent=2) + "\n")

    def test_preparation_and_allocation_controls(self):
        for observed in (False, True):
            mode = "observed" if observed else "linked"
            products = []
            for source in PROVIDERS:
                if observed and source == "src/nanovm/vm.c":
                    continue
                obj = self.artifacts / (mode + "-" + Path(source).stem + ".o")
                hooks = ["-DRA_ALLOC_WRAP", "-include", "tests/nanoisa/record_array_alloc.h"] if observed else []
                self.command(mode + "-" + Path(source).stem + "-build",
                    [*self.cc, *self.flags, *hooks, "-c", source, "-o", str(obj)])
                products.append(str(obj))
            exe = self.artifacts / (mode + "-preparation")
            self.command(mode + "-harness-build", [*self.cc, *self.flags,
                *(["-DVM_UF_WHITEBOX"] if observed else []),
                "tests/nanoisa/test_union_field_preparation.c", "tests/nanoisa/record_array_alloc.c",
                *products, *self.objects, *self.ldflags, "-o", str(exe)])
            output = self.command(mode + "-preparation-run", [str(exe), *(self.modules if not observed else [])])
            self.assertIn(b"union preparation checks; no execution admission", output)
            if observed:
                self.assertIn(b"transient/persistent with valid retry", output)
                self.assertIn(b"optional-growth-successes=2", output)
            print(output.decode(), flush=True)

if __name__ == "__main__":
    unittest.main()
