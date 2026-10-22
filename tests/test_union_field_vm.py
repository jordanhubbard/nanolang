"""I qualify private union execution; public selection stays closed."""
from tests.native_process import objects_before_archives
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic
from tests.test_record_array_vm import PROVIDERS

class UnionFieldVm(unittest.TestCase):
    command = test_file_cyclic.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-union-field-vm-"))
        print(f"I retain union execution artifacts at {cls.artifacts}", flush=True)
        cls.cc = shlex.split(os.environ.get("RECORD_ARRAY_VM_CC", "cc"))
        cls.flags = [*shlex.split(os.environ.get("RECORD_ARRAY_VM_CFLAGS", "")),
            "-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror",
            "-DNANO_UNION_FIELD_PRIVATE_PREPARATION", "-Isrc", "-Isrc/nanoisa"]
        names = {Path(source).stem for source in PROVIDERS}
        objects = [obj for obj in shlex.split(os.environ["RECORD_ARRAY_VM_OBJECTS"])
                   if Path(obj).stem not in names]
        # Only the known Make concatenation duplicate; retain other archive order.
        cls.objects = [obj for i, obj in enumerate(objects)
            if not (obj.endswith("/libnano_compiler_inputs.a") and obj in objects[i + 1:])]
        cls.objects = objects_before_archives(cls.objects)
        cls.ldflags = shlex.split(os.environ.get("RECORD_ARRAY_VM_LDFLAGS", "-lm -lcrypto -lffi -pthread"))
        cls.modules = shlex.split(os.environ.get("UNION_VM_MODULES", ""))
        if len(cls.modules) not in (0, 6):
            raise ValueError("I require zero or all six retained union original modules.")
        (cls.artifacts / "selection.json").write_text(json.dumps({
            "cc": cls.cc, "flags": cls.flags, "objects": cls.objects,
            "providers": PROVIDERS, "ldflags": cls.ldflags, "modules": cls.modules,
            "instrumentation": "Listed providers; common retained objects keep their recorded attribution."
        }, indent=2) + "\n")

    def qualify(self, threaded):
        for observed in (False, True):
            mode = ("observed" if observed else "linked") + ("-goto" if threaded else "-switch")
            dispatch = [] if threaded else ["-DNANO_NO_COMPUTED_GOTO"]
            flags = [*self.flags, *dispatch]
            macros = self.command(mode + "-macros", [*self.cc, *flags, "-dM", "-E", "src/nanovm/vm.c"])
            if threaded:
                self.assertIn(b"#define NANO_COMPUTED_GOTO 1", macros)
            else:
                self.assertNotIn(b"#define NANO_COMPUTED_GOTO ", macros)
            products = []
            for source in PROVIDERS:
                if source == "src/nanovm/vm.c":
                    continue  # Both modes include current vm.c for internal scope controls.
                obj = self.artifacts / (mode + "-" + Path(source).stem + ".o")
                hooks = ["-DRA_ALLOC_WRAP", "-include", "tests/nanoisa/record_array_alloc.h"] if observed else []
                self.command(mode + "-" + Path(source).stem + "-build",
                    [*self.cc, *flags, *hooks, "-c", source, "-o", str(obj)])
                products.append(str(obj))
            exe = self.artifacts / mode
            self.command(mode + "-harness-build", [*self.cc, *flags, "-DUNION_EXECUTION_WHITEBOX",
                *(["-DUNION_EXECUTION_ALLOC"] if observed else []),
                "tests/nanoisa/test_union_field_vm.c", "tests/nanoisa/record_array_alloc.c",
                *products, *self.objects, *self.ldflags, "-o", str(exe)])
            output = self.command(mode + "-run", [str(exe), *(self.modules if not observed else [])])
            self.assertIn(b"private union execution checks", output)
            self.assertIn(b"no public enrollment", output)
            if observed:
                for label in (b"tail18", b"call18", b"dynamic-string", b"retained-global", b"record-and-empty-union"):
                    self.assertIn(b"runtime prefixes " + label, output)
            print(output.decode(), flush=True)

    def test_switch(self):
        self.qualify(False)

    def test_goto(self):
        self.qualify(True)

if __name__ == "__main__":
    unittest.main()
