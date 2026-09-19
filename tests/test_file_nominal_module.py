"""I qualify service v2 metadata retention; every executing route still refuses."""
import os
from pathlib import Path
import tempfile
import unittest
from tests import test_service_bindings_module as support
ROOT = Path(__file__).resolve().parents[1]

class NominalModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        support.ServiceModule.setUpClass.__func__(cls)

    def command(self, *args, **kwargs):
        return support.ServiceModule.command(self, *args, **kwargs)

    def qualify(self, instrument):
        name = "nominal-allocation" if instrument else "nominal-linked"
        objects = list(self.objects)
        # I instrument the adapter, both bridges, retained-layout allocation and
        # the private map. Unchanged remaining objects retain normal build flags.
        sources = ("nvm_format", "nvm_v2_convert", "service_bindings_module",
                   "service_file_nominal_plan", "retained_layouts", "nvm_v2_layouts")
        for source in sources:
            old = next(p for p in objects if p.endswith(f"/nanoisa/{source}.o"))
            objects.remove(old)
            obj = self.artifacts / f"{name}-{source}.o"
            hooks = (["-include", "tests/nanoisa/service_alloc_hooks.h", "-Dmalloc=service_test_malloc",
                      "-Dcalloc=service_test_calloc", "-Drealloc=service_test_realloc"] if instrument else [])
            self.command(f"{name}-{source}-compile", [*self.compiler, *self.flags, *hooks,
                "-c", f"src/nanoisa/{source}.c", "-o", str(obj)])
            objects.append(str(obj))
        exe = self.artifacts / name
        self.command(f"{name}-build", [*self.compiler, *self.flags,
            *(["-DSERVICE_ALLOC_TEST"] if instrument else []),
            "tests/nanoisa/test_file_nominal_module.c", *objects, *self.linkflags, "-o", str(exe)])
        wire = self.artifacts / f"{name}.nvm"
        output = self.command(f"{name}-run", [str(exe), "tests/fixtures/nsi_file_plan.json"],
                              extra={"NOMINAL_MODULE_WIRE": str(wire)})
        self.assertIn("checks passed; no host execution", output)
        print(output.strip(), flush=True)
        if not instrument:
            commands = [
                ("facts", [str(ROOT / "bin/nanoisa_hl_facts"), str(wire)], False),
                ("native-c", [str(ROOT / "bin/nvm2c"), str(wire)], True),
                ("llvm", [str(ROOT / "bin/nvm2llvm"), str(wire)], True),
                ("wasm", ["python3", "scripts/nvm2wasm.py", str(wire)], True),
                ("recover-c", ["python3", "scripts/nvm2hl.py", "--language", "c", str(wire)], True),
                ("recover-nano", ["python3", "scripts/nvm2hl.py", "--language", "nano", str(wire)], True),
            ]
            for label, command, has_output in commands:
                destination = self.artifacts / f"nominal-{label}-preserved"
                destination.write_bytes(b"prior output\0")
                if has_output:
                    command += ["-o", str(destination)]
                stdout = self.command(f"nominal-cli-{label}", command, expected=1,
                    extra={"NANO_NVM2LLVM": str(ROOT / "bin/nvm2llvm")})
                self.assertEqual(stdout, "")
                self.assertEqual(destination.read_bytes(), b"prior output\0")

    def test_exact_retention_and_all_consumer_refusals(self):
        self.qualify(False)

    def test_adapter_attach_and_bridge_allocation_prefixes(self):
        self.qualify(True)

if __name__ == "__main__":
    unittest.main()
