"""I exercise the real origin binder and companion acquisition helper together."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServiceInputs(unittest.TestCase):
    def checked(self, command):
        result = subprocess.run(list(map(str, command)), cwd=ROOT, text=True,
                                capture_output=True, timeout=240,
                                env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_retained_origins_snapshots_and_partial_failure(self):
        with tempfile.TemporaryDirectory(prefix="nano-service-inputs-") as directory:
            work = Path(directory)
            catalog = (ROOT / "tests/fixtures/nsi_file_plan.json").read_bytes()
            stages = os.environ.get("NANO_SERVICE_INPUT_COMPILERS", "nano_virt,nanoc_stage1,nanoc_stage2").split(",")
            for stage in stages:
                module = work / (stage + ".nvm")
                self.checked([ROOT / "bin" / stage, ROOT / "tests/service_inputs.nano", "--emit-nvm", "-o", module])
                self.checked([ROOT / "bin/nano_vm", "--verify-only", module])
                source, native = module.with_suffix(".c"), module.with_suffix(".native")
                self.checked([ROOT / "bin/nvm2c", module, "-o", source])
                cc = shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", os.environ.get("CC", "cc")))
                self.checked([*cc, "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                              "-fsanitize=address,undefined", "-fno-sanitize-recover=all", source,
                              "-o", native, "-lm", *(["-ldl"] if sys.platform.startswith("linux") else [])])
                for mode in ("ok", "invalid", "missing", "symlink"):
                    for engine in ("vm", "native"):
                        with self.subTest(compiler=stage, mode=mode, engine=engine):
                            roots = [work / "one", work / "two"]
                            for root in roots:
                                root.mkdir(exist_ok=True)
                                (root / "binding.nano").write_text("# retained origin\n")
                                companion = root / "interface.nsi.json"
                                companion.unlink(missing_ok=True)
                                companion.write_bytes(catalog)
                            second = roots[1] / "interface.nsi.json"
                            if mode == "invalid":
                                second.write_text("{}")
                            elif mode == "missing":
                                second.unlink()
                            elif mode == "symlink":
                                second.unlink()
                                second.symlink_to(roots[0] / "interface.nsi.json")
                            command = [ROOT / "bin/nano_vm", module, "--"] if engine == "vm" else [native]
                            output = self.checked([*command, roots[0] / "binding.nano", roots[1] / "binding.nano", mode])
                            self.assertIn("PASS actual source companion acquisition", output)


    def test_actual_driver_acquisition_and_companion_output_protection(self):
        configured = os.environ.get("NANO_SERVICE_INPUT_DRIVER_MODULE")
        drivers = [[ROOT / "bin/nano_vm", Path(configured), "--"]] if configured else [
            [ROOT / "bin/nanoc_stage1"], [ROOT / "bin/nanoc_stage2"]]
        declaration = 'service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'
        catalog = (ROOT / "tests/fixtures/nsi_file_plan.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="nano-input-drivers-") as directory:
            work = Path(directory)
            imported = work / "library"
            imported.mkdir()
            binding = imported / "binding.nano"
            binding.write_text(declaration)
            bridge = work / "bridge.nano"
            bridge.write_text('module "library/binding.nano" as files\n')
            root = work / "root.nano"
            root.write_text('module "bridge.nano" as bridge\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
            companion = imported / "interface.nsi.json"
            output = work / "result.nvm"
            for driver in drivers:
                for mode in ("valid", "invalid", "missing", "symlink", "output-alias", "diagnostic-alias", "parse-error-alias"):
                    with self.subTest(driver=str(driver[0]), mode=mode):
                        binding.write_text(declaration + ('fn broken(\n' if mode == "parse-error-alias" else ''))
                        companion.unlink(missing_ok=True)
                        companion.write_bytes(catalog)
                        output.write_bytes(b"prior-output")
                        if mode == "invalid": companion.write_text("{}")
                        if mode == "missing": companion.unlink()
                        if mode == "symlink":
                            target = work / "catalog.json"
                            target.write_bytes(catalog)
                            companion.unlink()
                            companion.symlink_to(target)
                        before = companion.read_bytes() if companion.exists() else None
                        command = [*driver, root, "--emit-nvm", "-o", companion if mode == "output-alias" else output]
                        if mode in ("diagnostic-alias", "parse-error-alias"):
                            command += ["--llm-diags-json", companion]
                        run = subprocess.run(list(map(str, command)), cwd=ROOT, text=True,
                                             capture_output=True, timeout=90)
                        self.assertNotEqual(run.returncode, 0)
                        text = run.stdout + run.stderr
                        if mode in ("invalid", "missing", "symlink"):
                            self.assertIn("I cannot acquire the immutable companions", text)
                        elif mode.endswith("alias"):
                            self.assertIn("I will not overwrite a source file", text)
                        else:
                            self.assertIn("I have not resolved File service declarations", text)
                        self.assertEqual(output.read_bytes(), b"prior-output")
                        if before is not None: self.assertEqual(companion.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
