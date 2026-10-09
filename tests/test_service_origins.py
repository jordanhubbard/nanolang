"""I retain service source identity through the actual C loader and Nano binder."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServiceOrigins(unittest.TestCase):
    def run_checked(self, command):
        result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_c_loader_canonical_origin_lifetime(self):
        with tempfile.TemporaryDirectory(prefix="nano-service-origins-") as tmp:
            work = Path(tmp)
            declaration = 'service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'
            for name in ("one/binding.nano", "two/binding.nano", *[f"{i}.nano" for i in range(2, 17)]):
                path = work / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(declaration)
            (work / "alias.nano").symlink_to(work / "one/binding.nano")
            (work / "bridge.nano").write_text('module "alias.nano" as files\n')
            (work / "main.nano").write_text('module "bridge.nano" as wrapper\n')
            output = self.run_checked([ROOT / "obj/test_service_origins", work])
            self.assertIn("PASS service origins", output)

    def test_paired_merged_source_mapping(self):
        stages = os.environ.get("NANO_SERVICE_ORIGIN_COMPILERS", "nano_virt,nanoc_stage1,nanoc_stage2").split(",")
        with tempfile.TemporaryDirectory(prefix="nano-service-origin-products-") as tmp:
            for stage in stages:
                with self.subTest(compiler=stage):
                    module = Path(tmp) / (stage + ".nvm")
                    self.run_checked([ROOT / "bin" / stage, ROOT / "tests/service_origins.nano", "--emit-nvm", "-o", module])
                    self.run_checked([ROOT / "bin/nano_vm", "--verify-only", module])
                    self.assertIn("PASS merged service origins", self.run_checked([ROOT / "bin/nano_vm", module]))

    def test_actual_drivers_reject_duplicate_service_origins_before_lowering(self):
        with tempfile.TemporaryDirectory(prefix="nano-service-origin-drivers-") as tmp:
            work = Path(tmp)
            source = work / "binding.nano"
            declaration = 'service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"\n'
            source.write_text(declaration * 2)
            for compiler in ("nanoc_c", "nano_virt", "nanoc_stage1", "nanoc_stage2"):
                with self.subTest(compiler=compiler):
                    output = work / (compiler + ".out")
                    output.write_bytes(b"prior-output")
                    command = [ROOT / "bin" / compiler, source, "-o", output]
                    if compiler != "nanoc_c":
                        command.append("--emit-nvm")
                    result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                            capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("I cannot retain the original source", result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b"prior-output")


if __name__ == "__main__":
    unittest.main()
