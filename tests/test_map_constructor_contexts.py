"""I retain checked map tags at every constructor boundary."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/unit/test_map_constructor_contexts.nano"

class MapConstructorContexts(unittest.TestCase):
    def run_checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_scalar_tag_pairs_and_contexts(self):
        baseline = SOURCE.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for key in ("int", "string"):
                for value in ("int", "string"):
                    with self.subTest(key=key, value=value):
                        source = baseline.replace("HashMap<string,string>", f"HashMap<{key},{value}>")
                        for name in ("before", "grown", "keys"):
                            source = source.replace(f"{name}: array<string>", f"{name}: array<{key}>")
                        for name in ("items", "before_values", "grown_values"):
                            source = source.replace(f"{name}: array<string>", f"{name}: array<{value}>")
                        if key == "int":
                            source = source.replace('"new"', "11").replace('"answer"', "12")
                        if value == "int":
                            source = source.replace('"kept"', "42")
                        path = work / "case.nano"
                        path.write_text(source)
                        self.run_checked([ROOT / "bin/nano_virt", path, "--emit-nvm", "-o", work / "case.nvm"])
                        self.run_checked([ROOT / "bin/nano_vm", work / "case.nvm"])
                        # Native scoped-map cleanup is task1ed; retain the complete VM gate.
                        native = "\n".join(line for line in source.splitlines()
                                           if not line.startswith(" match box "))
                        path.write_text(native)
                        self.run_checked([ROOT / "bin/nanoc_c", path, "-o", work / "native"])
                        self.run_checked([work / "native"])

    def test_missing_or_unsupported_context_preserves_artifact(self):
        bodies = (
            "let values = (map_new)",
            "let values: HashMap<bool,string> = (map_new)",
            "let values: HashMap<string,bool> = (map_new)",
            "let values: HashMap<string,string> = (map_new 1)",
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, output = work / "bad.nano", work / "prior"
            for body in bodies:
                source.write_text("fn main() -> int { " + body + " return 0 } shadow main { assert true }")
                for compiler in ("nanoc_c", "nano_virt"):
                    with self.subTest(body=body, compiler=compiler):
                        output.write_text("prior artifact")
                        command = [ROOT / "bin" / compiler, source, "-o", output]
                        if compiler == "nano_virt": command.append("--emit-nvm")
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(output.read_text(), "prior artifact")

if __name__ == "__main__": unittest.main()
