import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "generate_root_package_json", ROOT / "scripts/generate_root_package_json.py"
)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class RootPackageJsonTests(unittest.TestCase):
    def test_projects_direct_dependencies_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.json"
            output = directory / "package.json"
            source.write_text(json.dumps({
                "dependencies": {"runtime": "^1.2.3"},
                "devDependencies": {"compiler": "^4.5.6"},
            }))
            generator.generate("3.5.0", source, output)
            package = json.loads(output.read_text())
            self.assertEqual(package["version"], "3.5.0")
            self.assertTrue(package["private"])
            self.assertEqual(package["dependencies"], {"runtime": "^1.2.3"})
            self.assertEqual(package["devDependencies"], {"compiler": "^4.5.6"})

    def test_generation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.json"
            output = directory / "package.json"
            source.write_text('{"dependencies":{"runtime":"1"}}')
            generator.generate("1.0.0", source, output)
            first = output.read_bytes()
            generator.generate("1.0.0", source, output)
            self.assertEqual(output.read_bytes(), first)

    def test_invalid_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "invalid release version"):
                generator.generate(
                    "release", ROOT / "vscode/package.json", Path(temporary) / "package.json"
                )


if __name__ == "__main__":
    unittest.main()
