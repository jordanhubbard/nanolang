"""I audit test dependencies without building or hiding missing tools."""
from pathlib import Path
import tempfile
import unittest

from scripts.check_test_prerequisites import ROOT, database, missing_prerequisites


class MakePrerequisites(unittest.TestCase):
    def test_repository_test_prerequisites_resolve(self):
        self.assertEqual(missing_prerequisites(database(ROOT), ROOT), [])

    def test_bare_tool_is_rejected_without_running_recipes(self):
        with tempfile.TemporaryDirectory(prefix="nano-make-prerequisites-") as tmp:
            root = Path(tmp)
            (root / "source.c").touch()
            (root / "Makefile.gnu").write_text(
                ".PHONY: build test-known\n"
                "build:\n\ttouch forbidden\n"
                "bin/nanoisa:\n\ttouch forbidden\n"
                "%.o: %.c\n\ttouch forbidden\n"
                "test-known: bin/nanoisa source.c source.o | build\n"
                "test-broken: nanoisa\n"
            )
            self.assertEqual(missing_prerequisites(database(root), root),
                             [("test-broken", "nanoisa")])
            self.assertFalse((root / "forbidden").exists())

    def test_actual_service_targets_reject_original_bare_tool_regression(self):
        for target in ("test-file-source-inputs", "test-service-inputs"):
            with self.subTest(target=target):
                text = database(ROOT, extra=f"{target}: nanoisa")
                self.assertIn((target, "nanoisa"), missing_prerequisites(text, ROOT))


if __name__ == "__main__":
    unittest.main()
