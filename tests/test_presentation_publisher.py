"""I name published presentation resources from my release version."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs" / "presentation" / "publish_google_workspace.py"
SPEC = importlib.util.spec_from_file_location("publish_google_workspace", MODULE_PATH)
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


class PresentationPublisher(unittest.TestCase):
    def version_header(self, text):
        directory = tempfile.TemporaryDirectory(prefix="nano-publisher-version-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "version.h"
        path.write_text(text)
        return path

    def test_release_edition_comes_from_version_owner(self):
        path = self.version_header("""#define NANOLANG_VERSION_MAJOR 5
#define NANOLANG_VERSION_MINOR 1
#define NANOLANG_VERSION_PATCH 0
""")
        self.assertEqual(PUBLISHER._release_edition(path), "5.1")

    def test_incomplete_version_metadata_is_refused(self):
        path = self.version_header("#define NANOLANG_VERSION_MAJOR 5\n")
        with self.assertRaisesRegex(SystemExit, "major and minor"):
            PUBLISHER._release_edition(path)

    def test_resource_names_use_the_selected_edition(self):
        self.assertEqual(PUBLISHER._resource_name("Overview", "5.1"),
                         "NanoLang Developer Overview (5.1 edition)")
        self.assertEqual(PUBLISHER._resource_name("Narrative", "5.1"),
                         "NanoLang Developer Narrative (5.1 edition)")


if __name__ == "__main__":
    unittest.main()
