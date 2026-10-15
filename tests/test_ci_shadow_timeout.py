"""I retain a finite bootstrap shadow deadline on every CI build mode."""
from pathlib import Path
import unittest
import yaml


ROOT = Path(__file__).resolve().parents[1]


class CiShadowTimeout(unittest.TestCase):
    def test_workflow_sets_global_finite_shadow_deadline(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        deadline=str(yaml.safe_load(workflow)["env"]["NANO_SHADOW_TIMEOUT_SECONDS"])
        self.assertRegex(deadline,r"^[0-9]+$")
        self.assertGreaterEqual(int(deadline),1)
        self.assertLessEqual(int(deadline),300)
        self.assertIn("run: make stage1", workflow)
        self.assertIn("run: make coverage", workflow)
        self.assertIn("make sanitize", workflow)


if __name__ == "__main__":
    unittest.main()
