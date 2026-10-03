"""I retain a finite bootstrap shadow deadline on every CI build mode."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CiShadowTimeout(unittest.TestCase):
    def test_workflow_sets_global_finite_shadow_deadline(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        prefix = workflow.split("jobs:", 1)[0]
        self.assertIn('NANO_SHADOW_TIMEOUT_SECONDS: "60"', prefix)
        self.assertIn("run: make stage1", workflow)
        self.assertIn("run: make coverage", workflow)
        self.assertIn("make sanitize", workflow)


if __name__ == "__main__":
    unittest.main()
