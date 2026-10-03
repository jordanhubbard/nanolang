import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CIWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / ".github/workflows/ci.yml").read_text()
        cls.workflow = yaml.safe_load(cls.text)
        cls.jobs = cls.workflow["jobs"]

    def test_required_test_jobs_use_bounded_suite(self):
        for job_name in ("build-and-test", "coverage"):
            commands = "\n".join(
                str(step.get("run", "")) for step in self.jobs[job_name]["steps"]
            )
            self.assertIn("make test-quick", commands, job_name)
            self.assertNotIn("make test ", commands, job_name)

    def test_sanitizers_have_instrumented_runtime_budget(self):
        job = self.jobs["sanitizers"]
        self.assertGreaterEqual(job["timeout-minutes"], 60)
        test_step = next(
            step for step in job["steps"] if step["name"] == "Run tests with sanitizers"
        )
        self.assertGreaterEqual(test_step["timeout-minutes"], 45)

    def test_sqlite_headers_are_installed_for_test_jobs(self):
        build_commands = "\n".join(
            str(step.get("run", "")) for step in self.jobs["build-and-test"]["steps"]
        )
        self.assertIn("libsqlite3-dev", build_commands)
        self.assertIn("sqlite", build_commands)
        for job_name in ("sanitizers", "coverage"):
            commands = "\n".join(
                str(step.get("run", "")) for step in self.jobs[job_name]["steps"]
            )
            self.assertIn("libsqlite3-dev", commands, job_name)

    def test_coverage_threshold_is_reported_consistently(self):
        self.assertIn("THRESHOLD=${COVERAGE_THRESHOLD:-70.0}", self.text)
        self.assertIn("Coverage threshold (70%) met", self.text)
        self.assertNotIn("Coverage threshold (80%) met", self.text)


if __name__ == "__main__":
    unittest.main()
