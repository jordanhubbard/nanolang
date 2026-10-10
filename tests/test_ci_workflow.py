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
        self.assertGreaterEqual(job["timeout-minutes"], 150)
        test_step = next(
            step for step in job["steps"] if step["name"] == "Run tests with sanitizers"
        )
        self.assertGreaterEqual(test_step["timeout-minutes"], 120)

    def test_sanitizers_bootstrap_real_generations_before_tests(self):
        job=self.jobs["sanitizers"]
        env=job["env"]
        self.assertIn("-O3",env["CFLAGS"].split())
        for flags in ("CFLAGS","LDFLAGS"):
            self.assertIn("-fsanitize=address,undefined",env[flags].split())
        commands=[str(step.get("run","")) for step in job["steps"]]
        bootstrap=next(i for i,c in enumerate(commands) if "make bootstrap3 " in c)
        tests=next(i for i,c in enumerate(commands) if "make test-units " in c)
        self.assertLess(bootstrap,tests)
        for command in commands:
            if any(target in command for target in ("make sanitize ","make bootstrap3 ","make test-units ")):
                self.assertIn('CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS"',command)

    def test_service_generations_have_a_budget_after_bootstrap(self):
        job = self.jobs["build-and-test"]
        steps = job["steps"]
        bootstrap = next(s for s in steps if "make -f Makefile.gnu bootstrap3" in s.get("run", ""))
        generations = next(s for s in steps if "for generation in 1 2" in s.get("run", ""))
        self.assertLess(steps.index(bootstrap), steps.index(generations))
        self.assertNotIn("bootstrap3", generations["run"])
        for suite in ("test_nano_service_driver", "test_socket_service_drivers", "test_mixed_service_drivers"):
            self.assertIn(suite, generations["run"])
        self.assertEqual(generations["env"]["NANO_SHADOW_TIMEOUT_SECONDS"], "10")
        self.assertGreaterEqual(job["timeout-minutes"], bootstrap["timeout-minutes"] + generations["timeout-minutes"] + 75)

    def test_instrumented_bootstrap_and_suite_budgets_fit_jobs(self):
        for name in ("sanitizers", "coverage"):
            job = self.jobs[name]
            steps = job["steps"]
            bootstrap = next(s for s in steps if "make bootstrap3 " in s.get("run", ""))
            tests = next(s for s in steps if any(t in s.get("run", "") for t in ("make test-units ", "make test-quick ")))
            self.assertLess(steps.index(bootstrap), steps.index(tests))
            command_limit = int(job["env"].get("BOOTSTRAP_NANOISA_TIMEOUT", "1800"))
            self.assertGreater(bootstrap["timeout-minutes"] * 60, 2 * command_limit)
            self.assertGreaterEqual(job["timeout-minutes"], bootstrap["timeout-minutes"] + tests["timeout-minutes"] + 20)
            self.assertIn('CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS"', bootstrap["run"])
            self.assertEqual(job["env"]["NANO_SHADOW_TIMEOUT_SECONDS"], "300")
        for flags in ("CFLAGS", "LDFLAGS"):
            self.assertIn("-fprofile-arcs", self.jobs["coverage"]["env"][flags].split())
            self.assertIn("-ftest-coverage", self.jobs["coverage"]["env"][flags].split())

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

    def test_coverage_dependency_retries_fit_setup_budget(self):
        job = self.jobs["coverage"]
        setup = next(step for step in job["steps"] if step["name"] == "Install dependencies")
        env = setup["env"]
        attempts = int(env["CI_APT_ATTEMPTS"])
        timeout = int(env["CI_APT_TIMEOUT_SECS"])
        backoff = int(env["CI_APT_BACKOFF_SECS"])
        retry_budget = attempts * timeout + backoff * attempts * (attempts - 1) // 2
        self.assertGreaterEqual(attempts, 2)
        self.assertGreater(timeout, 180)
        self.assertLess(retry_budget, setup["timeout-minutes"] * 60)
        self.assertGreaterEqual(job["timeout-minutes"] - setup["timeout-minutes"], 35)

    def test_coverage_threshold_is_reported_consistently(self):
        self.assertIn("THRESHOLD=${COVERAGE_THRESHOLD:-40.0}", self.text)
        self.assertIn("Coverage threshold (40%) met", self.text)
        self.assertNotIn("Coverage threshold (80%) met", self.text)


if __name__ == "__main__":
    unittest.main()
