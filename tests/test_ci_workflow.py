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

    def test_sanitizer_continuation_requires_the_successful_exact_checkpoint(self):
        first = self.jobs["sanitizers-stage1"]
        final = self.jobs["sanitizers"]
        self.assertIn("sanitizers-stage1", final["needs"])
        self.assertIn("always()", final["if"])
        first_commands = "\n".join(s.get("run", "") for s in first["steps"])
        self.assertIn("make bootstrap1 ", first_commands)
        self.assertNotIn("make bootstrap3 ", first_commands)
        self.assertIn("ci_bootstrap_checkpoint.py pack", first_commands)
        steps = final["steps"]
        commands = [s.get("run", "") for s in steps]
        restore = next(i for i,c in enumerate(commands) if "ci_bootstrap_checkpoint.py restore" in c)
        bootstrap = next(i for i,c in enumerate(commands) if "make bootstrap3 " in c)
        self.assertLess(restore, bootstrap)
        self.assertIn("verify-resume", commands[bootstrap])
        self.assertIn("needs.sanitizers-stage1.result", commands[0])
        self.assertEqual(first["env"], final["env"])
        upload = next(s for s in first["steps"] if s.get("id") == "stage1-checkpoint")
        download = next(s for s in steps if "actions/download-artifact@" in s.get("uses", "") and "checkpoint-artifact" in s.get("with", {}).get("artifact-ids", ""))
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertIn("needs.sanitizers-stage1.outputs.checkpoint-artifact", download["with"]["artifact-ids"])

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
        for name in ("sanitizers",):
            job = self.jobs[name]
            steps = job["steps"]
            bootstrap = next(s for s in steps if "make bootstrap3 " in s.get("run", ""))
            tests = next(s for s in steps if any(t in s.get("run", "") for t in ("make test-units ", "make test-quick ")))
            self.assertLess(steps.index(bootstrap), steps.index(tests))
            command_limit = int(job["env"].get("BOOTSTRAP_NANOISA_TIMEOUT", "1800"))
            generations = 1 if name == "sanitizers" else 2
            self.assertGreater(bootstrap["timeout-minutes"] * 60, generations * command_limit)
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
        recipe = (ROOT / ".github/containers/instrumented/Dockerfile").read_text()
        self.assertIn("libsqlite3-dev", recipe)
        self.assertIn("python3-yaml", recipe)
        for job_name in ("sanitizers-stage1", "sanitizers", "coverage-bootstrap", "coverage"):
            job = self.jobs[job_name]
            self.assertIn("instrumented-image", job["needs"])
            self.assertEqual(job["defaults"]["run"]["shell"], "python3 scripts/ci_container.py shell {0}")
            download = next(s for s in job["steps"] if "actions/download-artifact@" in s.get("uses", "") and "image-artifact" in s.get("with", {}).get("artifact-ids", ""))
            self.assertEqual(download["with"]["artifact-ids"], "${{ needs.instrumented-image.outputs.image-artifact }}")

    def test_coverage_dependency_setup_is_pinned_and_separately_bounded(self):
        image = self.jobs["instrumented-image"]
        self.assertGreaterEqual(image["timeout-minutes"], 30)
        self.assertLessEqual(image["timeout-minutes"], 35)
        recipe = (ROOT / ".github/containers/instrumented/Dockerfile").read_text()
        sources = (ROOT / ".github/containers/instrumented/ubuntu.sources").read_text()
        self.assertRegex(recipe, r"FROM ubuntu:24\.04@sha256:[0-9a-f]{64}")
        self.assertIn("Acquire::Retries=2", recipe)
        self.assertIn("APT::Update::Error-Mode=any", recipe)
        self.assertIn("https://snapshot.ubuntu.com/ubuntu/20261019T000000Z", sources)
        self.assertIn("Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg", sources)
        for name in ("coverage-bootstrap", "coverage", "sanitizers-stage1", "sanitizers"):
            commands = "\n".join(s.get("run", "") for s in self.jobs[name]["steps"])
            self.assertNotIn("apt-get", commands)
            self.assertNotIn("pip install", commands)
            self.assertIn("ci_container.py load", commands)
            for key, output in (("NANO_CI_IMAGE_ID", "image-id"), ("NANO_CI_ARCHIVE_SHA256", "archive-sha256"), ("NANO_CI_IMAGE_PLATFORM", "platform")):
                self.assertEqual(self.jobs[name]["env"][key], "${{ needs.instrumented-image.outputs." + output + " }}")

    def test_coverage_report_matches_the_recorded_report_only_policy(self):
        report = next(s for s in self.jobs["coverage"]["steps"] if s["name"] == "Report coverage (informational, not a gate)")
        self.assertIn("no threshold enforced", report["run"])
        self.assertNotIn("THRESHOLD=", self.text)
        comment = next(s for s in self.jobs["coverage"]["steps"] if s["name"] == "Comment coverage on PR")
        self.assertIn("no threshold is enforced", comment["with"]["script"])

    def test_coverage_bootstrap_and_suite_have_separate_sufficient_jobs(self):
        producer, consumer = self.jobs["coverage-bootstrap"], self.jobs["coverage"]
        bootstrap = next(s for s in producer["steps"] if "make bootstrap3 " in s.get("run", ""))
        suite = next(s for s in consumer["steps"] if "make test-quick " in s.get("run", ""))
        self.assertGreater(bootstrap["timeout-minutes"] * 60, 2 * int(producer["env"]["BOOTSTRAP_NANOISA_TIMEOUT"]))
        self.assertGreaterEqual(producer["timeout-minutes"], bootstrap["timeout-minutes"] + 30)
        self.assertGreaterEqual(consumer["timeout-minutes"], suite["timeout-minutes"] + 30)
        self.assertEqual(producer["env"], consumer["env"])
        self.assertIn("coverage-bootstrap", consumer["needs"])
        self.assertIn("needs.coverage-bootstrap.result", consumer["steps"][0]["run"])
        commands = "\n".join(s.get("run", "") for s in producer["steps"])
        self.assertIn("--phase stage2", commands)
        commands = [s.get("run", "") for s in consumer["steps"]]
        restore = next(i for i,c in enumerate(commands) if "ci_bootstrap_checkpoint.py restore" in c)
        self.assertLess(restore, next(i for i,c in enumerate(commands) if "make test-quick " in c))
        download = next(s for s in consumer["steps"] if "actions/download-artifact@" in s.get("uses", "") and "checkpoint-artifact" in s.get("with", {}).get("artifact-ids", ""))
        self.assertIn("needs.coverage-bootstrap.outputs.checkpoint-artifact", download["with"]["artifact-ids"])


if __name__ == "__main__":
    unittest.main()
