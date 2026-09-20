"""I check the built archive boundary, not an installed private cyclic API."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from tests import test_file_cyclic as retained_runner

ROOT = Path(__file__).resolve().parents[1]


class FileCyclicHostedIntegration(unittest.TestCase):
    command = retained_runner.FileCyclic.command

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix="nano-file-cyclic-hosted-integration-"))
        print(f"I retain integration artifacts at {cls.artifacts}", flush=True)

    def test_actual_public_archive_keeps_cyclic_refusal(self):
        archive = Path(os.environ["FILE_CYCLIC_HOSTED_ARCHIVE"]).resolve()
        objects = shlex.split(os.environ["FILE_CYCLIC_HOSTED_OBJECTS"])
        inputs = [archive, *(Path(p).resolve() for p in objects)]
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        (self.artifacts / "linked-inputs-before.json").write_text(json.dumps(before, indent=2) + "\n")
        shutil.copyfile(archive, self.artifacts / "qualified-public-runtime.a")
        executable = self.artifacts / "archive-boundary"
        compiler = shlex.split(os.environ.get("NANO_FILE_CYCLIC_HOSTED_CC", "cc"))
        flags = shlex.split(os.environ.get("NANO_FILE_CYCLIC_HOSTED_CFLAGS", ""))
        flags += ["-std=c11", "-D_DEFAULT_SOURCE", "-g", "-O1", "-Wall", "-Wextra", "-Werror", "-pthread"]
        ldflags = shlex.split(os.environ.get("FILE_CYCLIC_HOSTED_LDFLAGS", "-lm -lcrypto"))
        self.command("archive-boundary-build", [*compiler, *flags,
                     "tests/nanoisa/test_file_cyclic_hosted_integration.c",
                     *objects, str(archive), *ldflags, "-o", str(executable)])
        stdout = self.command("archive-boundary-run", [str(executable)])
        self.assertIn(b"linked archive cyclic refusal and acyclic scalar controls; no cyclic execution", stdout)
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        (self.artifacts / "linked-inputs-after.json").write_text(json.dumps(after, indent=2) + "\n")
        self.assertEqual(before, after)
        print(stdout.decode(errors="replace").strip(), flush=True)


if __name__ == "__main__":
    unittest.main()
