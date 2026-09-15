"""I do not count an unstarted build command as successful work."""
from pathlib import Path
import errno
import os
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MakeTimeouts(unittest.TestCase):
    def run_recipe(self, target, prefix, directory, env=None):
        lines = (ROOT / "Makefile.gnu").read_text().splitlines(keepends=True)
        start = next(i for i, line in enumerate(lines) if line.startswith(target + ":"))
        recipes = []
        current = ""
        for line in lines[start + 1:]:
            if not line.startswith("\t"):
                break
            current += line
            if not line.rstrip().endswith("\\"):
                recipes.append(current)
                current = ""
        selected = [recipe for recipe in recipes if recipe.startswith("\t@" + prefix)]
        self.assertEqual(len(selected), 1)
        return subprocess.run(
            ["make", "--no-print-directory", "-f", "-", "probe", "SHADOW_CHECK_TIMEOUT=2"],
            input="probe:\n" + selected[0], cwd=directory, env=env,
            capture_output=True, text=True, timeout=5,
        )

    def test_nsi_recipe_preserves_compiler_failure_after_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="nano-nsi-recipe-") as tmp:
            directory = Path(tmp)
            (directory / "bin").mkdir()
            (directory / "tests").mkdir()
            compiler = directory / "bin/nanoc"
            compiler.write_text('#!/bin/sh\ntouch tests/nsi_client_bin\nexit "$FAKE_STATUS"\n')
            compiler.chmod(0o700)
            for status in (0, 7):
                with self.subTest(status=status):
                    result = self.run_recipe("test-nsi-runtime", "if", tmp,
                                             dict(os.environ, FAKE_STATUS=str(status)))
                    self.assertEqual(result.returncode == 0, status == 0, result.stderr)
                    self.assertFalse((directory / "tests/nsi_client_bin").exists())

    def test_shadow_recipe_preserves_early_failure(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadow-recipe-") as tmp:
            directory = Path(tmp)
            (directory / "bin").mkdir()
            (directory / "scripts").mkdir()
            git = directory / "bin/git"
            git.write_text('#!/bin/sh\nprintf "%s\\n" first.nano second.nano\nexit "$GIT_STATUS"\n')
            git.chmod(0o700)
            for name in ("first.nano", "second.nano"):
                (directory / name).touch()
            checker = directory / "scripts/check_shadow_tests.sh"
            checker.write_text('echo "$1" >> checked\n'
                               'if [ "$1" = first.nano ]; then exit "$FIRST_STATUS"; fi\n')
            for first_status, git_status in ((0, 0), (7, 0), (0, 9)):
                with self.subTest(first_status=first_status, git_status=git_status):
                    checked = directory / "checked"
                    checked.unlink(missing_ok=True)
                    result = self.run_recipe(
                        "shadow-check", "files=", tmp,
                        dict(os.environ, PATH=str(directory / "bin") + os.pathsep + os.environ["PATH"],
                             FIRST_STATUS=str(first_status), GIT_STATUS=str(git_status)),
                    )
                    self.assertEqual(result.returncode == 0,
                                     first_status == 0 and git_status == 0, result.stderr)
                    if git_status:
                        self.assertFalse(checked.exists())
                    else:
                        expected = ["first.nano"] if first_status else ["first.nano", "second.nano"]
                        self.assertEqual(checked.read_text().splitlines(), expected)

    def test_timeout_wrappers_fail_closed(self):
        makefile = (ROOT / "Makefile.gnu").read_text()
        macros = re.findall(r"^(\w*TIMEOUT_CMD) \?= (.*)$", makefile, re.MULTILINE)
        self.assertEqual(len(macros), 4)
        wrappers = []
        for line_number, line in enumerate(makefile.splitlines(), 1):
            if "perl -e" not in line or "alarm" not in line:
                continue
            match = re.search(r"perl -e '([^']*)'", line)
            self.assertIsNotNone(match, f"I could not inspect timeout recipe at line {line_number}")
            wrappers.append((f"Makefile.gnu:{line_number}", match.group(1)))
        self.assertGreater(len(wrappers), len(macros))
        with tempfile.TemporaryDirectory(prefix="nano-make-timeout-") as tmp:
            denied = Path(tmp) / "not-executable"
            denied.write_text("#!/bin/sh\nexit 0\n")
            for name, value in wrappers:
                program = re.sub(r"\balarm\s+(?:\$\(\w+\)|\d+)", "alarm 1", value)
                expanded = subprocess.run(
                    ["make", "--no-print-directory", "--dry-run", "-f", "-", "probe"],
                    input=f"probe:\n\tperl -e '{program}'\n", text=True,
                    capture_output=True, timeout=3, check=True,
                )
                command = shlex.split(expanded.stdout)
                for args, expected in (([str(Path(tmp) / "missing")], None),
                                       ([str(denied)], None),
                                       (["sh", "-c", "exit 0"], 0),
                                       (["sh", "-c", "exit 7"], 7),
                                       (["perl", "-e", "sleep 5"], None)):
                    with self.subTest(wrapper=name, command=args):
                        result = subprocess.run(command + args, capture_output=True, timeout=3)
                        if expected is None:
                            self.assertNotEqual(result.returncode, 0, result.stderr)
                        else:
                            self.assertEqual(result.returncode, expected, result.stderr)
                        if args[0] in (str(Path(tmp) / "missing"), str(denied)):
                            error = errno.ENOENT if args[0].endswith("/missing") else errno.EACCES
                            self.assertIn(os.strerror(error).encode(), result.stderr)


if __name__ == "__main__":
    unittest.main()
