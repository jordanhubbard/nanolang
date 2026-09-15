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
