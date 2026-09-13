"""I do not count an unstarted build command as successful work."""
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MakeTimeouts(unittest.TestCase):
    def test_timeout_wrappers_fail_closed(self):
        makefile = (ROOT / "Makefile.gnu").read_text()
        wrappers = re.findall(r"^(\w*TIMEOUT_CMD) \?= (.*)$", makefile, re.MULTILINE)
        self.assertEqual(len(wrappers), 4)
        with tempfile.TemporaryDirectory(prefix="nano-make-timeout-") as tmp:
            denied = Path(tmp) / "not-executable"
            denied.write_text("#!/bin/sh\nexit 0\n")
            for name, value in wrappers:
                command = shlex.split(re.sub(r"\$\(\w+\)", "1", value).replace("$$", "$"))
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


if __name__ == "__main__":
    unittest.main()
