"""I retain leak checks where the sanitizer runtime supports them."""
import sys
import unittest
from unittest import mock
from pathlib import Path

from tests.sanitizer_options import asan_options


class SanitizerOptions(unittest.TestCase):
    def test_python_tests_do_not_force_leak_detection(self):
        root = Path(__file__).resolve().parent
        offenders = [str(path.relative_to(root)) for path in root.rglob("*.py")
                     if path != Path(__file__) and
                     "detect_leaks=1" in path.read_text()]
        self.assertEqual(offenders, [])

    def test_darwin_disables_unsupported_leak_detection(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(asan_options("halt_on_error=1"),
                             "detect_leaks=0:halt_on_error=1")

    def test_supported_hosts_retain_leak_detection(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(asan_options("abort_on_error=1"),
                             "detect_leaks=1:abort_on_error=1")


if __name__ == "__main__":
    unittest.main()
