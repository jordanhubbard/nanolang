"""I bound GNU tool reports without changing their bytes or refusal limits."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux', 'I exercise Linux GNU tool discovery.')
class GnuToolQueries(unittest.TestCase):
    def query(self, command, milliseconds=None):
        env = dict(os.environ, NANO_TRACE_BUILD='1')
        env.pop('NANO_CAPTURE_TIMEOUT_MS', None)
        if milliseconds is not None:
            env['NANO_CAPTURE_TIMEOUT_MS'] = str(milliseconds)
        return subprocess.run([str(ROOT / 'obj/test_module_generation_probe'), 'gnu-tool-query', command],
                              env=env, capture_output=True, timeout=5)

    def test_slow_report_stops_at_the_configured_capture_deadline(self):
        result = self.query("printf ready; sleep 0.3; printf late", 50)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, b'ready')
        self.assertRegex(result.stderr.decode(),
            r'phase=tool-query-ms expected=\d+ observed=\d+ accepted=0')

    def test_complete_report_preserves_output_and_actual_query_milestone(self):
        result = self.query("printf 'query output\\n'")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b'query output\n')
        self.assertRegex(result.stderr.decode(),
            r'phase=tool-query-ms expected=\d+ observed=\d+ accepted=1')

    def test_original_size_null_and_exit_refusals_remain(self):
        for command, accepted in (("printf '%0254d' 0", True),
                ("printf '%0255d' 0", False), (r"printf 'a\000b'", False),
                ("printf reported; exit 7", False)):
            with self.subTest(command=command):
                result = self.query(command)
                self.assertEqual(result.returncode, 0 if accepted else 1, result.stderr)
                if accepted:
                    self.assertEqual(len(result.stdout), 254)


if __name__ == '__main__':
    unittest.main()
