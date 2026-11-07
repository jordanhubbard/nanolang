"""I refuse empty, inconsistent or contradictory native JUnit evidence."""
from pathlib import Path
import tempfile
import unittest
from scripts.junit_route_audit import read_report


class JUnitEvidence(unittest.TestCase):
    def report(self, tests, failures, body, skipped=0):
        with tempfile.TemporaryDirectory(prefix='nano-junit-evidence-') as directory:
            path = Path(directory) / 'result.xml'
            path.write_text(f'<testsuites tests="{tests}" failures="{failures}" skipped="{skipped}">'
                            f'<testsuite>{body}</testsuite></testsuites>')
            return read_report(path)

    def test_failure_and_skip_remain_nonpassing_case_outcomes(self):
        result = self.report(3, 1, '<testcase classname="a" name="positive"/>'
                             '<testcase classname="b" name="refused"><failure message="original diagnostic"/></testcase>'
                             '<testcase classname="c" name="missing"><skipped/></testcase>', skipped=1)
        self.assertEqual(result['total'], 3)
        self.assertEqual(result['passed'], 1)
        self.assertEqual(result['failure'], 1)
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(result['cases'][1]['diagnostics'], ['original diagnostic'])

    def test_false_success_totals_refuse(self):
        with self.assertRaisesRegex(ValueError, 'totals'):
            self.report(1, 0, '<testcase classname="a" name="failure"><failure/></testcase>')

    def test_empty_or_truncated_corpus_refuses(self):
        for count in (0, 237):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.report(count, 0, '')

    def test_contradictory_case_refuses(self):
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            self.report(1, 1, '<testcase classname="a" name="bad"><failure/><skipped/></testcase>')


if __name__ == '__main__':
    unittest.main()
