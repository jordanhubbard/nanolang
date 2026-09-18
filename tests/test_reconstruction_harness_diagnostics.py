"""I retain compiler identity without running a failed compiler input."""
from pathlib import Path
import shlex
import subprocess
import unittest
from unittest import mock
from tests import test_scalar_reconstruction as harness


class HarnessDiagnostics(unittest.TestCase):
    def test_each_compiler_failure_retains_exact_command_and_output(self):
        for stage in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(stage=stage):
                args = [Path('/fixture tools')/stage, Path('/fixture inputs/ordinary.nano'), '-o', '/fixture output/program']
                argv = list(map(str, args))
                result = subprocess.CompletedProcess(argv, 7, 'fixture stdout\n', 'fixture stderr\n')
                case = harness.ScalarReconstruction()
                with mock.patch.object(harness.subprocess, 'run', return_value=result) as run:
                    with self.assertRaises(AssertionError) as error:
                        case.checked(args)
                self.assertIn(shlex.join(argv), str(error.exception))
                self.assertIn('fixture stdout\nfixture stderr', str(error.exception))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0], argv)


if __name__ == '__main__': unittest.main()
