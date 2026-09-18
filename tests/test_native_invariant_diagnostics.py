"""I retain useful generated-source evidence without changing invariant guards."""
import re
import signal
import tempfile
import unittest
from pathlib import Path

from tests import test_native_floats as floats

ROOT = floats.ROOT


class NativeInvariantDiagnostics(unittest.TestCase):
    run_command = floats.NativeFloats.run_command
    checked = floats.NativeFloats.checked
    assemble = floats.NativeFloats.assemble
    native = floats.NativeFloats.native

    def test_failed_helper_assertion_names_its_generated_source(self):
        helper = ('.function rejected 0 0 0 int 1\n'
                  'PUSH_BOOL 0\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-invariant-diagnostic-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, 'CALL rejected\nPOP\n', helper)
            vm = self.run_command([ROOT / 'bin/nano_vm', module])
            self.assertGreater(vm.returncode, 0, vm.stderr)
            binary = self.native(work, module, sanitize=True)
            result = self.run_command([binary])
            self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr)
            match = re.fullmatch(
                r'I stopped at a native invariant in ([A-Za-z_][A-Za-z_0-9]*) '
                r'at generated C line ([0-9]+)\.\n', result.stderr)
            self.assertIsNotNone(match, result.stderr)
            function, line = match.groups()
            self.assertIn('rejected', function)
            source = (work / 'input.c').read_text().splitlines()
            self.assertIn('NVM2C_ABORT();', source[int(line) - 1])
            self.assertEqual(result.stdout, '')

    def test_successful_assertion_keeps_output_and_silent_stderr(self):
        with tempfile.TemporaryDirectory(prefix='nano-invariant-success-') as tmp:
            work = Path(tmp)
            module = self.assemble(work, 'PUSH_BOOL 1\nASSERT\nPUSH_I64 17\nPRINTLN\n')
            vm = self.checked([ROOT / 'bin/nano_vm', module])
            native = self.checked([self.native(work, module, sanitize=True)])
            self.assertEqual(native.stdout, vm.stdout)
            self.assertEqual(native.stderr, '')


if __name__ == '__main__':
    unittest.main()
