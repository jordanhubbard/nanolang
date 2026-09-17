"""I preserve runtime tags when projected and returned strings meet."""
import tempfile
from pathlib import Path
import unittest
from tests import test_native_map_globals as maps
from tests.test_native_map_lifetimes import HELPERS


class NativeStringJoins(unittest.TestCase):
    command = maps.NativeMapGlobals.command
    check = maps.NativeMapGlobals.check

    def test_string_and_missing_branches_preserve_tags_and_roots(self):
        for first_tagged in (False, True):
            for condition in (False, True):
                for present in (False, True):
                    with self.subTest(first_tagged=first_tagged, condition=condition, present=present):
                        tagged = 'HM_NEW 5 5\n'
                        if present:
                            tagged += 'PUSH_STR key\nPUSH_STR text\nHM_SET\n'
                        tagged += 'PUSH_STR key\nHM_GET\n'
                        direct = 'CALL text_result\n'
                        first, second = (tagged, direct) if first_tagged else (direct, tagged)
                        selected_tagged = condition == first_tagged
                        expected_tag = 0 if selected_tagged and not present else 5
                        body = 'PUSH_I64 42\nPUSH_BOOL ' + str(int(condition)) + '\nJMP_FALSE other\n'
                        body += first + 'JMP join\nother:\n' + second + 'join:\n'
                        # An unrelated lower operand survives, and the only
                        # owner of the selected copied text is the join slot.
                        body += 'CALL churn\nDUP\nTYPE_CHECK ' + str(expected_tag) + '\nASSERT\n'
                        body += 'PUSH_STR text\nEQ\nASSERT\n' if expected_tag == 5 else 'POP\n'
                        body += 'PUSH_I64 42\nEQ\nASSERT\n'
                        self.check(body, HELPERS +
                                   '.function text_result 0 0 0 string 1\nPUSH_STR text\nRET\n.end\n')

    def test_conditional_edge_does_not_rewrite_fallthrough_stack(self):
        for condition in (False, True):
            with self.subTest(condition=condition):
                self.check('PUSH_STR text\nPUSH_BOOL ' + str(int(condition)) + '\nJMP_FALSE join\n'
                           'POP\nHM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPUSH_STR key\nHM_GET\n'
                           'join:\nCALL churn\nPUSH_STR text\nEQ\nASSERT\n', HELPERS)

    def test_backward_widening_remains_explicitly_refused(self):
        with tempfile.TemporaryDirectory(prefix='nano-loop-join-') as tmp:
            module, source = Path(tmp)/'loop.nvm', Path(tmp)/'loop.c'
            assembly = maps.ROOT/'tests/nanoisa/fixtures/tagged_string_loop_join.nasm'
            for command in ([maps.ROOT/'bin/nanoisa', 'asm', assembly, '-o', module],
                            [maps.ROOT/'bin/nano_vm', module]):
                result = self.command(command)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rejected = self.command([maps.ROOT/'bin/nvm2c', module, '-o', source])
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('backward stack edge', rejected.stderr)
            self.assertFalse(source.exists())

    def test_join_preserves_string_payload_through_return(self):
        self.check('PUSH_BOOL 1\nCALL choose\nPUSH_STR text\nEQ\nASSERT\n'
                   'PUSH_BOOL 0\nCALL choose\nPUSH_STR text\nEQ\nASSERT\n',
                   '.function choose 1 1 0 string 1\nLOAD_LOCAL 0\nJMP_FALSE other\n'
                   'HM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPUSH_STR key\nHM_GET\nJMP join\n'
                   'other:\nPUSH_STR text\njoin:\nRET\n.end\n')


if __name__ == '__main__':
    unittest.main()
