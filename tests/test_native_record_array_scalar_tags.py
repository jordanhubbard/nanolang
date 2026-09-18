"""I preserve plain and boxed present fields through record-array replacement."""
import unittest
from tests import test_native_optional_array_reads as optional

class RecordArrayScalarTags(unittest.TestCase):
    checked = optional.OptionalArrayReads.checked
    paired = optional.OptionalArrayReads.paired

    def test_both_storage_directions_and_alias_observation(self):
        for tag, first, second in [(1, 'PUSH_I64 17', 'PUSH_I64 42'),
                                   (4, 'PUSH_BOOL 0', 'PUSH_BOOL 1'),
                                   (5, 'PUSH_STR first', 'PUSH_STR second')]:
            for boxed_first in (False, True):
                with self.subTest(tag=tag, boxed_first=boxed_first):
                    initial = first + (f'\nARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET' if boxed_first else '')
                    replacement = second + ('' if boxed_first else f'\nARR_LITERAL {tag} 1\nPUSH_I64 0\nARR_GET')
                    body = initial + '\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n'
                    body += 'LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nPUSH_I64 0\n'
                    body += replacement + '\nAGG_PACK 0 0 0 1\nARR_SET\nPOP\n'
                    body += 'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n'
                    body += f'DUP\nTYPE_CHECK {tag}\nASSERT\n{second}\nEQ\nASSERT\n'
                    self.paired('.types 1 0 0\n.string first "before"\n.string second "after"\n.entry main\n.function main 0 2 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n')

if __name__ == '__main__':
    unittest.main()
