"""I retain exact map pairs and detached enumeration through my VM and C AOT."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.test_native_optional_array_reads import OptionalArrayReads

ROOT = Path(__file__).resolve().parents[1]
HEADER = '.entry main\n.string first "first"\n.string second "second"\n.string kept "retained"\n.string changed "changed"\n'
CHURN = ('.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nagain:\n'
         'HM_NEW 5 5\nPUSH_STR first\nPUSH_STR kept\nHM_SET\nPOP\n'
         'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\n'
         'LOAD_LOCAL 0\nPUSH_I64 5000\nI64_LT_S\nJMP_FALSE done\nJMP again\n'
         'done:\nRET\n.end\n')


class ScalarMapAot(unittest.TestCase):
    checked = OptionalArrayReads.checked
    paired = OptionalArrayReads.paired

    def test_all_pairs_empty_growth_optional_writes_and_detached_snapshots(self):
        for key in (1, 5):
            for value in (1, 5):
                with self.subTest(key=key, value=value):
                    push_key = 'PUSH_I64 -9223372036854775808\n' if key == 1 else 'PUSH_STR first\n'
                    push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                    changed = 'PUSH_I64 19\n' if value == 1 else 'PUSH_STR changed\n'
                    body = f'HM_NEW {key} {value}\nSTORE_LOCAL 0\n'
                    for opcode, tag, push in [('HM_KEYS', key, push_key), ('HM_VALUES', value, push_value)]:
                        body += 'LOAD_LOCAL 0\n' + opcode + '\nDUP\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                        body += push + 'ARR_PUSH\nPUSH_I64 0\nARR_GET\nDUP\n'
                        body += f'TYPE_CHECK {tag}\nASSERT\n' + push + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + push_value + 'HM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\nHM_KEYS\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nHM_VALUES\nSTORE_LOCAL 2\n'
                    body += 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nLOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\nHM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + changed + 'HM_SET\nPOP\n'
                    body += 'LOAD_LOCAL 0\n' + push_key + 'HM_GET\n' + changed + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nHM_DELETE\nPOP\n'
                    body += 'LOAD_LOCAL 0\nHM_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n'
                    body += f'HM_NEW {key} {value}\nSTORE_LOCAL 0\nCALL churn\n'
                    for slot, tag, push in [(1, key, push_key), (2, value, push_value)]:
                        body += f'LOAD_LOCAL {slot}\nDUP\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n'
                        body += f'PUSH_I64 0\nARR_GET\nDUP\nTYPE_CHECK {tag}\nASSERT\n' + push + 'EQ\nASSERT\n'
                    self.paired(HEADER + '.function main 0 3 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n' + CHURN)

    def test_forward_map_parameters_and_returned_snapshot_arrays(self):
        for key in (1, 5):
            for value in (1, 5):
                with self.subTest(key=key, value=value):
                    push_key = 'PUSH_I64 9223372036854775807\n' if key == 1 else 'PUSH_STR first\n'
                    push_value = 'PUSH_I64 -7\n' if value == 1 else 'PUSH_STR kept\n'
                    body = f'HM_NEW {key} {value}\n' + push_key + push_value + 'HM_SET\nSTORE_LOCAL 0\n'
                    body += 'LOAD_LOCAL 0\nCALL keys\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nCALL values\nSTORE_LOCAL 2\n'
                    body += f'HM_NEW {key} {value}\nSTORE_LOCAL 0\nCALL churn\n'
                    body += 'LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\n' + push_key + 'EQ\nASSERT\n'
                    body += 'LOAD_LOCAL 2\nPUSH_I64 0\nARR_GET\n' + push_value + 'EQ\nASSERT\n'
                    helpers = ('.function keys 1 1 0 array 1\nLOAD_LOCAL 0\nHM_KEYS\nRET\n.end\n'
                               '.function values 1 1 0 array 1\nLOAD_LOCAL 0\nHM_VALUES\nRET\n.end\n')
                    self.paired(HEADER + '.function main 0 3 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n' + helpers + CHURN)

    def test_known_wrong_pairs_and_nonmap_snapshots_preserve_output(self):
        invalid = ['HM_NEW 4 1\nPOP\n', 'HM_NEW 1 3\nPOP\n',
                   'HM_NEW 5 1\nPUSH_I64 7\nHM_HAS\nPOP\n',
                   'HM_NEW 1 1\nPUSH_STR first\nHM_HAS\nPOP\n',
                   'HM_NEW 1 1\nPUSH_I64 7\nPUSH_STR kept\nHM_SET\nPOP\n',
                   'PUSH_I64 7\nHM_KEYS\nPOP\n', 'PUSH_STR first\nHM_VALUES\nPOP\n']
        with tempfile.TemporaryDirectory(prefix='nano-map-aot-refusals-') as tmp:
            source, module, output = (Path(tmp) / name for name in ('input.nasm', 'input.nvm', 'prior.c'))
            for body in invalid:
                with self.subTest(body=body):
                    source.write_text(HEADER + '.function main 0 0 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n')
                    self.checked([ROOT / 'bin/nanoisa', 'asm', source, '-o', module])
                    output.write_text('I retain prior output.\n')
                    result = subprocess.run([ROOT / 'bin/nvm2c', module, '-o', output],
                                            capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('I ', result.stderr)
                    self.assertEqual(output.read_text(), 'I retain prior output.\n')
