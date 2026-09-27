"""I select managed union payload layouts only from proved constructor values."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_native_variant_scalar_carriers as scalar_carriers

HEADER = scalar_carriers.HEADER

ROOT = Path(__file__).resolve().parents[1]


class IndexedUnionPayloads(unittest.TestCase):
    checked = scalar_carriers.VariantScalarCarriers.checked
    paired = scalar_carriers.VariantScalarCarriers.paired

    def test_record_scalar_and_unit_through_storage_routes(self):
        routes = {
            'local': 'LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nRET\n',
            'global': 'LOAD_LOCAL 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n',
            'tail': 'LOAD_LOCAL 0\nTAIL_CALL identity\n',
            'join': 'LOAD_LOCAL 0\nPUSH_BOOL 1\nJMP_FALSE other\nJMP joined\nother:\nNOP\njoined:\nRET\n',
        }
        for route, relay in routes.items():
            for reverse in (False, True):
                with self.subTest(route=route, reverse=reverse):
                    record = 'PUSH_STR kept\nPUSH_STR truth\nSTR_CONCAT\nAGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\nCALL relay\nCALL read\nPUSH_STR kept\nPUSH_STR truth\nSTR_CONCAT\nEQ\nASSERT\n'
                    scalar = 'PUSH_STR truth\nAGG_PACK 1 0 1 1\nCALL relay\nCALL read\nPUSH_STR truth\nEQ\nASSERT\n'
                    body = scalar + record if reverse else record + scalar
                    self.paired(HEADER + '.function main 0 0 0 int 1\n' + body +
                        'AGG_PACK 1 0 2 0\nCALL relay\nCALL read\nPUSH_STR kept\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n' +
                        '.function relay 1 2 0 union 1\n' + relay + '.end\n' +
                        '.function identity 1 1 0 union 1\nLOAD_LOCAL 0\nRET\n.end\n' +
                        '.function read 1 1 0 string 1\nLOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE text\n'
                        'AGG_GET 0\nAGG_GET 0\nRET\ntext:\nDUP\nAGG_TAG\nPUSH_I64 1\nNE\nJMP_TRUE empty\n'
                        'AGG_GET 0\nRET\nempty:\nPOP\nPUSH_STR kept\nRET\n.end\n')

    def test_invalid_selected_payloads_preserve_output(self):
        consumers = {
            'unguarded': 'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nRET\n',
            'wrong_variant': 'LOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 1\nEQ\nJMP_FALSE other\nAGG_GET 0\nAGG_GET 0\nRET\nother:\nPOP\nPUSH_I64 0\nRET\n',
            'wrong_field': 'LOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE other\nAGG_GET 0\nAGG_GET 0\nSTR_LEN\nRET\nother:\nPOP\nPUSH_I64 0\nRET\n',
            'bypass': 'LOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_TRUE selected\nJMP joined\nselected:\nJMP joined\njoined:\nAGG_GET 0\nAGG_GET 0\nRET\n',
        }
        for name, consumer in consumers.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='indexed-union-refusal-') as tmp:
                root = Path(tmp)
                assembly, module, output = root/'input.nasm', root/'input.nvm', root/'output.c'
                assembly.write_text(HEADER + '.function main 0 0 0 int 1\n'
                    'PUSH_I64 7\nAGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\nCALL read\nPOP\n'
                    'PUSH_STR kept\nAGG_PACK 1 0 1 1\nCALL read\nPOP\nPUSH_I64 0\nRET\n.end\n'
                    '.function read 1 1 0 int 1\n' + consumer + '.end\n')
                self.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                output.write_text('previous')
                result = subprocess.run([ROOT/'bin/nvm2c', module, '-o', output],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertTrue('selected constructor' in result.stderr or
                                'shape' in result.stderr, result.stderr)
                self.assertEqual(output.read_text(), 'previous')
