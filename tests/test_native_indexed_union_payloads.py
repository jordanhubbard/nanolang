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
            'array': 'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nCALL array_bridge\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nRET\n',
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
                        ('.function array_bridge 1 1 0 array 1\nLOAD_LOCAL 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_BOOL 1\nJMP_FALSE other\nJMP joined\nother:\nNOP\njoined:\nTAIL_CALL array_identity\n.end\n' +
                        '.function array_identity 1 1 0 array 1\nLOAD_LOCAL 0\nRET\n.end\n' if route == 'array' else '') +
                        '.function read 1 1 0 string 1\nLOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE text\n'
                        'AGG_GET 0\nAGG_GET 0\nRET\ntext:\nDUP\nAGG_TAG\nPUSH_I64 1\nNE\nJMP_TRUE empty\n'
                        'AGG_GET 0\nRET\nempty:\nPOP\nPUSH_STR kept\nRET\n.end\n')

    def test_nested_record_fields_and_shared_values(self):
        for depth in (1, 3):
            for reverse in (False, True):
                for route in ('local', 'global', 'tail'):
                    with self.subTest(depth=depth, reverse=reverse, route=route):
                        values = [('PUSH_STR kept\nPUSH_STR truth\nSTR_CONCAT\nAGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\n',
                                   'PUSH_STR kept\nPUSH_STR truth\nSTR_CONCAT\n'),
                                  ('PUSH_STR truth\nAGG_PACK 1 0 1 1\n', 'PUSH_STR truth\n')]
                        if reverse: values.reverse()
                        values.append(('AGG_PACK 1 0 2 0\n', 'PUSH_STR kept\n'))
                        wrap = 'DUP\nAGG_PACK 0 0 0 2\n' + 'AGG_PACK 0 0 0 1\n' * (depth - 1)
                        body = ''.join(value + wrap + 'CALL relay\nCALL unwrap\nCALL read\n' + expected + 'EQ\nASSERT\n'
                                       for value, expected in values)
                        relay = {'local': 'LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nRET\n',
                                 'global': 'LOAD_LOCAL 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n',
                                 'tail': 'LOAD_LOCAL 0\nTAIL_CALL identity\n'}[route]
                        self.paired(HEADER + '.function main 0 0 0 int 1\n' + body + 'PUSH_I64 0\nRET\n.end\n' +
                            '.function relay 1 2 0 struct 1\n' + relay + '.end\n' +
                            '.function identity 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n' +
                            '.function unwrap 1 2 0 union 1\nLOAD_LOCAL 0\n' + 'AGG_GET 0\n' * (depth - 1) +
                            'DUP\nAGG_GET 1\nCALL read\nSTORE_LOCAL 1\nAGG_GET 0\nDUP\nCALL read\nLOAD_LOCAL 1\nEQ\nASSERT\nRET\n.end\n' +
                            '.function read 1 1 0 string 1\nLOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE text\n'
                            'AGG_GET 0\nAGG_GET 0\nRET\ntext:\nDUP\nAGG_TAG\nPUSH_I64 1\nEQ\nJMP_FALSE empty\n'
                            'AGG_GET 0\nRET\nempty:\nPOP\nPUSH_STR kept\nRET\n.end\n')

    def test_recursive_constructor_facts_converge(self):
        unpack = ''.join(f'DUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE bad\nAGG_GET 0\n' for _ in range(3))
        self.paired(HEADER + '.function main 0 2 0 int 1\nAGG_PACK 1 0 1 0\nSTORE_LOCAL 0\n'
            'PUSH_I64 0\nSTORE_LOCAL 1\nloop:\nLOAD_LOCAL 1\nPUSH_I64 3\nLT\nJMP_FALSE done\n'
            'LOAD_LOCAL 0\nCALL wrap\nSTORE_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 1\nJMP loop\n'
            'done:\nLOAD_LOCAL 0\n' + unpack + 'AGG_TAG\nPUSH_I64 1\nEQ\nASSERT\nPUSH_I64 0\nRET\n'
            'bad:\nPOP\nPUSH_BOOL 0\nASSERT\nPUSH_I64 1\nRET\n.end\n'
            '.function wrap 1 1 0 union 1\nLOAD_LOCAL 0\nAGG_PACK 1 0 0 1\nRET\n.end\n')

    def test_invalid_selected_payloads_preserve_output(self):
        consumers = {
            'unguarded': 'LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nRET\n',
            'wrong_variant': 'LOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 1\nEQ\nJMP_FALSE other\nAGG_GET 0\nAGG_GET 0\nRET\nother:\nPOP\nPUSH_I64 0\nRET\n',
            'wrong_field': 'LOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_FALSE other\nAGG_GET 0\nAGG_GET 0\nSTR_LEN\nRET\nother:\nPOP\nPUSH_I64 0\nRET\n',
            'bypass': 'LOAD_LOCAL 0\nDUP\nAGG_TAG\nPUSH_I64 0\nEQ\nJMP_TRUE selected\nJMP joined\nselected:\nJMP joined\njoined:\nAGG_GET 0\nAGG_GET 0\nRET\n',
        }
        cases = [(name, consumer, nested) for name, consumer in consumers.items() for nested in (False, True, 'array')]
        cases.extend(('unknown_parent', consumers['wrong_variant'].replace('PUSH_I64 1', 'PUSH_I64 0'), nested) for nested in (True, 'array'))
        cases.append(('unknown_nested_array_write', consumers['wrong_variant'].replace('PUSH_I64 1', 'PUSH_I64 0'), 'array'))
        for name, consumer, nested in cases:
            with self.subTest(case=name, nested=nested), tempfile.TemporaryDirectory(prefix='indexed-union-refusal-') as tmp:
                root = Path(tmp)
                assembly, module, output = root/'input.nasm', root/'input.nvm', root/'output.c'
                wrap = 'AGG_PACK 0 0 0 1\n' if nested else ''
                if nested: consumer = consumer.replace('LOAD_LOCAL 0\n', 'LOAD_LOCAL 0\nAGG_GET 0\n')
                if nested == 'array':
                    wrap += 'ARR_LITERAL 8 1\n'
                    consumer = consumer.replace('LOAD_LOCAL 0\n', 'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n')
                if name == 'unknown_nested_array_write':
                    wrap += 'AGG_PACK 0 0 0 1\n'
                    consumer = consumer.replace('LOAD_LOCAL 0\n',
                        'LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nPUSH_I64 9\n'
                        'AGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\nAGG_PACK 0 0 0 1\nARR_SET\n')
                address = 'FUNCREF read\nPOP\n' if name.startswith('unknown_') else ''
                assembly.write_text(HEADER + '.function main 0 0 0 int 1\n' + address +
                    'PUSH_I64 7\nAGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\n' + wrap + 'CALL read\nPOP\n' +
                    'PUSH_STR kept\nAGG_PACK 1 0 1 1\n' + wrap + 'CALL read\nPOP\nPUSH_I64 0\nRET\n.end\n' +
                    '.function read 1 1 0 int 1\n' + consumer + '.end\n')
                self.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                output.write_text('previous')
                result = subprocess.run([ROOT/'bin/nvm2c', module, '-o', output],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertTrue('selected constructor' in result.stderr or
                                'shape' in result.stderr, result.stderr)
                self.assertEqual(output.read_text(), 'previous')
