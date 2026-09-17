"""I resolve producer markers without duplicating instruction sizing in frontends."""
from pathlib import Path
import tempfile
import unittest
import tests.test_passive_inputs as input_tests


class PassiveDirectives(unittest.TestCase):
    command = input_tests.PassiveInputs.command
    assemble = input_tests.PassiveInputs.assemble
    paired_roundtrip = input_tests.PassiveInputs.paired_roundtrip

    def test_markers_match_exact_guarded_record(self):
        raw = input_tests.fixture()
        marked = raw[:raw.index('.passive')]
        for i in range(4):
            code = f'LOAD_LOCAL {i}\nSTORE_LOCAL {4+i}\n'
            marked = marked.replace(code, f'.par_node {4+i} {i}\n' + code)
        marked = marked.replace('.par_node 4', '.par_begin\n.par_node 4')
        marked = marked.replace('STORE_LOCAL 7\n', 'STORE_LOCAL 7\n.par_end\n')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            expected = self.assemble(directory, raw).read_bytes()
            actual = self.assemble(directory, marked).read_bytes()
            self.assertEqual(actual, expected)
        self.paired_roundtrip(marked, b'42\ntrue\nguarded\n1.5\n')

    def test_multiple_blocks_and_function_offsets(self):
        source = '.entry 2\n.function first 0 1 0 int 1\n'
        source += '.par_begin\n.par_node 0\nPUSH_I64 42\nSTORE_LOCAL 0\n.par_end\n'
        source += 'LOAD_LOCAL 0\nRET\n.end\n'
        source += '.function second 0 2 0 int 1\n'
        source += '.par_begin\n.par_node 0\nPUSH_I64 11\nSTORE_LOCAL 0\n.par_end\n'
        source += '.par_begin\n.par_node 1\nPUSH_I64 31\nSTORE_LOCAL 1\n.par_end\n'
        source += 'LOAD_LOCAL 0\nLOAD_LOCAL 1\nADD\nRET\n.end\n'
        source += '.function main 0 0 0 int 1\nCALL first\nPRINTLN\nCALL second\nPRINTLN\nPUSH_I64 0\nRET\n.end\n'
        self.paired_roundtrip(source, b'42\n42\n')

    def test_flow_forward_dependency_matches_exact_record(self):
        import struct
        source = ('.entry 1\n.function helper 1 3 0 int 1\n'
                  'LOAD_LOCAL 0\nTYPE_CHECK 1\nASSERT\nJMP entry\nentry:\n'
                  '.flow_begin 2\n.flow_node 1 1 0 1 0\n'
                  'LOAD_LOCAL 0\nSTORE_LOCAL 1\n'
                  '.flow_node 0 2 1 1 0\n'
                  'LOAD_LOCAL 1\nPUSH_I64 1\nADD\nSTORE_LOCAL 2\n.flow_end\n'
                  'LOAD_LOCAL 2\nPRINTLN\nPUSH_I64 0\nRET\n.end\n'
                  '.parameters 0 int\n.function main 0 0 0 int 1\n'
                  'PUSH_I64 42\nCALL helper\nRET\n.end\n')
        words = [2, 1, 2, 0, 11, 33, 2,
                 17, 33, 2, 1, 0, 0, 0, 1,
                 11, 17, 1, 0, 1, 0, 0, 0]
        raw = ''.join(line + '\n' for line in source.splitlines() if not line.startswith('.flow_'))
        raw += '.passive "' + struct.pack('<' + 'I'*len(words), *words).hex() + '"\n'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            expected = self.assemble(directory, raw).read_bytes()
            self.assertEqual(self.assemble(directory, source).read_bytes(), expected)
        self.paired_roundtrip(source, b'43\n')

    def test_flow_diamond_keeps_source_ids_and_closed_calls(self):
        source = ('.entry main\n.function square 1 1 0 int 1\n'
                  'LOAD_LOCAL 0\nDUP\nI64_MUL\nRET\n.end\n'
                  '.function main 0 4 0 int 1\n.flow_begin 4\n'
                  '.flow_node 3 3 0 0\nPUSH_I64 3\nSTORE_LOCAL 3\n'
                  '.flow_node 1 1 1 3 0\nLOAD_LOCAL 3\nCALL square\nSTORE_LOCAL 1\n'
                  '.flow_node 2 2 1 3 0\nLOAD_LOCAL 3\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 2\n'
                  '.flow_node 0 0 2 1 2 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nI64_ADD\nSTORE_LOCAL 0\n'
                  '.flow_end\nLOAD_LOCAL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n')
        self.paired_roundtrip(source, b'13\n')

    def test_flow_and_par_blocks_remain_distinct(self):
        source = ('.entry main\n.function main 0 4 0 int 1\n'
                  '.par_begin\n.par_node 0\nPUSH_I64 10\nSTORE_LOCAL 0\n.par_end\n'
                  '.flow_begin 2\n.flow_node 1 1 0 0\nPUSH_I64 20\nSTORE_LOCAL 1\n'
                  '.flow_node 0 2 1 1 0\nLOAD_LOCAL 1\nPUSH_I64 2\nADD\nSTORE_LOCAL 2\n.flow_end\n'
                  '.flow_begin 1\n.flow_node 0 3 0 0\nPUSH_I64 10\nSTORE_LOCAL 3\n.flow_end\n'
                  'LOAD_LOCAL 0\nLOAD_LOCAL 2\nADD\nLOAD_LOCAL 3\nADD\nPRINTLN\n'
                  'PUSH_I64 0\nRET\n.end\n')
        self.paired_roundtrip(source, b'42\n')

    def test_flow_text_requires_complete_distinct_nodes(self):
        body = '.function main 0 2 0 int 1\n'
        node = '.flow_node 0 0 0 0\nPUSH_I64 42\nSTORE_LOCAL 0\n'
        finish = 'LOAD_LOCAL 0\nRET\n.end\n'
        cases = ['.flow_begin 1\n', body + '.flow_end\n' + finish,
                 body + '.flow_begin 0\n' + finish,
                 body + '.flow_begin 1 extra\n' + finish,
                 body + '.flow_begin 1\n' + node + finish,
                 body + '.flow_begin 2\n' + node + '.flow_end\n' + finish,
                 body + '.flow_begin 1\n' + node + node + '.flow_end\n' + finish,
                 body + '.flow_begin 1\n.par_node 0\n' + finish,
                 body + '.par_begin\n' + node + finish,
                 body + '.flow_begin 1\n.flow_node 0 0 invalid\n' + finish]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for i, source in enumerate(cases):
                with self.subTest(case=i):
                    text, module = directory/'source.nasm', directory/'source.nvm'
                    text.write_text(source)
                    result = self.command(input_tests.ROOT/'bin/nanoisa', 'asm', text, '-o', module)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(module.exists())

    def test_incomplete_or_mixed_markers_are_refused(self):
        function = '.function main 0 1 0 int 1\n'
        node = '.par_node 0\nPUSH_I64 42\nSTORE_LOCAL 0\n'
        finish = 'LOAD_LOCAL 0\nRET\n.end\n'
        cases = [
            '.par_begin\n',
            function + '.par_node 0\n' + finish,
            function + '.par_end\n' + finish,
            function + '.par_begin\n.par_begin\n' + node + '.par_end\n' + finish,
            function + '.par_begin\n.par_end\n' + finish,
            function + '.par_begin\n' + node + finish,
            function + '.par_begin\n.par_node invalid\n' + finish,
            function + '.par_begin extra\n' + node + '.par_end\n' + finish,
            function + '.par_begin\n' + node + '.par_end extra\n' + finish,
            '.passive "00"\n' + function + '.par_begin\n' + node + '.par_end\n' + finish,
            function + '.par_begin\n' + node + '.par_end\n' + finish + '.passive "00"\n',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for i, source in enumerate(cases):
                with self.subTest(case=i):
                    text, module = directory/'source.nasm', directory/'source.nvm'
                    text.write_text(source)
                    result = self.command(Path(__file__).resolve().parents[1]/'bin/nanoisa',
                                          'asm', text, '-o', module)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(module.exists())


if __name__ == '__main__':
    unittest.main()
