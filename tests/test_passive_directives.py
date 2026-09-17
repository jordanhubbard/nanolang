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
