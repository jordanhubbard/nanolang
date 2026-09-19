"""I retain exact byte tags while reconstructing explicit scalar transport."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
from tests import test_reconstructed_integer_addition as addition

ROOT = Path(__file__).resolve().parents[1]
SHADOWS = '''shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }
shadow nlr_f1_identity { let value: u8 = 255 assert (== (cast_int (nlr_f1_identity value)) 255) }
shadow nlr_f2_choose { assert (== (cast_int (nlr_f2_choose false)) 128) assert (== (cast_int (nlr_f2_choose true)) 255) }
'''


class U8Reconstruction(unittest.TestCase):
    checked = addition.IntegerReconstruction.checked
    assemble = addition.IntegerReconstruction.assemble
    inspect_sources = False

    def paired(self, text, expected, shadows):
        with tempfile.TemporaryDirectory(prefix='nano-reconstruct-u8-') as tmp:
            directory = Path(tmp); module = self.assemble(directory, text)
            original = module.read_bytes()
            self.checked([ROOT/'bin/nano_vm', module], expected)
            for language in ('c', 'nano'):
                output = directory/('result.'+language)
                self.checked([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
                source = output.read_text()
                for forbidden in (r'\bgoto\b', r'nano_vm', r'nvm_blob', r'\bswitch\b', r'\bdispatch\b'):
                    self.assertNotRegex(source, forbidden)
                self.assertIn('return', source)
                if language == 'c':
                    executable = directory/'c-program'
                    self.checked(shlex.split(os.environ.get('CC', 'cc'))+['-std=c11', '-O1', '-Wall', '-Wextra',
                                 '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                 output, '-o', executable])
                    self.checked([executable], expected)
                    continue
                output.write_text(source+'\n'+shadows+f'\nshadow main {{ assert (== (main) {expected}) }}\n')
                for producer in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                    compiler = (ROOT/'bin'/producer if producer == 'nano_virt' else
                                Path(os.environ.get('NANO_HL_COMPILER_DIR', ROOT/'bin'))/producer)
                    recovered = directory/(producer+'.nvm')
                    self.checked([compiler, output, '--emit-nvm', '-o', recovered])
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', recovered])
                    self.checked([ROOT/'bin/nano_vm', recovered], expected)
                    c_source = directory/(producer+'.c'); native = directory/(producer+'-native')
                    self.checked([ROOT/'bin/nvm2c', recovered, '-o', c_source])
                    self.checked(shlex.split(os.environ.get('CC', 'cc'))+['-std=c11', '-O1', '-Wall', '-Wextra',
                                 '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                 c_source, '-o', native])
                    self.checked([native], expected)
                native = directory/'nanoc-c-native'
                self.checked([ROOT/'bin/nanoc_c', output, '-o', native])
                self.checked([native], expected)
            self.assertEqual(module.read_bytes(), original)

    def checked(self, args, expected=0):
        result = addition.IntegerReconstruction.checked(self, args, expected)
        if self.inspect_sources and Path(args[0]).name == 'nvm2hl':
            source = Path(args[-1]).read_text()
            if '--language' in args and args[args.index('--language') + 1] == 'c':
                self.assertIn('uint8_t', source)
                self.assertIn('UINT8_C(255)', source)
                self.assertIn('(int64_t)', source)
            else:
                self.assertIn(': u8', source)
                self.assertIn('(cast_int ', source)
        return result

    def test_constants_locals_calls_branches_and_explicit_casts(self):
        checks = ''
        for value in (0, 1, 127, 128, 254, 255):
            checks += (f'PUSH_U8 {value}\nCALL identity\nCAST_INT\nPUSH_I64 {value}\n'
                       'I64_EQ\nJMP_FALSE bad\n')
        text = '''.entry main
.function main 0 1 0 int 1
''' + checks + '''PUSH_U8 255
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_U8 1
STORE_LOCAL 0
CAST_INT
PUSH_I64 255
I64_EQ
JMP_FALSE bad
PUSH_U8 0
CAST_BOOL
JMP_TRUE bad
PUSH_U8 255
CAST_BOOL
JMP_FALSE bad
PUSH_BOOL 0
CALL choose
CAST_INT
PUSH_I64 128
I64_EQ
JMP_FALSE bad
PUSH_BOOL 1
CALL choose
CAST_INT
PUSH_I64 255
I64_EQ
JMP_FALSE bad
PUSH_I64 0
RET
bad:
PUSH_I64 1
RET
.end
.function identity 1 1 0 u8 1
.parameters identity u8
LOAD_LOCAL 0
RET
.end
.function choose 1 1 0 u8 1
.parameters choose bool
LOAD_LOCAL 0
JMP_FALSE low
PUSH_U8 255
RET
low:
PUSH_U8 128
RET
.end
'''
        self.inspect_sources = True
        try:
            self.paired(text, 0, SHADOWS)
        finally:
            self.inspect_sources = False

    def test_cast_bool_controls_a_pretest_loop(self):
        text = '''.entry main
.function main 0 1 0 int 1
PUSH_U8 255
STORE_LOCAL 0
loop:
LOAD_LOCAL 0
CAST_BOOL
JMP_FALSE done
PUSH_U8 0
STORE_LOCAL 0
JMP loop
done:
LOAD_LOCAL 0
CAST_INT
RET
.end
'''
        self.paired(text, 0, 'shadow nlr_f0_main { assert (== (nlr_f0_main) 0) }')

    def test_u8_entry_result_preserves_prior_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-hl-u8-refusal-') as tmp:
            directory = Path(tmp)
            module = self.assemble(directory, '.entry main\n.function main 0 0 0 u8 1\nPUSH_U8 7\nRET\n.end\n')
            for language in ('c', 'nano'):
                output = directory/('previous.'+language); output.write_text('previous')
                result = self.checked([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output], 1)
                self.assertIn('I did not publish reconstructed source', result.stderr)
                self.assertEqual(output.read_text(), 'previous')

    def test_contextual_source_literals_keep_exact_byte_tags(self):
        source_text = '''fn identity(value: u8) -> u8 { return value }
shadow identity { let value: u8 = 255 assert (== (cast_int (identity value)) 255) }
fn choose(flag: bool) -> u8 { if flag { return 128 } return 254 }
shadow choose { assert (== (cast_int (choose true)) 128) assert (== (cast_int (choose false)) 254) }
fn main() -> int {
    let zero: u8 = 0
    let mut value: u8 = 1
    set value 127
    let direct: u8 = (identity 255)
    let low: u8 = (choose true)
    let high: u8 = (choose false)
    assert (== (cast_int direct) 255)
    assert (== (cast_int low) 128)
    assert (== (cast_int high) 254)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-u8-source-') as tmp:
            directory = Path(tmp); source = directory/'literal.nano'; source.write_text(source_text)
            for producer in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(producer=producer):
                    module = directory/(producer+'.nvm')
                    self.checked([ROOT/'bin'/producer, source, '--emit-nvm', '-o', module])
                    dump = self.checked([ROOT/'bin/nanoisa', 'dump', module]).stdout
                    for value in (0, 1, 127, 128, 254, 255):
                        self.assertIn(f'PUSH_U8 {value}', dump)
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', module])
                    self.checked([ROOT/'bin/nano_vm', module])
                    c_source = directory/(producer+'.c'); executable = directory/(producer+'-native')
                    self.checked([ROOT/'bin/nvm2c', module, '-o', c_source])
                    self.checked(shlex.split(os.environ.get('CC', 'cc'))+['-std=c11', '-O1', '-Wall', '-Wextra',
                                 '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                 c_source, '-o', executable])
                    self.checked([executable])
            for compiler in ('nanoc_c',):
                with self.subTest(compiler=compiler):
                    executable = directory/(compiler+'-legacy')
                    self.checked([ROOT/'bin'/compiler, source, '-o', executable])
                    self.checked([executable])

    def test_contextual_source_literal_refusals_preserve_output(self):
        cases = ('256', '(- 0 1)', '(+ 127 1)')
        with tempfile.TemporaryDirectory(prefix='nano-u8-source-refusal-') as tmp:
            directory = Path(tmp); source = directory/'wrong.nano'; output = directory/'previous.nvm'
            for expression in cases:
                source.write_text(f'fn octet() -> u8 {{ return {expression} }}\nshadow octet {{ assert true }}\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n')
                for producer in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                    with self.subTest(producer=producer, expression=expression):
                        output.write_bytes(b'previous')
                        result = subprocess.run([ROOT/'bin'/producer, source, '--emit-nvm', '-o', output],
                                                cwd=ROOT, capture_output=True, text=True, timeout=180)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertRegex(result.stdout+result.stderr, '(?i)(u8|byte)')
                        self.assertEqual(output.read_bytes(), b'previous')


if __name__ == '__main__':
    unittest.main()
