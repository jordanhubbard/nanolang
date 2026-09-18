"""I keep profile decisions shared without widening ordinary or target admission."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VerifierProfiles(unittest.TestCase):
    def test_shared_admission_and_publication(self):
        entry = '.entry main\n.function main 0 0 0 int 1\n'
        end = 'PUSH_I64 0\nRET\n.end\n'
        cases = {
            'integer': (entry + end, True, True),
            'float_bool_u8_void': (entry + 'PUSH_F64 1.5\nPOP\nPUSH_BOOL 1\nPOP\nPUSH_U8 255\nPOP\nPUSH_VOID\nPOP\n' + end, True, True),
            'implicit': (entry + 'PUSH_I64 0\n.end\n', True, True),
            'advisory_does_not_select': ('.string key "profile"\n.string value "gpu"\n.metadata 0 1\n' + entry + end, True, True),
            'string_opcode': ('.string text "ordinary"\n' + entry + 'PUSH_STR text\nPOP\n' + end, True, False),
            'global_opcode': (entry + 'PUSH_I64 9\nSTORE_GLOBAL 0\n' + end, True, False),
            'import': ('.import "" "get_argc" int\n' + entry + end, True, False),
            'nominal_table': ('.types 1 0 0\n' + entry + end, True, False),
            'nonscalar_parameter': (entry + end + '.function helper 1 1 0 int 1\n.parameters helper string\nPUSH_I64 0\nRET\n.end\n', True, False),
            'initializer': (entry + end + '.function __init__ 0 0 0 void 0\nRET\n.end\n', True, False),
            'no_entry': (entry.replace('.entry main\n', '') + end, True, False),
            'float_entry': ('.entry main\n.function main 0 0 0 float 1\nPUSH_F64 0\nRET\n.end\n', True, False),
        }
        with tempfile.TemporaryDirectory(prefix='nano-profiles-') as tmp:
            work = Path(tmp)
            for name, (assembly, general, scalar) in cases.items():
                with self.subTest(case=name):
                    source, module = work/'input.nasm', work/'input.nvm'
                    source.write_text(assembly)
                    built = subprocess.run([ROOT/'bin/nanoisa', 'asm', source, '-o', module], capture_output=True, text=True, timeout=30)
                    self.assertEqual(built.returncode, 0, built.stderr)
                    probe = subprocess.run([ROOT/'obj/test_verifier_profiles', module], capture_output=True, text=True, timeout=30)
                    self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
                    self.assertEqual(probe.stdout.splitlines()[0], f'{int(general)} {int(scalar)} {int(scalar)} 1 1 1')
                    # Successful Wasm execution belongs to test-nvm2wasm.
                    # Refused inputs must stop before requiring external LLVM tools.
                    tools = [('nvm2llvm', 'll')]
                    if not scalar:
                        tools.append(('nvm2wasm', 'wasm'))
                    for tool, suffix in tools:
                        output = work/f'previous.{suffix}'
                        output.write_bytes(b'previous artifact')
                        result = subprocess.run([ROOT/'bin'/tool, module, '-o', output], capture_output=True, timeout=30)
                        self.assertEqual(result.returncode == 0, scalar, result.stderr)
                        if scalar:
                            self.assertNotEqual(output.read_bytes(), b'previous artifact')
                        else:
                            self.assertEqual(output.read_bytes(), b'previous artifact')
                    self.assertEqual(list(work.glob('.nano-wasm-*')), [])


if __name__ == '__main__':
    unittest.main()
