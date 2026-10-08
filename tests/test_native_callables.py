"""I check native callable dispatch, tagged refusal and aggregate lifetimes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeCallables(unittest.TestCase):
    def checked(self, command, **kwargs):
        result = subprocess.run([str(x) for x in command], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def emit(self, work, text):
        assembly, module, source = (work / name for name in ('input.nasm', 'input.nvm', 'input.c'))
        assembly.write_text(text)
        self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
        self.checked([ROOT / 'bin/nano_vm', module])
        self.checked([ROOT / 'bin/nvm2c', module, '-o', source])
        return source

    def sanitized(self, source, binary):
        self.checked(['cc', '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
        self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def test_many_targets_share_exclusive_dispatch_temporaries(self):
        branches = ''.join(f'LOAD_LOCAL 0\nPUSH_I64 {i}\nEQ\nJMP_FALSE next{i}\n'
                           f'FUNCREF leaf{i}\nRET\nnext{i}:\n' for i in range(60))
        functions = ''.join(f'.function leaf{i} 1 1 0 int 1\nLOAD_LOCAL 0\n'
                            f'PUSH_I64 {i}\nI64_ADD\nRET\n.end\n' for i in range(60))
        text = ('.entry main\n.function main 0 1 0 int 1\n'
                'PUSH_I64 59\nCALL choose\nSTORE_LOCAL 0\nPUSH_I64 1\nLOAD_LOCAL 0\n'
                'CALL_INDIRECT 1 1\nPUSH_I64 60\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n'
                '.function choose 1 1 0 function 1\n' + branches +
                'FUNCREF leaf0\nRET\n.end\n' + functions)
        with tempfile.TemporaryDirectory(prefix='nano-callable-targets-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            self.sanitized(source, work / 'program')


    def test_callable_tags_and_target_membership_are_checked(self):
        text = ('.entry main\n'
                '.function leaf 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n'
                '.function apply 2 2 0 int 1\nLOAD_LOCAL 1\nLOAD_LOCAL 0\nCALL_INDIRECT 1 1\nRET\n.end\n'
                '.function main 0 0 0 int 1\nFUNCREF leaf\nPUSH_I64 42\nCALL apply\n'
                'PUSH_I64 42\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-callable-tags-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            source.write_text('#define main fixture_main\n' + source.read_text() +
                '\n#undef main\nint main(int argc, char **argv) {\n'
                '    if (argc != 3) return 2;\n'
                '    nmap_value f = {(uint8_t)atoi(argv[1]), strtoll(argv[2], NULL, 10), NULL};\n'
                '    return nl_apply(f, 42) == 42 ? 0 : 3;\n}\n')
            binary = work / 'program'
            self.checked(['cc', '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-o', binary])
            env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'}
            self.checked([binary, 11, 0], env=env)
            for tag, target in ((1, 0), (0, 0), (11, -1), (11, 999), (11, 1)):
                with self.subTest(tag=tag, target=target):
                    result = subprocess.run([str(binary), str(tag), str(target)],
                                            capture_output=True, text=True, timeout=10, env=env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('native invariant', result.stderr)
                    self.assertNotIn('ERROR: AddressSanitizer', result.stderr)
                    self.assertNotIn('runtime error:', result.stderr)

    def test_indirect_record_and_array_results_retain_roots(self):
        churn = ('.function churn 0 1 0 void 0\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
                 'LOAD_LOCAL 0\nPUSH_I64 2000\nLT\nJMP_FALSE done\n'
                 'PUSH_STR keep\nPUSH_STR alive\nSTR_CONCAT\nPOP\n'
                 'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP loop\ndone:\nRET\n.end\n')
        for array in (False, True):
            with self.subTest(array=array), tempfile.TemporaryDirectory(prefix='nano-callable-roots-') as tmp:
                work = Path(tmp)
                packed = 'ARR_LITERAL 5 1\n' if array else 'AGG_PACK 0 0 0 1\n'
                get = 'PUSH_I64 0\nARR_GET\n' if array else 'AGG_GET 0\n'
                tag = 'array' if array else 'struct'
                calls = ''.join('PUSH_STR keep\nPUSH_STR alive\nSTR_CONCAT\n' + packed +
                                f'PUSH_BOOL {choice}\nCALL choose\nCALL_INDIRECT 1 1\n' + get +
                                'CALL churn\nPUSH_STR expected\nEQ\nASSERT\n' for choice in (0, 1))
                text = ('.string keep "keep"\n.string alive "alive"\n.string expected "keepalive"\n'
                        '.entry main\n.function main 0 0 0 int 1\n' + calls + 'PUSH_I64 0\nRET\n.end\n'
                        '.function choose 1 1 0 function 1\nLOAD_LOCAL 0\nJMP_FALSE other\n'
                        'FUNCREF first\nRET\nother:\nFUNCREF second\nRET\n.end\n'
                        f'.function first 1 1 0 {tag} 1\nCALL churn\nLOAD_LOCAL 0\nRET\n.end\n'
                        f'.function second 1 1 0 {tag} 1\nLOAD_LOCAL 0\nCALL churn\nRET\n.end\n' + churn)
                source = self.emit(work, text)
                generated = source.read_text()
                marker = 'static void nmap_collect(void) {\n'
                self.assertIn(marker, generated)
                generated = generated.replace(marker,
                    'static size_t callable_collections;\n' + marker + '    ++callable_collections;\n')
                source.write_text('#define main fixture_main\n' + generated +
                    '\n#undef main\nint main(void) {\n'
                    '    int result = fixture_main();\n'
                    '    if (callable_collections < 2) abort();\n    return result;\n}\n')
                self.sanitized(source, work / 'program')


if __name__ == '__main__':
    unittest.main()
