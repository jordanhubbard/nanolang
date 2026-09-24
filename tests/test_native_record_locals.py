"""I keep record locals off the C stack without changing their value or roots."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.native_toolchain import native_cc

ROOT = Path(__file__).resolve().parents[1]


class NativeRecordLocals(unittest.TestCase):
    def checked(self, command, **kwargs):
        result = subprocess.run([str(x) for x in command], capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def emit(self, work, text):
        assembly, module, source = (work / x for x in ('input.nasm', 'input.nvm', 'input.c'))
        assembly.write_text(text)
        self.checked([ROOT / 'bin/nanoisa', 'asm', assembly, '-o', module])
        self.checked([ROOT / 'bin/nano_vm', module])
        self.checked([ROOT / 'bin/nvm2c', module, '-o', source])
        return source

    def sanitized(self, source, binary):
        self.checked([*native_cc(), '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                      source, '-o', binary])
        self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})

    def test_wide_local_frames_preserve_recursive_values_with_small_static_stack(self):
        # I retain different records across a call and check every value on unwind.
        setup = ''.join(f'PUSH_I64 {i}\nAGG_PACK 0 0 0 1\nSTORE_LOCAL {i}\n'
                        for i in range(1, 97))
        checks = ''.join(f'LOAD_LOCAL {i}\nAGG_GET 0\nPUSH_I64 {i}\nEQ\nASSERT\n'
                         for i in range(1, 97))
        text = ('.entry main\n.types 2 0 0\n'
                '.function wide 0 0 0 struct 1\n' + 'PUSH_I64 0\n' * 75 +
                'AGG_PACK 0 1 0 75\nRET\n.end\n'
                '.function walk 1 97 0 int 1\n' + setup +
                'LOAD_LOCAL 0\nPUSH_I64 0\nEQ\nBOOL_NOT\nJMP_FALSE done\n'
                'LOAD_LOCAL 0\nPUSH_I64 1\nI64_SUB\nCALL walk\nPOP\ndone:\n' + checks +
                'LOAD_LOCAL 0\nRET\n.end\n'
                '.function main 0 0 0 int 1\nPUSH_I64 32\nCALL walk\n'
                'PUSH_I64 32\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-record-locals-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            # I measure stack use without executing an old overflowing compiler.
            self.checked([*native_cc(), '-std=c11', '-O0', '-fstack-usage', '-c', source,
                          '-o', work / 'frame.o'])
            lines = (work / 'frame.su').read_text().splitlines()
            frames = [int(line.split('\t')[1]) for line in lines
                      if line.split('\t')[0].endswith(':nl_walk')]
            self.assertEqual(len(frames), 1)
            self.assertLess(frames[0], 64 * 1024, 'I keep wide record locals off the C stack')
            self.sanitized(source, work / 'program')

    def test_record_tail_swap_preserves_owned_edges_and_array_aliases(self):
        # Each call allocates an owned string; loop collection must trace records
        # in the current frame, including swapped parameters and copied locals.
        text = (
            '.entry main\n.types 1 0 0\n.string key "key"\n.string text "retained"\n'
            '.function value 0 0 0 string 1\nHM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\n'
            'HM_SET\nPUSH_STR key\nHM_GET\nRET\n.end\n'
            '.function swap 3 4 0 struct 1\nLOAD_LOCAL 0\nSTORE_LOCAL 3\n'
            'LOAD_LOCAL 2\nPUSH_I64 0\nEQ\nBOOL_NOT\nJMP_FALSE done\nCALL value\nPOP\n'
            'LOAD_LOCAL 3\nAGG_GET 1\nPUSH_I64 0\nLOAD_LOCAL 2\nARR_SET\nPOP\n'
            'LOAD_LOCAL 1\nLOAD_LOCAL 0\nLOAD_LOCAL 2\nPUSH_I64 1\nI64_SUB\n'
            'TAIL_CALL swap\ndone:\nLOAD_LOCAL 3\nRET\n.end\n'
            '.function main 0 2 0 int 1\nPUSH_I64 0\nARR_LITERAL 1 1\nSTORE_LOCAL 0\n'
            'CALL value\nLOAD_LOCAL 0\nPUSH_I64 7\nAGG_PACK 0 0 0 3\n'
            'CALL value\nLOAD_LOCAL 0\nPUSH_I64 9\nAGG_PACK 0 0 0 3\n'
            'PUSH_I64 1001\nCALL swap\nSTORE_LOCAL 1\n'
            'LOAD_LOCAL 1\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n'
            'LOAD_LOCAL 1\nAGG_GET 2\nPUSH_I64 9\nEQ\nASSERT\n'
            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 1\nEQ\nASSERT\n'
            'PUSH_I64 0\nRET\n.end\n')
        with tempfile.TemporaryDirectory(prefix='nano-record-local-roots-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            self.sanitized(source, work / 'program')
