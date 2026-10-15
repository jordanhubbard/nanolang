"""I retain callable environments through mutable array owners and cycles."""
from pathlib import Path
import tempfile
import unittest
from tests import test_native_closures as closures

ROOT = closures.ROOT


class NativeClosureArrays(unittest.TestCase):
    checked = closures.NativeClosures.checked
    emit = closures.NativeClosures.emit
    sanitized = closures.NativeClosures.sanitized
    run_module = closures.NativeClosures.run_module

    def test_mixed_arrays_preserve_growth_aliases_tags_and_roots(self):
        for tag in (11, 15):
            for constructor in ('literal', 'empty', 'new'):
                with self.subTest(tag=tag, constructor=constructor):
                    if constructor == 'literal':
                        initial = f'PUSH_I64 41\nCLOSURE_NEW read 1\nFUNCREF named\nARR_LITERAL {tag} 2\n'
                    else:
                        initial = (f'ARR_NEW {tag}\n' if constructor == 'new' else f'ARR_LITERAL {tag} 0\n')
                        initial += 'PUSH_I64 41\nCLOSURE_NEW read 1\nARR_PUSH\nFUNCREF named\nARR_PUSH\n'
                    text = ('.string a "a"\n.string b "b"\n.entry main\n'
                            '.function read 0 0 1 int 1\nLOAD_UPVALUE 0 0\nRET\n.end\n'
                            '.function named 0 0 0 int 1\nPUSH_I64 7\nRET\n.end\n'
                            '.function pick 2 2 0 function 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nARR_GET\nRET\n.end\n'
                            '.function main 0 3 0 int 1\n' + initial + 'STORE_LOCAL 0\n'
                            'LOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 1\n'
                            'LOAD_LOCAL 1\nAGG_GET 0\nSTORE_GLOBAL 0\n'
                            'LOAD_GLOBAL 0\nPUSH_I64 0\nPUSH_I64 42\nCLOSURE_NEW read 1\nARR_SET\nPOP\n'
                            'CALL churn\nLOAD_LOCAL 0\nPUSH_I64 0\nCALL pick\nCALL_INDIRECT 0 1\nPUSH_I64 42\nEQ\nASSERT\n'
                            'LOAD_GLOBAL 0\nPRINTLN\nPUSH_I64 0\nSTORE_LOCAL 2\ngrow:\n'
                            'LOAD_LOCAL 2\nPUSH_I64 40\nLT\nJMP_FALSE grown\n'
                            'LOAD_LOCAL 0\nLOAD_LOCAL 2\nCLOSURE_NEW read 1\nARR_PUSH\nPOP\n'
                            'LOAD_LOCAL 2\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 2\nJMP grow\ngrown:\nCALL churn\n'
                            'LOAD_LOCAL 1\nAGG_GET 0\nARR_LEN\nPUSH_I64 42\nEQ\nASSERT\n'
                            'LOAD_GLOBAL 0\nPUSH_I64 41\nARR_GET\nCALL_INDIRECT 0 1\nPUSH_I64 39\nEQ\nASSERT\n'
                            'LOAD_LOCAL 0\nPUSH_I64 2\nARR_GET\nLOAD_LOCAL 0\nPUSH_I64 3\nARR_GET\nNE\nASSERT\n'
                            'LOAD_LOCAL 1\nAGG_GET 0\nPUSH_I64 0\nFUNCREF named\nARR_SET\nPOP\nCALL churn\n'
                            'LOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nTYPE_CHECK 11\nASSERT\n'
                            'LOAD_GLOBAL 0\nPUSH_I64 2\nARR_GET\nTYPE_CHECK 15\nASSERT\n'
                            'LOAD_LOCAL 0\nPUSH_I64 -1\nARR_GET\nTYPE_CHECK 0\nASSERT\n'
                            'LOAD_LOCAL 0\nPUSH_I64 42\nARR_GET\nTYPE_CHECK 0\nASSERT\n'
                            'LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nCALL_INDIRECT 0 1\nPUSH_I64 7\nEQ\nASSERT\n'
                            'PUSH_I64 0\nRET\n.end\n' + closures.CHURN)
                    self.run_module(text, '[closure(0), fn(1)]\n', collections=True)

    def test_unreachable_closure_array_cycles_are_collected(self):
        text = ('.string a "a"\n.string b "b"\n.entry main\n'
                '.function length 0 0 1 int 1\nLOAD_UPVALUE 0 0\nARR_LEN\nRET\n.end\n'
                '.function create 0 1 0 void 0\nARR_NEW 11\nSTORE_LOCAL 0\n'
                'LOAD_LOCAL 0\nLOAD_LOCAL 0\nCLOSURE_NEW length 1\nARR_PUSH\nPOP\n'
                'CALL churn\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nCALL_INDIRECT 0 1\nPUSH_I64 1\nEQ\nASSERT\nRET\n.end\n'
                '.function main 0 1 0 int 1\nPUSH_I64 0\nSTORE_LOCAL 0\nloop:\n'
                'LOAD_LOCAL 0\nPUSH_I64 20\nLT\nJMP_FALSE done\nCALL create\n'
                'LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP loop\ndone:\n'
                'CALL churn\nPUSH_I64 0\nRET\n.end\n' + closures.CHURN)
        with tempfile.TemporaryDirectory(prefix='nano-closure-array-cycles-') as tmp:
            work = Path(tmp)
            source = self.emit(work, text)
            source.write_text('#define main fixture_main\n' + source.read_text() +
                '\n#undef main\nint main(void) {\n'
                '    int result = (int)nl_main(); nmap_collect();\n'
                '    if (nrec_owned_head || narr_owners || nagg_live_bytes) abort();\n'
                '    return result;\n}\n')
            self.sanitized(source, work / 'program')

    def test_source_closures_through_returned_function_arrays(self):
        text = '''fn make(n: int) -> fn() -> int {
    return fn() -> int { return n }
}
shadow make { let value: fn() -> int = (make 42) assert (== (value) 42) }
fn functions(n: int) -> array<fn() -> int> { return [(make n), (make (+ n 1))] }
shadow functions {
    let values: array<fn() -> int> = (functions 40)
    let value: fn() -> int = (at values 1)
    assert (== (value) 41)
}
fn main() -> int {
    let values: array<fn() -> int> = (functions 40)
    let alias: array<fn() -> int> = values
    (array_set alias 0 (make 42))
    let chosen: fn() -> int = (at values 0)
    assert (== (chosen) 42)
    (println (chosen))
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-source-closure-arrays-') as tmp:
            work = Path(tmp)
            source, module, native = work / 'input.nano', work / 'input.nvm', work / 'input.c'
            source.write_text(text)
            self.checked([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module])
            self.assertEqual(self.checked([ROOT / 'bin/nano_vm', module]).stdout, '42\n')
            self.checked([ROOT / 'bin/nvm2c', module, '-o', native])
            self.assertEqual(self.sanitized(native, work / 'program').stdout, '42\n')


if __name__ == '__main__':
    unittest.main()
