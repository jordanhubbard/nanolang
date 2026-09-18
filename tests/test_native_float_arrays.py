"""I preserve exact float elements through native array transport."""
from tests import test_native_optional_array_reads as array_reads
from pathlib import Path
import os
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
import unittest

class FloatArrays(unittest.TestCase):
    checked = array_reads.OptionalArrayReads.checked
    paired = array_reads.OptionalArrayReads.paired
    def test_float_transport(self):
        self.paired(""".entry main
.function main 0 2 0 int 1
PUSH_F64 1.5
ARR_LITERAL 3 1
STORE_LOCAL 0
LOAD_LOCAL 0
CALL relay
STORE_LOCAL 1
LOAD_LOCAL 1
PUSH_F64 2.5
ARR_PUSH
POP
LOAD_LOCAL 0
ARR_LEN
PUSH_I64 2
I64_EQ
ASSERT
LOAD_LOCAL 0
PUSH_I64 0
PUSH_F64 -0.0
ARR_SET
POP
LOAD_LOCAL 1
PUSH_I64 1
ARR_GET
DUP
TYPE_CHECK 3
ASSERT
PUSH_F64 2.5
F64_EQ
ASSERT
LOAD_LOCAL 1
PUSH_I64 -1
ARR_GET
TYPE_CHECK 0
ASSERT
ARR_NEW 3
PUSH_F64 9.25
ARR_PUSH
PUSH_I64 0
ARR_GET
PUSH_F64 9.25
F64_EQ
ASSERT
PUSH_I64 0
RET
.end
.function relay 1 1 0 array 1
LOAD_LOCAL 0
RET
.end
""")
    def test_signed_zero_and_scalar_return(self):
        self.paired(""".string negative_zero "-0"
.entry main
.function main 0 0 0 int 1
PUSH_F64 -0.0
ARR_LITERAL 3 1
CALL first
CAST_STRING
PUSH_STR negative_zero
EQ
ASSERT
PUSH_I64 0
RET
.end
.function first 1 1 0 float 1
LOAD_LOCAL 0
PUSH_I64 0
ARR_GET
RET
.end
""")

    def test_growth_alias_and_record_roots(self):
        body = 'ARR_NEW 3\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 1\n'
        for i in range(200):
            body += f'LOAD_LOCAL 0\nPUSH_F64 {i}.5\nARR_PUSH\nPOP\n'
            body += 'CALL garbage\nPOP\n'
        body += 'LOAD_LOCAL 1\nAGG_GET 0\nPUSH_I64 199\nARR_GET\nPUSH_F64 199.5\nF64_EQ\nASSERT\n'
        self.paired('.types 1 0 0\n.entry main\n.function main 0 2 0 int 1\n'+body+
                    'PUSH_I64 0\nRET\n.end\n.function garbage 0 0 0 int 1\n'+
                    ('PUSH_F64 4.5\n' * 256)+'ARR_LITERAL 3 256\nPOP\nPUSH_I64 0\nRET\n.end\n')

    def test_source_float_arrays_and_preserved_refusals(self):
        source_text = """fn first(xs: array<float>) -> float { return (at xs 0) }
shadow first { assert (== (first [1.5]) 1.5) }
fn filled() -> array<float> { return (array_new 3 2.5) }
shadow filled { assert (== (array_length (filled)) 3) }
fn main() -> int {
 let inferred = [1.25, 2.75]
 assert (== (first inferred) 1.25)
 assert (== (first [3.75]) 3.75)
 let values: array<float> = (filled)
 let alias: array<float> = values
 (array_set values 0 4.5)
 assert (== (first alias) 4.5)
 let empty: array<float> = []
 let appended: array<float> = (array_push empty 8.5)
 assert (== (first appended) 8.5)
 assert (== (array_length values) 3)
 return 0
}
shadow main { assert (== (main) 0) }
"""
        with tempfile.TemporaryDirectory(prefix='nano-float-source-') as tmp:
            path=Path(tmp); source=path/'input.nano'; module=path/'input.nvm'
            generated=path/'output.c'; binary=path/'native'
            source.write_text(source_text)
            for producer in ('nano_virt','nanoisa_emit'):
                self.checked([ROOT/'bin'/producer,source,'--emit-nvm','-o',module])
                self.checked([ROOT/'bin/nano_vm','--verify-only',module])
                self.checked([ROOT/'bin/nano_vm',module])
                self.checked([ROOT/'bin/nvm2c',module,'-o',generated])
                self.checked([os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror',
                              '-fsanitize=address,undefined','-fno-sanitize-recover=all',generated,'-lm','-o',binary])
                self.checked([binary])
            for declaration in ('let xs: array<float> = [1]', 'let xs: array<int> = [1.5]',
                                'let xs: array<float> = [true]',
                                'let ints: array<int> = [1] let xs: array<float> = ints',
                                'let floats: array<float> = [1.5] let xs: array<int> = floats'):
                source.write_text('fn main() -> int { '+declaration+' return 0 }\nshadow main { assert true }\n')
                module.write_bytes(b'previous')
                result=subprocess.run([ROOT/'bin/nanoisa_emit',source,'--emit-nvm','-o',module],
                                      cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(module.read_bytes(),b'previous')

if __name__ == '__main__': unittest.main()
