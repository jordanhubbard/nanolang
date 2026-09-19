"""I reconstruct exact float representations using integer bit observations."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from scripts.nanoisa_reconstruction import Analyze, Emit, Expr, INT, BOOL, FLOAT, Refusal, ARITHMETIC, COMPARE, UNSIGNED_COMPARE, GENERIC_COMPARE
from tests.test_canonical_f64_bits import PATTERNS
ROOT=Path(__file__).resolve().parents[1]
TOOLS=Path(os.environ.get('NANO_HL_COMPILER_DIR',ROOT/'bin'))

class FloatReconstruction(unittest.TestCase):
    def setUp(self):
        # I retain the first ordinary failure's source, module, command and output.
        self.work=Path(tempfile.mkdtemp(prefix='nano-reconstruct-f64-'))
        self.sequence=0
    def command(self,args,success=True):
        self.sequence+=1
        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180)
        (self.work/f'command-{self.sequence}.log').write_text(shlex.join(list(map(str,args)))+'\n'+p.stdout+p.stderr)
        self.assertEqual(p.returncode==0,success,f'{args}\n{p.stdout}\n{p.stderr}\nI retained {self.work}')
        return p
    def module(self,text):
        asm=self.work/'input.nasm';module=self.work/'input.nvm';asm.write_text(text)
        self.command([ROOT/'bin/nanoisa','asm',asm,'-o',module])
        return module
    def native(self,module,name):
        c=self.work/(name+'.c');exe=self.work/name
        self.command([ROOT/'bin/nvm2c',module,'-o',c])
        self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-o',exe])
        self.command([exe])
    def paired(self,text,shadows=''):
        module=self.module(text);original=hashlib.sha256(module.read_bytes()).hexdigest()
        self.command([ROOT/'bin/nano_vm',module]);self.native(module,'original')
        c=self.work/'reconstructed.c';nano=self.work/'reconstructed.nano'
        for language,output in [('c',c),('nano',nano)]:
            self.command([ROOT/'bin/nvm2hl',module,'--language',language,'-o',output])
        self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-o',self.work/'reconstructed'])
        self.command([self.work/'reconstructed'])
        source=nano.read_text()
        nano.write_text(source+'\n'+shadows+'\nshadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\nshadow main { assert (== (main) 0) }\n')
        for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            exe=self.work/name
            self.command([TOOLS/name,nano,'-o',exe]);self.command([exe])
        for name in ('nano_virt','nanoc_stage1','nanoc_stage2'):
            emitted=self.work/(name+'.nvm')
            self.command([TOOLS/name,nano,'--emit-nvm','-o',emitted])
            self.command([ROOT/'bin/nano_vm',emitted]);self.native(emitted,name+'-native')
        self.assertEqual(hashlib.sha256(module.read_bytes()).hexdigest(),original)
        return source,c.read_text()
    @staticmethod
    def check(body,ordinal):
        return body+f'I64_EQ\nJMP_TRUE okay{ordinal}\nPUSH_I64 {ordinal+1}\nRET\nokay{ordinal}:\n'
    def test_small_exact_constant_chunks(self):
        for start in range(0,len(PATTERNS),4):
            body=''
            for i,bits in enumerate(PATTERNS[start:start+4]):
                signed=bits if bits<(1<<63) else bits-(1<<64)
                body+=self.check(f'PUSH_F64 bits:{bits:016x}\nF64_TO_BITS\nF64_FROM_BITS\nF64_TO_BITS\nPUSH_I64 {signed}\n',i)
            self.paired('.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
    def test_signed_integer_boundary_nan_payloads(self):
        body=''
        for i,bits in enumerate((0x7fffffffffffffff,0xffffffffffffffff)):
            signed=bits if bits<(1<<63) else bits-(1<<64)
            body+=self.check(f'PUSH_F64 bits:{bits:016x}\nF64_TO_BITS\nF64_FROM_BITS\nF64_TO_BITS\nPUSH_I64 {signed}\n',i)
        self.paired('.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')

    def test_float_helpers_snapshots_local_joins_and_loop(self):
        text='''.entry main
.function main 0 2 0 int 1
PUSH_F64 bits:fff0000000000042
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_F64 bits:0000000000000000
STORE_LOCAL 0
DUP
CALL relay
SWAP
POP
F64_TO_BITS
PUSH_I64 -4503599627370430
I64_EQ
JMP_TRUE snapshot_ok
PUSH_I64 1
RET
snapshot_ok:
PUSH_F64 bits:7ff0000000000001
CALL relay
POP
PUSH_BOOL 1
CALL choose
F64_TO_BITS
PUSH_I64 -4503599627370430
I64_EQ
JMP_TRUE true_ok
PUSH_I64 2
RET
true_ok:
PUSH_BOOL 0
CALL choose
F64_TO_BITS
PUSH_I64 1
I64_EQ
JMP_TRUE false_ok
PUSH_I64 3
RET
false_ok:
PUSH_I64 0
STORE_LOCAL 1
loop:
LOAD_LOCAL 1
PUSH_I64 2
I64_LT_S
PUSH_F64 bits:fff0000000000042
F64_TO_BITS
PUSH_I64 -4503599627370430
I64_EQ
BOOL_AND
JMP_FALSE done
PUSH_I64 -4503599627370430
F64_FROM_BITS
STORE_LOCAL 0
LOAD_LOCAL 1
PUSH_I64 1
I64_ADD
STORE_LOCAL 1
JMP loop
done:
LOAD_LOCAL 0
F64_TO_BITS
PUSH_I64 -4503599627370430
I64_EQ
JMP_TRUE final_ok
PUSH_I64 4
RET
final_ok:
PUSH_I64 0
RET
.end
.function relay 1 1 0 float 1
.parameters relay float
LOAD_LOCAL 0
RET
.end
.function choose 1 2 0 float 1
.parameters choose bool
LOAD_LOCAL 0
JMP_FALSE other
PUSH_F64 bits:fff0000000000042
STORE_LOCAL 1
JMP joined
other:
PUSH_F64 bits:0000000000000001
STORE_LOCAL 1
joined:
LOAD_LOCAL 1
RET
.end
'''
        nano,c=self.paired(text,'''shadow nlr_f1_relay { assert (== (float_to_bits (nlr_f1_relay (float_from_bits 1))) 1) }
shadow nlr_f2_choose { assert (== (float_to_bits (nlr_f2_choose true)) -4503599627370430) assert (== (float_to_bits (nlr_f2_choose false)) 1) }''')
        self.assertEqual(nano.count('= (nlr_f1_relay '),2)
        self.assertEqual(c.count('= nlr_f1_relay('),2)
        self.observe_call_count(c)
    def observe_call_count(self,c):
        # I instrument the reconstructed helper as an observer. This adds no
        # global/counter admission to reconstruction's original closed profile.
        c='static int observed_calls;\n'+c.replace('double nlr_f1_relay(double nlr_a0) {',
                    'double nlr_f1_relay(double nlr_a0) { ++observed_calls;')
        c=c.replace('return (int)nlr_f0_main();',
                    'int result=(int)nlr_f0_main(); return result ? result : observed_calls == 2 ? 0 : 99;')
        source=self.work/'observed.c';source.write_text(c);exe=self.work/'observed'
        self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-o',exe])
        self.command([exe])
        nano=(self.work/'reconstructed.nano').read_text()
        nano='let mut observed_calls: int = 0\n'+nano.replace('fn nlr_f1_relay(nlr_a0: float) -> float {',
                    'fn nlr_f1_relay(nlr_a0: float) -> float { set observed_calls (+ observed_calls 1)')
        nano=nano.replace('return (nlr_f0_main)',
                    'set observed_calls 0\nlet result: int = (nlr_f0_main)\nassert (== observed_calls 2)\nreturn result')
        source=self.work/'observed.nano';source.write_text(nano)
        for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            exe=self.work/('observed-'+name)
            self.command([TOOLS/name,source,'-o',exe]);self.command([exe])
        for name in ('nano_virt','nanoc_stage1','nanoc_stage2'):
            module=self.work/('observed-'+name+'.nvm')
            self.command([TOOLS/name,source,'--emit-nvm','-o',module])
            self.command([ROOT/'bin/nano_vm',module]);self.native(module,'observed-'+name+'-native')

    def test_float_operations_remain_refused_without_publication(self):
        for body,operation in [('PUSH_F64 1.0\nCAST_INT\nPOP\n','cast'),
                               ('PUSH_F64 1.0\nCAST_BOOL\nPOP\n','truth'),
                               ('PUSH_F64 1.0\nPUSH_F64 2.0\nEQ\nPOP\n','comparison'),
                               ('PUSH_F64 1.0\nPUSH_F64 2.0\nADD\nPOP\n','operand')]:
            module=self.module('.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
            for language in ('c','nano'):
                output=self.work/('prior.'+language);output.write_bytes(b'retained')
                p=self.command([ROOT/'bin/nvm2hl',module,'--language',language,'-o',output],False)
                self.assertIn(operation,p.stderr)
                self.assertEqual(output.read_bytes(),b'retained')
    def test_float_entry_mixed_local_and_stack_join_refusals(self):
        cases = [
            ('.entry main\n.function main 0 0 0 float 1\nPUSH_F64 1.0\nRET\n.end\n','require'),
            ('.entry main\n.function main 0 1 0 int 1\nPUSH_F64 1.0\nSTORE_LOCAL 0\nPUSH_I64 1\nSTORE_LOCAL 0\nPUSH_I64 0\nRET\n.end\n','one scalar type'),
            ('.entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nJMP_FALSE other\nPUSH_F64 1.0\nJMP joined\nother:\nPUSH_F64 2.0\njoined:\nPOP\nPUSH_I64 0\nRET\n.end\n','empty stacks'),
        ]
        for text,diagnostic in cases:
            module=self.module(text)
            for language in ('c','nano'):
                output=self.work/('boundary.'+language);output.write_bytes(b'retained')
                p=self.command([ROOT/'bin/nvm2hl',module,'--language',language,'-o',output],False)
                self.assertIn(diagnostic,p.stderr)
                self.assertEqual(output.read_bytes(),b'retained')

    def test_all_existing_operator_families_keep_float_refusals(self):
        operations = set(ARITHMETIC) | set(COMPARE) | set(UNSIGNED_COMPARE) | set(GENERIC_COMPARE) | {
            'CAST_INT','CAST_BOOL','AND','OR','NOT','BOOL_AND','BOOL_OR','BOOL_NOT',
            'I64_MUL_WIDE_S','I64_MUL_WIDE_U'}
        for op in sorted(operations):
            for position in (0,1):
                with self.subTest(op=op,position=position):
                    module={'functions':[{'params':[],'result':INT,'locals':0,'size':1,
                                         'code':[{'op':op,'pc':0,'arg':0}]}]}
                    analyzer=Analyze(module,0)
                    other=Expr(BOOL,'constant',True) if op.startswith('BOOL_') else Expr(INT,'constant',1)
                    stack=[other,other]
                    stack[position]=Expr(FLOAT,'float_bits',0)
                    # Unary operations consume only the top slot.
                    if op in ('NEG','I64_NEG','I64_INVERT','CAST_INT','CAST_BOOL','NOT','BOOL_NOT'):
                        stack=[Expr(FLOAT,'float_bits',0)]
                    with self.assertRaises(Refusal):
                        analyzer.simple(0,stack,set(),[])

    def test_missing_or_nonhex_fact_and_unmapped_tags_refuse(self):
        module=self.module('.entry main\n.function main 0 0 0 int 1\nPUSH_F64 1.0\nF64_TO_BITS\nRET\n.end\n')
        facts=json.loads(self.command([ROOT/'bin/nanoisa_hl_facts',module]).stdout)
        for value in (None,'0','0'*17,'g'*16,1.0):
            changed=copy.deepcopy(facts)
            changed['functions'][0]['code'][0]['f64_bits']=value
            with self.assertRaisesRegex(Refusal,'sixteen hexadecimal'):
                Analyze(changed,0).run()
        with self.assertRaisesRegex(Refusal,'explicit scalar'):
            Emit([],'c').type(7)
if __name__=='__main__':unittest.main()
