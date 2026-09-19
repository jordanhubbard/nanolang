"""I validate two executable source surfaces from the same retained module."""
from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

LOOP = '''.entry main
.function main 0 0 0 int 1
PUSH_I64 {start}
CALL walk
RET
.end
.function walk 1 2 0 int 1
.local_begin 0 "state"
PUSH_BOOL 0
STORE_LOCAL 1
.local_begin 1 "state"
loop:
LOAD_LOCAL 0
PUSH_I64 3
I64_LT_S
JMP_FALSE done
inner:
LOAD_LOCAL 1
BOOL_NOT
JMP_FALSE inner_done
PUSH_BOOL 1
STORE_LOCAL 1
JMP inner
inner_done:
LOAD_LOCAL 0
PUSH_I64 0
I64_EQ
JMP_FALSE second
PUSH_I64 1
STORE_LOCAL 0
JMP next
second:
LOAD_LOCAL 0
PUSH_I64 1
I64_EQ
JMP_FALSE third
PUSH_I64 2
STORE_LOCAL 0
JMP next
third:
PUSH_I64 3
STORE_LOCAL 0
next:
JMP loop
done:
LOAD_LOCAL 0
RET
.end
.parameters walk int
'''
DIAMOND = '''.entry main
.function main 0 0 0 int 1
PUSH_BOOL {flag}
PUSH_I64 41
PUSH_I64 42
CALL choose
RET
.end
.function choose 3 4 0 int 1
LOAD_LOCAL 0
JMP_FALSE otherwise
LOAD_LOCAL 1
STORE_LOCAL 3
JMP done
otherwise:
LOAD_LOCAL 2
STORE_LOCAL 3
done:
LOAD_LOCAL 3
RET
.end
.parameters choose bool int int
'''


class ScalarReconstruction(unittest.TestCase):
    def checked(self, args, expected=0):
        argv = list(map(str, args))
        p = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=120,
                           env={**os.environ,'NANOLANG_SHADOW_TIMEOUT_SECONDS':'60'})
        self.assertEqual(p.returncode, expected,
                         'I ran: '+shlex.join(argv)+'\n'+p.stdout+p.stderr)
        return p

    def assemble(self, directory, text):
        source=directory/'input.nasm'; source.write_text(text)
        module=directory/'input.nvm'
        self.checked([ROOT/'bin/nanoisa','asm',source,'-o',module])
        source.unlink()  # My reconstructor has only the module.
        return module

    def paired(self, text, expected, shadows, structured=False, names=()):
        with tempfile.TemporaryDirectory(prefix='nano-reconstruct-') as d:
            directory=Path(d);module=self.assemble(directory,text)
            original=module.read_bytes()
            self.checked([ROOT/'bin/nano_vm',module],expected)
            for language in ('c','nano'):
                output=directory/('result.'+language)
                self.checked([ROOT/'bin/nvm2hl','--language',language,module,'-o',output])
                source=output.read_text()
                for forbidden in (r'\bgoto\b',r'nano_vm',r'nvm_blob',r'\bswitch\b',r'\bdispatch\b'):
                    self.assertNotRegex(source,forbidden)
                self.assertIn('return',source)
                for name in names:
                    self.assertIn(name,source)
                if structured:
                    self.assertIn('while',source);self.assertIn('if',source)
                    self.assertIn('nlr_l0_state',source);self.assertIn('nlr_l1_state',source)
                if language=='c':
                    binary=directory/'c-program'
                    self.checked(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-Wall','-Wextra','-Werror',
                                  '-fsanitize=address,undefined','-fno-sanitize-recover=all',output,'-o',binary])
                    self.checked([binary],expected)
                else:
                    # My fixture assertions are independent validation, not recovered source shadows.
                    output.write_text(source+'\n'+shadows+f'\nshadow main {{ assert (== (main) {expected}) }}\n')
                    for driver in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                        compiler=Path(os.environ.get('NANO_HL_COMPILER_DIR',ROOT/'bin'))/driver
                        binary=directory/driver
                        self.checked([compiler,output,'-o',binary])
                        self.checked([binary],expected)
            self.assertEqual(module.read_bytes(),original)

    def test_nested_loops_zero_one_and_multiple_iterations(self):
        for start in (3,2,0):
            with self.subTest(start=start):
                shadows=f'''shadow nlr_f0_main {{ assert (== (nlr_f0_main) 3) }}
shadow nlr_f1_walk {{ assert (== (nlr_f1_walk 0) 3) assert (== (nlr_f1_walk 2) 3) assert (== (nlr_f1_walk 3) 3) }}'''
                self.paired(LOOP.format(start=start),3,shadows,True)

    def test_diamond_both_results_and_definite_assignment(self):
        for flag in (0,1):
            with self.subTest(flag=flag):
                expected=41 if flag else 42
                shadows=f'''shadow nlr_f0_main {{ assert (== (nlr_f0_main) {expected}) }}
shadow nlr_f1_choose {{ assert (== (nlr_f1_choose true 11 12) 11) assert (== (nlr_f1_choose false 11 12) 12) }}'''
                self.paired(DIAMOND.format(flag=flag),expected,shadows)

    def test_loaded_value_snapshot_survives_later_store(self):
        text='''.entry main
.function main 0 1 0 int 1
PUSH_I64 42
STORE_LOCAL 0
LOAD_LOCAL 0
PUSH_I64 7
STORE_LOCAL 0
DUP
POP
RET
.end
'''
        self.paired(text,42,'shadow nlr_f0_main { assert (== (nlr_f0_main) 42) }')

    def test_optional_names_do_not_alias_slots(self):
        text=''' .string "nano.local.v99"
.string "ignored"
.metadata 0 1
.function main 0 2 0 int 1
PUSH_I64 42
STORE_LOCAL 0
.local_begin 0 "if return"
PUSH_I64 7
STORE_LOCAL 1
.local_begin 1 "if return"
LOAD_LOCAL 0
RET
.end
.entry main
'''
        self.paired(text,42,'shadow nlr_f0_main { assert (== (nlr_f0_main) 42) }',
                    names=('nlr_l0_if_return','nlr_l1_if_return'))

    def test_int_boundaries_and_boolean_call(self):
        text='''.entry main
.function main 0 0 0 int 1
PUSH_I64 -9223372036854775808
PUSH_I64 9223372036854775807
CALL smaller
JMP_FALSE no
PUSH_I64 42
RET
no:
PUSH_I64 1
RET
.end
.function smaller 2 2 0 bool 1
LOAD_LOCAL 0
LOAD_LOCAL 1
I64_LT_S
BOOL_NOT
BOOL_NOT
PUSH_BOOL 1
BOOL_AND
RET
.end
.parameters smaller int int
'''
        shadows='''shadow nlr_f0_main { assert (== (nlr_f0_main) 42) }
shadow nlr_f1_smaller { assert (nlr_f1_smaller -2 3) assert (not (nlr_f1_smaller 3 -2)) }'''
        self.paired(text,42,shadows)

    def test_refusal_preserves_prior_outputs(self):
        cases={
            'no_entry':'.function helper 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n',
            'unsupported_cast':'.function main 0 0 0 int 1\nPUSH_I64 1\nCAST_FLOAT\nRET\n.end\n.entry main\n',
            'unknown_parameter':DIAMOND.format(flag=0).replace('.parameters choose bool int int\n',''),
            'uninitialized':DIAMOND.format(flag=0).replace('LOAD_LOCAL 2\nSTORE_LOCAL 3\n',''),
            'mixed_local':DIAMOND.format(flag=0).replace('LOAD_LOCAL 2\nSTORE_LOCAL 3','PUSH_BOOL 1\nSTORE_LOCAL 3'),
            'stack_join':'.function main 0 0 0 int 1\nPUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nJMP done\nother:\nPUSH_I64 2\ndone:\nRET\n.end\n.entry main\n',
            'unstructured':'.function main 0 0 0 int 1\nJMP done\nPUSH_I64 1\nRET\ndone:\nPUSH_I64 2\nRET\n.end\n.entry main\n',
            'irreducible':'.function main 0 0 0 int 1\nPUSH_BOOL 1\nJMP_FALSE b\na:\nNOP\nJMP c\nb:\nNOP\nJMP c\nc:\nPUSH_BOOL 0\nJMP_TRUE a\nPUSH_I64 0\nRET\n.end\n.entry main\n',
            'recursive':'.function main 0 0 0 int 1\nCALL main\nRET\n.end\n.entry main\n',
            'global':'.function main 0 0 0 int 1\nPUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_I64 1\nRET\n.end\n.entry main\n',
        }
        for name,text in cases.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory(prefix='nano-reconstruct-refusal-') as d:
                directory=Path(d);module=self.assemble(directory,text)
                for language in ('c','nano'):
                    output=directory/'prior';output.write_text('I remain intact.\n')
                    p=subprocess.run([ROOT/'bin/nvm2hl','--language',language,module,'-o',output],capture_output=True,text=True,timeout=30)
                    self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
                    self.assertIn('I did not publish',p.stderr)
                    self.assertEqual(output.read_text(),'I remain intact.\n')


if __name__=='__main__':
    unittest.main()
