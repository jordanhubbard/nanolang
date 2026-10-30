"""I retain FLOAT caller locals inside the existing exclusive path profile."""
import copy
import unittest
from tests import test_owned_nested_path_reconstruction as support
from nanoisa_owned_reconstruction import OwnedAnalyze, OwnedEmit, reconstruct_owned
from nanoisa_reconstruction import Expr, FLOAT, Refusal

ROOT = support.ROOT
BASELINE = ROOT/'docs/evidence/borrowed-float-local-baseline-20261019'


def float_expressions(node):
    if isinstance(node, Expr):
        return ([node] if node.tag == FLOAT else []) + [item for arg in node.args for item in float_expressions(arg)]
    if isinstance(node, (list, tuple)):
        return [item for arg in node for item in float_expressions(arg)]
    return []


class BorrowedFloatLocalReconstruction(unittest.TestCase):
    setUpClass = support.OwnedNestedPathReconstruction.__dict__['setUpClass']
    command = support.OwnedNestedPathReconstruction.command
    facts = support.OwnedNestedPathReconstruction.facts
    native = support.OwnedNestedPathReconstruction.native
    module = support.OwnedNestedPathReconstruction.module
    qualify = support.OwnedNestedPathReconstruction.qualify

    def original(self):
        return self.module(self.compilers[0], (BASELINE/'original.nano').read_text(), 'borrowed-float')

    def test_unchanged_ordinary_and_independent_shadow(self):
        for compiler in self.compilers:
            module = self.module(compiler, (BASELINE/'original.nano').read_text(), 'borrowed-float')
            self.qualify(module)
            facts = self.facts(module)
            emitter = OwnedEmit([OwnedAnalyze(facts,i).run() for i in range(2)], 'nano', facts['layouts'])
            self.assertEqual([emitter.nano_local_cost(fn) for fn in emitter.functions], [98,21])
        shadow = self.work/'borrowed-float-original-shadow.nvm'
        self.command([ROOT/'bin/nanoisa','asm',BASELINE/'original-shadows.nasm','-o',shadow])
        self.command([ROOT/'bin/nano_vm','--verify-only',shadow]); self.command([ROOT/'bin/nano_vm',shadow])
        self.qualify(shadow)

    def test_planner_preserves_float_order_and_prior_helper(self):
        facts = self.facts(self.original())
        functions = [OwnedAnalyze(facts,i).run() for i in range(2)]
        original = copy.deepcopy(functions)
        prior = copy.deepcopy(functions); prior[0].borrowed_float_caller = False
        old = OwnedEmit(prior,'nano',facts['layouts'])
        new = OwnedEmit(functions,'nano',facts['layouts'])
        self.assertEqual([old.nano_local_cost(fn) for fn in old.functions], [131,21])
        self.assertEqual(float_expressions(functions[0].body),float_expressions(new.functions[0].body))
        self.assertEqual(functions[0].body,original[0].body)
        self.assertEqual(new.functions[1].body,old.functions[1].body)
        self.assertFalse(new.scratch_functions)
        self.assertIsNotNone(new.borrowed_float_plan)
        self.assertEqual(OwnedEmit(functions,'c',facts['layouts']).run(0),OwnedEmit(prior,'c',facts['layouts']).run(0))

    def test_forbidden_float_operations_and_helper_locals(self):
        facts = self.facts(self.original())
        caller = facts['functions'][0]
        code_key = 'code' if 'code' in caller else 'instructions'
        position = next(i for i,ins in enumerate(caller[code_key]) if ins['op']=='F64_EQ')
        for opcode in ('F64_ADD','F64_NE','F64_TO_BITS','F64_FROM_BITS'):
            bad = copy.deepcopy(facts); bad['functions'][0][code_key][position]['op'] = opcode
            for language in ('c','nano'):
                with self.subTest(opcode=opcode,language=language), self.assertRaises(Refusal):
                    reconstruct_owned(bad,language)
        bad = copy.deepcopy(facts)
        helper = bad['functions'][1]
        helper['local_types'][-1] = [FLOAT,4294967295]
        for language in ('c','nano'):
            with self.assertRaises(Refusal): reconstruct_owned(bad,language)

    def test_generated_consistency_shadow(self):
        facts = self.facts(self.original())
        text = reconstruct_owned(facts,'nano')
        selected = support.OwnedNestedPathReconstruction.selected_shadow(text)
        for compiler in self.compilers:
            module = self.module(compiler,selected,'borrowed-float-consistency')
            regenerated = self.facts(module)
            self.assertLessEqual(max(fn['locals'] for fn in regenerated['functions']),128)
            native = module.with_suffix('.c')
            self.command([ROOT/'bin/nvm2c',module,'-o',native]); self.native(native,0)

    def test_actual_local_cap_and_atomic_refusal(self):
        for count in (30,31):
            text = (BASELINE/'original.nano').read_text()
            prefix, main = text.split('fn main()',1)
            insertion = ''.join(f' let boundary_{i}: int = {i}\n' for i in range(count))
            self.assertEqual(main.count('return 0'),1)
            text = prefix+'fn main()'+main.replace('return 0',insertion+' return 0',1)
            for compiler in self.compilers:
                module = self.module(compiler,text,'borrowed-float-cap-'+str(count))
                output = module.with_suffix('.nano'); output.write_text('prior checked source')
                result = self.command([ROOT/'bin/nvm2hl','--language','nano',module,'-o',output],None)
                if count == 31:
                    self.assertNotEqual(result.returncode,0); self.assertIn('128 local slots',result.stderr)
                    self.assertEqual(output.read_text(),'prior checked source')
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    rebuilt = self.module(compiler,output.read_text(),'borrowed-float-128')
                    self.assertEqual(self.facts(rebuilt)['functions'][0]['locals'],128)

    def test_finite_negative_zero_equality(self):
        text = (BASELINE/'original.nano').read_text().replace('2.0','-0.0')
        for compiler in self.compilers:
            self.qualify(self.module(compiler,text,'borrowed-negative-zero'))


if __name__ == '__main__':
    unittest.main()
