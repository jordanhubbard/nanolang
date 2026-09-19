import pathlib,sys,tempfile,unittest
root=pathlib.Path('/home/jkh/Src/nanolang-mutation-builtin-corrected')
sys.path.insert(0,str(root))
from tests.test_mutation_builtin_identity import MutationBuiltinIdentity
MutationBuiltinIdentity.work=pathlib.Path(tempfile.mkdtemp(prefix='nanolang-mutation-nested-'))
print('I retain fresh nested artifacts in '+str(MutationBuiltinIdentity.work),flush=True)
MutationBuiltinIdentity.drivers=[pathlib.Path('/tmp/nanolang-mutation-identity-7i1p1jr6')/('checker-'+name) for name in ('nanoc_c','nanoc_stage1','nanoc_stage2')]
case=MutationBuiltinIdentity('test_nested_scope_restores_declaration')
result=unittest.TextTestRunner(verbosity=2).run(case)
raise SystemExit(0 if result.wasSuccessful() else 1)
