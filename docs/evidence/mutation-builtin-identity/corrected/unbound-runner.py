import pathlib,sys,tempfile,unittest
root=pathlib.Path('/home/jkh/Src/nanolang-mutation-builtin-corrected')
sys.path.insert(0,str(root))
from tests.test_mutation_builtin_identity import MutationBuiltinIdentity
MutationBuiltinIdentity.work=pathlib.Path(tempfile.mkdtemp(prefix='nanolang-mutation-unbound-corrected-'))
print('I retain fresh unbound artifacts in '+str(MutationBuiltinIdentity.work),flush=True)
case=MutationBuiltinIdentity('test_unbound_mutation')
result=unittest.TextTestRunner(verbosity=2).run(case)
raise SystemExit(0 if result.wasSuccessful() else 1)
