import importlib.util,unittest,sys
from pathlib import Path
out=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('contract',out/'test_nanoisa_introspection.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=Path('/private/tmp/nanolang-match-guards-20261009');m.CALLABLE_FIXTURE=out/'module_introspection_callables.nano.txt'
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));sys.exit(not r.wasSuccessful())
