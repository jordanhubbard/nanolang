from pathlib import Path
import importlib.util,unittest
w=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prepared_binding_tests',w/'integration/tests/test_selfhost_module_bindings.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=w/'candidate';module.COMPILER=w/'indexed-native'
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
raise SystemExit(not result.wasSuccessful())
