import importlib.util,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(sys.argv[2]).resolve()))
spec=importlib.util.spec_from_file_location('owner_authority_corrected',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=pathlib.Path(sys.argv[2]).resolve()
suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.OwnedArrayAuthority)
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
