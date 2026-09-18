import hashlib,json,pathlib,sys,unittest
from tests.test_source_borrow_emission import SourceBorrowEmission
report=pathlib.Path(sys.argv[1]); names=sys.argv[2:]
def retained(cls):
    cls.temporary._finalizer.detach()
    tools={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*cls.emitters,*cls.shadow_tools] if p.is_file()}
    report.write_text(json.dumps({'work':str(cls.work),'generated_producers':tools},indent=2)+'\n')
    print('I retain fresh source-wrapper fixtures at',cls.work,flush=True)
SourceBorrowEmission.tearDownClass=classmethod(retained)
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
