import hashlib,json,pathlib,sys,unittest
from tests.test_source_borrow_emission import SourceBorrowEmission
report=pathlib.Path(sys.argv[1]); names=sys.argv[2:]
original=SourceBorrowEmission.setUpClass.__func__
def inventory(cls):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*cls.emitters,*cls.shadow_tools] if p.is_file()}
def prepared(cls):
    original(cls)
    cls.frozen_producers=inventory(cls)
    report.with_suffix('.before.json').write_text(json.dumps({'work':str(cls.work),'generated_producers':cls.frozen_producers},indent=2)+'\n')
def retained(cls):
    cls.temporary._finalizer.detach()
    after=inventory(cls)
    report.write_text(json.dumps({'work':str(cls.work),'generated_producers':after,'unchanged':after==cls.frozen_producers},indent=2)+'\n')
    print('I retain fresh float fixtures at',cls.work,flush=True)
    assert after==cls.frozen_producers
SourceBorrowEmission.setUpClass=classmethod(prepared)
SourceBorrowEmission.tearDownClass=classmethod(retained)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(0 if result.wasSuccessful() else 1)
