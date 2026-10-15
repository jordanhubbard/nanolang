import hashlib,json,os,subprocess,time,unittest,sys
from pathlib import Path
root=Path('/private/tmp/nanolang-capture-ae92c0488-cow')
os.chdir(root);sys.path.insert(0,str(root))
from tests.test_source_snapshots import SourceSnapshots
class XassemblerDiagnostic(SourceSnapshots):
    def runTest(self):
        self.assembler_search_order_phases_and_recovery(('xassembler',))
out=Path('/private/tmp/nanolang-snapshot-xassembler-diagnostic-20261008');out.mkdir()
probe=root/'obj/test_module_generation_probe'
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def sha():return hashlib.sha256(probe.read_bytes()).hexdigest()
m={'source_commit':git('rev-parse','HEAD'),'initial_status':git('status','--porcelain'),'probe_sha256':sha(),'runner_pid':os.getpid(),'scope':'original xassembler helper: all placements, external selections and shared choices; not full corpus acceptance'}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
start=time.monotonic()
with (out/'gate.log').open('w') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite([XassemblerDiagnostic()]))
m.update(exit_code=0 if result.wasSuccessful() else 1,seconds=time.monotonic()-start,final_status=git('status','--porcelain'),head_unchanged=git('rev-parse','HEAD')==m['source_commit'],probe_unchanged=sha()==m['probe_sha256'])
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True)
raise SystemExit(m['exit_code'])
