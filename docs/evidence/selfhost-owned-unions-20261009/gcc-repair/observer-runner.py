from pathlib import Path
import os,subprocess,unittest,json,hashlib
from tests.test_native_array_pop import NativeArrayPop
out=Path('/private/tmp/nanolang-ci-gcc-observers');out.mkdir(exist_ok=True)
original=NativeArrayPop.sanitized
records=[]
def qualify(self,source,binary):
    retained=out/(self._testMethodName+'.c');retained.write_bytes(source.read_bytes())
    command=['/opt/homebrew/bin/gcc-16','-std=c11','-O0','-g','-Wall','-Wextra','-Werror','-c',str(retained),'-o',str(retained.with_suffix('.o'))]
    result=subprocess.run(command,capture_output=True,text=True)
    retained.with_suffix('.log').write_text(result.stdout+result.stderr)
    records.append({'method':self._testMethodName,'command':command,'exit':result.returncode,'source_sha256':hashlib.sha256(retained.read_bytes()).hexdigest()})
    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    return original(self,source,binary)
NativeArrayPop.sanitized=qualify
suite=unittest.TestSuite(NativeArrayPop(name) for name in ('test_popped_closure_survives_owner_reuse_and_collection','test_record_children_survive_owner_reuse_and_collection'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
