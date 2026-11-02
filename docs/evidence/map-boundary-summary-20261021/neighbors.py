from pathlib import Path
import os,json,sys,time
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
w=Path(__file__).parent
e=json.loads((root/'docs/evidence/map-private-native-20261021/neighbor-environment.json').read_text())

e['RECORD_ARRAY_CFLAGS']+=' -O0'
(w/'neighbor-environment.json').write_text(json.dumps(e,indent=2)+'\n')
t=time.monotonic()
p=run_captured([sys.executable,'-m','unittest','-v','tests.test_record_map_execution','tests.test_record_map_image'],cwd=root,env=dict(os.environ,**e),timeout=900,build=True)
(w/'neighbor-stdout.log').write_bytes(p.stdout);(w/'neighbor-stderr.log').write_bytes(p.stderr)
(w/'neighbor-status.json').write_text(json.dumps({'returncode':p.returncode,'seconds':time.monotonic()-t}))
print(p.stdout.decode());print(p.stderr.decode());sys.exit(p.returncode)
