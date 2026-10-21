from pathlib import Path
import os,json,sys,time
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
w=Path(__file__).parent
e=json.loads((root/'docs/evidence/map-private-native-20261021/environment.json').read_text())
for suffix in ('CC','CFLAGS','OBJECTS','LDFLAGS'):e['RECORD_ARRAY_'+suffix]=e['RECORD_ARRAY_VM_'+suffix]
e['RECORD_ARRAY_CFLAGS']+=' -O0'
(w/'environment.json').write_text(json.dumps(e,indent=2)+'\n')
t=time.monotonic()
p=run_captured([sys.executable,'-m','unittest','-v','tests.test_record_map_boundary'],cwd=root,env=dict(os.environ,**e),timeout=900,build=True)
(w/'stdout.log').write_bytes(p.stdout);(w/'stderr.log').write_bytes(p.stderr)
(w/'status.json').write_text(json.dumps({'returncode':p.returncode,'seconds':time.monotonic()-t}))
print(p.stdout.decode());print(p.stderr.decode());sys.exit(p.returncode)
