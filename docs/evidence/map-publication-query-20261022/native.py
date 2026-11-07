import json,os,shlex,subprocess
from pathlib import Path
out=Path(__file__).parent;s=json.loads(Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-native-8twxf6mp/selection.json').read_text())
env=dict(os.environ,RECORD_ARRAY_VM_CC=shlex.join(s['cc']),RECORD_ARRAY_VM_CFLAGS=shlex.join(s['flags']),RECORD_ARRAY_VM_OBJECTS=shlex.join(s['objects']),RECORD_ARRAY_VM_LDFLAGS=shlex.join(s['ldflags']),RECORD_MAP_MODULES=shlex.join(s['modules']))
with (out/'native-final.log').open('w') as f:p=subprocess.run(['python3','-m','unittest','-f','-v','tests.test_record_map_native'],env=env,stdout=f,stderr=subprocess.STDOUT)
(out/'native-final-status.json').write_text(json.dumps(dict(status=p.returncode))+'\n');print((out/'native-final.log').read_text());raise SystemExit(p.returncode)
