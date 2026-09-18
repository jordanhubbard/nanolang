#!/usr/bin/env python3
from pathlib import Path
import os,sys,shutil,subprocess,json,hashlib,time
assert len(sys.argv)==3 and sys.argv[1]=='--check-shadows'
p=Path(os.environ['NANO_SHADOW_CAPTURE'])
shutil.copyfile(sys.argv[2],p.with_suffix('.nvm'))
t=time.monotonic()
r=subprocess.run([os.environ['NANO_SHADOW_REAL_VM'],*sys.argv[1:]])
p.with_suffix('.json').write_text(json.dumps({'arguments':sys.argv[1:],'module_sha256':hashlib.sha256(p.with_suffix('.nvm').read_bytes()).hexdigest(),'vm_status':r.returncode,'seconds':round(time.monotonic()-t,3)},indent=2)+'\n')
sys.exit(r.returncode)
