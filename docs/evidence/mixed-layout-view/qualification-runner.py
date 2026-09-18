from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path.cwd();pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();d=Path('/tmp/nanolang-mixed-view-'+pin[:8])
cases=[('focused-clang',['make','test-mixed-layout-view'],{'CC':'/tmp/nanolang-projection-clang','MIXED_CFLAGS':'-fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'}),('regression',['make','-j2','test-retained-layouts','test-ownership-contracts','test-verifier','test-managed-record-plan'],{'NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'})]
for name,cmd,extra in cases:
 start=time.monotonic()
 with (d/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**extra},timeout=240);status=r.returncode
  except subprocess.TimeoutExpired:status='timeout'
 (d/(name+'.json')).write_text(json.dumps({'pin':pin,'command':cmd,'environment':extra,'status':status,'seconds':time.monotonic()-start},indent=2)+'\n')
 print(name,status,flush=True)
 if status:break
before=json.loads((d/'before.json').read_text());after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
(d/'after.json').write_text(json.dumps(after,indent=2)+'\n');assert before==after
print('unchanged',len(after),flush=True)
