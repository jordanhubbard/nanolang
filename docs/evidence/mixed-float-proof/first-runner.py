from pathlib import Path
import hashlib,json,subprocess,time
pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();d=Path('/tmp/nanolang-mixed-proof-'+pin[:8]);d.mkdir(exist_ok=True)
files=[p for p in subprocess.check_output(['git','ls-files','src','tests','Makefile.gnu'],text=True).splitlines() if Path(p).is_file()]
(d/'before.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2)+'\n')
start=time.monotonic()
with (d/'gcc.log').open('w') as log:
 try:r=subprocess.run(['make','-j2','test-mixed-float-proof'],stdout=log,stderr=subprocess.STDOUT,timeout=180);status=r.returncode
 except subprocess.TimeoutExpired:status='timeout'
(d/'result.json').write_text(json.dumps({'pin':pin,'command':['make','-j2','test-mixed-float-proof'],'status':status,'seconds':time.monotonic()-start},indent=2)+'\n')
print(d,status)
