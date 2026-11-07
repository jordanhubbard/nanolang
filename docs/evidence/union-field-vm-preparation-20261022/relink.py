from pathlib import Path
import json,subprocess,hashlib,time,sys
p=Path('/private/tmp/nanolang-union-vm-preparation-20261022/retry');label=sys.argv[1]
cmd=json.load(open(p/'observed-link.argv.json'));cmd[-1]=str(p/label)
(p/(label+'-test.sha256')).write_text(hashlib.sha256(Path('tests/nanoisa/test_union_field_preparation.c').read_bytes()).hexdigest()+'\n')
for name,args in [(label+'-link',cmd),(label+'-run',[cmd[-1]])]:
 (p/(name+'.argv.json')).write_text(json.dumps(args)+'\n');t=time.monotonic()
 with open(p/(name+'.stdout'),'wb') as o,open(p/(name+'.stderr'),'wb') as e:r=subprocess.run(args,stdout=o,stderr=e,timeout=180)
 elapsed=time.monotonic()-t;print(name,r.returncode,elapsed,flush=True);print((p/(name+'.stdout')).read_text(),flush=True);print((p/(name+'.stderr')).read_text(),flush=True)
 (p/(name+'.status.json')).write_text(json.dumps({'status':r.returncode,'seconds':elapsed})+'\n')
 if r.returncode:sys.exit(r.returncode)
