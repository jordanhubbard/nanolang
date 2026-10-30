from pathlib import Path
import os,sys,json,time
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
w=Path(__file__).parent;env=os.environ.copy()
e=json.loads((r/'docs/evidence/map-copied-image-20261021/vm-environment.json').read_text());env.update(e)
env['RECORD_MAP_MODULES']=' '.join(str(Path('/private/tmp/nanolang-map-baseline-20261021')/('string-int-'+p+'.nvm')) for p in ['seed','c-hosted-nano','stage1','stage2'])
(w/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('RECORD_ARRAY_VM_','RECORD_MAP_'))},indent=2))
old=json.loads((r/'docs/evidence/map-declarations-20261021/environment.json').read_text())
for k in ['CC','CFLAGS','OBJECTS','LDFLAGS']:env['RECORD_ARRAY_'+k]=old['DECLARATION_'+k]
args=[sys.executable,'-m','unittest','-f','-v','tests.test_record_map_image','tests.test_record_map_execution','tests.test_record_array_execution','tests.test_record_array_vm'];t=time.monotonic();p=run_captured(args,cwd=r,env=env,timeout=900)
(w/'neighbors.stdout').write_bytes(p.stdout);(w/'neighbors.stderr').write_bytes(p.stderr);(w/'neighbors.json').write_text(json.dumps({'argv':args,'exit':p.returncode,'seconds':time.monotonic()-t},indent=2));print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
