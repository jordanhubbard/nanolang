from pathlib import Path
import json,os,sys,time
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
w=Path(__file__).parent;env=os.environ.copy();env.update(json.loads((r/'docs/evidence/map-copied-image-20261021/vm-environment.json').read_text()));env['ASAN_OPTIONS']='detect_leaks=0';env['LSAN_OPTIONS']=''
env['RECORD_MAP_MODULES']=' '.join(str(Path('/private/tmp/nanolang-map-baseline-20261021')/('string-int-'+p+'.nvm')) for p in ['seed','c-hosted-nano','stage1','stage2'])
for name,modules,combined in [('combined-preparation',['tests.test_record_map_preparation'],True),('old-array-vm',['tests.test_record_array_vm'],False)]:
 e=env.copy()
 if combined:e['RECORD_ARRAY_VM_CFLAGS']+=' -DNANO_RECORD_MAP_PRIVATE_HEAP -DNANO_RECORD_ARRAY_PRIVATE_RUNTIME'
 (w/(name+'-environment.json')).write_text(json.dumps({k:v for k,v in e.items() if k.startswith(('RECORD_','ASAN','LSAN'))},indent=2))
 args=[sys.executable,'-m','unittest','-f','-v',*modules];t=time.monotonic();p=run_captured(args,cwd=r,env=e,timeout=900)
 (w/(name+'.stdout')).write_bytes(p.stdout);(w/(name+'.stderr')).write_bytes(p.stderr);(w/(name+'.json')).write_text(json.dumps({'argv':args,'exit':p.returncode,'seconds':time.monotonic()-t},indent=2));print(name,p.returncode,p.stdout.decode(),p.stderr.decode(),flush=True)
 if p.returncode:sys.exit(p.returncode)
