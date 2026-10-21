from pathlib import Path
import json,sys,os,time
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
w=Path(__file__).parent;env=os.environ.copy();env.update(json.loads((r/'docs/evidence/map-copied-image-20261021/vm-environment.json').read_text()));env['LSAN_OPTIONS']=''
env['RECORD_MAP_MODULES']=' '.join(str(Path('/private/tmp/nanolang-map-baseline-20261021')/('string-int-'+p+'.nvm')) for p in ['seed','c-hosted-nano','stage1','stage2'])
(w/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('RECORD_','LSAN'))},indent=2))
args=[sys.executable,'-m','unittest','-f','-v','tests.test_record_map_observation','tests.test_record_map_runtime','tests.test_record_array_vm','tests.test_record_map_heap'];t=time.monotonic();p=run_captured(args,cwd=r,env=env,timeout=900)
(w/'final-corrected.stdout').write_bytes(p.stdout);(w/'final-corrected.stderr').write_bytes(p.stderr);(w/'final-corrected.json').write_text(json.dumps({'argv':args,'exit':p.returncode,'seconds':time.monotonic()-t},indent=2));print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
