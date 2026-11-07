from pathlib import Path
import os,sys,time,json
root=Path.cwd();sys.path.insert(0,str(root));from tests.native_process import run_captured
p=Path(__file__).parent;consumers=Path('/private/tmp/nanolang-map-delete-consumer-20261022');env=dict(os.environ,NANO_VM=str(consumers/'nano_vm'),LSAN_OPTIONS='')
for name in ['nanoisa','nano_vm','nvm2c']:env['MAP_PUBLICATION_'+name.upper()]=str(consumers/name)
for key,name in [('MAP_PUBLICATION_NANO_VIRT','nano_virt'),('MAP_PUBLICATION_C_SHADOW','c-shadow'),('MAP_PUBLICATION_COMPILERS','canonical'),('MAP_PUBLICATION_RAW_DRIVERS','raw')]:env[key]=str(p/name)
env['MAP_PUBLICATION_CC']='/opt/homebrew/opt/llvm/bin/clang'
args=[sys.executable,'-m','unittest','-f','-v','tests.test_map_field_publication'];start=time.monotonic();r=run_captured(args,cwd=root,env=env,timeout=900)
(p/'corpus.stdout').write_bytes(r.stdout);(p/'corpus.stderr').write_bytes(r.stderr);(p/'corpus.json').write_text(json.dumps({'argv':args,'status':r.returncode,'seconds':time.monotonic()-start,'selectors':{k:v for k,v in env.items() if k.startswith('MAP_PUBLICATION_') or k=='NANO_VM'}},indent=2));print(r.stdout.decode(),r.stderr.decode(),flush=True);sys.exit(r.returncode)
