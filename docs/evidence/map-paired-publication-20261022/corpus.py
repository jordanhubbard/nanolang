import os,sys,json,time
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
out=Path(__file__).parent
env=os.environ.copy()
for name in ('nano_virt','nanoisa','nano_vm','nvm2c'):env['MAP_PUBLICATION_'+name.upper()]=str(out/name)
env['MAP_PUBLICATION_COMPILERS']=str(out/'canonical');env['MAP_PUBLICATION_RAW_DRIVERS']=str(out/'raw_driver')
env['MAP_PUBLICATION_C_SHADOW']=str(out/'c-shadow')
env['MAP_PUBLICATION_CC']='/opt/homebrew/opt/llvm/bin/clang'
args=[sys.executable,'-m','unittest','-f','-v']+['tests.test_map_field_publication.MapFieldPublication.'+x for x in ['test_unchanged_and_concat','test_nested_u8','test_independent_shadow_omission','test_mixed_omission']]
t=time.monotonic();p=run_captured(args,cwd=root,env=env,timeout=900)
(out/'corpus.stdout').write_bytes(p.stdout);(out/'corpus.stderr').write_bytes(p.stderr);(out/'corpus.json').write_text(json.dumps(dict(argv=args,exit=p.returncode,seconds=time.monotonic()-t,selectors={k:v for k,v in env.items() if k.startswith('MAP_PUBLICATION_')}),indent=2)+'\n')
print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
