from pathlib import Path
import subprocess,json,time
out=Path(__file__).parent;rows=[]
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for mode,macros in [('default',[]),('private',['-DNANO_RECORD_MAP_PRIVATE_NATIVE','-DNANO_RECORD_MAP_PRIVATE_RUNTIME','-DNANO_RECORD_MAP_PRIVATE_PREPARATION','-DNANO_RECORD_MAP_PRIVATE_HEAP','-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME'])]:
  for src in ['src/nanoisa/managed_array_shapes.c','src/nanoisa/nvm2c.c','tests/nanoisa/test_record_map_publication.c']:
   name=Path(cc).name+'-'+mode+'-'+Path(src).stem;args=[cc,'-std=c11','-D_DEFAULT_SOURCE','-Isrc','-Isrc/nanoisa','-Wall','-Wextra','-Werror',*macros,'-fsyntax-only',src];r=subprocess.run(args,capture_output=True);(out/(name+'.stderr')).write_bytes(r.stderr);rows.append(dict(name=name,argv=args,status=r.returncode));(out/'strict.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0 and not r.stderr,(name,r.stderr)
print('PASS strict12 Clang/GCC16 default/private')
