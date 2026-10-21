from pathlib import Path
import subprocess,json,time
out=Path(__file__).parent;rows=[]
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src in ['src/nanovirt/codegen.c','modules/nanoisa/nanoisa.c','tests/nanovirt/test_map_publication.c']:
   name=Path(cc).name+'-'+Path(src).stem;args=[cc,'-std=c11','-D_DEFAULT_SOURCE','-Isrc','-Isrc/nanoisa','-Wall','-Wextra','-Werror','-fsyntax-only',src];r=subprocess.run(args,capture_output=True);(out/(name+'.stderr')).write_bytes(r.stderr);rows.append(dict(name=name,argv=args,status=r.returncode));(out/'strict.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0 and not r.stderr,(name,r.stderr)
print('PASS strict6 Clang/GCC16 changed publisher/facade/include harness')
