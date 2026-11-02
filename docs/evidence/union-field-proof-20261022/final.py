from pathlib import Path
import os,subprocess,json,time
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path('/private/tmp/nanolang-union-proof-20261022');rows=[]
e=os.environ.copy();e.update(json.loads((out/'target-env.json').read_text()))
commands=[('final-union',['python3','-m','unittest','-f','-v','tests.test_union_field_proof'])]
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for observed in [False,True]:
  commands.append((Path(cc).name+('-observed' if observed else ''),[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-fsyntax-only',*(['-DUFP_ALLOC'] if observed else []),'tests/nanoisa/test_union_field_proof.c']))
for name,cmd in commands:
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env=e,capture_output=True,timeout=240)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
 rows.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'final-status.json').write_text(json.dumps(rows,indent=2))
 print(name,p.returncode,p.stdout.decode()[-3200:],p.stderr.decode()[-3000:],flush=True)
 if p.returncode:break
