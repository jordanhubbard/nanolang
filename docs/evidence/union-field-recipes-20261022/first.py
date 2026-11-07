from pathlib import Path
import subprocess,os,json,time,hashlib
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path('/private/tmp/nanolang-union-recipes-20261022');rows=[]
e=os.environ.copy();env=json.loads((root/'docs/evidence/union-field-proof-20261022/target-env.json').read_text());env['UNION_EXECUTION_MODULES']=env['UNION_PROOF_MODULES'];e.update(env);(out/'target-env.json').write_text(json.dumps(env,indent=2))
sources=['src/nanoisa/managed_array_shapes.c','src/nanoisa/managed_record_array_execution.inc','src/nanoisa/managed_union_field_execution.inc','src/nanoisa/managed_union_field_execution_private.h','src/nanoisa/union_field_analysis.inc','tests/nanoisa/test_union_field_execution.c','tests/nanoisa/test_union_field_proof.c','tests/test_union_field_execution.py']
(out/'first-source-pins.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},indent=2))
commands=[]
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src,flags in [('src/nanoisa/managed_array_shapes.c',[]),('tests/nanoisa/test_union_field_execution.c',[]),('tests/nanoisa/test_union_field_execution.c',['-DUFP_ALLOC'])]:
  commands.append((Path(cc).name+'-'+Path(src).stem+('-observed' if flags else ''),[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc',*flags,'-fsyntax-only',src]))
commands.append(('first-recipes',['python3','-m','unittest','-f','-v','tests.test_union_field_execution']))
for name,cmd in commands:
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env=e,capture_output=True,timeout=240)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'first-status.json').write_text(json.dumps(rows,indent=2));print(name,p.returncode,p.stdout.decode()[-2500:],p.stderr.decode()[-4000:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
