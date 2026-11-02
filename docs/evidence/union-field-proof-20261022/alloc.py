from pathlib import Path
import subprocess,json,shlex,time,hashlib
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path('/private/tmp/nanolang-union-proof-20261022');rows=[]
cc='/opt/homebrew/opt/llvm/bin/clang';flags=['-std=c11','-D_DEFAULT_SOURCE','-Isrc','-Wall','-Wextra','-Werror','-g','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all']
def run(name,cmd):
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,capture_output=True,timeout=180)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
 rows.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'alloc-status.json').write_text(json.dumps(rows,indent=2))
 print(name,p.returncode,p.stdout.decode(errors='replace')[-2500:],p.stderr.decode(errors='replace')[-3000:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
e=json.loads(Path('/private/tmp/nanolang-union-declaration-20261022/target-env.json').read_text());objs=shlex.split(e['DECLARATION_OBJECTS']);objs=[p for p in objs if Path(p).stem!='managed_array_shapes']
for stem in ['verifier','verifier_types','ownership_contracts','nvm_v2_layouts','retained_layouts','vm_decode']:
 src=('src/nanovm/' if stem=='vm_decode' else 'src/nanoisa/')+stem+'.c';obj=str(out/(stem+'-alloc.o'))
 run('alloc-compile-'+stem,[cc,*flags,'-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c',src,'-o',obj])
 if any(Path(p).stem==stem for p in objs):objs=[obj if Path(p).stem==stem else p for p in objs]
 else:objs.append(obj)
cmd=[cc,*flags,'-DUFP_ALLOC','tests/nanoisa/test_union_field_proof.c','tests/nanoisa/record_array_alloc.c',*objs,*shlex.split(e['DECLARATION_LDFLAGS']),'-o',str(out/'alloc')]
(out/'alloc-object-pins.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in objs},indent=2))
run('alloc-build',cmd);run('alloc-run',[str(out/'alloc')])
