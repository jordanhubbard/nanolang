from pathlib import Path
import os,subprocess,json,time,hashlib
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path('/private/tmp/nanolang-union-recipes-20261022');rows=[]
e=os.environ.copy();e.update(json.loads((out/'target-env.json').read_text()))
def run(name,cmd):
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env=e,capture_output=True,timeout=240)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'final-status.json').write_text(json.dumps(rows,indent=2));print(name,p.returncode,p.stdout.decode()[-3500:],p.stderr.decode()[-2800:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('final-recipes-proof-neighbors',['python3','-m','unittest','-f','-v','tests.test_union_field_execution','tests.test_union_field_proof','tests.test_record_array_execution','tests.test_record_map_execution'])
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src,flags in [('src/nanoisa/managed_array_shapes.c',[]),('tests/nanoisa/test_union_field_execution.c',[]),('tests/nanoisa/test_union_field_execution.c',['-DUFP_ALLOC'])]:
  run(Path(cc).name+'-'+Path(src).stem+('-observed' if flags else ''),[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc',*flags,'-fsyntax-only',src])
base=json.loads((root/'docs/evidence/map-string-consumer-20261022/budget-commands.json').read_text())
for row in base:
 cmd=[x.replace('/private/tmp/nanolang-map-string-consumer-20261022/budget-',str(out/'budget-')) for x in row['command']]
 run(row['name'],cmd)
actual=(out/'old-six-budget.stdout').read_bytes();expected=(root/'docs/evidence/map-string-consumer-20261022/old-six-budget.stdout').read_bytes()
assert actual==expected,'old budget mismatch'
print('six old budget rows byte-identical',flush=True)
