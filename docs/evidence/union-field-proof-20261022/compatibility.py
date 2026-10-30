from pathlib import Path
import os,subprocess,json,shlex,time
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path('/private/tmp/nanolang-union-proof-20261022');rows=[]
def run(name,cmd,env=None):
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env=env,capture_output=True,timeout=240)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
 rows.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'compatibility-status.json').write_text(json.dumps(rows,indent=2))
 print(name,p.returncode,p.stdout.decode(errors='replace')[-3000:],p.stderr.decode(errors='replace')[-3000:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
line=subprocess.check_output(['make','-f','Makefile.gnu','-n','test-union-field-proof'],cwd=root,text=True).splitlines()[-1]
e=dict(x.split('=',1) for x in shlex.split(line) if x.startswith('RECORD_ARRAY_'))
e['RECORD_ARRAY_OBJECTS']=' '.join(str(Path('/Users/jkh/Src/nanolang-51-harness')/p) for p in shlex.split(e['RECORD_ARRAY_OBJECTS']))
e['RECORD_ARRAY_CC']='/opt/homebrew/opt/llvm/bin/clang';e['RECORD_ARRAY_CFLAGS']='-Isrc -fsanitize=address,undefined -fno-sanitize-recover=all'
e['UNION_PROOF_MODULES']=' '.join(str(Path('/private/tmp/nanolang-union-field-baseline-20261022')/(name+'.nvm')) for name in ['seed','c-shadow','canonical','canonical-shadow','raw-program','raw-shadow'])
env=os.environ.copy();env.update(e);(out/'target-env.json').write_text(json.dumps(e,indent=2))
run('permanent-and-neighbors',['python3','-m','unittest','-f','-v','tests.test_union_field_proof','tests.test_record_array_origins','tests.test_record_map_origins'],env)
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src in ['src/nanoisa/managed_array_shapes.c','src/nanoisa/verifier.c','tests/nanoisa/test_union_field_proof.c']:
  for instrument in ([False,True] if 'tests/' in src else [False]):
   name=Path(cc).name+'-'+Path(src).stem+('-observed' if instrument else '')
   run(name,[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-fsyntax-only',*(['-DUFP_ALLOC'] if instrument else []),src])
base=json.loads((root/'docs/evidence/map-string-consumer-20261022/budget-commands.json').read_text())
for row in base:
 cmd=[x.replace('/private/tmp/nanolang-map-string-consumer-20261022/budget-',str(out/'budget-')) for x in row['command']]
 run(row['name'],cmd)
actual=(out/'old-six-budget.stdout').read_bytes();expected=(root/'docs/evidence/map-string-consumer-20261022/old-six-budget.stdout').read_bytes()
if actual!=expected:raise SystemExit('old budget mismatch')
print('six old budget rows byte-identical',flush=True)
