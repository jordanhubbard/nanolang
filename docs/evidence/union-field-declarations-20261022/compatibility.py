from pathlib import Path
import subprocess,os,shlex,json,time
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path('/private/tmp/nanolang-union-declaration-20261022')
status=[]
def run(name,cmd,env=None):
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env=env,capture_output=True,timeout=180)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
 status.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'compatibility-status.json').write_text(json.dumps(status,indent=2))
 print(name,p.returncode,p.stdout.decode(errors='replace')[-1600:],p.stderr.decode(errors='replace')[-1800:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
line=subprocess.check_output(['make','-f','Makefile.gnu','-n','test-ownership-declaration-projection'],cwd=root,text=True).splitlines()[-1]
e=dict(x.split('=',1) for x in shlex.split(line) if x.startswith('DECLARATION_'))
e['DECLARATION_OBJECTS']=' '.join(str(Path('/Users/jkh/Src/nanolang-51-harness')/p) for p in shlex.split(e['DECLARATION_OBJECTS']))
e['DECLARATION_CC']='/opt/homebrew/opt/llvm/bin/clang';e['DECLARATION_CFLAGS']='-Isrc -fsanitize=address,undefined -fno-sanitize-recover=all'
for k,v in list(e.items()):e[k.replace('DECLARATION_','ORDINARY_ARRAY_')]=v
env=os.environ.copy();env.update(e);(out/'target-env.json').write_text(json.dumps(e,indent=2))
run('final-and-neighbors',['python3','-m','unittest','-f','-v','tests.test_union_field_declarations','tests.test_ownership_declaration_projection','tests.test_ordinary_array_authority','tests.test_map_declarations'],env)
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src in ['src/nanoisa/ownership_contracts.c','tests/nanoisa/test_union_field_declarations.c']:
  for instrument in ([False,True] if 'tests/' in src else [False]):
   name=Path(cc).name+'-'+Path(src).stem+('-observed' if instrument else '')
   run(name,[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-fsyntax-only',*(['-DOAA_INSTRUMENT'] if instrument else []),src])
base=json.loads((root/'docs/evidence/map-string-consumer-20261022/budget-commands.json').read_text())
for row in base:
 cmd=[x.replace('/private/tmp/nanolang-map-string-consumer-20261022/budget-',str(out/'budget-')) for x in row['command']]
 run(row['name'],cmd)
