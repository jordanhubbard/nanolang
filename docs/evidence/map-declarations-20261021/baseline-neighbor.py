import pathlib,subprocess,json,shlex,sys,time
child=pathlib.Path.cwd();sys.path.insert(0,str(child));from tests.native_process import run_captured
w=pathlib.Path(__file__).parent;base=w/'baseline';src=base/'src/nanoisa';src.mkdir(parents=True,exist_ok=True)
for p in (child/'src').iterdir():
 if p.name!='nanoisa' and not (base/'src'/p.name).exists():(base/'src'/p.name).symlink_to(p,target_is_directory=p.is_dir())
for p in (child/'src/nanoisa').iterdir():
 q=src/p.name
 if not q.exists():q.symlink_to(p,target_is_directory=p.is_dir())
changed=['ownership_contracts.c','ownership_contracts.h','ordinary_array_authority.inc','ownership_declaration_projection.inc','ownership_layouts_private.h','nvm_v2_layouts.c']
for name in changed:
 q=src/name
 if q.is_symlink():q.unlink()
 q.write_bytes(subprocess.check_output(['git','show','HEAD:src/nanoisa/'+name],cwd=child))
test=w/'baseline-test.c';text=(child/'tests/nanoisa/test_ownership_declaration_projection.c').read_text().replace('#include "test_ordinary_array_authority.c"','#include "'+str(child/'tests/nanoisa/test_ordinary_array_authority.c')+'"').replace('#include "../../src/', '#include "'+str(child/'src')+'/').replace('#define BAD(change,status) do {mixed_make', '#define BAD(change,status) do {fprintf(stderr,"I probe BAD line%u\\n",__LINE__);mixed_make')
test.write_text(text);env=json.loads((w/'environment.json').read_text());cc=env['DECLARATION_CC'];flags=['-std=c11','-D_DEFAULT_SOURCE','-O1','-g','-Wall','-Wextra','-Werror',*shlex.split(env['DECLARATION_CFLAGS'])]
commands=[]
for name,args in [('baseline-neighbor-build',[cc,*flags,str(test),str(src/'ownership_contracts.c'),str(src/'nvm_v2_layouts.c'),*shlex.split(env['DECLARATION_OBJECTS']),*shlex.split(env['DECLARATION_LDFLAGS']),'-o',str(w/'baseline-neighbor')]),('baseline-neighbor',[str(w/'baseline-neighbor')])]:
 r=run_captured(args,cwd=child,timeout=180);(w/(name+'.stdout')).write_bytes(r.stdout);(w/(name+'.stderr')).write_bytes(r.stderr);commands.append({'name':name,'argv':args,'exit':r.returncode});(w/'baseline-neighbor.json').write_text(json.dumps(commands,indent=2));print(name,r.returncode,r.stdout.decode(),r.stderr.decode());
 if name.endswith('build'):assert r.returncode==0
