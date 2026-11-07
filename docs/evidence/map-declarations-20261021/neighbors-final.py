import pathlib,json,sys,os,time,shlex
child=pathlib.Path.cwd();sys.path.insert(0,str(child));from tests.native_process import run_captured
w=pathlib.Path(__file__).parent;env=os.environ.copy();env.update(json.loads((w/'environment.json').read_text()));env['OWNERSHIP_LAYOUT_CC']=env['DECLARATION_CC'];env['OWNERSHIP_LAYOUT_CFLAGS']=env['DECLARATION_CFLAGS'];receipt=[]
def run(name,args):
 t=time.monotonic();r=run_captured(list(map(str,args)),cwd=child,env=env,timeout=180);(w/(name+'.stdout')).write_bytes(r.stdout);(w/(name+'.stderr')).write_bytes(r.stderr);receipt.append({'name':name,'argv':list(map(str,args)),'exit':r.returncode,'seconds':time.monotonic()-t});(w/'neighbors-final.json').write_text(json.dumps(receipt,indent=2));print(name,r.returncode,flush=True);assert r.returncode==0,r.stderr.decode();return r
r=run('declaration-neighbors-final',[sys.executable,'-m','unittest','-v','tests.test_ownership_declaration_projection','tests.test_ownership_layouts_private']);print(r.stdout.decode(),r.stderr.decode(),flush=True)
cc=env['DECLARATION_CC'];flags=['-std=c11','-D_DEFAULT_SOURCE','-g','-O1','-Wall','-Wextra','-Werror',*shlex.split(env['DECLARATION_CFLAGS'])]
run('old-contract-build',[cc,*flags,'tests/nanoisa/test_ownership_contracts.c','src/nanoisa/ownership_contracts.c','src/nanoisa/nvm_v2_layouts.c',*shlex.split(env['DECLARATION_OBJECTS']),*shlex.split(env['DECLARATION_LDFLAGS']),'-o',w/'old-contract'])
r=run('old-contract',[w/'old-contract',w/'old-ordinary.nvm',w/'old-borrowed.nvm']);print(r.stdout.decode(),flush=True)
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for source in ['src/nanoisa/ownership_contracts.c','src/nanoisa/nvm_v2_layouts.c','tests/nanoisa/test_map_declarations.c']:
  run(pathlib.Path(cc).name+'-'+pathlib.Path(source).stem+'-syntax',[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa','-fsyntax-only',source])
print('ALL NEIGHBORS/STRICT PASS',flush=True)
