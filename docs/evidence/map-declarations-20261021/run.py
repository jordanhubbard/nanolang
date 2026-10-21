import pathlib,subprocess,shlex,json,sys,os,time,hashlib
child=pathlib.Path.cwd();root=pathlib.Path('/Users/jkh/Src/nanolang-51-harness');work=pathlib.Path(__file__).parent
sys.path.insert(0,str(child))
from tests.native_process import run_captured
text='include Makefile.gnu\n.PHONY: map-link-inputs\nmap-link-inputs:\n\t@echo $(sort $(NANOISA_OBJECTS)) $(NANOISA_UTF8)\n\t@echo $(LDFLAGS)\n'
r=subprocess.run(['make','--no-print-directory','-s','-f','-','map-link-inputs'],input=text,text=True,cwd=root,capture_output=True,timeout=20);assert r.returncode==0,r.stderr
lines=r.stdout.splitlines();objects=[str(root/p) for p in shlex.split(lines[0]) if pathlib.Path(p).stem not in ('ownership_contracts','nvm_v2_layouts')]
cc='/opt/homebrew/opt/llvm/bin/clang';flags='-fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -Isrc/nanoisa'
env=os.environ.copy();env.update(DECLARATION_CC=cc,DECLARATION_CFLAGS=flags,DECLARATION_OBJECTS=shlex.join(objects),DECLARATION_LDFLAGS=lines[1])
pins={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in [cc,*objects]}
(work/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('DECLARATION_')},indent=2));(work/'before.json').write_text(json.dumps(pins,indent=2))
start=time.monotonic();args=[sys.executable,'-m','unittest','-v','tests.test_map_declarations']
r=run_captured(args,cwd=child,env=env,timeout=900)
(work/'first.stdout').write_bytes(r.stdout);(work/'first.stderr').write_bytes(r.stderr)
(work/'first.json').write_text(json.dumps({'argv':args,'exit':r.returncode,'seconds':time.monotonic()-start},indent=2))
print(r.stdout.decode(),r.stderr.decode());print('TERMINAL',r.returncode,flush=True)
