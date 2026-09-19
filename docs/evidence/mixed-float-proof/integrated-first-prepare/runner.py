from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();d=Path('/tmp/nanolang-mixed-proof-final-'+pin[:8]);d.mkdir(exist_ok=True)
def run(name,cmd,extra):
 start=time.monotonic()
 with (d/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**extra},timeout=240);status=r.returncode
  except subprocess.TimeoutExpired:status='timeout'
 result={'name':name,'command':cmd,'environment':extra,'status':status,'seconds':time.monotonic()-start}
 (d/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,status,flush=True);return status
files=[p for p in subprocess.check_output(['git','ls-files','src','tests','Makefile.gnu'],text=True).splitlines() if Path(p).is_file()]
sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files};(d/'sources-before.json').write_text(json.dumps(sources,indent=2)+'\n')
if run('prepare',['make','-j2','nano_vm','nvm2c','nvm2wasm','nanoisa_dump'],{'CC':'cc'}):raise SystemExit(1)
paths={Path('/tmp/nanolang-projection-clang')}
for name in ['cc','gcc','clang','make','python3','ld','as']:paths.add(Path(shutil.which(name)).resolve())
paths.add(Path(subprocess.check_output(['cc','-print-prog-name=cc1'],text=True).strip()).resolve())
paths.update(Path('obj/nanoisa').glob('*.o'));paths.update([Path('obj/utf8.o'),Path('obj/nanovm/vm_decode.o'),Path('obj/nanovm/vm_dispatch.o')])
paths.update(Path('bin')/n for n in ['nano_vm','nvm2c','nvm2llvm','nvm2wasm','nanoisa_dump'])
before={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(d/'tools-before.json').write_text(json.dumps(before,indent=2)+'\n')
(d/'environment.json').write_text(json.dumps({'pin':pin,'PATH':os.environ.get('PATH'),'compilers':{name:subprocess.check_output([name,'--version'],text=True).splitlines()[0] for name in ['cc','/tmp/nanolang-projection-clang'] }},indent=2)+'\n')
for name,cmd,extra in [('gcc',['make','test-mixed-float-proof'],{'CC':'cc'}),('clang',['make','test-mixed-float-proof'],{'CC':'/tmp/nanolang-projection-clang','MIXED_CFLAGS':'-fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'}),('adjacent',['make','-j2','test-mixed-layout-view','test-retained-layouts','test-ownership-contracts','test-verifier','test-managed-record-plan'],{'CC':'cc','NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'})]:
 if run(name,cmd,extra):break
after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before};(d/'tools-after.json').write_text(json.dumps(after,indent=2)+'\n')
added={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('obj/nanoisa').glob('*.o') if str(p.resolve()) not in before};(d/'additional-objects.json').write_text(json.dumps(added,indent=2)+'\n')
end={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources};(d/'sources-after.json').write_text(json.dumps(end,indent=2)+'\n')
assert before==after;assert sources==end
(d/'identity-result.json').write_text(json.dumps({'unchanged_tools':len(before),'unchanged_sources':len(sources),'additional_objects':len(added)},indent=2)+'\n')
print('unchanged',len(before),len(sources),'additional',len(added),flush=True)
