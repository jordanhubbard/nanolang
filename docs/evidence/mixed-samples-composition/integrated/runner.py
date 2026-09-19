from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path.cwd();pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();dest=Path('/tmp/nanolang-composition-'+pin[:8]);dest.mkdir(exist_ok=False)
files=subprocess.check_output(['git','ls-files','src','tests','Makefile.gnu'],text=True).splitlines()
def hashes(paths):return {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}
sources=hashes(files);(dest/'sources-before.json').write_text(json.dumps(sources,indent=2)+'\n')
tools={Path(shutil.which(n)).resolve() for n in ['cc','clang','make','python3','ld','as']};tools.add(Path(subprocess.check_output(['cc','-print-prog-name=cc1'],text=True).strip()).resolve());tools.add(Path('/tmp/nanolang-projection-clang'));tools_before=hashes(tools);(dest/'tools-before.json').write_text(json.dumps(tools_before,indent=2)+'\n')
result=[]
def run(name,cmd,extra):
 start=time.monotonic()
 with (dest/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**extra},timeout=180);status=r.returncode
  except subprocess.TimeoutExpired:status='timeout'
 item={'name':name,'command':cmd,'environment':extra,'status':status,'seconds':time.monotonic()-start};result.append(item);(dest/(name+'.json')).write_text(json.dumps(item,indent=2)+'\n');print(name,status,flush=True);return status
try:
 if run('gcc',['make','-j2','test-mixed-samples'],{'CC':'cc'}):raise SystemExit(1)
 objects=hashes([*Path('obj/nanoisa').glob('*.o'),Path('obj/utf8.o'),Path('obj/nanovm/vm_decode.o'),Path('obj/nanovm/vm_dispatch.o')]);(dest/'objects-after-build.json').write_text(json.dumps(objects,indent=2)+'\n')
 if run('clang',['make','test-mixed-samples'],{'CC':'/tmp/nanolang-projection-clang','MIXED_CFLAGS':'-fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'}):raise SystemExit(1)

finally:
 (dest/'results.json').write_text(json.dumps({'pin':pin,'results':result},indent=2)+'\n');(dest/'sources-after.json').write_text(json.dumps(hashes(files),indent=2)+'\n');(dest/'tools-after.json').write_text(json.dumps(hashes(tools),indent=2)+'\n')
 assert sources==hashes(files);assert tools_before==hashes(tools)
 if 'objects' in globals():
  (dest/'objects-after.json').write_text(json.dumps(hashes(objects),indent=2)+'\n');assert objects==hashes(objects)
 print(dest,flush=True)
