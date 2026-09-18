from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();d=Path('/tmp/nanolang-mixed-view-tools-'+pin[:8]);d.mkdir(exist_ok=True)
def tool_inputs():
 paths={Path('/tmp/nanolang-projection-clang')}
 for name in ['cc','gcc','clang','make','python3','ld','as']:
  paths.add(Path(shutil.which(name)).resolve())
 cc1=subprocess.check_output(['cc','-print-prog-name=cc1'],text=True).strip()
 paths.add(Path(cc1).resolve())
 paths.update(Path('obj/nanoisa').glob('*.o'));paths.add(Path('obj/utf8.o'))
 paths.update([Path('obj/nanovm/vm_decode.o'),Path('obj/nanovm/vm_dispatch.o')])
 return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
before=tool_inputs();(d/'tools-before.json').write_text(json.dumps(before,indent=2)+'\n')
files=[p for p in subprocess.check_output(['git','ls-files','src','tests','Makefile.gnu'],text=True).splitlines() if Path(p).is_file()]
sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files};(d/'sources-before.json').write_text(json.dumps(sources,indent=2)+'\n')
cases=[('gcc',['make','test-mixed-layout-view'],{}),('clang',['make','test-mixed-layout-view'],{'CC':'/tmp/nanolang-projection-clang','MIXED_CFLAGS':'-fsanitize=address,undefined -fno-sanitize-recover=all','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'}),('adjacent',['make','-j2','test-retained-layouts','test-ownership-contracts','test-verifier','test-managed-record-plan'],{'NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'})]
results=[]
for name,cmd,extra in cases:
 start=time.monotonic()
 with (d/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**extra},timeout=240);status=r.returncode
  except subprocess.TimeoutExpired:status='timeout'
 results.append({'name':name,'command':cmd,'environment':extra,'status':status,'seconds':time.monotonic()-start});print(name,status,flush=True)
 if status:break
(d/'result.json').write_text(json.dumps({'pin':pin,'results':results},indent=2)+'\n')
after=tool_inputs();(d/'tools-after.json').write_text(json.dumps(after,indent=2)+'\n')
end={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files};(d/'sources-after.json').write_text(json.dumps(end,indent=2)+'\n')
assert before==after;assert sources==end
print('unchanged tools/objects',len(before),'sources',len(sources),flush=True)
