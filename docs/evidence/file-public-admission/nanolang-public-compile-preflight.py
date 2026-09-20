import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir();os.chdir(root)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
files=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
def sources():return {p:sha(p) for p in files if p and Path(p).is_file()}
before=sources();save('sources-before.json',before)
darwin=sys.platform=='darwin';sdk=os.environ.get('SDKROOT','')
compilers=[('apple','/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang',[]),('homebrew','/opt/homebrew/opt/llvm/bin/clang',[])] if darwin else [('gcc','/usr/bin/gcc',[]),('clang','/usr/local/bin/clang',['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'])]
save('tools.json',{n:{'path':str(Path(c).resolve()),'sha256':sha(c)} for n,c,_ in compilers})
results=[]
for name,cc,extra in compilers:
 for instrument in (False,True):
  for capture in (False,True):
   label=f'{name}-'+('instrumented' if instrument else 'linked')+'-'+('capture' if capture else 'replay')
   argv=[cc,*extra,'-std=c11','-D_DEFAULT_SOURCE','-DNVM_FILE_VM_PRIVATE','-DNVM_FILE_NATIVE_PRIVATE','-DNVM_FILE_PUBLIC_ENGINE','-g','-O1','-Wall','-Wextra','-Werror','-I.','-Isrc','-Isrc/nanoisa','-pthread']
   if darwin:argv+=['-I'+sdk+'/usr/include/ffi','-I/opt/homebrew/opt/openssl@3/include']
   if instrument:argv+=['-DHOSTED_INSTRUMENT']
   if capture:argv+=['-DFILE_NATIVE_CAPTURE']
   argv+=['-c','tests/nanoisa/test_file_public.c','-o',str(out/(label+'.o'))]
   save(label+'-command.json',argv);start=time.monotonic()
   with (out/(label+'.log')).open('w') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=120)
   results.append({'label':label,'status':r.returncode,'seconds':round(time.monotonic()-start,3)});save('results.json',results)
   print(results[-1],flush=True)
   if r.returncode:sys.exit(r.returncode)
after=sources();save('sources-after.json',after);assert before==after
save('artifacts.json',{p.name:sha(p) for p in out.glob('*.o')})
