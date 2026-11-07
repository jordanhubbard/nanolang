from pathlib import Path
import json,subprocess,sys,time
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
w=Path(__file__).parent;old=Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-admission-ua9vp_uj')
vm=w/'baseline-vm.c';vm.write_bytes(subprocess.check_output(['git','show','HEAD:src/nanovm/vm.c']))
s=(r/'tests/nanovm/test_ordinary_admission.c').read_text().replace('"../../src/nanovm/vm.c"','"'+str(vm)+'"').replace('"../../','"'+str(r)+'/')
h=w/'baseline-admission.c';h.write_text(s)
base=json.loads((old/'001-command.json').read_text())['argv'];rows=[]
for mode in ['baseline-O1']:
 args=base[:];idx=args.index('tests/nanovm/test_ordinary_admission.c');args[idx]=str(h) if mode=='baseline-O1' else 'tests/nanovm/test_ordinary_admission.c';args[args.index('-o')+1]=str(w/mode)
 if mode=='baseline-O1':args+=['-iquote',str(r/'src/nanovm'),'-iquote',str(r/'tests/nanovm')]
 if mode=='current-noinline':args+=['-fno-inline-functions']
 if mode=='current-normal':
  # Match normal test target's empty harness CFLAGS; retained providers still need sanitizer runtime linkage.
  idx=args.index('-std=c11');args=[args[0],*args[idx:]];args+=['-fsanitize=address,undefined']
  # This enables instrumentation again: instead use link-only sanitized runtime below.
  args=args[:-1]+['-Wl,-rpath,/opt/homebrew/Cellar/llvm/23.1.2/lib/clang/23/lib/darwin','/opt/homebrew/Cellar/llvm/23.1.2/lib/clang/23/lib/darwin/libclang_rt.asan_osx_dynamic.dylib','/opt/homebrew/Cellar/llvm/23.1.2/lib/clang/23/lib/darwin/libclang_rt.ubsan_osx_dynamic.dylib']
 for phase,argv in [('build',args),('run',[str(w/mode)])]:
  t=time.monotonic();p=run_captured(argv,cwd=r,timeout=180);(w/(mode+'-corrected-'+phase+'.stdout')).write_bytes(p.stdout);(w/(mode+'-corrected-'+phase+'.stderr')).write_bytes(p.stderr);rows.append({'mode':mode,'phase':phase,'argv':argv,'exit':p.returncode,'seconds':time.monotonic()-t});(w/'ordinary-baseline-corrected.json').write_text(json.dumps(rows,indent=2));print(mode,phase,p.returncode,p.stderr.decode()[-300:],flush=True)
  if phase=='build' and p.returncode:break
