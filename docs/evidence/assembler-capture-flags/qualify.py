from pathlib import Path
import subprocess,os,json,hashlib,sys,time,ctypes
root=Path('/home/jkh/Src/nanolang-capture-flags');out=Path('/tmp/nanolang-capture-flags-qualified');out.mkdir(exist_ok=False)
inputs=['Makefile.gnu','Makefile','src/runtime/assembler_capture.c','src/runtime/assembler_capture.h','tests/test_assembler_capture_records.py']
def hashes():return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in inputs}
def save(n,d):(out/n).write_text(json.dumps(d,indent=2)+'\n')
def run(n,args,env=None):
 save(n+'-command.json',{'argv':args,'cwd':str(root),'selected_environment':env or {}});start=time.monotonic()
 with (out/(n+'.log')).open('wb') as log:
  try:p=subprocess.run(args,cwd=root,env=dict(os.environ,**(env or {})),stdout=log,stderr=subprocess.STDOUT,timeout=90)
  except subprocess.TimeoutExpired:save(n+'-status.json',{'timeout':True});raise
 save(n+'-status.json',{'returncode':p.returncode,'seconds':time.monotonic()-start,'timeout':False});assert p.returncode==0,n
save('inputs-before.json',hashes())
probe=out/'flag-probe.h';probe.write_text('#if NANO_CAPTURE_CPP_PROBE != 17 || NANO_CAPTURE_C_PROBE != 25\n#error I require caller preprocessing and compiler flags\n#endif\nconst int nano_capture_flag_probe = NANO_CAPTURE_CPP_PROBE + NANO_CAPTURE_C_PROBE;\n')
base='-Wall -Wextra -Werror -std=c99 -O2 -D_GNU_SOURCE -DNANO_CAPTURE_C_PROBE=25'
cases=[('gcc-default',[]),('clang-selected',['CC=/usr/local/bin/clang','CFLAGS='+base+' --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','CPPFLAGS=-DNANO_CAPTURE_CPP_PROBE=17 -include '+str(probe)]),('gcc-parent-sanitizers',['CC=/bin/gcc','CFLAGS='+base+' -fsanitize=address,undefined','LDFLAGS=-fsanitize=address,undefined','CPPFLAGS=-DNANO_CAPTURE_CPP_PROBE=17 -include '+str(probe)]),('gcc-helper-ubsan',['CC=/bin/gcc','NANO_AS_CAPTURE_CFLAGS='+base+' -fsanitize=undefined -fno-sanitize-recover=all','NANO_AS_CAPTURE_LDFLAGS=-fsanitize=undefined','CPPFLAGS=-DNANO_CAPTURE_CPP_PROBE=17 -include '+str(probe)])]
for name,args in cases:
 directory=out/name;helper=directory/'nano_as_capture.so'
 run(name+'-build',['make','-B',str(helper),'BIN_DIR='+str(directory),*args])
 if name!='gcc-default':
  library=ctypes.CDLL(str(helper));assert ctypes.c_int.in_dll(library,'nano_capture_flag_probe').value==42
 log=(out/(name+'-build.log')).read_text();commands=[line for line in log.splitlines() if ' -shared ' in line and 'assembler_capture.c' in line];assert len(commands)==1
 if name=='gcc-parent-sanitizers':assert '-fsanitize=' not in commands[0]
 if name=='gcc-helper-ubsan':assert '-fsanitize=undefined' in commands[0]
 run(name+'-corpus',[sys.executable,'-m','unittest','-v','tests.test_assembler_capture_records'],{'NANO_AS_CAPTURE_TEST_HELPER':str(helper),'UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
 save(name+'-product.json',{'path':str(helper),'sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'bytes':helper.stat().st_size})
save('inputs-after.json',hashes());assert json.loads((out/'inputs-before.json').read_text())==hashes();print('PASS four actual Make builds and capture/replay corpus runs',flush=True)
