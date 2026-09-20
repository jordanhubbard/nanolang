from pathlib import Path
import subprocess,os,json,hashlib,time,signal
out=Path('/tmp/nanolang-darwin-asan-startup-control');out.mkdir(exist_ok=False)
source=out/'hello.c';source.write_text('#include <stdio.h>\nint main(void) { puts("I entered main"); return 0; }\n')
def run(label,args,timeout=30):
 (out/(label+'-command.json')).write_text(json.dumps({'argv':args,'timeout':timeout})+'\n');start=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
  p=subprocess.Popen(args,stdout=stdout,stderr=stderr,start_new_session=True)
  try:rc=p.wait(timeout=timeout);timed=False
  except subprocess.TimeoutExpired:timed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=10)
 try:os.killpg(p.pid,0);gone=False
 except ProcessLookupError:gone=True
 d={'returncode':rc,'timeout':timed,'seconds':time.monotonic()-start,'group_disappeared':gone};(out/(label+'-status.json')).write_text(json.dumps(d)+'\n');print(label,d,flush=True);return d
for label,cc in [('apple','/usr/bin/cc'),('homebrew','/opt/homebrew/opt/llvm/bin/clang')]:
 run(label+'-version',[cc,'--version']);run(label+'-resource-dir',[cc,'-print-resource-dir'])
 exe=out/label
 assert run(label+'-build',[cc,'-std=c11','-O1','-g','-fno-omit-frame-pointer','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(source),'-o',str(exe)])['returncode']==0
 run(label+'-libraries',['/usr/bin/otool','-L',str(exe)])
 run(label+'-run',[str(exe)],10)
(out/'environment.json').write_text(json.dumps({k:os.environ.get(k) for k in ['PATH','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','SDKROOT','DEVELOPER_DIR']},indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps({str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()},indent=2)+'\n')
