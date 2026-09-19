import os,subprocess,pathlib,json,hashlib,time,signal
r=pathlib.Path('/private/tmp/nanolang-owner-array-bits-15ae');o=pathlib.Path('/private/tmp/nanolang-owner-array-bits-15ae-evidence');o.mkdir(exist_ok=False)
env=os.environ.copy();env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['SDKROOT']=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
def run(name,args,cwd=None):
 start=time.monotonic()
 with (o/(name+'.log')).open('w') as f:
  p=subprocess.Popen(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try: rc=p.wait(timeout=600)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc=124
 (o/(name+'.json')).write_text(json.dumps({'command':args,'status':rc,'seconds':time.monotonic()-start},indent=2)+'\n');print(name,rc,flush=True)
 if rc:raise SystemExit(rc)
run('clone',['git','clone','--no-hardlinks','/Users/jkh/Src/nanolang',str(r)])
run('remote',['git','remote','set-url','origin','https://github.com/jordanhubbard/nanolang.git'],r)
run('fetch',['git','fetch','origin','test/owned-array-bits-boundaries'],r)
run('checkout',['git','checkout','--detach','6db8e572b'],r)
paths=[pathlib.Path('/usr/bin/clang'),pathlib.Path(subprocess.check_output(['xcrun','--find','clang'],text=True).strip()),pathlib.Path('/opt/homebrew/opt/llvm/bin/clang').resolve(),pathlib.Path('/opt/homebrew/bin/python3').resolve(),pathlib.Path('/usr/bin/make'),pathlib.Path('/opt/homebrew/bin/pkg-config').resolve()]
llvm=pathlib.Path('/opt/homebrew/opt/llvm').resolve();paths+=list((llvm/'etc/clang').glob('*.cfg'));paths+=list((llvm/'lib/clang').glob('*/lib/darwin/*asan*'));paths+=list((llvm/'lib/clang').glob('*/lib/darwin/*ubsan*'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(o/'external-tools.json').write_text(json.dumps({str(p):sha(p) for p in paths if p.is_file()},indent=2)+'\n')
run('identities',['/bin/sh','-c','hostname; sw_vers; uname -a; /usr/bin/clang --version; /opt/homebrew/opt/llvm/bin/clang --version; xcrun --show-sdk-path; xcrun --show-sdk-version; /usr/bin/make --version; /opt/homebrew/bin/python3 --version'],r)
files=subprocess.check_output(['git','ls-files','src','src_nano','tests','Makefile','Makefile.gnu'],cwd=r,text=True).splitlines();(o/'setup-source-before.json').write_text(json.dumps({f:sha(r/f) for f in files if (r/f).is_file()},indent=2)+'\n')
supp=o/'prepare.mk';supp.write_text('include Makefile.gnu\n.PHONY: mutation-prepare\nmutation-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)\n')
run('prepare',['/usr/bin/make','-f',str(supp),'-j4','CC=/usr/bin/clang','mutation-prepare'],r)
