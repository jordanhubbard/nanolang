import pathlib,subprocess,hashlib,json,os,time
r=pathlib.Path('/home/jkh/Src/nanolang-union-metadata-ownership'); o=pathlib.Path('/tmp/nanolang-union-owner-10fc-gates');o.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=r,text=True).strip()
files=git('ls-files','src','tests','Makefile','Makefile.gnu','modules').splitlines()
def sources():return {p:sha(r/p) for p in files if (r/p).is_file()}
b=sources();(o/'source-before.json').write_text(json.dumps(b,indent=2));(o/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());pin=git('rev-parse','HEAD');report={'pin':pin,'steps':[]}
env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';env.pop('LSAN_OPTIONS',None)
steps=[('normal',['make','-j2','test-union-metadata-ownership']),('gcc-sanitized',['make','-j2','test-union-metadata-ownership','OBJ_DIR=obj-union-gcc','CC=gcc','CFLAGS=-Wall -Wextra -Werror -std=c99 -g -O0 -fPIC -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all','LDFLAGS=-lm -fsanitize=address,undefined']),('adjacent',['make','-j2','test-module-metadata','test-env-scoping','NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'])]
status=1
try:
 for name,cmd in steps:
  assert git('rev-parse','HEAD')==pin
  print('START',name,flush=True);t=time.monotonic()
  with (o/(name+'.log')).open('wb') as f:p=subprocess.run(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
  status=p.returncode;report['steps'].append(dict(name=name,command=cmd,status=status,seconds=round(time.monotonic()-t,3),log_sha256=sha(o/(name+'.log'))));(o/'manifest.json').write_text(json.dumps(report,indent=2));print('END',name,status,flush=True)
  if status:break
finally:
 a=sources();(o/'source-after.json').write_text(json.dumps(a,indent=2));report['source_unchanged']=a==b;report['head_unchanged']=git('rev-parse','HEAD')==pin;report['artifacts']={str(p.relative_to(r)):sha(p) for d in ('obj','obj-union-gcc','bin') for p in (r/d).rglob('*') if p.is_file()};(o/'manifest.json').write_text(json.dumps(report,indent=2))
raise SystemExit(status)
