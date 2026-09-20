import pathlib,subprocess,hashlib,json,os,time,signal,shutil
r=pathlib.Path('/home/jkh/Src/nanolang-checker-metadata-ownership');o=pathlib.Path('/tmp/nanolang-checker-owner-f7c-gates');o.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=r,text=True).strip()
files=git('ls-files').splitlines()
def sources():return {p:sha(r/p) for p in files if (r/p).is_file()}
def tools():
 paths=['/usr/bin/gcc','/usr/bin/make',shutil.which('python3'),'/usr/bin/ld','/usr/bin/as']
 paths += [subprocess.check_output(['/usr/bin/gcc','-print-prog-name='+n],text=True).strip() for n in ['cc1','collect2']]
 return {p:{'resolved':str(pathlib.Path(p).resolve()),'sha256':sha(pathlib.Path(p).resolve())} for p in paths}
def providers():return {str(p.relative_to(r)):sha(p) for d in ('obj','obj-checker-gcc','bin') for p in (r/d).rglob('*') if p.is_file()}
def write(n,d):(o/n).write_text(json.dumps(d,indent=2)+'\n')
archive=o/'artifacts';archive.mkdir();wrapper=o/'gcc-retain';wrapper.write_text('''#!/usr/bin/python3
import subprocess,sys,pathlib,hashlib,json,uuid
args=sys.argv[1:];rc=subprocess.run(['/usr/bin/gcc',*args]).returncode
if rc==0 and '-o' in args:
 p=pathlib.Path(args[args.index('-o')+1])
 if p.is_file():
  b=p.read_bytes();s=hashlib.sha256(b).hexdigest();root=pathlib.Path(__file__).parent/'artifacts';(root/s).write_bytes(b)
  (root/(str(uuid.uuid4())+'.json')).write_text(json.dumps({'arguments':args,'path':str(p.resolve()),'sha256':s}))
sys.exit(rc)
''');wrapper.chmod(0o755)
(o/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());pin=git('rev-parse','HEAD');before=sources();write('source-before.json',before);write('tools-before.json',tools());write('environment.json',{'uname':subprocess.check_output(['uname','-a'],text=True),'gcc':subprocess.check_output(['/usr/bin/gcc','--version'],text=True)})
env=os.environ.copy();env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',CC=str(wrapper));env.pop('LSAN_OPTIONS',None)
flags='-Wall -Wextra -Werror -std=c99 -g -O0 -fPIC -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all'
steps=[('normal',['make','-j2','test-checker-metadata-ownership','test-union-metadata-ownership','CC='+str(wrapper)]),('gcc-sanitized',['make','-j2','test-checker-metadata-ownership','OBJ_DIR=obj-checker-gcc','CC='+str(wrapper),'CFLAGS='+flags,'LDFLAGS=-lm -fsanitize=address,undefined']),('adjacent',['make','-j2','test-module-metadata','test-env-scoping','CC='+str(wrapper),'NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'])]
report={'pin':pin,'steps':[]};status=1
try:
 for name,cmd in steps:
  assert git('rev-parse','HEAD')==pin and sources()==before
  tb=tools();pb=providers();write(name+'-tools-before.json',tb);write(name+'-providers-before.json',pb)
  print('START',name,flush=True);t=time.monotonic()
  with (o/(name+'.log')).open('wb') as f:
   p=subprocess.Popen(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:status=p.wait(timeout=1800)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    status=124
  ta=tools();pa=providers();write(name+'-tools-after.json',ta);write(name+'-providers-after.json',pa)
  for relative,s in pa.items():
   p=r/relative;dest=archive/s
   if not dest.exists():shutil.copyfile(p,dest)
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-t,3),'tools_unchanged':tb==ta,'log_sha256':sha(o/(name+'.log'))});write('manifest.json',report);print('END',name,status,flush=True)
  if status or tb!=ta:break
finally:
 after=sources();write('source-after.json',after);write('tools-after.json',tools());report.update(source_unchanged=before==after,head_unchanged=git('rev-parse','HEAD')==pin);write('manifest.json',report)
raise SystemExit(status)
