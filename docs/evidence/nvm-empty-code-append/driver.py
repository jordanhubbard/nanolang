from pathlib import Path
import hashlib,json,subprocess,os,signal,time,sys,shutil
root=Path('/home/jkh/Src/nanolang-empty-code-append');report=Path('/tmp/nanolang-empty-append-qualification');report.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=subprocess.check_output(['git','ls-files','src','modules','tests/nanoisa','scripts','schema','Makefile.gnu','Makefile','GNUmakefile'],cwd=root,text=True).splitlines()
def inputs():return {p:{'sha256':sha(root/p),'bytes':(root/p).stat().st_size} for p in paths if (root/p).is_file()}
def dump(name,obj):(report/name).write_text(json.dumps(obj,indent=2)+'\n')
before=inputs();dump('inputs-before.json',before);dump('source.json',{'pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()});shutil.copyfile(__file__,report/'driver.py');shutil.copyfile('/tmp/nvm-empty-append-original-sanitizers.log.gz',report/'original-ci-sanitizers.log.gz')
tools=['/usr/bin/gcc-13','/usr/local/bin/clang','/usr/bin/make',sys.executable,'/usr/bin/python3'];dump('tools.json',{p:{'realpath':str(Path(p).resolve()),'sha256':sha(Path(p).resolve())} for p in tools})
env=dict(os.environ,PATH='/usr/bin:/bin:'+os.environ['PATH'],PYTHONDONTWRITEBYTECODE='1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
rows=[]
try:
 for label,cc in [('gcc','/usr/bin/gcc-13'),('clang','/usr/local/bin/clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13')]:
  assert shutil.disk_usage(root).free>3*1024**3
  flags='-Wall -Wextra -Werror -std=c99 -g -O3 -ftree-vectorize -fPIC -Isrc -D_GNU_SOURCE -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer'
  (root/'obj').mkdir(exist_ok=True)
  argv=['/usr/bin/make','-j2','test-nanoisa','test-verifier','OBJ_DIR=obj-empty-'+label,'CC='+cc,'CFLAGS='+flags,'LDFLAGS=-lm -lcrypto -fsanitize=undefined','PYTHON_WITH_YAML=/usr/bin/python3']
  dump(label+'-command.json',{'argv':argv,'cwd':str(root),'environment':{k:env[k] for k in ['PATH','UBSAN_OPTIONS','LSAN_OPTIONS','PYTHONDONTWRITEBYTECODE']}})
  started=time.monotonic();timeout=False
  with (report/(label+'.stdout')).open('wb') as out,(report/(label+'.stderr')).open('wb') as err:
   p=subprocess.Popen(argv,cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
   try:rc=p.wait(timeout=900)
   except subprocess.TimeoutExpired:timeout=True;rc=None
   for sig in [signal.SIGTERM,signal.SIGKILL]:
    try:os.killpg(p.pid,0)
    except ProcessLookupError:break
    os.killpg(p.pid,sig)
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
     p.poll()
     try:os.killpg(p.pid,0)
     except ProcessLookupError:break
     time.sleep(.05)
   p.wait(timeout=3)
   try:os.killpg(p.pid,0);gone=False
   except ProcessLookupError:gone=True
  row={'configuration':label,'returncode':p.returncode,'timeout':timeout,'group_gone':gone,'leader_reaped':p.poll() is not None,'seconds':time.monotonic()-started};rows.append(row);dump(label+'-terminal.json',row);dump('results.json',rows);print(row,flush=True)
  products={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (root/('obj-empty-'+label)).rglob('*') if p.is_file()};dump(label+'-products.json',products)
  assert row['returncode']==0 and gone and not timeout,row
finally:
 after=inputs();dump('inputs-after.json',after);dump('identity.json',{'unchanged':before==after,'inputs':len(before)})
