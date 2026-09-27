from pathlib import Path
import subprocess,hashlib,json,tarfile,os,signal,time,shutil
base=Path(__file__).resolve().parent;repo=Path('/tmp/nanolang-record-literal-source-order');pin='a88e4fd85'
assert shutil.disk_usage(base).free >= 2*1024**3, 'I require 2GiB before preparation'
root=base/'source';reports=base/'reports';root.mkdir();reports.mkdir()
def dump(n,x):(reports/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=subprocess.check_output(['git','-C',str(repo),'ls-tree','-rlz',pin],text=True).split('\0')
paths=[r.split('\t',1)[1] for r in rows if r and r.split()[1]=='blob' and not r.split('\t',1)[1].startswith('docs/evidence/')]
archive=base/'source.tar'
with archive.open('wb') as f:subprocess.run(['git','-C',str(repo),'archive',pin,*paths],stdout=f,check=True)
with tarfile.open(archive) as t:t.extractall(root,filter='data')
subprocess.run(['git','init','-q'],cwd=root,check=True)
subprocess.run(['git','add','-f','.'],cwd=root,check=True)
subprocess.run(['git','-c','user.name=Qualification','-c','user.email=qualification@localhost','commit','-qm','Frozen source inputs'],cwd=root,check=True)
def inventory():return {p:{'sha256':sha(root/p),'bytes':(root/p).stat().st_size} for p in paths}
before=inventory();dump('inputs-before.json',before)
dump('source.json',{'pin':subprocess.check_output(['git','-C',str(repo),'rev-parse',pin],text=True).strip(),'archive_sha256':sha(archive),'files':len(paths),'scope':'All tracked inputs except docs/evidence; fresh GCC ordinary VM and native LLVM/Wasm translation admission controls only.'})
tools=['/usr/bin/gcc','/usr/bin/make','/usr/bin/as','/usr/bin/ld','/usr/bin/python3','/usr/local/bin/clang','/usr/local/bin/opt']
for name in ['cc1','collect2']:tools.append(subprocess.check_output(['/usr/bin/gcc','-print-prog-name='+name],text=True).strip())
def toolmap():return {p:{'realpath':str(Path(p).resolve()),'sha256':sha(Path(p).resolve())} for p in tools}
original_tools=toolmap();dump('tools-before.json',original_tools)
cmd=['/usr/bin/make','-f','Makefile.gnu','-j2','CC=/usr/bin/gcc','NMS_RUNTIME_CLANG=/usr/local/bin/clang','NMS_RUNTIME_OPT=/usr/local/bin/opt','test-capture-transport-consumers']
dump('command.json',{'argv':cmd,'cwd':str(root),'timeout_seconds':600})
start=time.monotonic();status={'timeout':False,'cleanup_errors':[]}
with (reports/'stdout.log').open('wb') as out,(reports/'stderr.log').open('wb') as err:
 p=subprocess.Popen(cmd,cwd=root,stdout=out,stderr=err,start_new_session=True)
 try:rc=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  status['timeout']=True;os.killpg(p.pid,signal.SIGTERM)
  try:rc=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=10)
status.update(returncode=rc,seconds=time.monotonic()-start,leader_reaped=p.poll() is not None)
try:os.killpg(p.pid,0);status['group_gone']=False
except ProcessLookupError:status['group_gone']=True
if not status['group_gone']:
 os.killpg(p.pid,signal.SIGKILL);status['cleanup_errors'].append('Residual group required termination')
dump('terminal.json',status)
after=inventory();dump('inputs-after.json',after);dump('tools-after.json',toolmap())
products={}
for p in root.glob('obj/**/*'):
 if p.is_file():products[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
dump('products.json',products)
dump('checks.json',{'terminal':status,'source_unchanged':before==after,'tools_unchanged':original_tools==toolmap(),'products':len(products)})
print(json.dumps(status),flush=True)
raise SystemExit(rc or (before!=after) or (original_tools!=toolmap()))
