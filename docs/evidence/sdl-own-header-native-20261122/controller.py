import gzip,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
primary=Path('/home/jkh/Src/nanolang');root=Path.home()/'.worktrees/nanolang/own-header-1181-puck-20261122';pin='6d8eeacce5e9dbdda7121cd0910d81d70c4772ce'
def sample():
 values=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
 memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))}
 return values,memory
previous,_=sample();busy=[];memory=[]
for _ in range(3):
 time.sleep(5);current,m=sample();delta=[b-a for a,b in zip(previous,current)];busy.append(100*(1-(delta[3]+delta[4])/sum(delta)));memory.append(m);previous=current
volumes={p:dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in (str(primary),str(root.parent),'/tmp')}
health={'cpus':os.cpu_count(),'affinity':len(os.sched_getaffinity(0)),'busy_percent':busy,'memory':memory,'volumes':volumes}
if all(v>=90 for v in busy) or any(m['MemAvailable']/m['MemTotal']<=0.05 for m in memory) or any(v['free']<4*1024**3 or v['free']/v['total']<0.02 for v in volumes.values()):
 print(json.dumps({'held':True,'health':health}),flush=True);sys.exit(2)
assert not root.exists(),'I preserve an existing qualification root.'
r=subprocess.run(['git','-c','credential.helper=!gh auth git-credential','fetch','--no-tags','https://github.com/jordanhubbard/nanolang.git',pin],cwd=primary,capture_output=True,timeout=180)
assert r.returncode==0,r.stderr.decode()
subprocess.run(['git','worktree','add','--detach','--no-checkout',str(root),pin],cwd=primary,check=True,capture_output=True,timeout=60)
subprocess.run(['git','sparse-checkout','init','--no-cone'],cwd=root,check=True,capture_output=True,timeout=60)
subprocess.run(['git','sparse-checkout','set','--no-cone','--stdin'],cwd=root,input='/*\n!/docs/evidence/*\n!/docs/validation/*\n'.replace('\\n','\n'),text=True,check=True,capture_output=True,timeout=180)
subprocess.run(['git','checkout','--detach',pin],cwd=root,check=True,capture_output=True,timeout=180)
out=root/'.test_output/own-header-1181';out.mkdir(parents=True,exist_ok=False)
(out/'worker-before.json').write_text(json.dumps(health,indent=2)+'\n')
print(json.dumps({'phase':'prepared','root':str(root),'source_pin':pin,'health':health}),flush=True)
paths=subprocess.check_output(['git','ls-files','-z','src','src_nano','tests','scripts','modules','std','stdlib','.github','GNUmakefile','Makefile.gnu'],cwd=root).split(b'\0')
# I use the original Git path list, not sparse evidence omissions, for native source inputs.
if len(paths)==1:
 paths=subprocess.check_output(['git','ls-files','-z','src','src_nano','tests','scripts','modules','std','stdlib','.github','GNUmakefile','Makefile.gnu'],cwd=root).split(bytes([0]))
paths=[os.fsdecode(p) for p in paths if p]
inputs={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
(out/'source-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
source_dir=out/'source-inputs';source_dir.mkdir()
for name,h in inputs.items():
 p=source_dir/(h+'.gz')
 if not p.exists():p.write_bytes(gzip.compress((root/name).read_bytes(),mtime=0))
def tools():
 result={}
 for name in ('cc','gcc','clang','as','ld','make','python3'):
  p=shutil.which(name)
  if p:
   resolved=Path(p).resolve();result[name]={'path':str(resolved),'sha256':hashlib.sha256(resolved.read_bytes()).hexdigest()}
 return result
before=tools();(out/'tools-before.json').write_text(json.dumps(before,indent=2)+'\n')
reports=[]
def run(label,argv,timeout):
 start=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
  try:
   child=subprocess.run(argv,cwd=root,stdout=stdout,stderr=stderr,timeout=timeout);code=child.returncode;error=None
  except subprocess.TimeoutExpired:
   code=None;error='I retain the original command timeout.'
 report={'label':label,'argv':argv,'timeout':timeout,'returncode':code,'error':error,'elapsed_seconds':time.monotonic()-start};reports.append(report);(out/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return code==0
ok=run('build',['make','-f','Makefile.gnu','-j1','bin/nanoc_c'],900)
if ok:
 shutil.copy2(root/'bin/nanoc_c',out/'nanoc_c')
 ok=run('imported-c-header-controls',[sys.executable,'-m','unittest','-v','tests.test_imported_callback_typedefs'],120)
 deps=subprocess.run(['pkg-config','--exists','sdl2','SDL2_image','SDL2_mixer','SDL2_ttf'],cwd=root).returncode==0
 (out/'sdl-dependencies.json').write_text(json.dumps({'original_example_dependencies_available':deps})+'\n')
 if ok and deps:ok=run('original-escaped-examples',['make','-f','Makefile.gnu','test-examples-regressions'],120)
 else:reports.append({'label':'original-escaped-examples','executed':False,'reason':'I keep original dependency requirements; full hosted execution remains required.'})
after=tools();unchanged=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in inputs.items())
record={'source_pin':pin,'inputs':len(inputs),'all_source_hashes_unchanged':unchanged,'tools_unchanged':before==after,'reports':reports,'success':ok and unchanged and before==after,'original_full_examples_or_release_accepted':False}
(out/'tools-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
