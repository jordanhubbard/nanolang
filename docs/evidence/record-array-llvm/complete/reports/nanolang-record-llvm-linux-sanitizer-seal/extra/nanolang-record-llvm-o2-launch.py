import hashlib,json,os,re,shlex,shutil,signal,stat,subprocess,sys,tarfile,time
from pathlib import Path
base=Path('/run/user/1000/nanolang-llvm-o2-aed76-20260921');original=Path('/tmp/nanolang-record-llvm-qualified-78042')
assert not base.exists();assert shutil.disk_usage(base.parent).free>=2147483648
base.mkdir(mode=0o700);root=base/'source';report=base/'reports';scratch=base/'temporary';scratch.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):(base/n).write_text(json.dumps(v,indent=2)+'\n')
def checked(argv,**kw):return subprocess.run(argv,check=True,timeout=1800,**kw)
# I revalidate the last completed source/tool/provider endpoint before copying.
checked([sys.executable,'/tmp/verify-record-backend-current.py','--root',str(original),'--report','/tmp/nanolang-record-llvm-78042-linux-clang-continuation-r2','--output',str(base/'original-current.json')])
manifest=json.loads(Path('/tmp/nanolang-record-llvm-78042-source.json').read_text())
root.mkdir();copied={}
for name,digest in manifest['files'].items():
 p=original/name;assert sha(p)==digest,name;dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==digest
shutil.copy2(original/'.qualification-tracked',root/'.qualification-tracked')
for directory in ('obj','bin','lib'):
 for p in (original/directory).rglob('*'):
  if p.is_dir():continue
  assert not p.is_symlink(),str(p)
  name=str(p.relative_to(original));dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
  before=sha(p);mode=stat.S_IMODE(p.stat().st_mode);shutil.copy2(p,dest)
  assert sha(dest)==before and stat.S_IMODE(dest.stat().st_mode)==mode;copied[name]={'sha256':before,'mode':mode}
save('copied-providers.json',copied)
fixture='tests/test_record_array_llvm.py';pin='aed76cdb3'
content=subprocess.check_output(['git','-C','/home/jkh/Src/nanolang-mixed-generated-llvm','show',pin+':'+fixture]);(root/fixture).write_bytes(content)
actual={name:sha(root/name) for name in manifest['files']};changed=[n for n,h in actual.items() if h!=manifest['files'][n]];assert changed==[fixture],changed
save('source-overlay.json',{'production':manifest['pin'],'fixture_commit':pin,'fixture':fixture,'fixture_sha256':sha(root/fixture),'source':actual})
for file in ['/tmp/nanolang-record-llvm-o2-driver.py',__file__,'/tmp/verify-record-llvm-o2-archive.py','/tmp/record-llvm-clang-o0-complete-audit.json']:
 shutil.copy2(file,base/Path(file).name)
env=dict(os.environ,LSAN_OPTIONS='',TMPDIR=str(scratch),CARRIER_PHASES='configuration,clang-sanitizer-native',RECORD_LLVM_NATIVE_OPTIMIZATIONS='O2',CC='/bin/gcc-13',CARRIER_SAN_CC='/bin/gcc-13',NMS_RUNTIME_CLANG='/usr/local/bin/clang',NMS_RUNTIME_OPT='/usr/local/bin/opt',NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',RECORD_LLVM_WASMTIME='/home/jkh/.local/bin/wasmtime',RECORD_LLVM_NODE='/home/linuxbrew/.linuxbrew/bin/node',RECORD_LLVM_C_BASELINE='/tmp/nano-record-array-generated-x9fej2n6/corpus',CARRIER_EXTRA_TOOLS=json.dumps({'wasm_ld':'/usr/local/bin/wasm-ld'}))
for n in ('clang','opt','llc','nm'):env['RECORD_LLVM_'+n.upper()]='/usr/local/bin/'+('llvm-nm' if n=='nm' else n)
record={'started':time.time(),'source':manifest['pin'],'fixture':pin,'native_optimizations':['O2'],'output_filesystem':str(base),'prephase_bytes':2147483648,'active_minimum_bytes':1610612736,'capacity_signals':[]}
save('launch.json',record)
with (base/'outer.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(base/'nanolang-record-llvm-o2-driver.py'),str(root),str(report)],env=env,stdout=log,stderr=subprocess.STDOUT)
 while p.poll() is None:
  free=shutil.disk_usage(base).free
  if free<1610612736 and not record['capacity_signals']:
   for line in subprocess.check_output(['ps','-eo','pid=,ppid=,pgid=,args='],text=True).splitlines():
    row=line.strip().split(None,3)
    if len(row)==4 and int(row[1])==p.pid and 'unittest' in row[3]:
     assert row[0]==row[2];os.killpg(int(row[2]),signal.SIGTERM);record['capacity_signals'].append({'group':int(row[2]),'free_bytes':free})
  time.sleep(5)
 record.update(returncode=p.returncode,finished=time.time());save('outer.json',record)
# I compare all generated inputs against the completed O0 run before acceptance.
def canonical(data):
 identities={}
 def sub(match):
  if match.group(0)==b'identity_count=0;':identities.clear();return match.group(0)
  value=int(match.group(1))
  if not value:return match.group(0)
  if value not in identities:identities[value]=len(identities)+1
  return b'identity(UINT64_C('+str(identities[value]).encode()+b'),o.identity)'
 return re.sub(rb'identity_count=0;|identity\(UINT64_C\((\d+)\),o\.identity\)',sub,data)
correspondence={'status':'FAIL','rows':{}}
try:
 candidates=list(scratch.glob('nano-record-array-llvm-*/native-corpus'));assert len(candidates)==1
 prior=Path('/tmp/nano-record-array-llvm-6gzgdsmh/native-corpus')
 for i in range(73):
  for suffix in ('ll','replay.c'):
   name=f'product-{i:04d}.'+suffix;a=(prior/name).read_bytes();b=(candidates[0]/name).read_bytes()
   assert (canonical(a)==canonical(b)) if suffix=='replay.c' else a==b,name
   correspondence['rows'][name]={'prior':hashlib.sha256(a).hexdigest(),'current':hashlib.sha256(b).hexdigest(),'comparison':'identity bijection' if suffix=='replay.c' else 'exact'}
 correspondence['status']='PASS'
except Exception as error:correspondence['error']=repr(error)
save('corpus-correspondence.json',correspondence)
# I retain all inputs, reports and products even when the gate fails.
entries={}
for p in base.rglob('*'):
 if p.is_dir():continue
 assert p.is_file() and not p.is_symlink(),p
 entries[str(p.relative_to(base))]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
archive=base.parent/(base.name+'-evidence.tar.gz');index=base.parent/(base.name+'-manifest.json');index.write_text(json.dumps(entries,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as tar:
 for name in sorted(entries):tar.add(base/name,arcname=name,recursive=False)
verifier=base/'verify-record-llvm-o2-archive.py'
local=json.loads(subprocess.check_output([sys.executable,str(verifier),str(archive),str(index)],text=True))
remote='/Users/jkh/nanolang-qualification/'+base.name
checked(['ssh','-o','ConnectTimeout=15','puck.local','mkdir -p '+shlex.quote(remote)])
for p in (archive,index,verifier):checked(['scp','-q',str(p),'puck.local:'+remote+'/'+p.name])
command='python3 '+shlex.quote(remote+'/'+verifier.name)+' '+shlex.quote(remote+'/'+archive.name)+' '+shlex.quote(remote+'/'+index.name)
remote_check=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=15','puck.local',command],text=True));assert remote_check==local
retained={'local':local,'remote':remote_check,'durable_remote':remote,'gate_returncode':record['returncode'],'corpus_correspondence':correspondence['status'],'temporary_data_removed':False}
save('durable-retention.json',retained);checked(['scp','-q',str(base/'durable-retention.json'),'puck.local:'+remote+'/durable-retention.json'])
print(json.dumps(retained),flush=True)

sys.exit(0 if record['returncode']==0 and correspondence['status']=='PASS' else 1)
