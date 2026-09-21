from pathlib import Path
import hashlib,json,subprocess,time
start=time.monotonic();repo=Path('/home/jkh/Src/nanolang-mixed-generated-llvm');pin='3d4afccb726e18c8b59813c1de1b710e3b56636f';prefix='docs/evidence/record-array-llvm/complete';seal=Path('/run/user/1000/nanolang-record-llvm-complete-seal')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def gitfile(path):return subprocess.check_output(['git','show',pin+':'+prefix+'/'+path],cwd=repo)
index=json.loads(gitfile('report-sha256.json'));store=json.loads(gitfile('artifact-store.json'));summary=json.loads(gitfile('summary.json'))
assert index==json.loads((seal/'report-sha256.json').read_text())
assert store==json.loads((seal/'artifact-store.json').read_text())
actual=subprocess.check_output(['git','ls-tree','-rz','--name-only',pin,'--',prefix+'/reports'],cwd=repo).decode().split('\0');actual={p[len(prefix+'/reports/'):] for p in actual if p};assert actual==set(index)
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE);parsed={};bytes_reports=0
for name,row in index.items():
 proc.stdin.write((pin+':'+prefix+'/reports/'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().decode().split();assert len(header)==3 and header[1]=='blob',name
 size=int(header[2]);data=proc.stdout.read(size);assert proc.stdout.read(1)==b'\n'
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
 assert (seal/'reports'/name).read_bytes()==data,name;bytes_reports+=size
 if name.endswith('.json'):parsed[name]=json.loads(data)
proc.stdin.close();assert proc.wait()==0
objects=seal/'objects';assert {p.name for p in objects.iterdir()}==set(store)
total=0
for digest,row in store.items():
 p=objects/digest;assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==digest,digest;total+=p.stat().st_size
references=0
def visit(value):
 global references
 if isinstance(value,dict):
  if 'artifact' in value and 'sha256' in value:
   digest=value['sha256'];assert digest in store,digest;assert Path(value['artifact']).name==digest,value;references+=1
  for child in value.values():visit(child)
 elif isinstance(value,list):
  for child in value:visit(child)
for value in parsed.values():visit(value)
pairs=[]
for name,value in parsed.items():
 if name.endswith(('-source-before.json','-tools-before.json')):
  other=name.replace('-before.json','-after.json')
  if other in parsed:assert value==parsed[other],name;pairs.append(name)
terminals=[{'path':name,**value} for name,value in parsed.items() if name.endswith('-terminal.json')]
assert len(index)==summary['reports'] and len(store)==summary['objects'] and total==summary['object_bytes']
result={'status':'PASS','pin':pin,'reports':len(index),'report_bytes':bytes_reports,'objects':len(store),'object_bytes':total,'verified_artifact_references':references,'equal_source_tool_pairs':len(pairs),'terminal_reports':len(terminals),'nonzero_terminals':[v for v in terminals if v.get('returncode') not in (0,None)],'unlaunched_terminals':[v for v in terminals if v.get('launched') is False],'seconds':time.monotonic()-start,'scope':'Independent exact committed report set and bytes, local sealed report bytes, every retained content-addressed object, referenced object identity and before/after pairs. Includes failed histories; not semantic corpus coverage, current integration or release acceptance.'}
Path('/home/jkh/nanolang-qualification/llvm-root-integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('nonzero_terminals','unlaunched_terminals')},indent=2))
