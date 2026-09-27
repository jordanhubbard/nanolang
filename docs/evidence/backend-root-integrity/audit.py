from pathlib import Path
import hashlib,json,subprocess,time
start=time.monotonic();repo=Path('/home/jkh/Src/nanolang-mixed-generated-llvm');pin='a1bb1cb85e5551f9e5bb10491f1608877ef6b7b5';prefix='docs/evidence/record-array-llvm/complete';seal=Path('/home/jkh/nanolang-qualification/backend-integration-a1bb-combined-seal')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
index=json.loads((seal/'report-sha256.json').read_text());store=json.loads((seal/'artifact-store.json').read_text());summary=json.loads((seal/'summary.json').read_text())
assert {str(p.relative_to(seal/'reports')) for p in (seal/'reports').rglob('*') if p.is_file()}==set(index)
parsed={};bytes_reports=0
for name,row in index.items():
 data=(seal/'reports'/name).read_bytes()
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
 bytes_reports+=len(data)
 if name.endswith('.json'):parsed[name]=json.loads(data)
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
result={'status':'PASS','pin':pin,'reports':len(index),'report_bytes':bytes_reports,'objects':len(store),'object_bytes':total,'verified_artifact_references':references,'equal_source_tool_pairs':len(pairs),'terminal_reports':len(terminals),'nonzero_terminals':[v for v in terminals if v.get('returncode') not in (0,None)],'unlaunched_terminals':[v for v in terminals if v.get('launched') is False],'seconds':time.monotonic()-start,'scope':'Independent exact local sealed report set and bytes; Git publication checked separately, every retained content-addressed object, referenced object identity and before/after pairs. Includes failed histories; not semantic corpus coverage, current integration or release acceptance.'}
Path('/home/jkh/nanolang-qualification/backend-root-integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('nonzero_terminals','unlaunched_terminals')},indent=2))
