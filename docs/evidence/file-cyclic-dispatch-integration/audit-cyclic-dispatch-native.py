import pathlib,json,hashlib,subprocess,tarfile,time
repo=pathlib.Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-plan');r=repo/'docs/evidence/file-cyclic-dispatch';pin='31f42c4f7f09382a15714ee27e41f1ad12c07c51';out={}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((r/n).read_text())
def save():pathlib.Path('/tmp/file-cyclic-dispatch-native-audit.json').write_text(json.dumps(out,indent=2)+'\n')
m=read('report-sha256.json');p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in m.items():
 p.stdin.write((pin+':docs/evidence/file-cyclic-dispatch/'+name+'\n').encode());p.stdin.flush();hdr=p.stdout.readline();data=p.stdout.read(int(hdr.split()[-1]));assert p.stdout.read(1)==b'\n';assert hashlib.sha256(data).hexdigest()==h,name;assert sha(r/name)==h
p.stdin.close();assert p.wait()==0;out['committed_and_worktree_reports']=len(m);save();print('reports',len(m),flush=True)
a=read('artifact-store.json')
for h,v in a.items():assert pathlib.Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==h,h
out['cas']={'count':len(a),'bytes':sum(v['bytes'] for v in a.values())};save();print('CAS',len(a),flush=True)
z=read('artifact-archive.json');assert sha(z['path'])==z['sha256'];assert pathlib.Path(z['path']).stat().st_size==z['bytes'];seen=set()
with tarfile.open(z['path']) as t:
 for member in t:
  if member.isdir():continue
  h=pathlib.Path(member.name).name;assert h in a and h not in seen;f=t.extractfile(member);d=hashlib.sha256()
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
  assert d.hexdigest()==h and member.size==a[h]['bytes'];seen.add(h)
assert seen==set(a);out['archive']={**z,'members':len(seen)};save();print('archive',len(seen),flush=True)
s=read('qualification-summary.json');refs=0;pairs=[];term=[];current=[];remote={}
for group,phases in s['phases'].items():
 d=r/group
 for row in phases:
  phase=row['phase'];b=json.loads((d/(phase+'-source-before.json')).read_text());after=json.loads((d/(phase+'-source-after.json')).read_text());assert b==after
  tb=json.loads((d/(phase+'-tools-before.json')).read_text());ta=json.loads((d/(phase+'-tools-after.json')).read_text());assert tb==ta
  inputs=json.loads((d/(phase+'-inputs-before.json')).read_text());products=json.loads((d/(phase+'-artifacts.json')).read_text());assert not set(inputs)-set(products)
  changed=[n for n,v in inputs.items() if v['sha256']!=products[n]['sha256']];assert not changed or phase=='public'
  for collection in [inputs,products]:
   for n,v in collection.items():assert v['sha256'] in a;refs+=1
  t=json.loads((d/(phase+'-terminal.json')).read_text());assert t['returncode']==row['status'];assert t['leader_reaped'] and t['group_disappeared'] and not t['timeout'] and not t['cleanup_signals'] and not t['errors']
  term.append({'group':group,'phase':phase,'returncode':t['returncode']});pairs.append({'group':group,'phase':phase,'sources':len(b),'tools':len(tb),'changed_providers':len(changed)})
 phase=phases[-1]['phase'];sources=json.loads((d/(phase+'-source-after.json')).read_text());tools=json.loads((d/(phase+'-tools-after.json')).read_text());products=json.loads((d/(phase+'-artifacts.json')).read_text());prefix=group.split('-')[0]
 tree=pathlib.Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-'+prefix) if group.endswith('linux') else pathlib.Path('/private/tmp/nanolang-file-cyclic-dispatch-'+prefix)
 want={str(tree/n):h for n,h in sources.items()};want.update({v['path']:v['sha256'] for v in tools.values()});providers={n:v for n,v in products.items() if n.startswith(str(tree)+'/')};want.update({n:v['sha256'] for n,v in providers.items()})
 if group.endswith('linux'):
  for n,h in want.items():assert sha(n)==h,n
  current.append({'group':group,'source':len(sources),'tool_labels':len(tools),'providers':len(providers),'paths':len(want)})
 else:remote[group]={'paths':want,'source':len(sources),'tool_labels':len(tools),'providers':len(providers)}
assert refs==36997 and len(pairs)==49;out.update(references=refs,pairs=pairs,terminals=term,current_linux=current);save()
pathlib.Path('/tmp/cyclic-dispatch-remote-expected.json').write_text(json.dumps(remote))
prod=read('production-identity.json')
for n,h in prod.items():
 for rev in ['4a12db5e0',pin,'9284','7913','100c','6a65']:
  if len(rev)==4:
   tree='/home/jkh/Src/nanolang-file-cyclic-dispatch-'+rev;actual=subprocess.check_output(['git','rev-parse','HEAD:'+n],cwd=tree,text=True).strip()
  else:actual=subprocess.check_output(['git','rev-parse',rev+':'+n],cwd=repo,text=True).strip()
  assert actual==h,(n,rev)
out['production_blobs']=len(prod);save();print('pairs/current/production complete',flush=True)
