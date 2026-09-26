from pathlib import Path
import hashlib,json,gzip,tarfile,io,subprocess,shutil
repo=Path('/home/jkh/Src/nanolang-eval-callback-ownership');base=Path('/home/jkh/nanolang-qualification');out=repo/'docs/evidence/evaluator-integer-callback';out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
sets=[('first','integer-negation-1b800','1b80030ce4991670f43ff50f5a74957170c8df74',('linux',)),('callback','callback-dfd7','dfd7e53a3cb5b4499708e8ae6c798f5c5dc88f31',('linux','darwin')),('combined','integer-combined-439c','439cdb8ab89ac6ec810d24a4edd7b1fd013e4881',('linux','darwin')),('snapshot','integer-combined-62a','62a77f512110fcabaa26b4d30f832a04733a5c86',('linux','darwin'))]
seal={'lanes':{},'objects':{},'source_git_checks':{},'retained_executables':{}}
for label,directory,pin,hosts in sets:
 root=base/directory;expected_source=None
 for host in hosts:
  for config in ('ordinary','sanitizer'):
   r=root/(host+'-'+config+'-reports');lane=label+'-'+host+'-'+config
   objects={}
   for p in sorted((r/'artifacts').iterdir()):
    if not p.is_file():continue
    h=sha(p);assert h==p.name
    objects[h]={'bytes':p.stat().st_size,'path':str(p)};seal['objects'].setdefault(h,objects[h])
   pairs=0
   for p in sorted(r.glob('*-source-before.json')):
    before=json.loads(p.read_text());assert before==json.loads((r/p.name.replace('-before','-after')).read_text());pairs+=1
    if expected_source is None:expected_source=before
    assert before==expected_source
   for p in r.glob('*-tools-before.json'):
    assert json.loads(p.read_text())==json.loads((r/p.name.replace('-before','-after')).read_text());pairs+=1
   refs=0
   for p in r.glob('*-artifacts.json'):
    for v in json.loads(p.read_text()).values():assert v['sha256'] in objects;refs+=1
   terminals={p.name:json.loads(p.read_text()) for p in r.glob('*-terminal.json')}
   for t in terminals.values():assert t['leader_reaped'] and t['group_disappeared'] and not t['errors'] and not t['timeout']
   for p in r.glob('*-nested-processes.json'):assert not json.loads(p.read_text())['remaining']
   results=json.loads((r/'results.json').read_text())
   expected_failure=label=='first' and config=='sanitizer'
   assert all(x['status']==(2 if expected_failure else 0) for x in results)
   if not expected_failure:assert json.loads((r/'final.json').read_text())['status']=='PASS'
   if not expected_failure:
    for result in results:
     log=(r/(result['phase']+'.log')).read_text()
     if result['phase']=='test-integer-binary-eval':assert 'I retain 405 exact binary integer results' in log
     else:
      assert 'I retain 30 exact integer negation results' in log
      if label!='first':assert 'I preserve owned and borrowed callbacks across six aliases' in log
      if label=='snapshot':
       assert 'I retain callback loop identity, snapshot refusal and fresh recovery in 24 cases.' in log
       assert log.count('I could not retain the array callback name.')==24

   for p in r.rglob('retention.json'):
    rec=json.loads(p.read_text());exe=p.parent/Path(rec['source']).name;assert sha(exe)==rec['sha256'] and exe.stat().st_size==rec['bytes'];assert exe.stat().st_mode&511==rec['mode'];seal['retained_executables'][lane+':'+exe.name]=dict(rec,local_path=str(exe))
   if expected_failure:
    exe=root/'linux-sanitizer/tests/test_integer_negation_eval';h=sha(exe);seal['retained_executables'][lane+':'+exe.name]={'local_path':str(exe),'bytes':exe.stat().st_size,'mode':exe.stat().st_mode&511,'sha256':h,'retention':'failed target stopped before original rm'}
   reports=[];bundle=out/(lane+'.tar.gz')
   with bundle.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z,tarfile.open(fileobj=z,mode='w|') as t:
    for p in sorted(r.rglob('*')):
     if not p.is_file() or 'artifacts' in p.relative_to(r).parts:continue
     if p.name.startswith('test_') and 'retained' in p.relative_to(r).parts:continue
     b=p.read_bytes();name=str(p.relative_to(r));e=tarfile.TarInfo(name);e.size=len(b);e.mode=p.stat().st_mode&511;t.addfile(e,io.BytesIO(b));reports.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
   with tarfile.open(bundle) as t:
    for item in reports:
     b=t.extractfile(item['path']).read();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
   seal['lanes'][lane]={'source':pin,'bundle':bundle.name,'sha256':sha(bundle),'reports':reports,'objects':len(objects),'equal_source_tool_pairs':pairs,'artifact_references':refs,'results':results,'terminals':terminals}
 names=sorted(expected_source);payload=''.join(pin+':'+p+'\n' for p in names).encode();data=subprocess.check_output(['git','cat-file','--batch'],input=payload,cwd=repo);stream=io.BytesIO(data)
 for name in names:
  fields=stream.readline().split();assert fields[1]==b'blob';b=stream.read(int(fields[2]));assert stream.read(1)==b'\n';assert hashlib.sha256(b).hexdigest()==expected_source[name],name
 assert stream.read()==b'';seal['source_git_checks'][pin]={'paths':len(names),'status':'PASS'}
seal['totals']={'reports':sum(len(x['reports']) for x in seal['lanes'].values()),'bundles':len(seal['lanes']),'unique_objects':len(seal['objects']),'object_bytes':sum(x['bytes'] for x in seal['objects'].values()),'equal_source_tool_pairs':sum(x['equal_source_tool_pairs'] for x in seal['lanes'].values()),'terminals':sum(len(x['terminals']) for x in seal['lanes'].values()),'retained_executables':len(seal['retained_executables'])}
(out/'seal.json').write_text(json.dumps(seal,indent=2)+'\n');shutil.copy2(__file__,out/'seal.py');print(seal['totals'])
