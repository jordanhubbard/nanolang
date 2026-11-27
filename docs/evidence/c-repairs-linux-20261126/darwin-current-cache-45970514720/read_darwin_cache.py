"""I verify exact hosted artifacts without executing any compiler or product."""
import argparse,gzip,hashlib,json,re,subprocess,sys,tempfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='ceb65fb29e15cc354eb969c49cad2a937181857328fe6b0308ed9359bcb2c2b5'
m=json.loads(raw);assert m['version']==1 and m['run']==45970514720 and not m['whole_release']
assert m['revision']=='6e09c07a9f1651294e4669282cdc17ee14126775'
assert m['artifact_ids']=={'cache':15191184169}
seen=set();used=set();results={}
with tempfile.TemporaryDirectory(prefix='current-hosted-darwin-cache-reader-',dir=a.workspace) as temporary:
 root=Path(temporary)
 for row in m['files']:
  relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts and len(relative.parts)==1
  assert row['path'] not in seen;seen.add(row['path']);h=hashlib.sha256();size=0
  with (root/relative).open('wb') as f:
   for key in row['chunks']:
    assert re.fullmatch('[0-9a-f]{64}',key)
    packed=(a.archive/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];block=gzip.decompress(packed)
    assert len(block)==meta['bytes'] and sha(block)==key
    f.write(block);h.update(block);size+=len(block);used.add(key)
  assert size==row['size'] and h.hexdigest()==row['sha256'],row['path']
 assert used==set(m['chunks'])
 run=json.loads((root/'api-run.json').read_bytes());assert run['id']==m['run'] and run['head_sha']==m['revision']
 jobs=json.loads((root/'api-jobs.json').read_bytes())['jobs'];artifacts=json.loads((root/'api-artifacts.json').read_bytes())['artifacts']
 for kind in ('cache',):
  selected=[j for j in jobs if j['name']=='C repair original gates (xcode-27, '+kind+')'];assert len(selected)==1
  job=selected[0];assert job['status']=='completed' and job['conclusion']=='success'
  steps={s['name']:s for s in job['steps']}
  required=['Require a hosted worker before provisioning','Require storage headroom','Record source and tool identities','Build fresh production tools','Verify source and selected tool guards','Retain terminal logs, products and actual host closures']
  required+=['Execute the original bootstrap with its default limits','Execute the original native compiler and emitter suites'] if kind=='bootstrap' else ['Execute my array repair regression','Execute the complete original cache and worker gates']
  for name in required:assert steps[name]['conclusion']=='success',(kind,name)
  selected=[v for v in artifacts if v['id']==m['artifact_ids'][kind]];assert len(selected)==1
  artifact=selected[0];archive=root/(kind+'.zip');assert not artifact['expired']
  assert artifact['name']=='c-repair-original-xcode-27-'+kind+'-'+m['revision']
  assert artifact['size_in_bytes']==archive.stat().st_size
  assert artifact['digest']=='sha256:'+sha(archive.read_bytes())
  destination=root/kind;destination.mkdir();members=set()
  with zipfile.ZipFile(archive) as z:
   for item in z.infolist():
    rel=Path(item.filename);assert item.orig_filename==item.filename and not rel.is_absolute() and '..' not in rel.parts
    assert str(rel) not in members;members.add(str(rel))
    assert (item.external_attr>>16)&0o170000 != 0o120000,item.filename
   z.extractall(destination)
  command=[sys.executable,str(root/('check_'+kind+'.py')),'--artifact',str(destination/m['artifact_roots'][kind]),'--repo',str(a.repo)]
  checked=subprocess.run(command,capture_output=True,text=True,timeout=120)
  assert checked.returncode==0,checked.stdout+checked.stderr
  assert checked.stderr==''
  result=json.loads(checked.stdout);assert result['verified'] and not result['whole_release']
  assert result==json.loads((root/(kind+'-check.stdout')).read_bytes())
  assert (root/(kind+'-check.stderr')).read_bytes()==b''
  results[kind]=result
 print(json.dumps({'manifest_sha256':sha(raw),'archive_records_verified':len(seen),'archive_chunks_verified':len(used),'github_artifact_zip_digests_match':True,'cache':results['cache'],'current_paired_platforms':False,'cast_reconciliation_decision_pending':True,'whole_release':False}))
