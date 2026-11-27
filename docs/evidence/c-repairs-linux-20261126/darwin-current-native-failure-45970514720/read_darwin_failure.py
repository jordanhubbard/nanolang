"""I verify my failed current Darwin component artifact without executing any compiler or product."""
import argparse,gzip,hashlib,json,re,subprocess,sys,tempfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='6b984b9bf5236a4c3c87cfee8ae6240d3c1aeb30249e87546e177cfb9407ab58'
m=json.loads(raw);assert m['version']==1 and m['run']==45970514720 and not m['whole_release']
assert m['revision']=='6e09c07a9f1651294e4669282cdc17ee14126775'
assert m['artifact']==15188575446 and m['job']==140409516288
seen=set();used=set();results={}
with tempfile.TemporaryDirectory(prefix='current-darwin-failure-reader-',dir=a.workspace) as temporary:
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
 job=json.loads((root/'job-terminal.json').read_bytes());assert job['id']==m['job'] and job['run_id']==m['run'] and job['head_sha']==m['revision']
 assert job['status']=='completed' and job['conclusion']=='failure'
 steps={v['name']:v for v in job['steps']}
 for name in ['Require a hosted worker before provisioning','Require storage headroom','Record source and tool identities','Build fresh production tools','Execute the original bootstrap with its default limits','Localize a refused emitter setup at unchanged limits','Verify source and selected tool guards','Retain terminal logs, products and actual host closures']:
  assert steps[name]['conclusion']=='success',name
 assert steps['Execute the original native compiler and emitter suites']['conclusion']=='failure'
 artifacts=json.loads((root/'api-artifacts.json').read_bytes())['artifacts'];selected=[v for v in artifacts if v['id']==m['artifact']];assert len(selected)==1
 artifact=selected[0];archive=root/'bootstrap.zip';assert not artifact['expired']
 assert artifact['name']=='c-repair-original-xcode-27-bootstrap-'+m['revision']
 assert artifact['size_in_bytes']==archive.stat().st_size and artifact['digest']=='sha256:'+sha(archive.read_bytes())
 destination=root/'bootstrap';destination.mkdir();members=set()
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   rel=Path(item.filename);assert item.orig_filename==item.filename and not rel.is_absolute() and '..' not in rel.parts
   assert str(rel) not in members;members.add(str(rel));assert (item.external_attr>>16)&0o170000 != 0o120000
  z.extractall(destination)
 for script,label in [('check_bootstrap.py','bootstrap'),('analyze_trace.py','trace')]:
  command=[sys.executable,str(root/script),'--artifact',str(destination/m['artifact_root'])]
  if label=='bootstrap':command+=['--repo',str(a.repo)]
  checked=subprocess.run(command,capture_output=True,text=True,timeout=120)
  assert checked.returncode==0,checked.stdout+checked.stderr
  assert checked.stderr=='' and (root/(label+'-check.stderr')).read_bytes()==b''
  result=json.loads(checked.stdout);assert not result['whole_release'] and result==json.loads((root/(label+'-check.stdout')).read_bytes())
  results[label]=result
 assert results['bootstrap']['verified'] and not results['bootstrap']['full_native_component_gate_passed']
 assert results['trace']['parent_timeout'] and not results['trace']['qualification'] and not results['trace']['actual_child_exit_time_known']
 print(json.dumps({'manifest_sha256':sha(raw),'archive_records_verified':len(seen),'archive_chunks_verified':len(used),'github_artifact_zip_digest_matches':True,'zip_members_verified':len(members),'bootstrap':results['bootstrap'],'trace':results['trace'],'current_paired_platforms':False,'original_emitter_gate_failed':True,'whole_release':False}))
