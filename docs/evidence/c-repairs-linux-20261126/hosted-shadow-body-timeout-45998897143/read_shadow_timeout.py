"""I replay my hosted body-timeout evidence without executing a compiler."""
import argparse,gzip,hashlib,importlib.util,json,re,subprocess,tempfile,zipfile,collections
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();a.repo=a.repo.resolve()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='20213c0ad070ef267a2501fe1d4bfa690e3a1d0ecb5ffe030314bf637f4adbd2'
m=json.loads(raw);assert m['version']==1 and not m['qualification'] and not m['whole_release'] and m['whole_wrapper_zip_reconstructed']
assert (m['run'],m['job'],m['artifact'])==(45998897143,140505211888,15203409836)
assert m['source_revision']=='51c82debb858145371cd9f28005d683e0e79184f' and m['product_revision']=='6e09c07a9f1651294e4669282cdc17ee14126775'
with tempfile.TemporaryDirectory(prefix='hosted-body-timeout-reader-',dir=a.workspace) as temporary:
 root=Path(temporary);used=set();seen=set()
 for row in m['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts and str(name) not in seen;seen.add(str(name));dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with dest.open('wb') as f:
   for k in row['chunks']:
    assert re.fullmatch('[a-f0-9]{64}',k);meta=m['chunks'][k];packed=(a.archive/'objects'/(k+'.gz')).read_bytes();assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed);assert len(data)==meta['bytes'] and sha(data)==k;f.write(data);h.update(data);size+=len(data);used.add(k)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 assert used==set(m['chunks']) and len(seen)==7
 artifact=json.loads((root/'api-artifact.json').read_text());run=json.loads((root/'api-run-full.json').read_text());jobs=json.loads((root/'api-run.json').read_text())
 assert artifact['id']==m['artifact'] and artifact['digest']=='sha256:'+m['wrapper_zip_sha256'] and artifact['size_in_bytes']==75900597
 assert sha((root/'artifact.zip').read_bytes())==m['wrapper_zip_sha256']
 assert run['id']==m['run'] and run['head_sha']==m['source_revision'] and run['conclusion']=='failure'
 assert jobs['headSha']==m['source_revision'] and jobs['conclusion']=='failure' and len(jobs['jobs'])==1
 job=jobs['jobs'][0];assert job['databaseId']==m['job'];steps={s['name']:s['conclusion'] for s in job['steps']}
 assert steps['Check my isolated cache namespace restore']=='success' and steps['Observe original C seed with calibrated tail observer']=='failure'
 f=root/'failure';f.mkdir();members={r['path']:r for r in m['members']};assert len(members)==845
 with zipfile.ZipFile(root/'artifact.zip') as z:
  assert set(z.namelist())==set(members) and len(z.namelist())==845
  for name in z.namelist():
   rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts;data=z.read(name);assert len(data)==members[name]['bytes'] and sha(data)==members[name]['sha256'];dest=f/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 q=json.loads((f/'result.json').read_text());assert q['terminal'] and not q['qualification'] and not q['whole_release']
 assert (q['command_timeout_seconds'],q['shadow_seconds'],q['compiler_exit_code'],q['observer_exit_code'])==(300,10,1,2)
 assert not q['flush_observed'] and 'trigger_phase' not in q and 'command_timeout' not in q
 before=json.loads((f/'identities-before.json').read_text());assert len(before)==1402 and before==json.loads((f/'identities-after.json').read_text())
 hosts=json.loads((f/'host-libraries-before.json').read_text());assert hosts==json.loads((f/'host-libraries-after.json').read_text())
 sources=json.loads((f/'observer-sources.json').read_text());assert len(sources)==4
 for name,h in sources.items():
  data=(f/'observer-sources'/Path(name).name).read_bytes();assert sha(data)==h
  assert data==subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=a.repo)
 def product(revision):
  result={}
  for entry in subprocess.check_output(['git','ls-tree','-rz',revision],cwd=a.repo).split(b'\0'):
   if entry:
    meta,name=entry.split(b'\t',1);name=name.decode()
    if not name.startswith('docs/') and name not in sources:result[name]=meta
  return result
 assert product(m['product_revision'])==product(m['source_revision'])
 guard=json.loads((f/'product-tree-guard.json').read_text());assert guard['matches'] and guard['unchanged_product_entries']==4695
 build=json.loads((f/'observer-build.json').read_text());assert sha((f/'lifecycle-observer').read_bytes())==build['sha256']
 assert build['source_sha256']==sources['scripts/ci_darwin_shadow_lifecycle.c']
 restore=json.loads((f/'cache-current-restoration.json').read_text());assert restore['complete'] and not restore['original_link_metadata_available'] and not restore['qualification'];assert len(restore['entries'])==9
 restored=0
 for entry in restore['entries']:
  assert entry['restored'] and entry['basis']=='unique-complete-byte-equal-generation' and re.fullmatch(r'\.nano-gen-[A-Za-z0-9]{6}',entry['target'])
  backup=Path(entry['backup']);assert backup.parts[:2]==('.test_output','shadow-lifecycle') and '..' not in backup.parts
  location=f/Path(*backup.parts[2:]);actual={v.name:{'size':v.stat().st_size,'sha256':sha(v.read_bytes())} for v in location.iterdir() if v.is_file()};assert actual==entry['files'];restored+=len(actual)
 assert restored==812
 lines=[]
 for line in (f/'compiler.stderr').read_text().splitlines():
  if line.startswith('{'):
   try:lines.append(json.loads(line))
   except ValueError:pass
 starts=[v for v in lines if v['event']=='shadow-start'];done=[v for v in lines if v['event']=='shadow-complete'];phases=[v for v in lines if v['event']=='cseed-shadow-phase']
 assert len(starts)==843 and len(done)==842 and all(v['failures']==0 for v in done)
 assert [(v['source'],v['line'],v['name']) for v in starts[:-1]]==[(v['source'],v['line'],v['name']) for v in done]
 assert starts[-1]['name']=='nb_owner_array_fixture' and starts[-1]['line']==3554
 assert [v['phase'] for v in phases]==['dispatch-start','supervisor-status'];dispatch,status=phases
 assert dispatch['pid']==q['shadow_pid'] and status['pid']==q['compiler_pid'] and status['child_pid']==q['shadow_pid']
 assert status['parent_timeout'] and status['reaped'] and status['signal']==9 and status['wait_status']==9 and not status['completed'] and status['deadline_seconds']==10
 ns=lambda x,prefix:x[prefix+'seconds']*10**9+x[prefix+'nanoseconds']
 elapsed=ns(status,'monotonic_')-ns(status,'parent_start_');assert elapsed==10590942000
 totals=collections.defaultdict(float)
 for v in done:totals[v['source'].split('/')[-1]]+=v['elapsed_seconds']
 analysis=json.loads((root/'analysis.json').read_text());assert analysis['starts']==843 and analysis['completions']==842 and analysis['completed_failures']==0
 assert analysis['phases']==phases and analysis['unfinished_shadow']==starts[-1] and analysis['completed_elapsed_by_file']==dict(totals)
 assert abs(analysis['completed_elapsed_sum']-9.320307)<1e-9 and not analysis['exclusive_cost_attribution'] and not analysis['old_failure_exclusively_explained']
 events=[json.loads(l) for l in (f/'lifecycle.jsonl').read_text().splitlines()];assert [v['event'] for v in events]==['clock','sample','armed','cancelled-before-trigger']
 assert events[1]['pid']==q['shadow_pid'] and events[1]['parent']==q['compiler_pid'] and events[1]['usage_result']==0 and events[1]['exit_absolute']==0
 # I replay the exact original control predicates with compiler/process execution forbidden.
 spec=importlib.util.spec_from_file_location('retained_lifecycle',f/'observer-sources/ci_darwin_shadow_lifecycle.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 controls=root/'control-replay';controls.mkdir()
 for kind in ['success','late','blocked']:(controls/('control-'+kind+'.jsonl')).write_bytes((f/('control-'+kind+'.jsonl')).read_bytes())
 commands=[]
 def captured(argv,out,name,**kwargs):
  assert kwargs=={'timeout':5} and name=='control-'+argv[2] and argv[1]=='--control';commands.append(argv)
 with patch.object(subprocess,'Popen',side_effect=AssertionError('I forbid product execution.')),patch.object(module,'captured',captured):module.controls(Path('/retained/observer'),controls)
 assert len(commands)==3 and (controls/'controls.json').read_bytes()==(f/'controls.json').read_bytes()
 print(json.dumps({'verified':True,'manifest_sha256':sha(raw),'wrapper_bytes':75900597,'wrapper_members':845,'records':7,'chunks':len(used),'native_controls_replayed':3,'restore_entries':9,'restored_files':restored,'identity_guards':len(before),'starts':len(starts),'completions':len(done),'recorded_assertion_failures':0,'unfinished_shadow':starts[-1]['name'],'supervisor_elapsed_ns':elapsed,'flush_observed':False,'original_10_second_failure':True,'old_failure_exclusively_explained':False,'product_execution':False,'qualification':False,'whole_release':False}))
