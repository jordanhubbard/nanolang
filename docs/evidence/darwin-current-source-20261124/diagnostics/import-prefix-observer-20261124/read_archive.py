"""I check retained diagnostic bytes and keep the cutoff distinct from qualification."""
from pathlib import Path
import gzip,hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((base/'manifest.json').read_bytes())
assert sha((base/'manifest.json').read_bytes())=='e85889d7e431fdcdf7245cb4d517245e7059af83cbbce8ac10583805cdebc294'
parent=base/manifest['base_manifest']['path']
assert sha(parent.read_bytes())==manifest['base_manifest']['sha256']
original=json.loads(parent.read_bytes());prior={r['path']:r for r in original['files']}
assert not manifest['qualification'] and not manifest['whole_release']
rows={r['path']:r for r in manifest['files']};assert len(rows)==len(manifest['files'])
for h,c in manifest['chunks'].items():
 p=(base/c['artifact']).resolve();assert p.is_relative_to(base.parent.parent)
 z=p.read_bytes();assert len(z)==c['compressed_bytes'] and sha(z)==c['compressed_sha256']
 raw=gzip.decompress(z);assert len(raw)==c['bytes'] and sha(raw)==h
 if c['artifact'].startswith('../../'):assert c['compressed_sha256']==original['chunks'][h]['compressed_sha256']
def blocks(r):
 for h in r['chunks']:yield gzip.decompress((base/manifest['chunks'][h]['artifact']).read_bytes())
def data(n):return b''.join(blocks(rows[n]))
for r in rows.values():
 digest=hashlib.sha256();size=0
 for raw in blocks(r):digest.update(raw);size+=len(raw)
 assert digest.hexdigest()==r['sha256'] and size==r['bytes'],r['path']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
summary=[]
for label in ['direct-import','observed-import','source-prefix']:
 def js(name):return json.loads(data(label+'/'+name))
 q=js('result.json');cfg=js('config.json');pre=js('preflight.json')
 assert q['terminal'] and q['controller_exit']==0 and not q['source_edited']
 assert q['capture_timeout']=='original default' and q['shadow_timeout']=='original default10'
 assert pre['admitted'] and pre['system']=='Darwin' and pre['machine']=='arm64' and not pre['native']
 assert len(pre['rounds'])==3 and all(r['cpu']>=1 and r['share']>=.75 for r in pre['rounds'])
 assert q['source_pin']==cfg['pin']==manifest['source_pin']==original['source_pin']
 assert q['source_before']==q['source_after']==cfg['sources'] and len(cfg['sources'])==3520
 assert q['tools_before']==q['tools_after']
 assert q['qualified_image_before']==q['qualified_image_after']==q['copied_image']['sha256']==cfg['qualified_sha256']
 assert rows[label+'/source/bin/nanoc_c']['sha256']==cfg['qualified_sha256']
 assert cfg['qualified_sha256']==prior['qualification/source-normal/bin/nanoc_c']['sha256']
 for name,h in cfg['sources'].items():
  assert rows[label+'/source/'+name]['sha256']==h
  proc.stdin.write((cfg['pin']+':'+name+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob';raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h
 for name,h in q['tools_before'].items():assert original['tool_guards'][name]['actual']==h
 case=q['case'];assert case['timeout']==120 and case['user_seconds']>0 and case['system_seconds']>0
 events=[]
 for line in data(label+'/import-case.stderr').splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 starts=[e for e in events if e.get('event')=='shadow-start'];ends=[e for e in events if e.get('event')=='shadow-complete']
 if label=='source-prefix':
  assert q['full_source_input'] and q['source_prefix_observation'] and not q['qualification_attempt']
  assert case['exit']==-9 and case['timed_out'] and 120<=case['seconds']<125 and not starts and not ends
  assert not any(e.get('event')=='cseed-shadow-phase' for e in events)
  assert label+'/native' not in rows and 'native_run' not in q
  assert len(q['provider_copies'])==494
  for name,h in q['provider_copies'].items():
   assert rows[label+'/source/'+name]['sha256']==h==prior['qualification/source-normal/'+name]['sha256']
  assert rows[label+'/source/src_nano/nanoc_v06.nano']['sha256']==q['fixture_sha256']
 else:
  assert not q['full_source_attempt'] and case['exit']==0 and not case['timed_out']
  assert len(starts)==len(ends)==3
  assert q['native_run']['exit']==0 and q['native_run']['timeout']==10
  assert rows[label+'/native']['sha256']==q['product']['sha256']
  assert rows[label+'/source/trace-import.nano']['sha256']==q['fixture_sha256']
 if label!='direct-import':
  assert rows[label+'/trace-cc']['sha256']==q['observer']['sha256']
  assert rows[label+'/cc-observer.c']['sha256']==q['observer']['source_sha256']
  for c in q['controls']:
   expected=0 if c['label']=='version' else 1
   assert c['direct']['exit']==c['observed']['exit']==expected
   for suffix in ['stdout','stderr']:assert data(label+'/direct-'+c['label']+'.'+suffix)==data(label+'/observed-'+c['label']+'.'+suffix)
  cc=[json.loads(l) for l in data(label+'/cc-events.jsonl').splitlines()]
  begin={r['pid']:r for r in cc if r['event']=='start'};finish={r['pid']:r for r in cc if r['event']=='end'}
  assert len(begin)+len(finish)==len(cc) and set(finish)<=set(begin)
  for pid,e in finish.items():assert e['argv']==begin[pid]['argv'] and e['clock']>=e['start']==begin[pid]['start']
  assert (len(begin),len(finish))==((26,25) if label=='source-prefix' else (26,26))
  controls={c['observed']['pid'] for c in q['controls']}
  completed=[r for pid,r in finish.items() if pid not in controls]
  summary.append(dict(root=label,completed_tool_calls=len(completed),inflight=len(begin)-len(finish),child_wall=sum(r['clock']-r['start'] for r in completed),child_cpu=sum(r['user_seconds']+r['system_seconds'] for r in completed)))
proc.stdin.close();assert proc.wait()==0
print('PASS integrity: %d diagnostic records; two small imports/native runs pass; original-source prefix ends at120s with native-9 and zero shadow events. Qualification remains open.'%len(rows))
print(json.dumps(summary,indent=2))
