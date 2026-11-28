"""I verify retained repair evidence; I do not qualify a release or current host."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,subprocess
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
sha=lambda x:hashlib.sha256(x).hexdigest();base=a.archive;m=json.loads((base/'manifest.json').read_bytes());assert m['format']==1 and not m['whole_release'];rows={v['path']:v for v in m['files']};assert len(rows)==len(m['files'])
for h,r in m['chunks'].items():
 assert len(h)==64 and all(c in '0123456789abcdef'for c in h)
 data=(base/'objects'/(h+'.gz')).read_bytes();assert len(data)==r['compressed_bytes'] and sha(data)==r['compressed_sha256'];raw=gzip.decompress(data);assert len(raw)==r['bytes'] and sha(raw)==h

def raw(name):
 r=rows[name];data=b''.join(gzip.decompress((base/'objects'/(h+'.gz')).read_bytes())for h in r['chunks']);assert len(data)==r['bytes'] and sha(data)==r['sha256'];return data
for n in rows:
 path=Path(n);assert not path.is_absolute() and '..' not in path.parts;raw(n)
load=lambda n:json.loads(raw('job/'+n))
q=load('restored-pure-components/retained-callables-result.json');assert q['terminal'] and not q['whole_release'];assert q['all_guards_match'] and not q['functional_gate_passed'] and q['tests_run']==14 and q['failures']==0 and q['errors']==31
core=load('restored-pure-components/core-acceptance.json');text=load('text-companion-components/retained-callables-result.json');summary=load('component-acceptance.json')
assert core['core_methods_passed']==11 and len(set(core['core_methods']))==11 and not core['original_whole_run_passed']
assert core['original_result_sha256']==sha(raw('job/restored-pure-components/retained-callables-result.json')) and core['stderr_sha256']==sha(raw('job/restored-pure-components/stderr'))
import re
method_ids=re.findall(r'^test_\w+ \((tests\.test_nanolang_translator_[^)]+)\) \.\.\. ok$',raw('job/restored-pure-components/stderr').decode(),re.M)
assert method_ids==core['core_methods'] and len(method_ids)==11
assert text['terminal'] and text['functional_gate_passed'] and text['all_guards_match'] and text['tests_run']==3 and text['failures']==text['errors']==0
for kind in ('source','tools','images'):assert q[kind+'_after']==text[kind+'_before']==text[kind+'_after']
assert summary['terminal'] and not summary['whole_release'] and not summary['original_fourteen_method_observer_passed'] and not summary['broad_structured_gate_passed']
array=load('expanded-array-regression/result.json');assert array['terminal'] and array['passed'] and array['guards_match'] and array['tests_run']==5
assert sha(raw('job/expanded-array-regression/qualified-test-source.py'))==array['source_sha256']
assert summary['new_nonempty_native_cases']==60 and summary['new_nonempty_profile_refusals']==36 and summary['expected_runtime_traps']==250

for k in ('source','tools','images'):assert q[k+'_before']==q[k+'_after']
# I check pinned original bytes and each exact repair overlay against the cohort.
assert sha((base/'baseline-tree.txt').read_bytes())==m['baseline_tree_sha256']
assert subprocess.check_output(['git','ls-tree','-r',m['baseline']],cwd=a.repo)==(base/'baseline-tree.txt').read_bytes()
names=list(q['source_after']);needed=[n for n in names if n not in m['source_overlays']['cache-source']]
commands=''.join(m['baseline']+':'+n+'\n'for n in needed).encode();r=subprocess.run(['git','cat-file','--batch'],input=commands,cwd=a.repo,capture_output=True,check=True);data=r.stdout;offset=0
for n in needed:
 end=data.index(b'\n',offset);head=data[offset:end].split();assert head[1]==b'blob';size=int(head[2]);body=data[end+1:end+1+size];assert sha(body)==q['source_after'][n],n;offset=end+size+2
assert offset==len(data)
for n,h in m['source_overlays']['cache-source'].items():
 if n in q['source_after']:assert sha(raw('source-overlays/cache-source/'+n))==h==q['source_after'][n]
for k in ('tools_after','images_after'):
 for n,h in q[k].items():assert sha(raw('selected-final-tools/'+n.lstrip('/')))==h
for row in q['commands'][core['core_commands']:]:assert row.get('exit')==0 and 'bytes-like object' in row['exception']
assert len(q['commands'])-core['core_commands']==31
for command in q['commands'][:core['core_commands']]+text['commands']:
 assert command.get('exit')==0 and 'exception' not in command
 assert command['effective_timeout']==(600 if command['build'] else command['timeout'])
 for stream in ('stdout','stderr'):
  v=command[stream];n='job/'+str(Path(v['path']).relative_to(m['raw_job']));assert sha(raw(n))==v['sha256']
# I verify the complete recorded protocols and six original report hashes.
protocols={}
for path in dict.fromkeys(q['temporary_roots']):
 rel='job/'+str(Path(path).relative_to(m['raw_job']))
 if rel+'/manifest.json' not in rows:continue
 v=json.loads(raw(rel+'/manifest.json'))
 if not isinstance(v,dict) or 'cases' not in v or 'routes' not in v:continue
 groups=collections.Counter(r.get('group')for r in v['cases'])
 if groups.get('original')!=1001:continue
 extra=next(k for k in ('analysis','public','cache')if k in groups);expected={'analysis':308,'public':845,'cache':3317}[extra];assert groups[extra]==expected
 for row in v['cases']:
  source='job/'+str(Path(row['path']).relative_to(m['raw_job']));assert sha(raw(source))==row['sha256']
 for group in ('original',extra):
  selected=[r for r in v['routes']if r['group']==group];assert len(selected)==6
  reports=[]
  for r in selected:
   value=raw(rel+'/'+group+'-'+r['label']+'.reports');assert len(value.splitlines())==r['cases']==groups[group];assert sha(value)==r['sha256'];reports.append(value)
   if group=='original':assert r['sha256']=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
  assert len(set(reports))==1
 protocols[extra]=expected
assert protocols=={'analysis':308,'public':845,'cache':3317},protocols
for n in ('backend-comparison/result.json','backend-comparison-nanoisa/result.json'):
 v=load(n)
 for module in v['modules'].values():
  assert module['identical_output_bytes'];assert module['measurements'][0]['output_hashes']==module['measurements'][1]['output_hashes'];assert all(r['ok']==1 for r in module['measurements'])
reconstruction=load('f28-reconstruction.json');assert reconstruction['byte_exact'] and not reconstruction['rebuild'] and reconstruction['reconstructed_files']==7045
assert sha((a.repo/m['prerequisite_manifest']).read_bytes())==m['prerequisite_manifest_sha256']==reconstruction['archive_manifest_sha256']
for path,h in reconstruction['hosts'].items():assert q['tools_before'][path]==q['tools_after'][path]==h
old_inputs=load('retained-original-countercases/receipt.json');translated=load('recovered-original-translations/result.json');assert len(old_inputs)==len(translated['cases'])==7 and translated['all_passed'] and translated['terminal'] and translated['tool_before']==translated['tool_after']
for row,case in zip(old_inputs,translated['cases']):
 assert row['label']==case['label'] and row['sha256']==case['before']==case['after']==case['expected'] and case['exit']==0
 assert sha(raw('job/retained-original-countercases/'+row['label']+'.nvm'))==row['sha256']
profile=load('cache-profile/result.json');assert profile['terminal'] and profile['all_guards_match'] and profile['four_positive_hits'] and not profile['whole_release']
phases=load('cache-profile/phase-analysis.json');assert [r['compiler_calls']for r in phases]==[331,331,172,172] and [r['assembly_calls']for r in phases]==[85,85,43,43] and all(r['maximum_assembly_overlap']==8 and r['all_statuses_zero']for r in phases)
failed=load('combined-affine/retained-callables-result.json');assert failed['terminal'] and failed['failures']==3 and failed['all_guards_match'] and not failed['functional_gate_passed']
stopped=load('final-pure-components/retained-callables-result.json');assert stopped['terminal'] and not stopped['all_guards_match'] and 'KeyboardInterrupt' in stopped['exception']
print(json.dumps(dict(verified=True,records=len(rows),chunks=len(m['chunks']),source_files=len(names),core_methods=11,companion_methods=3,array_methods=5,original_fourteen_method_observer_passed=False,protocols=protocols,original_frame_cases=1001,original_report_hash_preserved=True,component_gate_passed=True,whole_release=False,current_liveness_checked=False)))
