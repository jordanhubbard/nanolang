"""I verify three retained proposals without granting product or release acceptance."""
from pathlib import Path
import argparse,hashlib,json,subprocess,yaml
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--proof',type=Path,default=Path(__file__).with_name('source-proof.json'));a=p.parse_args()
sha=lambda d:hashlib.sha256(d).hexdigest();raw=a.proof.read_bytes();assert sha(raw)=='46e258eb0b21ddaacdc0cf42ac7fd8a5b47e92e384d70d90517fee8b70120db1';q=json.loads(raw)
assert not any(q[k] for k in ['release_gate_passed','all_69_requirements_complete','entire_open_inventory_reviewed','product_acceptance_granted']);assert q['replacement_pr']==1168
read=lambda rev,name:subprocess.check_output(['git','show',rev+':'+name],cwd=a.repo)
def version_block(text):
 start=text.index('    if (argc >= 2 && (strcmp(argv[1], "--version")');end=text.index('\n    }',start)+6;return text[start:end]
assert [r['pr'] for r in q['items']]==[1070,1071,1074]
for row in q['items']:
 assert row['closure_is_consolidation_only'] and not row['main_merge_granted']
 assert subprocess.run(['git','merge-base','--is-ancestor',row['head'],q['integration']],cwd=a.repo).returncode==0
 for item in row['full_files']:
  data=read(row['head'],item['path']);assert data==read(q['integration'],item['path']) and len(data)==item['bytes'] and sha(data)==item['sha256']
 for item in row['fragments']:
  old=read(row['head'],item['path']).decode();current=read(q['integration'],item['path']).decode();kind=item['kind']
  if kind=='original_job_except_explicit_mode_condition':
   original=yaml.safe_load(old)['jobs'][item['job']];workflow=yaml.safe_load(current);observed=dict(workflow['jobs'][item['job']]);assert observed.pop('if')==item['current_condition'];assert original==observed and len(original['steps'])==item['original_step_count']==8;assert sha(json.dumps(original,sort_keys=True).encode())==item['canonical_sha256']
   inputs=workflow.get('on',workflow.get(True))['workflow_dispatch']['inputs'];assert {key:inputs[key]['default'] for key in item['new_mode_defaults']}==item['new_mode_defaults']
  elif kind=='exact_literal_block':
   start=old.index(item['start_marker']);end=old.index(item['end_marker'],start);fragment=old[start:end];assert fragment in current and sha(fragment.encode())==item['sha256']
  elif kind=='required_literal':assert item['literal'] in old and item['literal'] in current
  elif kind=='retained_test_module_in_expanded_fast_gate':
   assert '\tpython3 -m unittest tests.test_wasmtime_prerequisite' in old
   assert item['literal'] in current and item['literal'].split()==['python3','-m','unittest','tests.test_make_prerequisites','tests.test_wasmtime_prerequisite']
  elif kind=='original_tail_section_retained':
   fragment=old[old.index(item['start_marker']):];assert fragment in current and sha(fragment.encode())==item['sha256']
  elif kind=='exact_version_branch':
   fragment=version_block(old);assert fragment==version_block(current) and sha(fragment.encode())==item['sha256'] and 'Built:' not in fragment
  else:raise AssertionError(kind)
assert all(q['related_issue_states'][str(i)]['state']=='CLOSED' for i in [1045,1073,1060]);assert all(q['related_issue_states'][str(i)]['state']=='OPEN' for i in [1049,1054,1039])
print('I verify eight complete source files and nine wiring/job/version fragments for proposals1070/1071/1074 at integrationf878d817b7. I grant no merge, product, whole-inventory or69-requirement release acceptance.')
