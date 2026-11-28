"""I check my partial review records; I grant no release acceptance."""
from pathlib import Path
import argparse, hashlib, json, subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
expected={'pr1069-make-wiring.json': 'a021e376f2918c0fcb575e38363929f0288743c160d56d9fa34fa2402f3332dd', 'README.md': '074d2d1eafb7b7bdd75843dd77a2184a18392593f12aa9e2abf07ca19700fe71', 'pr-1066-consolidation.md': '1c7996c8bb3e2ccce91562ed44c5dd51b3b7b2fccf3843bfc9054d53e31334fb', 'pr1069-consolidation.md': '16cff514c83c2c4e4330d69d0481fb844a566355c24037b73d488b439c870e8c', 'pr-1072-consolidation.md': 'a126e82b28ca3e8d451faf00c8ec6e3dc597710665d855e690003362ccdab5f4', 'review.json': 'dd6f30b1d57f802329d7e1fe871aa1548acb5196316123569c1a365879b5de4a'}
for name,digest in expected.items():assert sha((root/name).read_bytes())==digest,name
q=json.loads((root/'review.json').read_text())
assert q['release']=='5.1.0' and not q['release_gate_passed'] and not q['all_open_item_reviews_complete'] and not q['all_69_requirements_complete']
assert [r['number'] for r in q['consolidated_pull_requests']]==[1066,1072,1069]
for row in q['consolidated_pull_requests']:
 assert row['after']['state']=='CLOSED' and row['after']['headRefOid']==row['head']
 assert not row['source_changed'] and not row['branch_deleted'] and not row['acceptance_issue_closed']
 assert row['replacement']==1168
 body=(root/row['annotation_file']).read_text()
 assert body.startswith('Stale for release v5.1.0:') and row['head'] in body
for row in q['source_retention']:
 subprocess.run(['git','merge-base','--is-ancestor',row['head'],row['integration']],cwd=a.repo,check=True)
 for file in row['exact_files']:
  old=subprocess.check_output(['git','show',row['head']+':'+file['path']],cwd=a.repo)
  current=subprocess.check_output(['git','show',row['integration']+':'+file['path']],cwd=a.repo)
  assert old==current and sha(current)==file['sha256']
wiring=json.loads((root/'pr1069-make-wiring.json').read_text())
make=subprocess.check_output(['git','show',wiring['integration']+':Makefile.gnu'],cwd=a.repo,text=True)
for marker in ['python3 -m unittest tests.test_make_prerequisites tests.test_wasmtime_prerequisite','test-quick test-units: test-make-prerequisites','test-make-prerequisites:\n\tpython3 -m unittest -v tests.test_make_prerequisites']:
 assert marker in make
print('I verify three annotated PR consolidations and retained source/wiring. My full release-readiness gate remains open.')
