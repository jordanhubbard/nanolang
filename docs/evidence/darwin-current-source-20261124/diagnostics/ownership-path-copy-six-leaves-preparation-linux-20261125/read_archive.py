"""I verify static six-leaf preparation and refusal of a live global/array parent."""
from pathlib import Path
import hashlib,json,subprocess
base=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=(base/'manifest.json').read_bytes();assert sha(raw)=='ea0f65dbb79ba97d1a3ab582269bab2a6476e5606224ae3d87c149878e531e5b'
m=json.loads(raw);assert m['format']=='plain-controller-v1' and m['prepared'] and not m['execution_started'] and not m['source_created'] and not m['whole_release']
assert m['source_base']=='67d99e5121521428740d385d10071b9679e66013'
for r in m['files']:
 raw=(base/r['path']).read_bytes();assert len(raw)==r['bytes'] and sha(raw)==r['sha256']
for name in ['setup.py','cohort.py','cohort_journal.py','admit.py']:compile((base/name).read_text(),name,'exec')
p=json.loads((base/'preparation.json').read_text());refused=json.loads((base/'live-parent-refusal.json').read_text())
assert p['terminal'] and p['prepared'] and not p['execution_started'] and not p['source_created'] and p['refusal_exit']!=0
assert refused['terminal'] and refused['commands']==[] and not refused.get('admitted_setup')
assert 'I require complete accepted global/array qualification before source preparation' in (base/'live-parent-refusal.stderr').read_text()
for name,h in p['controller_sha256'].items():assert sha((base/name).read_bytes())==h
before=(base/'baseline-cohort.py').read_text();after=(base/'cohort.py').read_text()
assert after==before.replace('original_methods=0,new_methods=6,',"working_overlay=True,overlay_sha256=s['overlays'],query_evidence=s['query_evidence'],global_array_evidence=s['global_array_evidence'],foundation_evidence=s['foundation_evidence'],original_methods=0,new_methods=6,")
assert 'new_cases=dict(owners=14542,aggregates=15222,effects=8128,value_calls=15644,call_context=834,analysis_cache=3317)' in after
assert sum([14542,15222,8128,15644,834,3317])==57687
repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=base,text=True).strip());manifest=(repo/m['baseline_manifest']).read_bytes();assert sha(manifest)==m['baseline_manifest_sha256']
rows={r['path']:r for r in json.loads(manifest)['files']}
assert m['baseline_records'] and all(rows[n]['sha256']==sha((base/'baseline-cohort.py').read_bytes()) for n in m['baseline_records'])
setup=(base/'setup.py').read_text();guard=setup.index("assert parent.get('terminal')")
assert guard<setup.index("run('complete-global-array-reader'")<setup.index("root=j/'source';root.mkdir()")
assert "os.environ['NANOLANG_PATH_COPY_GLOBAL_ARRAY_EVIDENCE_COMMIT']" in setup and 'dec468a6bda61adf5d268c365832811c448d69f0' in setup
assert "assert s['source_before']==gate['source_before']==parent['source_before']" in setup
print('PASS 9 sealed preparation records; unchanged six methods/57687 new cases, static syntax and real live-parent refusal before any command or checkout. No positive admission, compilation, execution, component or release acceptance.')
