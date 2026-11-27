"""I check the actual current Darwin cache/worker/regression artifacts."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
root=a.artifact;guard=root/'.test_output/c-repair-original-gates'
sha=lambda b:hashlib.sha256(b).hexdigest()
rev=(guard/'revision.txt').read_text().strip();assert rev=='6e09c07a9f1651294e4669282cdc17ee14126775'
for kind in ('tools','images'):
 before=json.loads((guard/(kind+'-before.json')).read_bytes());assert before==json.loads((guard/(kind+'-after.json')).read_bytes())
 for i,(name,want) in enumerate(sorted(before.items())):
  f=root/'bin'/name if kind=='images' else guard/'selected-tools'/(str(i)+'-'+Path(name).name)
  assert sha(f.read_bytes())==want,(kind,name)
with subprocess.Popen(['git','-C',str(a.repo),'ls-tree','-r',rev],stdout=subprocess.PIPE) as process:
 h=hashlib.sha256()
 for block in iter(lambda:process.stdout.read(1024*1024),b''):h.update(block)
 assert process.wait()==0
assert h.hexdigest()==sha((guard/'source-tree.txt').read_bytes())
cache=(guard/'cache-workers.log').read_text();arrays=(guard/'array-regression.log').read_text()
assert 'FAILED (' not in cache+arrays and 'Traceback (most recent call last)' not in cache+arrays
runs=[int(n) for n in re.findall(r'^Ran (\d+) tests? in ',cache,re.M)]
skips=[int(n) for n in re.findall(r'^OK \(skipped=(\d+)\)',cache,re.M)]
assert sum(runs)==287,(runs,sum(runs))
assert sum(skips)==26,skips
assert runs[-1]==3,runs
assert re.search(r'^Ran 6 tests in [^\n]+\n\nOK(?:\n|$)',arrays,re.M)
assert len(re.findall(r'^Ran ',arrays,re.M))==1
assert 'test_native_record_pointer_storage' in arrays and 'test_native_array_write_inference' in arrays
print(json.dumps({'verified':True,'revision':rev,'cache_methods_reported':sum(runs)-3,'cache_existing_skips':sum(skips),'worker_methods_passed':3,'array_pointer_methods_passed':6,'source_tree_matches':True,'selected_tool_and_image_guards_match':True,'whole_release':False}))
