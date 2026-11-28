"""I verify my retained current Darwin bootstrap/native pass and emitter refusal."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
root=a.artifact;guard=root/'.test_output/c-repair-original-gates'
sha=lambda b:hashlib.sha256(b).hexdigest()
q=json.loads((root/'bin/nanoc_bootstrap.json').read_text())
rev=(guard/'revision.txt').read_text().strip();assert rev=='6e09c07a9f1651294e4669282cdc17ee14126775'
def retained(path):
 rel=Path(path).relative_to('/Users/runner/work/nanolang/nanolang');assert '..' not in rel.parts;return root/rel
assert q['phase']=='stage2' and q['raw_stage1_stage2_equal']
assert q['stage_timeout_seconds']==1800 and q['shadow_timeout_seconds']==30
assert len(q['steps'])==17 and all(r['exit_code']==0 for r in q['steps'])
for r in q['steps']:assert sha(retained(r['log']).read_bytes())==r['log_sha256'],r['label']
for name,value in q['artifacts'].items():assert sha((root/'bin'/name).read_bytes())==value,name
assert (root/'bin/nanoc_stage1.nvm').read_bytes()==(root/'bin/nanoc_stage2.nvm').read_bytes()
assert sha((root/'bin/nanoc').read_bytes())==q['artifacts']['nanoc_stage2']
assert 'bin/nanoc succeeded with bin/nanoc_c removed' in (guard/'bootstrap.log').read_text()
assert len(q['hosts'])==6
for name,value in q['hosts'].items():assert sha(retained(name).read_bytes())==value,name
for name,value in q['tools'].items():assert sha((root/'bin'/name).read_bytes())==value,name
for kind in ['tools','images']:
 before=json.loads((guard/(kind+'-before.json')).read_text());after=json.loads((guard/(kind+'-after.json')).read_text());assert before==after
 for i,(name,value) in enumerate(sorted(before.items())):
  f=(root/'bin'/name) if kind=='images' else guard/'selected-tools'/(str(i)+'-'+Path(name).name)
  assert sha(f.read_bytes())==value,(kind,name)
cmd=['git','-C',str(a.repo),'ls-tree','-r',rev]
with subprocess.Popen(cmd,stdout=subprocess.PIPE) as proc:
 h=hashlib.sha256()
 for block in iter(lambda:proc.stdout.read(1024*1024),b''):h.update(block)
 assert proc.wait()==0
assert h.hexdigest()==sha((guard/'source-tree.txt').read_bytes())
names=list(q['sources']);request=''.join(rev+':'+n+'\n' for n in names).encode()
data=subprocess.check_output(['git','-C',str(a.repo),'cat-file','--batch'],input=request)
offset=0
missing=[]
for name in names:
 end=data.index(b'\n',offset);header=data[offset:end].split()
 if len(header)==2 and header[-1]==b'missing':
  assert name=='modules/index.json',name
  assert sha((root/name).read_bytes())==q['sources'][name],name
  offset=end+1;continue
 assert len(header)==3,(name,header)
 size=int(header[-1]);blob=data[end+1:end+1+size];offset=end+size+2
 assert sha(blob)==q['sources'][name],name
assert offset==len(data)
assert missing==[],missing
assert sha((root/'modules/index.json').read_bytes())==q['sources']['modules/index.json']
log=(guard/'native-suites.log').read_text()
assert re.search(r'Ran 36 tests in [^\n]+\n\nOK(?:\n|$)',log)
assert 'Ran 0 tests in 96.600s' in log
assert 'FAILED (errors=1)' in log and 'I stopped shadow execution after 10 seconds.' in log
assert 'nanoc_c exited 1:' in log
assert len(re.findall(r'^Ran 36 tests in ',log,re.M))==len(re.findall(r'^Ran 0 tests in ',log,re.M))==1
print(json.dumps({'verified':True,'revision':rev,'bootstrap_steps':17,'git_sources_verified':len(names)-1,'generated_sources_retained':['modules/index.json'],'generated_sources_unretained':missing,'all_recorded_source_bytes_verified':True,'host_libraries':len(q['hosts']),'raw_module_bytes':(root/'bin/nanoc_stage1.nvm').stat().st_size,'raw_module_sha256':q['artifacts']['nanoc_stage1.nvm'],'installed_image_matches_stage2':True,'original_no_c_seed_probe_passed':True,'native_methods_passed':36,'emitter_methods_discovered':0,'emitter_methods_passed':0,'emitter_c_seed_setup_shadow_timeout':True,'full_native_component_gate_passed':False,'whole_release':False}))
