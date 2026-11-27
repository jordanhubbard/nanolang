"""I verify exact original C-repair cohort bytes without executing a product."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,zipfile
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);args=p.parse_args();base=args.root
REV='8864bd37aa933fa244a7a0d779e1ee757266064a';FROZEN='6e09c07a9f1651294e4669282cdc17ee14126775';sha=lambda v:hashlib.sha256(v).hexdigest()
def stream_hash(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(2*1024*1024),b''):h.update(block)
 return h.hexdigest()
with subprocess.Popen(['git','ls-tree','-r',REV],cwd=args.repo,stdout=subprocess.PIPE) as proc:
 tree_sha=stream_hash(proc.stdout);assert proc.wait()==0
for name in ['tests/test_native_shadows.py','tests/test_native_shadow_helpers.py','tests/test_native_shadow_emitter.py']:
 assert subprocess.check_output(['git','show',FROZEN+':'+name],cwd=args.repo)==subprocess.check_output(['git','show',REV+':'+name],cwd=args.repo),name
metadata=json.loads((base/'artifacts-with-darwin-bootstrap.json').read_text())['artifacts'];reports=[]
for label,artifact_id,kind in [('linux-bootstrap',15233274843,'bootstrap'),('darwin-bootstrap',15234441631,'bootstrap'),('linux-cache',15234725242,'cache')]:
 artifact=next(a for a in metadata if a['id']==artifact_id);path=base/(label+'.zip');job=json.loads((base/(label+'.job.json')).read_text());assert job['head_sha']==REV and job['conclusion']=='success'
 with path.open('rb') as f:digest=stream_hash(f)
 assert path.stat().st_size==artifact['size_in_bytes'] and 'sha256:'+digest==artifact['digest']
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(names)==len(set(names));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names);assert z.testzip() is None
  markers=[n for n in names if n.endswith('/.test_output/c-repair-original-gates/revision.txt')];assert len(markers)==1;prefix=markers[0][:-len('.test_output/c-repair-original-gates/revision.txt')];guard=prefix+'.test_output/c-repair-original-gates/'
  read=lambda n:z.read(prefix+n);load=lambda n:json.loads(read(n));assert z.read(guard+'revision.txt').decode().strip()==REV
  with z.open(guard+'source-tree.txt') as stream:assert stream_hash(stream)==tree_sha
  for category in ['tools','images']:
   before=json.loads(z.read(guard+category+'-before.json'));after=json.loads(z.read(guard+category+'-after.json'));assert before==after
   for i,(name,expected) in enumerate(sorted(before.items())):
    member=prefix+'bin/'+name if category=='images' else guard+'selected-tools/'+str(i)+'-'+Path(name).name
    assert sha(z.read(member))==expected,(label,category,name)
  regression=z.read(guard+'string-binding-regression.log').decode()
  for marker in ['I passed 17 paired depth cases and output boundaries.','I passed 14 private-profile depth and fixture controls.','I passed 9 transient verifier allocation positions and recovery.']:
   assert marker in regression,(label,marker)
  report={'cohort':label,'revision':REV,'job_success':True,'zip_bytes':path.stat().st_size,'zip_sha256':digest,'zip_members':len(names),'source_tree_sha256':tree_sha,'source_tool_image_guards_verified':True,'whole_release':False}
  if kind=='bootstrap':
   q=load('bin/nanoc_bootstrap.json');assert q['phase']=='stage2' and q['raw_stage1_stage2_equal'];assert q['stage_timeout_seconds']==1800 and q['shadow_timeout_seconds']==30
   labels=['seed','seed-verify','stage1','stage1-verify','stage1-translate','stage1-native','stage1-hello-compile','stage1-hello-execute','stage2','stage2-verify','stage2-translate','stage2-native','stage2-hello-compile','stage2-hello-execute','seed-final-verify','stage1-final-verify','stage2-final-verify']
   assert [s['label'] for s in q['steps']]==labels and all(s['exit_code']==0 and s['elapsed_seconds']<1800 for s in q['steps'])
   def retained(absolute):
    rel=Path(absolute).relative_to(q['root']);assert '..' not in rel.parts;return prefix+str(rel)
   for step in q['steps']:assert sha(z.read(retained(step['log'])))==step['log_sha256'],step['label']
   for name,expected in q['artifacts'].items():assert sha(read('bin/'+name))==expected,name
   for name,expected in q['tools'].items():assert sha(read('bin/'+name))==expected,name
   for name,expected in q['hosts'].items():assert sha(z.read(retained(name)))==expected,name
   assert len(q['hosts'])==6 and read('bin/nanoc_stage1.nvm')==read('bin/nanoc_stage2.nvm');assert sha(read('bin/nanoc'))==q['artifacts']['nanoc_stage2']
   assert 'bin/nanoc succeeded with bin/nanoc_c removed' in z.read(guard+'bootstrap.log').decode()
   source_names=list(q['sources']);assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\n' not in n for n in source_names)
   request=''.join(REV+':'+n+'\n' for n in source_names).encode();data=subprocess.check_output(['git','cat-file','--batch'],cwd=args.repo,input=request);offset=0;generated=[]
   for name in source_names:
    end=data.index(b'\n',offset);header=data[offset:end].split()
    if len(header)==2 and header[-1]==b'missing':generated.append(name);blob=read(name);offset=end+1
    else:
     assert len(header)==3;size=int(header[-1]);blob=data[end+1:end+1+size];offset=end+size+2
    assert sha(blob)==q['sources'][name],name
   assert offset==len(data) and generated==['modules/index.json']
   log=z.read(guard+'native-suites.log').decode();assert re.search(r'^Ran 36 tests in [^\n]+\n\nOK(?:\n|$)',log,re.M);assert re.search(r'^Ran 12 tests in [^\n]+\n\nOK(?:\n|$)',log,re.M);assert 'FAILED' not in log and 'skipped=' not in log
   report.update(bootstrap_steps=17,raw_module_equal=True,raw_module_sha256=q['artifacts']['nanoc_stage1.nvm'],installed_image_matches_stage2=True,no_c_seed_probe=True,host_libraries=6,all_recorded_source_bytes_verified=True,git_source_files=len(source_names)-1,retained_generated_sources=generated,native_methods=36,emitter_methods=12,full_original_native_gate_passed=True)
  else:
   log=z.read(guard+'cache-workers.log').decode();methods=[int(n) for n in re.findall(r'^Ran (\d+) tests? in ',log,re.M)];assert methods==[42,57,4,132,20,16,13,3];assert [int(n) for n in re.findall(r'^OK \(skipped=(\d+)\)',log,re.M)]==[9,24,1];assert 'FAILED' not in log
   arrays=z.read(guard+'array-regression.log').decode();assert re.search(r'^Ran 6 tests in [^\n]+\n\nOK(?:\n|$)',arrays,re.M)
   report.update(cache_reported_methods=sum(methods[:-1]),existing_skips=34,worker_methods=3,array_methods=6,full_original_cache_worker_gate_passed=True)
  reports.append(report)
(base/'verified-cohorts.json').write_text(json.dumps(reports,indent=2)+'\n')
for report in reports:print(json.dumps({k:v for k,v in report.items() if k not in ['source_tree_sha256','raw_module_sha256','retained_generated_sources']}))
