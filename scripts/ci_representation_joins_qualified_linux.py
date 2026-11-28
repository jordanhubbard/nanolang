"""I verify exact original C-repair cohort bytes without executing a product."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,zipfile,struct
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--cohort',choices=['linux-bootstrap','linux-cache','darwin-bootstrap','darwin-cache'],required=True);args=p.parse_args();base=args.root
REV='ddd3b5a4ff35678d1906c07c08fc6bbc39b041de';FROZEN='6e09c07a9f1651294e4669282cdc17ee14126775';sha=lambda v:hashlib.sha256(v).hexdigest()
def stream_hash(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(2*1024*1024),b''):h.update(block)
 return h.hexdigest()
with subprocess.Popen(['git','ls-tree','-r',REV],cwd=args.repo,stdout=subprocess.PIPE) as proc:
 tree_sha=stream_hash(proc.stdout);assert proc.wait()==0
for name in ['tests/test_native_shadows.py','tests/test_shadow_runner.py','tests/test_make_timeouts.py','tests/test_native_shadow_only_entry.py','tests/test_native_shadow_helpers.py','tests/test_native_shadow_emitter.py']:
 assert subprocess.check_output(['git','show',FROZEN+':'+name],cwd=args.repo)==subprocess.check_output(['git','show',REV+':'+name],cwd=args.repo),name
metadata=json.loads((base/'artifacts-current.json').read_text())['artifacts'];reports=[]
label=args.cohort
platform,kind=label.split('-');worker='ubuntu-latest' if platform=='linux' else 'xcode-27'
expected_name='c-repair-original-'+worker+'-'+kind+'-'+REV
selected=[a for a in metadata if a['name']==expected_name];assert len(selected)==1
for label,artifact_id,kind in [(label,selected[0]['id'],kind)]:
 artifact=next(a for a in metadata if a['id']==artifact_id);path=base/(label+'.zip');job=json.loads((base/(label+'.job.json')).read_text());assert job['head_sha']==REV and job['conclusion']=='success' and job['run_id']==46119624230
 assert job['id']=={'linux-bootstrap':140901037401,'linux-cache':140901037397,'darwin-bootstrap':140901037222,'darwin-cache':140901037356}[label]
 assert next(step for step in job['steps'] if step['name']=='Verify source and selected tool guards')['conclusion']=='success'
 assert next(step for step in job['steps'] if step['name']=='Require a hosted worker before provisioning')['conclusion']=='success'
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
  for marker in ['I preserve caller declarations across refusals, collect reachable children and hand off without owning borrowed results.', 'I preserve overlapping cyclic caller roots without adopting storage.', 'I refuse owned record graphs while scalar list cells and borrow-only keepalive roots grant no custody handoff.', 'I preserve custody declarations through 6 allocation failures and recover.']:
   assert regression.count(marker)==1,(label,marker)
  rows=[json.loads(line) for line in regression.splitlines() if line.startswith('{"case":')]
  rows=[r for r in rows if 'expected_reclaims' in r]
  assert len(rows)==10 and len({r['case'] for r in rows})==10 and sum(r['actual_reclaims'] for r in rows)==84
  assert all(r['actual_reclaims']==r['expected_reclaims'] for r in rows)
  report_caller=True
  root_groups={};key=None
  for line in regression.splitlines():
   match=re.match(r'(current|previous) (ordinary|sanitized) (\{"roots":.*)',line)
   if match:key=(match[1],match[2]);line=match[3];root_groups[key]=[]
   if line.startswith('{"roots":'):root_groups[key].append(json.loads(line))
  assert set(root_groups)=={(v,m) for v in ['current','previous'] for m in ['ordinary','sanitized']}
  for (variant,mode),rows in root_groups.items():
   expected=[dict(roots=n,fresh_lookups=(new if variant=='current' else old),reused_lookups=n*(1 if variant=='current' else 2),duplicate_lookups=n,allocations=k) for n,old,new,k in [(16,31,16,2),(32,79,49,4),(1024,3055,2038,14),(4096,12271,8184,18)]]
   assert rows==expected,(variant,mode,rows)
  assert regression.count('eight allocation-abort boundaries.')==4
  assert re.search(r'^Ran 4 tests in [^\n]+\n\nOK(?:\n|$)',regression,re.M)
  for name in ['test_growing_aliased_graph_has_bounded_lookup_work','test_current_preserves_identity_allocation_and_lookup_budget','test_missing_rehash_loses_required_membership','test_previous_helper_exceeds_single_lookup_budget']:assert name in regression,name


  flow_groups={};active=None
  for line in regression.splitlines():
   match=re.match(r'(current|previous|collisions) (ordinary|sanitized) (\{"semantic_cases":.*)',line)
   if match:
    key=(match[1],match[2]);active=[json.loads(match[3])];flow_groups.setdefault(key,[]).append(active)
   elif active is not None and line.startswith('{"nodes":'):active.append(json.loads(line))
  assert set(flow_groups)=={(v,m) for v in ['current','previous','collisions'] for m in ['ordinary','sanitized']}
  all_digests=set()
  for (variant,mode),batches in flow_groups.items():
   assert len(batches)==(3 if variant=='previous' else 1)
   for rows in batches:
    assert rows[0]['semantic_cases']==1800 and re.fullmatch('[0-9a-f]{16}',rows[0]['digest'])
    all_digests.add(rows[0]['digest']);assert len(rows)==7
    assert [row['nodes'] for row in rows[1:]]==[32,128,1024,4096,128,128]
    assert [row['cycle'] for row in rows[1:]]==[0,0,0,0,1,0]
    assert [row['wide'] for row in rows[1:]]==[0,0,0,0,0,1]
    assert [row['root_calls'] for row in rows[1:]]==([690,8898,529922,8411138,9033,17026] if variant=='previous' else [162,642,5122,20482,647,642])
    assert [row['optional_allocations'] for row in rows[1:]]==([0]*6 if variant=='previous' else [3,5,8,10,5,0])
    assert all(row['failure']==0 for row in rows[1:])
  assert len(all_digests)==1
  for name in ['test_previous_helper_preserves_graphs_and_exceeds_lookup_budget','test_current_preserves_graphs_and_bounds_root_lookup_work','test_collisions_preserve_all_pairs_and_flags','test_omitted_optional_flag_loses_a_distinct_pair']:
   assert name in regression
  assert len(re.findall(r'^Ran 4 tests in [^\n]+\n\nOK(?:\n|$)',regression,re.M))==2
  assert 'Shape constraints: 3890 passed, 0 failed' in regression

  for marker in ['I passed 17 paired depth cases and output boundaries.','I passed 14 private-profile depth and fixture controls.','I passed 9 transient verifier allocation positions and recovery.']:
   assert marker in regression,(label,marker)
  for label_name in ['empty','plain tail','newline boundaries','multi-byte delimiter','CRLF','UTF8 segments','empty-delimiter bytes','empty input and delimiter','nonoverlapping matches']:
   assert 'I retained '+label_name+' across input and frame retirement.' in regression,label_name
  gc_rows=[json.loads(line) for line in regression.splitlines() if line.startswith('{"owners":')]
  assert len(gc_rows)==12 and [r['owners'] for r in gc_rows]==[16,64,256,1024,4096,16]*2
  assert all(r['within_budget'] and r['maximum_bucket']<r['bucket_budget']==r['owners']*8 for r in gc_rows)
  for count in [15,16,17,31]:assert regression.count(f'I preserved both essential allocation failures around a {count}-owner boundary.')==2
  assert regression.count('I retained registry membership, payloads, logical accounting, release and reinitialization within my working-set budget.')==2
  assert regression.count('I passed header/registry refusal, three constructor families, retries, finalizers and reinitialization.')==3
  assert all(text not in regression for text in ['ERROR: AddressSanitizer','ERROR: LeakSanitizer','runtime error:'])
  for name in ['tests/test_gc_allocation_failure.c','tests/test_gc_struct.c','src/runtime/gc.h']:
   assert subprocess.check_output(['git','show','c8bd5c821ac6361e6bbdef2d1750721bfaa2ba22:'+name],cwd=args.repo)==subprocess.check_output(['git','show',REV+':'+name],cwd=args.repo)
  report={'cohort':label,'revision':REV,'job_success':True,'zip_bytes':path.stat().st_size,'zip_sha256':digest,'zip_members':len(names),'source_tree_sha256':tree_sha,'source_tool_image_guards_verified':True,'direct_split_lifetime_cases':9,'whole_release':False,'gc_working_set_epochs':12,'gc_growth_boundary_reports':8,'gc_injected_failures_normal_and_sanitized':48,'original_gc_sources_unchanged':True,'caller_root_controls_normal':report_caller,'root_insertion_methods':4,'root_insertion_normal_and_sanitized':True,'old_helper_and_missing_rehash_controls_passed':True,'root_lookup_4096_fresh':8184,'root_lookup_4096_reused':4096,'root_lookup_4096_duplicate':4096,'root_allocations_4096':18}
  report.update(shape_flow_methods=4,shape_flow_semantic_cases=1800,shape_flow_old_root_calls4096=8411138,shape_flow_current_root_calls4096=20482,shape_flow_same_host_old_current_digest=next(iter(all_digests)),shape_flow_collision_and_omitted_flag_controls=True,shape_flow_injected_index_allocation_failures=5,unchanged_original_shape_assertions=3890)
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
   native=read('bin/nanoc_stage2')
   if platform=='darwin':assert native[:4]==bytes.fromhex('cffaedfe') and struct.unpack_from('<I',native,4)[0]==0x0100000c
   else:assert native[:6]==b'\x7fELF\x02\x01' and int.from_bytes(native[18:20],'little')==62
   report['native_macho_arm64']=True
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
   log=z.read(guard+'cache-workers.log').decode();methods=[int(n) for n in re.findall(r'^Ran (\d+) tests? in ',log,re.M)];assert methods==[42,57,4,132,20,16,13,3];assert [int(n) for n in re.findall(r'^OK \(skipped=(\d+)\)',log,re.M)]==([9,24,1] if platform=='linux' else [4,22]);assert 'FAILED' not in log
   arrays=z.read(guard+'array-regression.log').decode();assert re.search(r'^Ran 6 tests in [^\n]+\n\nOK(?:\n|$)',arrays,re.M)
   report.update(cache_reported_methods=sum(methods[:-1]),existing_skips=(34 if platform=='linux' else 26),worker_methods=3,array_methods=6,full_original_cache_worker_gate_passed=True)
  if kind=='bootstrap':
   line_log=z.read(guard+'compiler-line-splitting.log').decode();assert re.search(r'^Ran 2 tests in [^\n]+\n\nOK(?:\n|$)',line_log,re.M)
   report['line_splitting_methods']=2
  executable=read('bin/nanoc_c')
  if platform=='darwin':assert executable[:8]==bytes.fromhex('cffaedfe0c000001')
  else:assert executable[:6]==b'\x7fELF\x02\x01' and int.from_bytes(executable[18:20],'little')==62
  reports.append(report)
(base/('verified-'+label+'.json')).write_text(json.dumps(reports,indent=2)+'\n')
for report in reports:print(json.dumps({k:v for k,v in report.items() if k not in ['source_tree_sha256','raw_module_sha256','retained_generated_sources']}))
