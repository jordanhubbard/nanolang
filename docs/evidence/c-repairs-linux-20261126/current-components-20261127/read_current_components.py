"""I check retained component bytes and replay their pure Python report oracles."""
import argparse,collections,gzip,hashlib,importlib,json,re,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
encoded=(a.archive/'manifest.json').read_bytes();assert sha(encoded)=='d0238220daab4d89488543b8b55e215b22064f07ccc6229b24e5a544a0ff3ada';m=json.loads(encoded)
assert m['version']==1 and m['revision']=='6e09c07a9f1651294e4669282cdc17ee14126775' and not m['whole_release']
seen=set();used=set()
with tempfile.TemporaryDirectory(prefix='current-components-reader-',dir=a.workspace) as temp:
 root=Path(temp)
 for r in m['files']:
  rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
  assert r['path'] not in seen;seen.add(r['path']);assert r['kind']=='file'
  target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with target.open('wb') as out:
   for key in r['chunks']:
    assert re.fullmatch('[0-9a-f]{64}',key)
    data=(a.archive/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key];assert sha(data)==meta['gzip_sha256']
    block=gzip.decompress(data);assert len(block)==meta['bytes'] and sha(block)==key
    out.write(block);h.update(block);size+=len(block);used.add(key)
  assert size==r['size'] and h.hexdigest()==r['sha256'],r['path']
 assert used==set(m['chunks'])
 load=lambda name:json.loads((root/name).read_bytes())
 core=load('core/retained-callables-result.json');companion=load('companion/retained-callables-result.json');accepted=load('core/component-acceptance.json')
 assert core['terminal'] and core['all_guards_match'] and core['qualified_guards_match']
 assert core['tests_run']==14 and core['failures']==0 and core['errors']==29 and not core['functional_gate_passed']
 assert companion['terminal'] and companion['all_guards_match'] and companion['qualified_guards_match']
 assert companion['tests_run']==3 and companion['errors']==companion['failures']==0 and companion['functional_gate_passed']
 assert not accepted['original_fourteen_method_observer_passed'] and not accepted['whole_release']
 assert accepted['core_result_sha256']==sha((root/'core/retained-callables-result.json').read_bytes())
 assert accepted['companion_result_sha256']==sha((root/'companion/retained-callables-result.json').read_bytes())
 assert accepted['core_methods_passed']==11 and accepted['companion_methods_passed']==3
 for kind in ('source','tools'):
  assert core[kind+'_before']==core[kind+'_after']==companion[kind+'_before']==companion[kind+'_after']
 for n,h in core['images_after'].items():assert core['images_before'][n]==companion['images_before'][n]==companion['images_after'][n]==h
 assert companion['images_before']==companion['images_after']
 for q,folder in ((core,'core'),(companion,'companion')):
  assert q['source_head']==m['revision'] and q['prerequisite_current_stage2'] and not q['prerequisite_reconstructed']
  assert q['observer_sha256']==sha((root/folder/'run.py').read_bytes())
  for kind in ('tools_after','images_after'):
   for name,want in q[kind].items():assert sha((root/'selected-inputs'/name.lstrip('/')).read_bytes())==want,name
  for command in q['commands']:
   assert command['effective_timeout']==(600 if command['build'] else command['timeout'])
   if 'exception' in command:continue
   assert command['exit']==0,command['argv']
   for stream in ('stdout','stderr'):
    data=command[stream];relative=Path(data['path']).relative_to(m[folder+'_raw'])
    assert sha((root/folder/relative).read_bytes())==data['sha256']
 errors=[(i,c) for i,c in enumerate(core['commands']) if 'exception' in c]
 assert len(errors)==29 and errors[0][0]==10381
 for i,c in errors:assert 'FileNotFoundError' in c['exception'] and c['argv'][0]==m['tree']+'/bin/nanoisa'
 assert all(c['exit']==0 and 'exception' not in c for c in core['commands'][:10381])
 stderr=(root/'core/stderr').read_text();methods=re.findall(r'^(test_\w+ \([^\n]+\)) \.\.\. ok$',stderr,re.M)
 expected=['test_original_frames_and_complete_private_body','test_original_frames_and_public_effect_globals','test_original_frames_and_completed_cache_paths','test_boxed_fields_below_exact_shared_record_views','test_imported_shadows','test_malformed_solver_packets','test_named_directional_countercases','test_original_api_trace','test_full_collector_matrix','test_imported_shadows','test_original_callable_suite']
 assert [v.split(' ')[0] for v in methods[:11]]==expected
 assert 'Ran 14 tests in 1911.659s' in stderr and 'FAILED (errors=29)' in stderr
 fix=load('core/assembler-prerequisite.json');assert fix['exit_code']==0 and fix['old_guards_match']
 assert fix['original_result_sha256']==accepted['core_result_sha256']
 assert fix['new_assembler_sha256']==companion['images_after'][m['tree']+'/bin/nanoisa']
 assert '\nRan 3 tests in 5.835s\n\nOK\n' in (root/'companion/stderr').read_text()
 # I bind the genuine producer to the already accepted immutable native receipt.
 prerequisite=subprocess.check_output(['git','-C',str(a.repo),'show','HEAD:'+m['prerequisite_manifest']])
 assert sha(prerequisite)==m['prerequisite_manifest_sha256'];prior=json.loads(prerequisite)
 entries={r['path']:r for r in prior['files']}
 native=(root/'core/qualified-native-result.json').read_bytes();receipt=(root/'core/qualified-bootstrap-receipt.json').read_bytes()
 assert sha(native)==entries['.test_output/frozen-native-qualification/result.json']['sha256']
 assert sha(receipt)==entries['bin/nanoc_bootstrap.json']['sha256']
 n=json.loads(native);b=json.loads(receipt)
 assert n['component_native_gate_passed'] and n['guards_match'] and n['generated_index_matches_receipt'] and n['host_closure_matches_receipt']
 assert len(b['steps'])==17 and all(s['exit_code']==0 for s in b['steps'])
 assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30 and b['raw_stage1_stage2_equal']
 producer_root=Path(prior['original_root'])
 for q in (core,companion):
  assert q['qualified_sources_after']==n['sources_after'] and q['qualified_tools_after']==n['tools_after']
  assert q['prerequisite_native_sha256']==sha(native) and q['prerequisite_receipt_sha256']==sha(receipt)
  assert q['tools_after'][str(producer_root/'bin/nanoc_stage2')]==b['artifacts']['nanoc_stage2']==entries['bin/nanoc_stage2']['sha256']
  for path,want in b['hosts'].items():assert q['tools_after'][path]==want
 # I check every actual selected source byte against Git, then use only its report helpers.
 source=root/'source';source.mkdir();names=list(core['source_after'])
 request=''.join(m['revision']+':'+name+'\n' for name in names).encode()
 data=subprocess.check_output(['git','-C',str(a.repo),'cat-file','--batch'],input=request);offset=0
 for name in names:
  end=data.index(b'\n',offset);header=data[offset:end].split();assert header[1]==b'blob',name
  size=int(header[-1]);blob=data[end+1:end+1+size];offset=end+size+2;assert sha(blob)==core['source_after'][name],name
  target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
 assert offset==len(data)
 def no_execution(event,args):
  if event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec'):raise RuntimeError('I forbid product execution during replay: '+event)
 sys.addaudithook(no_execution);sys.path.insert(0,str(source))
 private=importlib.import_module('tests.test_nanolang_translator_affine_analysis')
 public=importlib.import_module('tests.test_nanolang_translator_affine_public')
 cache=importlib.import_module('tests.test_nanolang_translator_affine_analysis_cache')
 frame=importlib.import_module('tests.test_nanolang_translator_affine_frame')
 originals=frame.flow_cases()[0]+frame.queue_cases()+frame.frame_cases();assert len(originals)==1001
 builders={'analysis':private.analysis_cases,'public':public.public_cases,'cache':cache.cache_cases}
 wanted={'analysis':308,'public':845,'cache':3317};protocols={};reports_checked=0
 for name in dict.fromkeys(core['temporary_roots']):
  path=root/'tree'/Path(name).relative_to(m['tree'])/'manifest.json'
  if not path.is_file():continue
  protocol=json.loads(path.read_text())
  if not isinstance(protocol,dict) or not all(k in protocol for k in ('cases','routes')):continue
  groups=collections.Counter(row.get('group') for row in protocol['cases'])
  if groups.get('original')!=1001:continue
  extra=next(k for k in wanted if k in groups);assert groups[extra]==wanted[extra]
  assert extra not in protocols;protocols[extra]=wanted[extra]
  directory=path.parent
  for group,constructed in [('original',originals),(extra,builders[extra]())]:
   recorded=[r for r in protocol['cases'] if r['group']==group];assert len(recorded)==len(constructed)
   for row,(name,payload,expected_value) in zip(recorded,constructed):
    actual=root/'tree'/Path(row['path']).relative_to(m['tree'])
    assert row['name']==name and actual.read_bytes()==payload and sha(payload)==row['sha256']
    if 'expected' in row:assert row['expected']==json.loads(json.dumps(expected_value))
   routes=[r for r in protocol['routes'] if r['group']==group];assert len(routes)==6
   streams=[]
   for route in routes:
    raw=(directory/(group+'-'+route['label']+'.reports')).read_bytes();assert sha(raw)==route['sha256']
    lines=raw.decode().splitlines();assert len(lines)==route['cases']==len(constructed)
    if group=='original':assert sha(raw)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
    for (name,_,expected_value),line in zip(constructed,lines):
     if group=='analysis':private.inspect_report(line,expected_value)
     elif group=='public':public.inspect_report(line,expected_value)
     elif group=='cache' or name.startswith(('global-algebra-','bounded-fifo-')):assert line==expected_value
     else:assert line.startswith(expected_value)
    reports_checked+=len(lines);streams.append(raw)
   assert len(set(streams))==1
 assert protocols==wanted and reports_checked==44838
 outputs=[]
 for command in core['commands'][:10381]:
  d=command['stdout'];outputs.append((root/'core'/Path(d['path']).relative_to(m['core_raw'])).read_bytes())
 assert any(b'Shape constraints: 3890 passed, 0 failed' in v for v in outputs)
 assert any(b'505 passed, 0 failed' in v for v in outputs)
 print(json.dumps({'manifest_sha256':sha(encoded),'archive_records_verified':len(seen),'archive_chunks_verified':len(used),'git_sources_verified':len(names),'current_qualified_stage2':True,'core_methods_passed':11,'companion_methods_passed':3,'affine_reports_replayed':reports_checked,'original_frames_per_consumer':1001,'additional_cases':protocols,'original_fourteen_method_observer_passed':False,'preserved_missing_assembler_errors':29,'inter_run_source_tool_existing_image_guards_match':True,'whole_release':False}))
