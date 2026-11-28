"""I check retained representation-to-shape adapter inputs/reports without executing a product."""
import argparse,gzip,hashlib,importlib,json,re,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();a.repo=a.repo.resolve()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='e3f330020eac23b15cec126d9d3eef0ca0e2195d35b4e58f388b7280b85fb85f'
m=json.loads(raw);assert m['version']==1 and not m['whole_release']
assert m['source_base']=='8a6a7a3add80ff1f9a474913980b895ef15ef8fb'
prerequisite=a.repo/'docs/evidence/c-repairs-linux-20261126/native-current-linux-20261127/archive/manifest.json'
prerequisite_bytes=prerequisite.read_bytes() if prerequisite.is_file() else subprocess.check_output(['git','show','HEAD:'+str(prerequisite.relative_to(a.repo))],cwd=a.repo)
assert sha(prerequisite_bytes)==m['prerequisite_manifest_sha256']=='4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f'
with tempfile.TemporaryDirectory(prefix='representation-shape-reader-',dir=a.workspace) as temporary:
 root=Path(temporary);seen=set();used=set()
 for row in m['files']:
  relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
  assert str(relative) not in seen;seen.add(str(relative));dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256();size=0
  with dest.open('wb') as f:
   for key in row['chunks']:
    assert re.fullmatch('[a-f0-9]{64}',key)
    data=(a.archive/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key]
    assert sha(data)==meta['gzip_sha256'];block=gzip.decompress(data)
    assert len(block)==meta['bytes'] and sha(block)==key
    f.write(block);h.update(block);size+=len(block);used.add(key)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 assert used==set(m['chunks'])
 for name,want in [('build',0),('first-compile',1),('second',1),('third',0),('fourth',0)]:
  prototype=json.loads((root/'prototype'/(name+'.json')).read_text())
  assert prototype['exit']==want
 # I retain both complete failed setup attempts and their exact source/stream guards.
 for label in ['first','second']:
  q=json.loads((root/label/'result.json').read_text())
  assert q['terminal'] and q['guards_match'] and not q['functional_gate_passed']
  assert (q['tests_run'],q['failures'],q['errors'])==(0,0,1)
  assert q['source_before']==q['source_after'] and q['tools_before']==q['tools_after']
  assert [c['exit'] for c in q['commands']]==[0,0,0,0,1]
  for name in q['new_sources']+['Makefile.gnu']:
   assert sha((root/label/'failed-source'/name).read_bytes())==q['source_before'][name]
  for index,c in enumerate(q['commands']):
   for stream in ['stdout','stderr']:
    assert sha((root/label/'streams'/f'{index:04d}.{stream}').read_bytes())==c[stream]['sha256']
 passed=json.loads((root/'passed/result.json').read_text())
 assert passed['terminal'] and passed['guards_match'] and passed['functional_gate_passed']
 assert (passed['tests_run'],passed['failures'],passed['errors'])==(8,0,0)
 assert len(passed['commands'])==472
 native=json.loads((root/'passed/qualified-native-result.json').read_text());receipt=json.loads((root/'passed/qualified-bootstrap-receipt.json').read_text())
 assert native['terminal'] and native['component_native_gate_passed'] and native['guards_match']
 assert native['head_after']==m['product_c_source'] and len(receipt['hosts'])==6
 assert native['generated_index_matches_receipt'] and native['host_closure_matches_receipt']
 for label,q in [('passed',passed)]:
  assert q['source_before']==q['source_after'] and q['tools_before']==q['tools_after']
  assert q['qualified_sources_after']==native['sources_after'] and q['qualified_tools_after']==native['tools_after']
  assert q['source_head']==m['source_base'] and not q['whole_release']
  for index,c in enumerate(q['commands']):
   for stream in ('stdout','stderr'):
    assert Path(c[stream]['path']).name==f'{index:04d}.{stream}'
    assert sha((root/label/'streams'/f'{index:04d}.{stream}').read_bytes())==c[stream]['sha256']
   assert c['effective_timeout']==(600 if c['build'] else c['timeout'])
   assert c['exit']==0
  for name in q['new_sources']+['Makefile.gnu']:
   path=root/'source'/name
   assert sha(path.read_bytes())==q['source_before'][name]
 assert passed['tools_before'][next(k for k in passed['tools_before'] if k.endswith('/nanoc_stage2'))]=='cc94609575ad0d626e696449f486ed04336dff6e0887113542b09bce222a4820'
 for name in ['nanoc_c','nano_virt','nano_vm','nvm2c','nano_aot_runtime.o']:
  key=next(k for k in passed['tools_before'] if '/representation-shapes-20261127/bin/' in k and k.endswith('/'+name))
  assert sha((root/'tools'/name).read_bytes())==passed['tools_before'][key]
 new=set(passed['new_sources']+['Makefile.gnu'])
 # I resolve all unchanged source bytes in one bounded Git batch, not from the live build tree.
 expected={n:h for n,h in passed['source_before'].items() if n not in new}
 tree=subprocess.check_output(['git','ls-tree','-rz',m['source_base']],cwd=a.repo)
 blobs={}
 for entry in tree.split(b'\0'):
  if entry:
   meta,name=entry.split(b'\t',1);mode,kind,oid=meta.split()
   if name.decode() in expected:assert kind==b'blob';blobs[name.decode()]=oid.decode()
 assert set(blobs)==set(expected)
 requests=''.join(blobs[n]+'\n' for n in expected).encode();batch=root/'git-objects'
 with batch.open('wb') as f:subprocess.run(['git','cat-file','--batch'],input=requests,stdout=f,cwd=a.repo,check=True,timeout=120)
 c_source=None
 with batch.open('rb') as f:
  for name,h in expected.items():
   oid,kind,size=f.readline().split();assert oid.decode()==blobs[name] and kind==b'blob'
   data=f.read(int(size));assert f.read(1)==b'\n' and sha(data)==h,name
   if name=='src/nanoisa/nvm2c.c':c_source=data
  assert f.read()==b''
 for name in new:assert (a.repo/name).read_bytes()==(root/'source'/name).read_bytes()
 reference=json.loads((root/'passed-products/reference-source.json').read_text())
 assert reference['functions']==16 and reference['logical_adapters']==12 and reference['minimal_owner_not_binary_layout'] and not reference['product_source_edited']
 assert sha(c_source)==reference['source_sha256']
 markers=[v.encode() for v in reference['markers']]
 assert len(markers)==18 and all(c_source.count(v)==1 for v in set(markers))
 positions=[c_source.index(v) for v in markers]
 actual=b''.join(c_source[x:y] for x,y in zip(positions[::2],positions[1::2]))
 assert actual.count(b'static ')==16 and sha(actual)==reference['verbatim_helper_sha256']
 assert actual==(root/'passed-products/nvm2c_representation_shapes_actual.h').read_bytes()
 policy=json.loads((root/'passed/policy.json').read_text())
 assert [v['name'] for v in policy]==['shadow-policy','identifiers','diff']
 assert all(v['exit']==0 for v in policy)
 assert 'Ran 5 tests' in (root/'passed/identifiers.stderr').read_text()
 # I reuse the original independent oracles and packet builders with process execution forbidden.
 sys.path.insert(0,str(a.repo));test=importlib.import_module('tests.test_nanolang_translator_representation_shapes')
 case=test.RepresentationShapes('test_complete_kind_and_inferred_field_domains');reports=0;packets=0
 consumed_matrix=[]
 source_tree=Path(next(k for k in passed['tools_before'] if '/representation-shapes-20261127/bin/' in k and k.endswith('/nanoc_c'))).parents[1]
 original_products=Path(passed['temporary_roots'][0])
 commands={tuple(c['argv']):(index,c) for index,c in enumerate(passed['commands'])}
 assert len(commands)==len(passed['commands'])
 route_commands={
  'ordinary':[original_products/'reference-ordinary'], 'sanitized':[original_products/'reference-sanitized'],
  'c-seed-native':[original_products/'decoder-native'],
  'c-producer-nanoisa-vm':[source_tree/'bin/nano_vm',original_products/'decoder.nvm','--'],
  'c-producer-sanitized-native':[original_products/'decoder-aot'],
  'nanolang-producer-nanoisa-vm':[source_tree/'bin/nano_vm',original_products/'source-decoder.nvm','--'],
  'nanolang-producer-sanitized-native':[original_products/'source-decoder-aot']}
 def compare(rows,label):
  global reports,packets
  manifest=json.loads((root/'passed-products'/(label+'-manifest.json')).read_text())
  assert not manifest['qualification'] and len(rows)==len(manifest['cases'])
  for (name,data),record in zip(rows,manifest['cases']):
   assert name==record['name'] and sha(data)==record['sha256']
   assert data==(root/'passed-products'/Path(record['path']).name).read_bytes();packets+=1
  baseline=None;names=[]
  for route in manifest['routes']:
   data=(root/'passed-products'/(label+'-'+route['name']+'.reports')).read_bytes()
   assert sha(data)==route['sha256'] and len(data.splitlines())==route['lines']
   observed=[]
   for start in range(0,len(rows),16):
    argv=tuple(map(str,route_commands[route['name']]))+tuple(v['path'] for v in manifest['cases'][start:start+16])
    index,command=commands[argv]
    assert command['exit']==0 and not command['build'] and command['timeout']==command['effective_timeout']==60
    observed.append((root/'passed/streams'/f'{index:04d}.stdout').read_bytes());consumed_matrix.append(index)
   assert b''.join(observed)==data
   if baseline is None:baseline=data
   assert data==baseline;reports+=len(data.splitlines());names.append(route['name'])
  assert names==list(route_commands)
  return baseline.decode()
 case.compare=compare
 source_tree=Path(next(k for k in passed['tools_before'] if '/representation-shapes-20261127/bin/' in k and k.endswith('/nanoc_c'))).parents[1]
 producer=next(k for k in passed['tools_before'] if k.endswith('/nanoc_stage2'))
 case.directory=Path(passed['temporary_roots'][0]);case.fixture=source_tree/'tests/nvm2c_representation_shapes_driver.nano'
 shadow_commands=[(i,c) for i,c in enumerate(passed['commands']) if any('/imported-' in v for v in c['argv'])]
 assert len(shadow_commands)==8
 consumed=[]
 def retained_checked(command,**kwargs):
  index,c=shadow_commands[len(consumed)]
  assert list(map(str,command))==c['argv']
  assert kwargs.get('build',False)==c['build'] and kwargs.get('timeout',60)==c['timeout']
  assert c['exit']==0;consumed.append(index)
  return (root/'passed/streams'/f'{index:04d}.stdout').read_bytes().decode()
 with patch.object(subprocess,'Popen',side_effect=AssertionError('I forbid product execution in my reader.')):
  case.test_complete_kind_and_inferred_field_domains();case.test_complete_carrier_mask_and_extraction_domains();case.test_every_single_record_field_pair();case.test_context_poison_disabled_and_source_diagnostics();case.test_map_payload_and_record_sequence_boundaries();case.test_recursive_shared_graph_and_conversion_sequences();case.test_malformed_and_prior_packet_boundaries()
  with patch.object(test,'ROOT',source_tree),patch.object(test,'checked',retained_checked),patch.object(test,'source_compiler_command',lambda:[producer]):
   case.test_imported_shadows()
 assert len(consumed)==8
 assert packets==967 and reports==25741198
 assert len(set(consumed_matrix))==len(consumed_matrix)
 assert set(consumed_matrix+[i for i in consumed if not passed['commands'][i]['build']])=={i for i,c in enumerate(passed['commands']) if not c['build']}
 print(json.dumps({'verified':True,'manifest_sha256':sha(raw),'records':len(seen),'chunks':len(used),
  'unchanged_git_sources':len(expected),'actual_c_helpers':16,'logical_adapters':12,'routes':7,
  'packet_inputs':packets,'report_lines_replayed':reports,'matrix_commands_bound':len(consumed_matrix),
  'imported_shadow_commands_replayed':len(consumed),'prototype_refusals_retained':True,
  'methods_passed':8,'commands_passed':len(passed['commands']),
  'source_tool_qualified_producer_guards':True,'shadow_identifier_policy':True,
  'native_bootstrap_dependency_bound':True,'product_execution':False,'whole_release':False}))
