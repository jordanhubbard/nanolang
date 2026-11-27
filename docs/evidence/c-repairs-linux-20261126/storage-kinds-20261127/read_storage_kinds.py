"""I check retained storage-helper inputs/reports without executing a product."""
import argparse,gzip,hashlib,importlib,json,re,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();a.repo=a.repo.resolve()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='b72ea956ff36450ee6ff20a0549775007e40a378329c6d34850a79c56ea46096'
m=json.loads(raw);assert m['version']==1 and not m['whole_release']
assert m['source_base']=='779b26abe7487c51c08f273ddabc1992536a0997'
prerequisite=a.repo/'docs/evidence/c-repairs-linux-20261126/native-current-linux-20261127/archive/manifest.json'
prerequisite_bytes=prerequisite.read_bytes() if prerequisite.is_file() else subprocess.check_output(['git','show','HEAD:'+str(prerequisite.relative_to(a.repo))],cwd=a.repo)
assert sha(prerequisite_bytes)==m['prerequisite_manifest_sha256']=='4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f'
with tempfile.TemporaryDirectory(prefix='storage-kind-reader-',dir=a.workspace) as temporary:
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
 first=json.loads((root/'first/result.json').read_text());passed=json.loads((root/'passed/result.json').read_text())
 assert first['terminal'] and first['guards_match'] and not first['functional_gate_passed']
 assert first['tests_run']==0 and first['errors']==1 and len(first['commands'])==5
 assert first['commands'][-1]['exit']==1
 assert 'cannot convert aggregate storage numeric to int' in (root/'first/streams/0004.stderr').read_text()
 assert passed['terminal'] and passed['guards_match'] and passed['functional_gate_passed']
 assert (passed['tests_run'],passed['failures'],passed['errors'])==(3,0,0)
 assert len(passed['commands'])==38
 native=json.loads((root/'passed/qualified-native-result.json').read_text());receipt=json.loads((root/'passed/qualified-bootstrap-receipt.json').read_text())
 assert native['terminal'] and native['component_native_gate_passed'] and native['guards_match']
 assert native['head_after']==m['product_c_source'] and len(receipt['hosts'])==6
 assert native['generated_index_matches_receipt'] and native['host_closure_matches_receipt']
 for label,q in [('first',first),('passed',passed)]:
  assert q['source_before']==q['source_after'] and q['tools_before']==q['tools_after']
  assert q['qualified_sources_after']==native['sources_after'] and q['qualified_tools_after']==native['tools_after']
  assert q['source_head']==m['source_base'] and not q['whole_release']
  for index,c in enumerate(q['commands']):
   for stream in ('stdout','stderr'):
    assert Path(c[stream]['path']).name==f'{index:04d}.{stream}'
    assert sha((root/label/'streams'/f'{index:04d}.{stream}').read_bytes())==c[stream]['sha256']
   assert c['effective_timeout']==(600 if c['build'] else c['timeout'])
   assert c['exit']==(1 if label=='first' and index==4 else 0)
  for name in q['new_sources']+['Makefile.gnu']:
   path=root/('first/first-source' if label=='first' else 'source')/name
   assert sha(path.read_bytes())==q['source_before'][name]
 assert passed['tools_before'][next(k for k in passed['tools_before'] if k.endswith('/nanoc_stage2'))]=='cc94609575ad0d626e696449f486ed04336dff6e0887113542b09bce222a4820'
 for name in ['nanoc_c','nano_virt','nano_vm','nvm2c','nano_aot_runtime.o']:
  key=next(k for k in passed['tools_before'] if '/storage-kinds-20261127/bin/' in k and k.endswith('/'+name))
  assert sha((root/'tools'/name).read_bytes())==passed['tools_before'][key]
 new=set(passed['new_sources']+['Makefile.gnu'])
 # I resolve all unchanged source bytes in one bounded Git batch, not from the live build tree.
 expected={n:h for n,h in passed['source_before'].items() if n not in new}
 assert expected=={n:h for n,h in first['source_before'].items() if n not in new}
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
 begin=b'#define NVM2C_VK_INT 0';end=b'typedef struct Nvm2cFieldBlock'
 actual=c_source[c_source.index(begin):c_source.index(end)]
 assert actual.count(b'static ')==11
 assert actual==(root/'passed-products/nvm2c_storage_actual.h').read_bytes()
 # I reuse the original independent oracles and packet builders with process execution forbidden.
 sys.path.insert(0,str(a.repo));test=importlib.import_module('tests.test_nanolang_translator_storage_kinds')
 case=test.StorageKinds('test_complete_physical_domains');reports=0;packets=0
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
   if baseline is None:baseline=data
   assert data==baseline;reports+=len(data.splitlines());names.append(route['name'])
  assert names==['ordinary','sanitized','c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
  return baseline.decode()
 case.compare=compare
 with patch.object(subprocess,'Popen',side_effect=AssertionError('I forbid product execution in my reader.')):
  case.test_complete_physical_domains();case.test_malformed_and_prior_packet_boundaries()
 assert packets==58 and reports==921508
 print(json.dumps({'verified':True,'manifest_sha256':sha(raw),'records':len(seen),'chunks':len(used),'unchanged_git_sources':len(expected),'actual_c_helpers':11,'physical_inputs':131584,'routes':7,'packet_inputs':packets,'report_lines_replayed':reports,'original_failure_retained':True,'methods_passed':3,'commands_passed':38,'source_tool_qualified_producer_guards':True,'native_bootstrap_dependency_bound':True,'product_execution':False,'whole_release':False}))
