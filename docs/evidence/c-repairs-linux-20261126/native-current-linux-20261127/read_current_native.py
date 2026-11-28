"""I verify one current Linux native component gate without accepting a release."""
import argparse,gzip,hashlib,json,re,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();manifest=json.loads(raw)
assert manifest['revision']=='6e09c07a9f1651294e4669282cdc17ee14126775' and manifest['whole_release'] is False
# I pin this manifest below before publishing the reader.
PIN='4c3fef1f342e7a6b2d0fe912100f9b1d299455d4db3db2d89783ab2b6c00dd8f'
assert sha(raw)==PIN
with tempfile.TemporaryDirectory(prefix='current-native-reader-',dir=a.workspace) as directory:
 root=Path(directory);seen=set();used=set();links={}
 for r in manifest['files']:
  rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts and r['path'] not in seen;seen.add(r['path'])
  if r['kind']=='symlink':links[r['path']]=r['target'];continue
  assert r['kind']=='file';f=root/rel;f.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with f.open('wb') as out:
   for key in r['chunks']:
    packed=(a.archive/'objects'/(key+'.gz')).read_bytes();meta=manifest['chunks'][key];assert sha(packed)==meta['gzip_sha256']
    block=gzip.decompress(packed);assert sha(block)==key and len(block)==meta['bytes'];out.write(block);h.update(block);size+=len(block);used.add(key)
  assert h.hexdigest()==r['sha256'] and size==r['size'],r['path']
 assert used==set(manifest['chunks'])
 load=lambda f:json.loads(f.read_text())
 q=load(root/'.test_output/frozen-native-qualification/result.json')
 assert q['terminal'] and q['component_native_gate_passed'] and q['guards_match']
 assert q['head_before']==q['head_after']==manifest['revision']
 assert q['sources_before']==q['sources_after'] and q['tools_before']==q['tools_after']
 assert q['generated_index_matches_receipt'] and q['host_closure_matches_receipt'] and q['whole_release'] is False
 assert len(q['steps'])==3 and [s['exit_code'] for s in q['steps']]==[0,0,0]
 assert [s['label'] for s in q['steps']]==['bootstrap','native','emitter']
 assert q['steps'][1]['reported_tests']==36 and q['steps'][2]['reported_tests']==12
 original=Path(manifest['original_root'])
 relative=lambda f:Path(f).relative_to(original)
 for name,value in q['tools_after'].items():
  f=root/relative(name) if Path(name).is_relative_to(original) else root/manifest['external_tools'][name]
  assert sha(f.read_bytes())==value,name
 assert sha((root/'observer.py').read_bytes())==q['observer_sha256']
 names=[str(relative(n)) for n in q['sources_after']]
 request=''.join(manifest['revision']+':'+n+'\n' for n in names).encode()
 data=subprocess.check_output(['git','-C',str(a.repo),'cat-file','--batch'],input=request)
 blobs={};offset=0
 for name in names:
  end=data.index(b'\n',offset);header=data[offset:end].split();assert len(header)==3,(name,header)
  size=int(header[-1]);blob=data[end+1:end+1+size];offset=end+size+2
  assert sha(blob)==q['sources_after'][str(original/name)],name;blobs[name]=blob
 assert offset==len(data)
 bpath=root/'bin/nanoc_bootstrap.json';b=load(bpath)
 assert sha(bpath.read_bytes())==q['bootstrap_receipt_sha256']
 assert b['phase']=='stage2' and len(b['steps'])==17 and all(s['exit_code']==0 for s in b['steps'])
 assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30 and b['raw_stage1_stage2_equal']
 for r in b['steps']:assert sha((root/relative(r['log'])).read_bytes())==r['log_sha256'],r['label']
 for name,value in b['artifacts'].items():assert sha((root/'bin'/name).read_bytes())==value,name
 for name,value in b['tools'].items():assert sha((root/'bin'/name).read_bytes())==value,name
 for name,value in b['hosts'].items():assert sha((root/relative(name)).read_bytes())==q['hosts_after'][name]==value,name
 index=(root/'modules/index.json').read_bytes();assert sha(index)==q['generated_index_sha256']==b['sources']['modules/index.json']
 for name,value in b['sources'].items():assert sha(index if name=='modules/index.json' else blobs[name])==value,name
 assert links['bin/nanoc']=='nanoc_stage2'
 first=(root/'bin/nanoc_stage1.nvm').read_bytes();second=(root/'bin/nanoc_stage2.nvm').read_bytes()
 assert first==second and len(first)==1064904 and sha(first)=='8920ee0546671f9fa13ea98f7aaf5030b235d6d7349b277cdb875b4a5aa46833'
 logdir=root/'.test_output/frozen-native-qualification'
 assert 'bin/nanoc succeeded with bin/nanoc_c removed' in (logdir/'bootstrap.stdout').read_text()
 for name,count in [('native',36),('emitter',12)]:
  log=(logdir/(name+'.stdout')).read_text()+(logdir/(name+'.stderr')).read_text()
  assert re.search(r'Ran '+str(count)+r' tests in [^\n]+\n\nOK(?:\n|$)',log)
  assert 'FAILED (' not in log
 print(json.dumps({'archive_verified':True,'revision':manifest['revision'],'records':len(seen),'chunks':len(used),'source_files_verified':len(names),'bootstrap_steps_passed':17,'native_methods_passed':36,'emitter_methods_passed':12,'raw_module_bytes':len(first),'raw_module_sha256':sha(first),'selected_guards_match':True,'generated_index_verified':True,'host_libraries_verified':len(b['hosts']),'linux_native_component_gate_passed':True,'paired_current_platforms_accepted':False,'broad_CAST_STRING_decision_pending':True,'whole_release':False,'current_liveness_checked':False}))
