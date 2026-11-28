"""I replay a lost cache-pointer namespace without executing any product."""
import argparse,gzip,re,errno,hashlib,importlib.util,io,json,os,subprocess,tempfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True);group.add_argument('--diagnostic-zip',type=Path);group.add_argument('--original-zip',type=Path);group.add_argument('--native-evidence-archive',type=Path);p.add_argument('--observer-source',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
if a.diagnostic_zip:
 raw=a.diagnostic_zip.read_bytes();assert sha(raw)=='523bf5391553a705aed8d3935693af6bbb4124cef892fbc896d81400b228e3fe'
 with zipfile.ZipFile(io.BytesIO(raw)) as outer:original=outer.read('original-artifact.zip')
elif a.original_zip:
 original=a.original_zip.read_bytes()
else:
 encoded=(a.native_evidence_archive/'manifest.json').read_bytes()
 assert sha(encoded)=='6b984b9bf5236a4c3c87cfee8ae6240d3c1aeb30249e87546e177cfb9407ab58'
 manifest=json.loads(encoded);rows=[r for r in manifest['files'] if r['sha256']=='77326ff42549c8f0d2ed7c757a3a58611668f592348ca756e715268cb7480de9'];assert len(rows)==1
 blocks=[]
 for key in rows[0]['chunks']:
  assert re.fullmatch('[0-9a-f]{64}',key)
  packed=(a.native_evidence_archive/'objects'/(key+'.gz')).read_bytes();meta=manifest['chunks'][key]
  assert sha(packed)==meta['gzip_sha256'];block=gzip.decompress(packed)
  assert len(block)==meta['bytes'] and sha(block)==key;blocks.append(block)
 original=b''.join(blocks);assert len(original)==rows[0]['size']
assert sha(original)=='77326ff42549c8f0d2ed7c757a3a58611668f592348ca756e715268cb7480de9'
spec=importlib.util.spec_from_file_location('namespace_replay_observer',a.observer_source);observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
report={'original_zip_sha256':sha(original),'observer_source_sha256':sha(a.observer_source.read_bytes()),'product_execution':False,'original_failed_syscall_recorded':False,'qualification':False,'whole_release':False}
with tempfile.TemporaryDirectory(prefix='actual-cache-namespace-',dir=a.workspace) as temporary:
 root=Path(temporary);before={};prefix=Path('nanolang/nanolang')
 with zipfile.ZipFile(io.BytesIO(original)) as z:
  receipt=json.loads(z.read(str(prefix/'bin/nanoc_bootstrap.json')))
  for entry in z.infolist():
   name=Path(entry.filename);assert not name.is_absolute() and '..' not in name.parts
   assert ((entry.external_attr>>16)&0o170000)!=0o120000
   if not name.is_relative_to(prefix) or entry.is_dir():continue
   relative=name.relative_to(prefix)
   if not (relative.is_relative_to('modules') or relative.is_relative_to('obj/module_cache')):continue
   data=z.read(entry);dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);before[str(relative)]=sha(data)
 cache=root/'modules/std/.build';current=cache/'current';probe=cache/'.namespace-publication-probe'
 target='.nano-gen-IkMv3X';probe.symlink_to(target,target_is_directory=True)
 try:os.replace(probe,current)
 except OSError as error:assert error.errno in (errno.EISDIR,errno.ENOTDIR);report['flattened_pointer_publication_errno']=error.errno
 else:raise AssertionError('I unexpectedly replaced a flattened directory with a symlink.')
 namespaces=observer.restore_cache_currents(root,root/'.test_output/observer')
 assert namespaces['complete'] and len(namespaces['entries'])==9
 assert sum(len(v['files']) for v in namespaces['entries'])==812
 assert all(v['restored'] and v['basis']=='unique-complete-byte-equal-generation' for v in namespaces['entries'])
 os.replace(probe,current);assert os.readlink(current)==target
 for name,h in before.items():assert sha((root/name).read_bytes())==h,name
 for name,h in receipt['hosts'].items():
  relative=Path(name).relative_to(receipt['root']);assert sha((root/relative).read_bytes())==h
 report.update(restored_currents=9,current_file_entries=812,original_file_lookup_hashes_verified=len(before),receipt_hosts_unchanged=len(receipt['hosts']),reconstructed_pointer_publication_succeeded=True,namespace_report=namespaces)
(a.output/'actual-archive-replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='namespace_report'}))
