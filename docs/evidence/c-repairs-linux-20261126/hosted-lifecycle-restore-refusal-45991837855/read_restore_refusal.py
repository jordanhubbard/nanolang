"""I check selected hosted refusal bytes and the tested namespace correction."""
import argparse,ast,hashlib,importlib.util,json,shutil,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='41a175aae3984717bd034dd5cc5d526190bef1e9d49498ad290b99e830a64eed'
m=json.loads(raw);assert m['version']==1 and not m['qualification'] and not m['whole_release'] and not m['whole_wrapper_zip_reconstructed']
assert m['run']==45991837855 and m['artifact']==15195792561
assert m['source_revision']=='5ca56494a9dc8eb84c66da61402343a2a34994f2' and m['fix_revision']=='51c82debb858145371cd9f28005d683e0e79184f'
seen=set()
for row in m['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen;seen.add(str(rel))
 data=(a.archive/rel).read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256']
assert len(seen)==43
run=json.loads((a.archive/'api-run.json').read_text());artifact=json.loads((a.archive/'api-artifact.json').read_text());assert run['id']==m['run'] and run['head_sha']==m['source_revision'] and run['conclusion']=='failure'
assert artifact['id']==m['artifact'] and artifact['digest']=='sha256:'+m['wrapper_zip_sha256']
f=a.archive/'failure';q=json.loads((f/'result.json').read_text());assert q['terminal'] and q['compiler_exit_code']==1 and not q['flush_observed']
assert q['command_timeout_seconds']==300 and q['shadow_seconds']==10 and not q['qualification'] and not q['whole_release']
assert 'shadow_pid' not in q and 'observer_pid' not in q
error=(f/'compiler.stderr').read_bytes();assert b'I could not publish complete C-library artifacts for std' in error and b'cseed-shadow-phase' not in error
assert json.loads((f/'identities-before.json').read_text())==json.loads((f/'identities-after.json').read_text())
assert json.loads((f/'host-libraries-before.json').read_text())==json.loads((f/'host-libraries-after.json').read_text())
old_sources=json.loads((f/'observer-sources.json').read_text())
for name,h in old_sources.items():
 data=(f/'observer-sources'/Path(name).name).read_bytes();assert sha(data)==h
 assert data==subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=a.repo)
fix=a.archive/'fix';sources=fix/'sources'
new_names=['scripts/ci_darwin_shadow_lifecycle.c','scripts/ci_darwin_shadow_lifecycle.py','.github/workflows/cseed-shadow-lifecycle.yml','tests/test_cseed_shadow_lifecycle_restore.py']
for name in new_names:assert (sources/name).read_bytes()==subprocess.check_output(['git','show',m['fix_revision']+':'+name],cwd=a.repo)
assert (sources/new_names[0]).read_bytes()==(f/'observer-sources/ci_darwin_shadow_lifecycle.c').read_bytes()
functions=lambda text:{v.name:ast.dump(v,include_attributes=False) for v in ast.parse(text).body if isinstance(v,ast.FunctionDef)}
old=functions((f/'observer-sources/ci_darwin_shadow_lifecycle.py').read_text());new=functions((sources/new_names[1]).read_text())
protected=[name for name in old if name not in ('main','product_tree')]
for name in protected:assert old[name]==new[name],name
allowed=set(new_names)
def tree(revision):
 result={}
 for entry in subprocess.check_output(['git','ls-tree','-rz',revision],cwd=a.repo).split(b'\0'):
  if entry:
   metadata,name=entry.split(b'\t',1);name=name.decode()
   if not name.startswith('docs/') and name not in allowed:result[name]=metadata
 return result
assert tree('6e09c07a9f1651294e4669282cdc17ee14126775')==tree(m['fix_revision'])
analysis=json.loads((a.archive/'flattened-current-analysis.json').read_text());replay=json.loads((fix/'actual-archive-replay.json').read_text())
assert analysis['zip_sha256']==replay['original_zip_sha256']==m['original_zip_sha256']
assert not replay['product_execution'] and not replay['original_failed_syscall_recorded'] and not replay['qualification']
assert replay['flattened_pointer_publication_errno']==21 and replay['reconstructed_pointer_publication_succeeded']
assert replay['restored_currents']==9 and replay['current_file_entries']==812 and replay['original_file_lookup_hashes_verified']==1846 and replay['receipt_hosts_unchanged']==6
assert replay['observer_source_sha256']==sha((sources/new_names[1]).read_bytes())
rows=replay['namespace_report']['entries'];assert replay['namespace_report']['complete'] and not replay['namespace_report']['original_link_metadata_available']
assert len(rows)==len(analysis['currents'])==9
for observed,row in zip(analysis['currents'],rows):
 assert observed['current']=='nanolang/nanolang/'+row['current']
 assert observed['files']==len(row['files']) and observed['matching_generations']==['nanolang/nanolang/'+str(Path(row['current']).parent/row['target'])]
 assert row['restored'] and row['basis']=='unique-complete-byte-equal-generation'
assert 'Ran 12 tests' in (fix/'unittest.stderr').read_text() and (fix/'unittest.stderr').read_text().rstrip().endswith('OK')
with tempfile.TemporaryDirectory(prefix='hosted-control-reader-',dir=a.workspace) as temporary:
 root=Path(temporary)
 for name in ('control-success.jsonl','control-late.jsonl','control-blocked.jsonl','controls.json'):shutil.copyfile(f/name,root/name)
 source=f/'observer-sources/ci_darwin_shadow_lifecycle.py';spec=importlib.util.spec_from_file_location('hosted_control_predicate_reader',source);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 expected=(root/'controls.json').read_bytes();calls=[]
 def captured(argv,out,label,*,timeout=300,**kwargs):
  kind=('success','late','blocked')[len(calls)];assert argv==[str(root/'observer'),'--control',kind,str(root/('control-'+kind+'.jsonl'))]
  assert out==root and label=='control-'+kind and timeout==5 and not kwargs;calls.append(kind)
 with patch.object(helper,'captured',captured),patch.object(subprocess,'Popen',side_effect=AssertionError('I forbid product execution in my reader.')):helper.controls(root/'observer',root)
 assert calls==['success','late','blocked'] and (root/'controls.json').read_bytes()==expected
print(json.dumps({'verified_selected_bytes_and_sources':True,'manifest_sha256':sha(raw),'files_verified':len(seen),'original_native_controls_replayed':calls,'compiler_exit_before_shadows':1,'restored_guard_snapshots_match':True,'unchanged_original_python_functions':protected,'unchanged_product_git_tree':True,'namespace_records_verified':9,'namespace_entries':812,'actual_archive_kernel_replay_journal_verified':True,'restorer_methods_passed':12,'whole_wrapper_zip_reconstructed':False,'original_failed_syscall_established':False,'product_execution':False,'hosted_original_emitter_qualification':False,'whole_release':False}))
