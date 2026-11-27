"""I verify one bounded native control receipt without executing a product."""
import argparse,hashlib,importlib.util,json,shutil,struct,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='a5b89954d8f58634a217bdd4b10223d5196291eeb493548603ae5263d24fd5da'
m=json.loads(raw);assert m['version']==1 and not m['qualification'] and not m['whole_release']
assert m['source_revision']=='5ca56494a9dc8eb84c66da61402343a2a34994f2'
seen=set()
for row in m['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts and str(relative) not in seen
 seen.add(str(relative));data=(a.archive/relative).read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256']
assert len(seen)==24
f=a.archive/'files';result=json.loads((f/'result.json').read_text());inputs=json.loads((a.archive/'input-identities.json').read_text());transfer=json.loads((a.archive/'transfer.json').read_text())
assert result['terminal'] and result['controls_passed'] and result['source_guards_match']
assert not result['compiler_product_invoked'] and not result['qualification'] and not result['whole_release']
assert result['source_revision']==inputs['revision']==transfer['source_revision']==m['source_revision']
assert result['source_before']==result['source_after']==inputs['sources'] and len(inputs['sources'])==3
for name,h in inputs['sources'].items():
 data=(f/Path(name).name).read_bytes();assert sha(data)==h
 assert data==subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=a.repo)
assert transfer['exit']==0 and transfer['remote_root']==result['root']
assert set(transfer['files'])=={str(Path(n).relative_to('files')) for n in seen if n.startswith('files/')}
for name,row in transfer['files'].items():
 data=(f/name).read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256']
assert (result['build_timeout_seconds'],result['control_timeout_seconds'])==(120,5)
assert result['build']['argv']==['/usr/bin/xcrun','clang','-std=c11','-Wall','-Wextra','-Werror','-O2',result['root']+'/ci_darwin_shadow_lifecycle.c','-lproc','-o',result['root']+'/lifecycle-observer']
assert result['build']['source_sha256']==inputs['sources']['scripts/ci_darwin_shadow_lifecycle.c']
binary=(f/'lifecycle-observer').read_bytes();assert sha(binary)==result['build']['sha256']
assert struct.unpack_from('<II',binary)==(0xfeedfacf,0x0100000c)
assert result['services']['mac_control_plane_state']=='running' and result['services']['mac_control_plane_pid']>0
assert result['services']['webmux']['status']=='ok'
identity=json.loads((f/'identity.json').read_text());assert identity['uname'][0]=='Darwin' and identity['uname'][4]=='arm64'
assert 'arm64-apple-darwin27.2.0' in identity['clang_version']
cleanup=json.loads((a.archive/'cleanup.json').read_text());assert cleanup['removed_owned_temporary_root']==result['root'] and not cleanup['exists_after']
with tempfile.TemporaryDirectory(prefix='puck-controls-reader-',dir=a.workspace) as temporary:
 root=Path(temporary)
 for path in f.rglob('*'):
  if path.is_file():
   dest=root/path.relative_to(f);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 spec=importlib.util.spec_from_file_location('retained_lifecycle_control_reader',root/'ci_darwin_shadow_lifecycle.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 expected=(root/'controls.json').read_bytes();commands=[]
 def captured(argv,out,label,*,timeout=300,**kwargs):
  kind=('success','late','blocked')[len(commands)]
  assert argv==[str(root/'lifecycle-observer'),'--control',kind,str(root/('control-'+kind+'.jsonl'))]
  assert out==root and label=='control-'+kind and timeout==5 and not kwargs
  commands.append(kind)
 with patch.object(helper,'captured',captured),patch.object(subprocess,'Popen',side_effect=AssertionError('I forbid product execution in my reader.')):
  helper.controls(root/'lifecycle-observer',root)
 assert commands==['success','late','blocked'] and (root/'controls.json').read_bytes()==expected
 summaries=json.loads(expected)
print(json.dumps({'verified':True,'manifest_sha256':sha(raw),'files_verified':len(seen),'exact_source_revision':m['source_revision'],'native_arm64_observer_verified':True,'original_control_predicates_replayed':commands,'controls':summaries,'temporary_root_removed_after_transfer':True,'product_execution':False,'original_compiler_observation':False,'hosted_qualification':False,'whole_release':False}))
