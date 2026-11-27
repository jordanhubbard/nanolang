from pathlib import Path
import base64,hashlib,json,subprocess,shlex,time
root=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp/c-repairs-followups-20261126/puck-lifecycle-controls-20261127');root.mkdir()
revision='5ca56494a9dc8eb84c66da61402343a2a34994f2'
inputs={}
for name in ['scripts/ci_darwin_shadow_lifecycle.c','scripts/ci_darwin_shadow_lifecycle.py','.github/workflows/cseed-shadow-lifecycle.yml']:
 data=subprocess.check_output(['git','show',revision+':'+name]);inputs[name]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()}
remote=r'''
from pathlib import Path
import base64,hashlib,importlib.util,json,os,re,subprocess,sys,tempfile,time,traceback
payload=json.load(sys.stdin);root=Path(tempfile.mkdtemp(prefix='nanolang-lifecycle-controls-'));started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'terminal':False,'controls_passed':False,'qualification':False,'whole_release':False,'source_revision':payload['revision'],'root':str(root),'build_timeout_seconds':120,'control_timeout_seconds':5,'compiler_product_invoked':False}
try:
 for name,row in payload['sources'].items():
  data=base64.b64decode(row['base64']);assert hashlib.sha256(data).hexdigest()==row['sha256'];(root/Path(name).name).write_bytes(data)
 result['source_before']={name:sha(root/Path(name).name) for name in payload['sources']}
 status=subprocess.check_output(['/bin/launchctl','print','system/com.mac.control-plane'],text=True,timeout=10)
 assert re.search(r'^\s*state = running$',status,re.M)
 pids=re.findall(r'^\s*pid = (\d+)$',status,re.M);assert len(pids)==1
 health=json.loads(subprocess.check_output(['/usr/bin/curl','-fsS','--max-time','5','http://127.0.0.1:8080/api/health'],text=True,timeout=10));assert health['status']=='ok'
 result['services']={'mac_control_plane_state':'running','mac_control_plane_pid':int(pids[0]),'webmux':health}
 identity={'uname':list(os.uname()),'sw_vers':subprocess.check_output(['/usr/bin/sw_vers'],text=True,timeout=10),'clang_version':subprocess.check_output(['/usr/bin/xcrun','clang','--version'],text=True,timeout=10),'sdk_path':subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True,timeout=10).strip()}
 (root/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
 spec=importlib.util.spec_from_file_location('native_lifecycle_controls',root/'ci_darwin_shadow_lifecycle.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 binary=root/'lifecycle-observer';argv=['/usr/bin/xcrun','clang','-std=c11','-Wall','-Wextra','-Werror','-O2',str(root/'ci_darwin_shadow_lifecycle.c'),'-lproc','-o',str(binary)]
 before=time.monotonic();helper.captured(argv,root,'build-observer',cwd=root,timeout=120)
 result['build']={'argv':argv,'seconds':time.monotonic()-before,'sha256':sha(binary),'source_sha256':sha(root/'ci_darwin_shadow_lifecycle.c')}
 helper.controls(binary,root);result['controls_passed']=True
except BaseException as error:
 result['exception']=repr(error);(root/'exception.txt').write_text(traceback.format_exc())
finally:
 result['source_after']={name:sha(root/Path(name).name) for name in payload['sources']};result['source_guards_match']=result.get('source_before')==result['source_after'];result['terminal']=True;result['seconds']=time.monotonic()-started
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 files={}
 for path in sorted(root.rglob('*')):
  if path.is_file() and '__pycache__' not in path.parts:
   data=path.read_bytes();files[str(path.relative_to(root))]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
 print(json.dumps({'result':result,'files':files}))
'''
(root/'remote.py').write_text(remote)
(root/'input-identities.json').write_text(json.dumps({'revision':revision,'sources':{n:r['sha256'] for n,r in inputs.items()}},indent=2)+'\n')
argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','jkh@puck.local','/usr/bin/python3 -c '+shlex.quote(remote)]
start=time.monotonic()
with (root/'response.json').open('wb') as out,(root/'ssh.stderr').open('wb') as err:r=subprocess.run(argv,input=json.dumps({'revision':revision,'sources':inputs}).encode(),stdout=out,stderr=err,timeout=240)
assert r.returncode==0
response=json.loads((root/'response.json').read_bytes());files=root/'files';files.mkdir()
for name,row in response['files'].items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts;data=base64.b64decode(row['base64']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];dest=files/p;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(data)
(root/'transfer.json').write_text(json.dumps({'exit':r.returncode,'seconds':time.monotonic()-start,'files':{n:{k:v for k,v in d.items() if k!='base64'} for n,d in response['files'].items()},'source_revision':revision,'remote_root':response['result']['root']},indent=2)+'\n')
print(json.dumps({'result':response['result'],'files_received':len(response['files'])}))
