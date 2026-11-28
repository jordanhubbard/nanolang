
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
