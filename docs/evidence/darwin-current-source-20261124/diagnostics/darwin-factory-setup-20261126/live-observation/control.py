"""I derive fresh Darwin products once at the original workflow limits."""
from pathlib import Path
import os,sys,json,hashlib,time,subprocess,signal,traceback,shlex,stat
root=Path(__file__).resolve().parent;request=json.loads((root/'request.json').read_bytes());pin=request['source_pin'];source=root/'source';lease=Path('/private/tmp/nanolang-native-qualification-reservation');owned=False
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(terminal=False,accepted=False,whole_release=False,component_only=True,source_pin=pin,root=str(root),source=str(source),pid=os.getpid(),commands=[],admissions=[],source_before={},images_before={},tools_before={},host_libraries_before={},phases=[],ordinary_seconds=10,stage_seconds=1800,compiler_shadow_seconds=30,emitter_helper_seconds=300,c_product_source_edit=False)
q['process_start']=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart='],text=True).strip()
def save():
 p=root/'result.next.json';p.write_text(json.dumps(q,indent=2)+'\n');os.replace(p,root/'result.json')
def run(label,args,*,cwd=None,timeout=180,input=None,environment=None):
 args=list(map(str,args));row=dict(label=label,argv=args,cwd=str(cwd or root),timeout=timeout);q['commands'].append(row);save();started=time.monotonic()
 with (root/(label+'.stdout')).open('wb') as out,(root/(label+'.stderr')).open('wb') as err:
  child=subprocess.Popen(args,cwd=cwd or root,env=environment or env,stdin=subprocess.PIPE if input is not None else None,stdout=out,stderr=err,start_new_session=True);row['pid']=child.pid;identity=subprocess.run(['ps','-p',str(child.pid),'-o','lstart='],capture_output=True,text=True);row['process_start']=identity.stdout.strip();row['identity_probe_exit']=identity.returncode;save()
  try:child.communicate(input=input,timeout=timeout);row['exit']=child.returncode
  except subprocess.TimeoutExpired:
   row['timed_out']=True
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()
   row['exit']=child.returncode
 row['seconds']=time.monotonic()-started;row['stdout_sha256']=sha(root/(label+'.stdout'));row['stderr_sha256']=sha(root/(label+'.stderr'));save()
 assert row['exit']==0 and not row.get('timed_out'),label
 return (root/(label+'.stdout')).read_bytes()
def admission(label):
 rows=[];raw=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
 parsed=[]
 for line in raw.splitlines():
  fields=line.split(None,2)
  if len(fields)!=3:continue
  pid,ppid,args=fields
  try:image=Path(shlex.split(args)[0]).name
  except (ValueError,IndexError):continue
  parsed.append((int(pid),int(ppid),image,args))
 ancestors={os.getpid()};changed=True
 while changed:
  changed=False
  for pid,ppid,image,args in parsed:
   if pid in ancestors and ppid not in ancestors and ppid>1:ancestors.add(ppid);changed=True
 candidates=[]
 for pid,ppid,image,args in parsed:
  if pid in ancestors:continue
  if image in ['nano_vm','nano_virt','nvm2c','nanoisa'] or image.startswith(('nanoc','nvm2c_')):rows.append(dict(pid=pid,image=image))
  elif image in ['make','gmake','cc','clang','clang++','gcc','bash','sh'] or image.startswith('python'):candidates.append(pid)
 if candidates:
  p=subprocess.run(['lsof','-bw','-a','-p',','.join(map(str,candidates)),'-d','cwd','-Fn'],capture_output=True,text=True,timeout=15);current=None
  for line in p.stdout.splitlines():
   if line.startswith('p') and line[1:].isdigit():current=int(line[1:])
   elif line.startswith('n') and 'nanolang' in line[1:].lower() and current not in ancestors:rows.append(dict(pid=current,cwd=line[1:]))
 df=subprocess.check_output(['/bin/df','-P',str(root)],text=True);device=df.splitlines()[1].split()[0]
 selected=[line for line in subprocess.check_output(['/sbin/mount'],text=True).splitlines() if line.startswith(device+' on ')]
 types={line.rsplit(' (',1)[1].split(',',1)[0].rstrip(')') for line in selected}
 filesystem='apfs' if types=={'apfs'} else ','.join(sorted(types))
 record=dict(label=label,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),other_nanolang_jobs=rows,free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,filesystem=filesystem,filesystem_probe=dict(device=device,mounts=selected),loadavg=os.getloadavg(),cpu=[])
 q['admissions'].append(record);save();assert record['filesystem']=='apfs' and record['free_bytes']>15*1024**3 and not rows,record
 for _ in range(2):
  started=time.monotonic();cpu=time.process_time();value=1
  while time.process_time()-cpu<1:
   for k in range(1024):value=(value*1664525+1013904223)&0xffffffff
  used=time.process_time()-cpu;wall=time.monotonic()-started;record['cpu'].append(dict(cpu=used,wall=wall,share=used/wall));save();assert used/wall>=.75,record
 record['passed']=True;save()
def source_state():
 result={};links={};raw=subprocess.check_output(['git','ls-files','-s','-z'],cwd=source)
 for item in raw.decode().split('\0'):
  if not item:continue
  metadata,name=item.split('\t',1);mode,oid,stage=metadata.split();p=source/name
  if p.is_symlink():
   payload=os.readlink(p).encode();links[name]=os.readlink(p)
  elif p.is_file():payload=p.read_bytes()
  else:continue
  assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==oid,name
  result[name]=hashlib.sha256(payload).hexdigest()
 q['tracked_symlinks']=links
 return result
try:
 env=dict(os.environ,TMPDIR=str(root/'tmp'),NANOLANG_ROOT=str(source));(root/'tmp').mkdir(exist_ok=True)
 forbidden=['NANO_SHADOW_TIMEOUT_SECONDS','BOOTSTRAP_NANOISA_TIMEOUT','NANO_BUILD_CACHE','NANO_MODULE_PATH','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANO_NATIVE_SHADOW_SOURCE_COMPILER','NANOLANG_BOOTSTRAP_NO_CC','CMD_TIMEOUT','NANO_COMPILER_PHASE_RECORDER']
 assert all(name not in env for name in forbidden),{name:env[name] for name in forbidden if name in env}
 q['initial_environment']={n:env[n] for n in ['PATH','CC','CFLAGS','LDFLAGS','SDKROOT','DEVELOPER_DIR','SOURCE_DATE_EPOCH','MACOSX_DEPLOYMENT_TARGET'] if n in env};save()
 admission('before-reservation')
 lease.mkdir(mode=0o700);owned=True;(lease/'owner.json').write_text(json.dumps(dict(root=str(root),pid=q['pid'],process_start=q['process_start'],source_pin=pin,owner='Sparky release recovery / #976 / #1253 / #1247'),indent=2)+'\n');q['reservation']=str(lease);save()
 run('clone',['git','clone','--filter=blob:none','--no-checkout','--single-branch','--branch','feat/5.1-service-module-integration','https://github.com/jordanhubbard/nanolang.git',source],timeout=600)
 assert run('publication',['git','ls-remote','origin','refs/heads/feat/5.1-service-module-integration'],cwd=source).decode().split()[0]==pin
 run('sparse',['git','sparse-checkout','set','--no-cone','--stdin'],cwd=source,input=(root/'sparse-patterns.txt').read_bytes(),timeout=600)
 run('checkout',['git','checkout','--detach',pin],cwd=source,timeout=600)
 assert run('source-pin',['git','rev-parse','HEAD'],cwd=source).decode().strip()==pin
 assert not run('initial-status',['git','status','--porcelain'],cwd=source)
 q['source_before']=source_state();save()
 run('tool-env',[sys.executable,'-m','venv',root/'tool-env'],timeout=180)
 python=root/'tool-env/bin/python3'
 run('pyyaml',[python,'-m','pip','install','--no-input','--disable-pip-version-check','--report',root/'pip-report.private.json','PyYAML'],timeout=300)
 package=json.loads((root/'pip-report.private.json').read_bytes());q['dependency_install']=[dict(name=v['metadata']['name'],version=v['metadata']['version'],archive=v['download_info'].get('archive_info',{})) for v in package['install']]
 env['PATH']=str(root/'tool-env/bin')+os.pathsep+env['PATH'];q['qualified_environment']={n:env[n] for n in ['PATH','TMPDIR','NANOLANG_ROOT']};save()
 tools=[sys.executable,str(python),'/usr/bin/make','/usr/bin/cc','/usr/bin/git','/usr/bin/xcrun','/usr/bin/ld','/usr/bin/codesign']
 selected=run('selected-clang',['xcrun','--find','clang']).decode().strip();tools.append(selected)
 sdk=run('selected-sdk',['xcrun','--sdk','macosx','--show-sdk-path']).decode().strip();q['sdk']=sdk
 run('clang-version',['xcrun','clang','--version']);run('python-yaml',[python,'-c','import sys,yaml; print(sys.version); print(yaml.__version__)']);run('make-version',['/usr/bin/make','--version']);run('openssl-prefix',['pkg-config','--variable=prefix','openssl'])
 q['tools_before']={name:sha(name) for name in tools};save()
 make=['/usr/bin/make','-j1','-f','Makefile.gnu','--no-print-directory','CC=/usr/bin/cc'];q['make']=make;save()
 run('schema',make+['src_nano/generated/compiler_schema.nano','src_nano/generated/compiler_ast.nano','src/generated/compiler_schema.h','src/nanoisa/generated_schema.h'],cwd=source)
 assert source_state()==q['source_before']
 run('cold-tools',make+['bin/nano','bin/nanoc_c','bin/nano_vm','bin/nvm2c','bin/nanoisa','bin/nano_virt','bin/nano_aot_runtime.o','modules-index'],cwd=source,timeout=1800)
 q['images_before']={name:sha(source/name) for name in ['bin/nano','bin/nanoc_c','bin/nano_vm','bin/nvm2c','bin/nanoisa','bin/nano_virt','bin/nano_aot_runtime.o']};assert source_state()==q['source_before'];q['phases'].append('cold-tools');save()
 for name in ['nanoc_stage1','nanoc_stage2','nanoc_seed.nvm','nanoc_stage1.nvm','nanoc_stage2.nvm','nanoc_bootstrap.json','nanoc']:assert not (source/'bin'/name).exists(),name
 admission('before-bootstrap')
 run('bootstrap',make+['bootstrap3'],cwd=source,timeout=None)
 receipt=json.loads((source/'bin/nanoc_bootstrap.json').read_bytes());assert receipt['phase']=='stage2' and receipt['raw_stage1_stage2_equal'] and receipt['stage_timeout_seconds']==1800 and receipt['shadow_timeout_seconds']==30
 assert len(receipt['steps'])==17 and all(step['exit_code']==0 for step in receipt['steps'])
 stage1=source/'bin/nanoc_stage1.nvm';stage2=source/'bin/nanoc_stage2.nvm';assert stage1.read_bytes()==stage2.read_bytes()
 q['raw_equality']=dict(bytes=stage1.stat().st_size,sha256=sha(stage1));q['bootstrap_receipt_sha256']=sha(source/'bin/nanoc_bootstrap.json');q['host_libraries_before']={name:sha(name) for name in receipt['hosts']};assert q['host_libraries_before']==receipt['hosts'];q['phases'].append('bootstrap17');save()
 admission('before-native')
 for label,target in [('native36','test-native-shadows'),('emitter12','test-native-shadow-emitter')]:
  run(label,make+[target],cwd=source,timeout=1800);q['phases'].append(label);save()
 q['accepted']=True
except BaseException as error:
 q['exception']=repr(error);(root/'exception.log').write_text(traceback.format_exc())
finally:
 if q['source_before']:
  try:q['source_after']=source_state();q['source_stable']=q['source_before']==q['source_after']
  except BaseException as error:q['source_guard_error']=repr(error);q['source_stable']=False
 q['images_after']={name:sha(source/name) for name in q['images_before']};q['tools_after']={name:sha(name) for name in q['tools_before']};q['host_libraries_after']={name:sha(name) for name in q['host_libraries_before']}
 q['remaining_owned_processes']=[]
 for line in subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines():
  fields=line.split(None,2)
  if len(fields)==3 and fields[0].isdigit() and int(fields[0])!=os.getpid() and str(root) in fields[2]:q['remaining_owned_processes'].append(dict(pid=int(fields[0]),command=fields[2]))
 q['guards_match']=q.get('source_stable',False) and q['images_before']==q['images_after'] and q['tools_before']==q['tools_after'] and q['host_libraries_before']==q['host_libraries_after'];q['accepted']=q['accepted'] and q['guards_match'] and not q['remaining_owned_processes']
 if owned and not q['remaining_owned_processes']:
  try:
   owner=json.loads((lease/'owner.json').read_bytes());assert owner['root']==str(root) and owner['pid']==os.getpid();(lease/'owner.json').unlink();lease.rmdir();q['reservation_released']=True
  except BaseException as error:q['reservation_release_error']=repr(error);q['accepted']=False
 q['terminal']=True;save()
if not q['accepted']:raise SystemExit(1)
