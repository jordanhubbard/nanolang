import os,subprocess,signal,time,json
from pathlib import Path
root=Path('/private/tmp/nanolang-asan-profile-4cfad3737')
command=['/private/tmp/nanolang-asan-profile-4cfad3737/bin/nano_vm',str(root/'bin/nanoc_seed.nvm'),'--','src_nano/nanoc_v06.nano','--verbose','--emit-nvm','-o','/private/tmp/nanolang-asan-provisioned-stage1.nvm']
env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0',NANO_BUILD_CACHE='/private/tmp/nanolang-asan-profile-cache',NANO_SHADOW_TRACE='1',NANO_VM='/private/tmp/nanolang-profile-shadow-vm')
start=time.monotonic()
with open('/private/tmp/nanolang-asan-provisioned-execution.log','wb') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 Path('/private/tmp/nanolang-asan-provisioned.pid').write_text(str(child.pid))
 try:
  result=child.wait(timeout=7200);status='completed'
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGKILL);result=child.wait(timeout=10);status='full generation timed out'
Path('/private/tmp/nanolang-asan-provisioned-status.json').write_text(json.dumps(dict(status=status,exit_code=result,elapsed=time.monotonic()-start,command=command),indent=2))
