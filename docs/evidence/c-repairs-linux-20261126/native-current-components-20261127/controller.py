from pathlib import Path
import json,os,signal,subprocess,time
root=Path(__file__).resolve().parent
tree=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/worktrees/c-repair-components-current-20261127')
qualified=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/worktrees/c-repair-native-gates-20261127')
b=json.loads((root/'build.json').read_text());assert b['terminal'] and b['exit_code']==0 and b['guards_match']
env=dict(os.environ,PATH='/usr/bin:/bin',TMPDIR=str(tree/'tmp'),NANO_BUILD_CACHE=str(tree/'obj/module_cache'),NANO_AS_CAPTURE_HELPER=str(tree/'bin/nano_as_capture.so'),NANOLANG_TRANSLATOR_SOURCE_COMPILER=str(qualified/'bin/nanoc_stage2'))
assert env.get('NANO_SHADOW_TIMEOUT_SECONDS','10')=='10' and env.get('CMD_TIMEOUT','600')=='600'
q=dict(terminal=False,outer_limit_seconds=7200,qualified_producer=str(qualified/'bin/nanoc_stage2'),environment={k:env[k] for k in ['PATH','TMPDIR','NANO_BUILD_CACHE','NANO_AS_CAPTURE_HELPER','NANOLANG_TRANSLATOR_SOURCE_COMPILER']})
save=lambda:(root/'controller.json').write_text(json.dumps(q,indent=2)+'\n')
start=time.monotonic()
with (root/'stdout').open('wb') as out,(root/'stderr').open('wb') as err:
 p=subprocess.Popen(['/usr/bin/python3',str(root/'run.py')],env=env,stdout=out,stderr=err,start_new_session=True);q['pid']=p.pid;save()
 try:q['exit_code']=p.wait(timeout=7200)
 except BaseException as e:
  if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
  p.wait();q['exit_code']=p.returncode;q['exception']=repr(e);raise
 finally:q['seconds']=time.monotonic()-start;q['terminal']=True;save()
raise SystemExit(q['exit_code'])
