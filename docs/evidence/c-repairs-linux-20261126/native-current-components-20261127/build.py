from pathlib import Path
import hashlib,json,os,signal,subprocess,time
out=Path(__file__).resolve().parent
tree=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/worktrees/c-repair-components-current-20261127')
env=dict(os.environ,PATH='/usr/bin:/bin',TMPDIR=str(tree/'tmp'),NANO_BUILD_CACHE=str(tree/'obj/module_cache'),NANO_AS_CAPTURE_HELPER=str(tree/'bin/nano_as_capture.so'))
assert env.get('NANO_SHADOW_TIMEOUT_SECONDS','10')=='10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=[n for n in subprocess.check_output(['git','ls-files','-z'],cwd=tree).decode().split('\0') if n and not n.startswith('docs/') and (tree/n).is_file()]
q=dict(terminal=False,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),source_before={n:sha(tree/n) for n in names})
q['tools_before']={n:sha(Path(n)) for n in ('/usr/bin/cc','/usr/bin/as','/usr/bin/ld','/usr/bin/python3','/usr/bin/make')}
args=['make','-f','Makefile.gnu','-j2','CC=/usr/bin/cc','bin/nanoc_c','nano_virt','nano_vm','nvm2c','nvm2c-runtime','obj/nvm2c_affine_analysis_reference','obj/nvm2c_affine_public_reference','obj/nvm2c_affine_analysis_cache_reference']
q['argv']=args
save=lambda:(out/'build.json').write_text(json.dumps(q,indent=2)+'\n')
save();start=time.monotonic()
try:
 with (out/'build.stdout').open('wb') as stdout,(out/'build.stderr').open('wb') as stderr:
  p=subprocess.Popen(args,cwd=tree,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  try:q['exit_code']=p.wait(timeout=1800)
  except BaseException as e:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
   p.wait();q['exit_code']=p.returncode;q['exception']=repr(e);raise
finally:
 q['seconds']=time.monotonic()-start;q['source_after']={n:sha(tree/n) for n in names};q['tools_after']={n:sha(Path(n)) for n in q['tools_before']};q['guards_match']=q['source_before']==q['source_after'] and q['tools_before']==q['tools_after'];q['terminal']=True;save()
raise SystemExit(not(q.get('exit_code')==0 and q['guards_match']))
