import subprocess,json,hashlib,time,os,signal,sys,shutil
from pathlib import Path
root=Path(sys.argv[1]); evidence=Path(sys.argv[2]); evidence.mkdir(exist_ok=True,parents=True)
manifest=json.loads(Path(sys.argv[3]).read_text())
def hashes():
 return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in manifest['files']}
assert hashes()==manifest['files']
mode=evidence/'switch.mk'; mode.write_text('CFLAGS += -DNANO_NO_COMPUTED_GOTO\n')
for name,cmd in [('ordinary',['make','-j2','test-nanovm']),('switch',['make','-f','GNUmakefile','-f',str(mode),'-j2','OBJ_DIR=obj-switch','test-nanovm'])]:
 assert shutil.disk_usage(root).free>1024**3
 before=hashes();start=time.monotonic();timeout=False
 (evidence/(name+'-command.json')).write_text(json.dumps(cmd))
 with (evidence/(name+'.log')).open('wb') as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=180)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
 result={'returncode':rc,'timeout':timeout,'seconds':time.monotonic()-start,'source_unchanged':before==hashes()==manifest['files'],'revision':manifest['revision']}
 (evidence/(name+'-status.json')).write_text(json.dumps(result,indent=2));print(name,result,flush=True)
 assert rc==0 and result['source_unchanged']
