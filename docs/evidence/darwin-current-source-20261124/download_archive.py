from pathlib import Path
import json,subprocess,tempfile,tarfile,hashlib,time
j=Path('.test_output/darwin-fixed-fixture-20261124');remote=json.loads((j/'package-terminal.stdout').read_text());assert remote['archive'].startswith('/private/tmp/nanolang-darwin-evidence-');local=Path(tempfile.mkdtemp(prefix='nanolang-darwin-archive-',dir='/var/tmp'));q=dict(remote=remote,local=str(local),terminal=False);(j/'download.json').write_text(json.dumps(q,indent=2)+'\n');start=time.monotonic();p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jkh@puck.local','/usr/bin/tar','-C',remote['archive'],'-cf','-','.'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
 with tarfile.open(fileobj=p.stdout,mode='r|') as tar:
  for member in tar:
   assert (local/member.name).resolve().is_relative_to(local)
   assert member.isfile() or member.isdir()
   tar.extract(member,local,filter='data')
 p.stdout.close();error=p.stderr.read();rc=p.wait(timeout=30);assert rc==0,error
 actual=hashlib.sha256((local/'manifest.json').read_bytes()).hexdigest();assert actual==remote['manifest_sha256'];q.update(terminal=True,exit=rc,seconds=time.monotonic()-start,manifest_sha256=actual);(j/'download.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
except BaseException:
 p.terminate();raise
