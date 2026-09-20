import pathlib,subprocess,json
code='''import sys,json,pathlib,hashlib
expected=json.load(sys.stdin);out=[]
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for group,row in expected.items():
 for p,h in row['paths'].items():assert sha(p)==h,(group,p)
 out.append({'group':group,'paths':len(row['paths']),'sources':row['source'],'tool_labels':row['tool_labels'],'providers':row['providers']})
print(json.dumps(out))
'''
import shlex
p=subprocess.run(['ssh','puck.local','/opt/homebrew/bin/python3 -c '+shlex.quote(code)],input=pathlib.Path('/tmp/cyclic-dispatch-remote-expected.json').read_bytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
pathlib.Path('/tmp/cyclic-dispatch-remote-audit-stderr.txt').write_bytes(p.stderr);assert p.returncode==0,p.stderr
pathlib.Path('/tmp/cyclic-dispatch-remote-audit.json').write_bytes(p.stdout);print(p.stdout.decode())
