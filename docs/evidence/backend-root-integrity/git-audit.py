from pathlib import Path
import subprocess,json,hashlib
repo='/home/jkh/Src/nanolang-mixed-generated-ready';pin='061e8a5fb';prefix='docs/evidence/record-array-backend-integration';base=Path('/home/jkh/nanolang-qualification/backend-integration-a1bb-combined-seal')
index=json.loads((base/'report-sha256.json').read_text());expected={prefix+'/'+host+'/'+name:row for full,row in index.items() for host,name in [full.split('/',1)]}
actual=set(subprocess.check_output(['git','ls-tree','-rz','--name-only',pin,'--',prefix],cwd=repo).decode().strip('\0').split('\0'))
indexed=set()
for host in ['linux','puck']:
 hostindex=json.loads(subprocess.check_output(['git','show',pin+':'+prefix+'/'+host+'/report-sha256.json'],cwd=repo))
 for name,row in hostindex.items():
  path=prefix+'/'+host+'/'+name;assert expected[path]==row;indexed.add(path)
assert indexed<=set(expected) and set(expected)<=actual
assert json.loads(subprocess.check_output(['git','show',pin+':'+prefix+'/seal-sha256.json'],cwd=repo))==index
p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE);total=0
for path,row in expected.items():
 p.stdin.write((pin+':'+path+'\n').encode());p.stdin.flush();header=p.stdout.readline().decode().split();assert header[1]=='blob';data=p.stdout.read(int(header[2]));assert p.stdout.read(1)==b'\n';assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],path;total+=len(data)
p.stdin.close();assert p.wait()==0
out={'status':'PASS','git_pin':subprocess.check_output(['git','rev-parse',pin],cwd=repo,text=True).strip(),'reports':len(expected),'report_bytes':total,'scope':'Exact committed report set and bytes correspond to independently audited combined seal; semantic coverage and source correspondence are separate.'}
Path('/home/jkh/nanolang-qualification/backend-root-git-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
