from pathlib import Path
import subprocess,json,hashlib,sys
root=Path('/Users/jkh/Src/nanolang-51-harness');out=Path(__file__).parent
records=json.loads((out/'commands.json').read_text()) if (out/'commands.json').exists() else []
sys.path.insert(0,str(root))
from tests.test_file_source_plan import FileSourcePlan
class Harness:
 work=out
 command=classmethod(FileSourcePlan.command.__func__)
def cmd(args,timeout=180):
 name='command-'+str(len(records)+1)
 try:
  stdout,stderr=Harness.command(name,args,timeout=timeout)
 except AssertionError:
  status=json.loads((out/(name+'-status.json')).read_text())
  if status['timeout'] or status['cleanup_errors'] or not status['group_absent'] or status.get('launch_or_wait_error'):raise
  stdout=(out/(name+'.stdout')).read_bytes();stderr=(out/(name+'.stderr')).read_bytes()
 status=json.loads((out/(name+'-status.json')).read_text())
 r=subprocess.CompletedProcess(list(map(str,args)),status['returncode'],stdout,stderr)
 records.append(dict(command=list(map(str,args)),returncode=r.returncode,stdout_hex=r.stdout.hex(),stderr=r.stderr.decode('utf8','replace'),retained_prefix=name))
 (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 return r

local=Path('/Users/jkh/Src/nanolang-51-owned-float-design')
for name in ('nano_virt','original-shadows'):
 original=out/(name+'.nvm');source=out/(name+'-reconstructed.nano')
 r=cmd([sys.executable,local/'scripts/nvm2hl.py','--language','nano',original,'-o',source]);assert r.returncode==0,r.stderr
 rebuilt=out/(name+'-rebuilt.nvm');r=cmd([root/'bin/nano_virt',source,'--emit-nvm','-o',rebuilt]);assert r.returncode==0,r.stderr
 assert cmd([root/'bin/nano_vm','--verify-only',rebuilt]).returncode==0
 assert cmd([root/'bin/nano_vm',rebuilt]).returncode==0
 r=cmd([root/'bin/nanoisa','dump',rebuilt]);assert r.returncode==0;rebuilt.with_suffix('.dump.nasm').write_bytes(r.stdout)
 import re
 counts=[int(x) for x in re.findall(rb'^\.function \S+ \d+ (\d+) ',r.stdout,re.M)];assert counts==[127,29],counts
 print(name,'actual regenerated seed counts',counts,'verify/VM PASS',flush=True)
