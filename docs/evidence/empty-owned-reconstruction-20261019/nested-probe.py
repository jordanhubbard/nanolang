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

for compiler in ('nano_virt','nanoc_stage1','nanoc_stage2'):
 module=out/(compiler+'-nested-result.nvm')
 r=cmd([root/'bin'/compiler,out/'nested-result.nano','--emit-nvm','-o',module]);assert r.returncode==0,r.stderr
 assert cmd([root/'bin/nano_vm','--verify-only',module]).returncode==0
 r=cmd([root/'bin/nano_vm',module]);assert r.returncode==0,r.stderr
 r=cmd([root/'bin/nanoisa','dump',module]);assert r.returncode==0;module.with_suffix('.dump.nasm').write_bytes(r.stdout)
 print(compiler,'nestedempty/consumingemptyresult original PASS',flush=True)
