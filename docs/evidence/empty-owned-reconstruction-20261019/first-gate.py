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
for name in ('nano_virt','nanoc_stage1','nanoc_stage2','original-shadows'):
 original=out/(name+'.nvm')
 for lang in ('c','nano'):
  output=out/(name+'-reconstructed.'+lang)
  r=cmd([sys.executable,local/'scripts/nvm2hl.py','--language',lang,original,'-o',output]);assert r.returncode==0,r.stderr
  if lang=='c':
   for opt in ('-O0','-O2'):
    executable=out/(name+opt)
    r=cmd(['cc','-std=c11','-pedantic-errors','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',opt,output,'-o',executable]);assert r.returncode==0,r.stderr
    r=cmd([executable]);assert r.returncode==0 and r.stdout==b'',r.stderr
  else:
   for compiler in ('nano_virt','nanoc_stage1','nanoc_stage2'):
    rebuilt=out/(name+'-'+compiler+'-rebuilt.nvm')
    r=cmd([root/'bin'/compiler,output,'--emit-nvm','-o',rebuilt]);assert r.returncode==0,r.stderr
    assert cmd([root/'bin/nano_vm','--verify-only',rebuilt]).returncode==0
    r=cmd([root/'bin/nano_vm',rebuilt]);assert r.returncode==0 and r.stdout==b'',r.stderr
    r=cmd([local/'bin/nanoisa_hl_facts',rebuilt]);assert r.returncode==0,r.stderr
    rebuilt.with_suffix('.facts.json').write_bytes(r.stdout)
 print(name,'all3 Nano and strictISO C O0/O2 PASS',flush=True)
