from pathlib import Path
import subprocess,json,hashlib,sys
root=Path('/Users/jkh/Src/nanolang-owned-nested-depth2');out=Path(__file__).parent
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
def qualify(m,expected):
 assert cmd([root/'bin/nano_vm','--verify-only',m]).returncode==0
 r=cmd([root/'bin/nano_vm',m]);assert r.returncode==0 and r.stdout==expected,(r.returncode,r.stdout,r.stderr)
 r=cmd([root/'bin/nanoisa','dump',m]);assert r.returncode==0;m.with_suffix('.dump.nasm').write_bytes(r.stdout)
 r=cmd([root/'bin/nanoisa_hl_facts',m]);assert r.returncode!=0
 for lang in ['c','nano']:
  target=m.with_suffix('.refused.'+lang);target.write_bytes(b'previous verified output\n')
  r=cmd([root/'bin/nvm2hl','--language',lang,m,'-o',target]);assert r.returncode!=0 and target.read_bytes()==b'previous verified output\n'
source=root/'examples/language/nl_affine_resource_demo.nano'
if sys.argv[1]=='ordinary':
 for compiler in [root/'bin/nano_virt',root/'bin/nanoc_u8_context']:
  module=out/(compiler.name+'.nvm');r=cmd([compiler,source,'--emit-nvm','-o',module]);assert r.returncode==0,r.stderr
  qualify(module,b'audit.log\nresource closed\n')
 print('ordinary both providers PASS',flush=True)
elif sys.argv[1]=='build':
 files=list((root/'src_nano').rglob('*.nano'))
 pins={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 (out/'import-source-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
 r=cmd([root/'bin/nanoc_string_baseline',out/'shadow_driver.nano','-o',out/'shadow_driver'],timeout=900);assert r.returncode==0,r.stderr
 assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in pins.items())
 print('shadow driver build PASS',flush=True)
else:
 r=cmd([out/'shadow_driver',source,0,'raw']);assert r.returncode==0,r.stderr
 asm=out/'original-shadows.nasm';asm.write_bytes(r.stdout)
 module=out/'original-shadows.nvm';assert cmd([root/'bin/nanoisa','asm',asm,'-o',module]).returncode==0
 qualify(module,b'audit.log\nresource closed\nresource closed\naudit.log\nresource closed\n')
 print('original shadow module PASS',flush=True)
