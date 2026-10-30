from pathlib import Path
import os,sys,json,hashlib
sys.path.insert(0,str(Path.cwd()))
from tests.native_process import run_captured
root=Path.cwd();out=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness');driver=Path('/private/tmp/owned-string-baseline-20261019/shadow_driver');source=root/'tests/nanoisa/fixtures/map_record_fields.nano';rows=[]
pins=[driver,driver.with_suffix('.nano'),source,agg/'bin/nanoisa',agg/'bin/nano_vm']
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in pins};(out/'shadow-pins-before.json').write_text(json.dumps(before,indent=2)+'\n')
assert driver.with_suffix('.nano').read_bytes()==(root/'tests/nanoisa/fixtures/shadow_module_driver.nano.txt').read_bytes()
def run(name,args):
 r=run_captured(list(map(str,args)),cwd=root,env=os.environ,timeout=180);(out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr);rows.append(dict(name=name,argv=list(map(str,args)),status=r.returncode));(out/'shadow-status.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0 and not r.stderr,(name,r.stderr);return r.stdout
raw=run('original-shadow-emission',[driver,source,0,'raw']);asm=out/'original-shadow.nasm';asm.write_bytes(raw);module=out/'original-shadow.nvm'
run('original-shadow-assemble',[agg/'bin/nanoisa','asm',asm,'-o',module]);run('original-shadow-verify',[agg/'bin/nano_vm','--verify-only',module]);run('original-shadow-vm',[agg/'bin/nano_vm',module]);run('original-shadow-dump',[agg/'bin/nanoisa','dump',module])
print(run('original-shadow-query',[out/'module-closure',module]).decode())
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in pins};assert before==after;(out/'shadow-pins-after.json').write_text(json.dumps(after,indent=2)+'\n')
print('PASS independently selected original shadow raw verifier/VM then annotated publication query; retained driver, no rebuild')
