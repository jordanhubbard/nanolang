from pathlib import Path
import os,subprocess,json,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_generic_selected_patterns import GenericSelectedPatterns
root=Path.cwd();work=Path(os.environ.get('MIXED_WORK','/private/tmp/nl51-selected-patterns/c-mixed-matrix'));work.mkdir(exist_ok=True)
report=[]
def check(source,accepted):
 case=work/str(len(report));case.mkdir(exist_ok=True);src=case/'main.nano';out=case/'main.nvm';src.write_text(source);out.write_bytes(b'prior artifact')
 entry={'accepted':accepted,'steps':[]};report.append(entry)
 def run(cmd,expected=0):
  r=subprocess.run([str(x) for x in cmd],capture_output=True,timeout=120,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
  n=len(entry['steps']);(case/f'{n}.stdout').write_bytes(r.stdout);(case/f'{n}.stderr').write_bytes(r.stderr)
  entry['steps'].append({'argv':[str(x) for x in cmd],'returncode':r.returncode});(work/'results.json').write_text(json.dumps(report,indent=2))
  assert r.returncode==expected,r.stderr.decode()
 run([os.environ.get('MIXED_DRIVER',root/'bin/nano_virt'),src,'--emit-nvm','-o',out],0 if accepted else 1)
 if not accepted: assert out.read_bytes()==b'prior artifact';return
 run([root/'bin/nano_vm','--verify-only',out]);run([root/'bin/nano_vm',out])
 run([root/'bin/nvm2c',out,'-o',case/'main.c'])
 run(['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer',case/'main.c','-o',case/'main'])
 run([case/'main'])
test=GenericSelectedPatterns();test.check=check;test.test_copyable_pattern_with_resource_functions();print('passed',len(report),'mixed-profile cases')
