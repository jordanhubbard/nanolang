import sys,json,hashlib
from pathlib import Path
root=Path('/Users/jkh/Src/nanolang-51-harness');work=Path('/Users/jkh/Src/nanolang-51-owned-float-design')
sys.path.insert(0,str(work))
from tests.test_owned_call_pretest_reconstruction import OwnedCallPretestReconstruction
from tests.native_process import run_captured
out=Path('/private/tmp/double-call-pretest-baseline-20261020');out.mkdir(exist_ok=True)
source=out/'original.nano';source.write_text(OwnedCallPretestReconstruction.source(double=True))
commands=[];results=[]
def run(args,expected=0):
 r=run_captured(args,cwd=root,timeout=30,text=True)
 commands.append({'argv':list(map(str,args)),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 if expected is not None:assert r.returncode==expected,r.stderr
 return r
for name in ('nano_virt','nanoc_stage1','nanoc_stage2'):
 module=out/(name+'.nvm')
 run([root/'bin'/name,source,'--emit-nvm','-o',module]);run([root/'bin/nano_vm','--verify-only',module]);run([root/'bin/nano_vm',module])
 dump=run([root/'bin/nanoisa','dump',module]).stdout;(out/(name+'.nasm')).write_text(dump)
 facts=json.loads(run([root/'bin/nanoisa_hl_facts',module]).stdout);(out/(name+'.facts.json')).write_text(json.dumps(facts))
 for language in ('c','nano'):
  output=out/(name+'.'+language);output.write_text('prior verified source')
  r=run([root/'bin/nvm2hl','--language',language,module,'-o',output],None)
  assert r.returncode!=0 and 'one exact three-path reference call' in r.stderr
  assert output.read_text()=='prior verified source'
 results.append({'producer':name,'locals':[f['locals'] for f in facts['functions']],'calls':[sum(i['op']=='CALL_REF' for i in f['code']) for f in facts['functions']]})
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
