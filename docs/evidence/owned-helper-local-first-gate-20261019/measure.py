from pathlib import Path
import json,sys,hashlib
root=Path('/Users/jkh/Src/nanolang-owned-nested-depth2')
base=Path('/private/tmp/owned-helper-locals-baseline-20261019')
out=Path(__file__).parent
# I reuse the retained process-group runner, not its executable probe section.
namespace={'__file__':str(out/'measure.py')}
exec((base/'probe.py').read_text().split("source=root/")[0],namespace)
cmd=namespace['cmd']
sys.path.insert(0,str(root/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze,OwnedEmit
for name in ['nano_virt','original-shadows']:
 module=base/(name+'.nvm')
 r=cmd([root/'bin/nanoisa_hl_facts',module]);assert r.returncode==0,r.stderr
 (out/(name+'.facts.json')).write_bytes(r.stdout)
 facts=json.loads(r.stdout)
 functions=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
 emitter=OwnedEmit(functions,'nano',facts['layouts'])
 measured={'input':name,'input_locals':[f['locals'] for f in facts['functions']],
           'estimated':[emitter.nano_local_cost(fn) for fn in emitter.functions],
           'shadow_estimated':emitter.multi_shadow_local_cost(),'proof':emitter.control_proof}
 (out/(name+'.estimate.json')).write_text(json.dumps(measured,indent=2)+'\n')
 for lang in ['c','nano']:
  target=out/(name+'.'+lang);target.write_bytes(b'prior accepted output\n')
  result=cmd([root/'bin/nvm2hl','--language',lang,module,'-o',target])
  if result.returncode:
   assert target.read_bytes()==b'prior accepted output\n'
   print(name,lang,'ATOMIC REFUSAL',result.stderr.decode(),measured,flush=True)
   raise SystemExit(2)
 print(name,'ordinary CLI publication PASS',measured,flush=True)
 if '--publication-only' in sys.argv:
  continue
 for producer in ['nano_virt','nanoc_stage1','nanoc_stage2']:
  regenerated=out/(name+'-'+producer+'.nvm')
  r=cmd([root/'bin'/('helper_baseline_'+producer),out/(name+'.nano'),'--emit-nvm','-o',regenerated]);assert r.returncode==0,r.stderr
  assert cmd([root/'bin/nano_vm','--verify-only',regenerated]).returncode==0
  r=cmd([root/'bin/nano_vm',regenerated]);assert r.returncode==0 and r.stdout==b'',r.stderr
  r=cmd([root/'bin/nanoisa_hl_facts',regenerated]);assert r.returncode==0,r.stderr
  actual=[f['locals'] for f in json.loads(r.stdout)['functions']]
  (out/(name+'-'+producer+'.count.json')).write_text(json.dumps({'actual':actual,**measured},indent=2)+'\n')
  assert actual==measured['estimated'],(actual,measured)
 r=cmd([base/'shadow_driver',out/(name+'.nano'),0,'raw']);assert r.returncode==0,r.stderr
 asm=out/(name+'.generated-shadow.nasm');asm.write_bytes(r.stdout)
 shadow=out/(name+'.generated-shadow.nvm');assert cmd([root/'bin/nanoisa','asm',asm,'-o',shadow]).returncode==0
 assert cmd([root/'bin/nano_vm','--verify-only',shadow]).returncode==0
 r=cmd([root/'bin/nano_vm',shadow]);assert r.returncode==0 and r.stdout==b'',r.stderr
 r=cmd([root/'bin/nanoisa_hl_facts',shadow]);assert r.returncode==0,r.stderr
 actual=[f['locals'] for f in json.loads(r.stdout)['functions']]
 (out/(name+'.shadow-count.json')).write_text(json.dumps({'actual':actual,**measured},indent=2)+'\n')
 assert actual[0]<=measured['shadow_estimated'] and actual[1]==measured['estimated'][1],(actual,measured)
 print(name,'actual regenerated/shadow measurement PASS',flush=True)
