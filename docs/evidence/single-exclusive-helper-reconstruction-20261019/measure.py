import sys,json,subprocess
from pathlib import Path
root=Path('/Users/jkh/Src/nanolang-51-owned-float-design');w=Path(__file__).parent
sys.path.insert(0,str(root/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze,OwnedEmit
facts=json.loads((w/'nano_virt.facts.stdout').read_text())
functions=[OwnedAnalyze(facts,i).run() for i in range(2)]
result={}
for lang in ('c','nano'):
 emitter=OwnedEmit(functions,lang,facts['layouts'])
 result[lang]={'cost':[emitter.nano_local_cost(f) for f in emitter.functions],'work':emitter.control_proof}
 if lang=='nano':result[lang]['shadow_cost']=emitter.nano_local_cost(emitter.path_shadow_function())
 output=w/('measured.'+lang);output.write_text('prior source')
 r=subprocess.run([sys.executable,root/'scripts/nvm2hl.py','--language',lang,w/'nano_virt.nvm','-o',output],capture_output=True,text=True,timeout=60)
 result[lang].update(returncode=r.returncode,stderr=r.stderr,sentinel_preserved=output.read_text()=='prior source')
 print(lang,result[lang],flush=True)
(w/'measurement.json').write_text(json.dumps(result,indent=2)+'\n')
