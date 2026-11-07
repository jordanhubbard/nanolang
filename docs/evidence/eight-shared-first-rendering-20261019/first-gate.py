from pathlib import Path
import subprocess,json,sys,traceback
w=Path('/Users/jkh/Src/nanolang-51-owned-float-design');out=Path(__file__).parent
sys.path.insert(0,str(w/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze,OwnedEmit
rows=[]
for name in ('nano_virt','original-shadows'):
 module=out/(name+'.nvm')
 r=subprocess.run([str(w/'bin/nanoisa_hl_facts'),str(module)],capture_output=True,timeout=30)
 (out/(name+'-changed-facts.json')).write_bytes(r.stdout);assert r.returncode==0,r.stderr
 facts=json.loads(r.stdout);functions=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
 emitter=OwnedEmit(functions,'nano',facts['layouts'])
 row={'input':name,'proof':emitter.eight_shared_proof,'locals':[emitter.nano_local_cost(fn) for fn in emitter.functions], 'shadow_estimate':2+4*len(emitter.witness)+6*sum(map(len,emitter.witness))}
 for lang in ('c','nano'):
  path=out/(name+'-first.'+lang);path.write_text('previous verified output\n')
  p=subprocess.run([sys.executable,str(w/'scripts/nvm2hl.py'),'--language',lang,str(module),'-o',str(path)],capture_output=True,timeout=30)
  row[lang]={'returncode':p.returncode,'stderr':p.stderr.decode(),'sentinel_preserved':path.read_text()=='previous verified output\n'}
 rows.append(row);(out/'first-gate.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
