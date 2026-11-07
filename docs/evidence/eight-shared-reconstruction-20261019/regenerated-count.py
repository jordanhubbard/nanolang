from pathlib import Path
import sys,json,subprocess
w=Path('/Users/jkh/Src/nanolang-51-owned-float-design');root=Path('/Users/jkh/Src/nanolang-51-harness');out=Path(__file__).parent
sys.path.insert(0,str(w))
from tests.test_owned_nested_path_reconstruction import OwnedNestedPathReconstruction
rows=[]
for name in ('nano_virt','original-shadows'):
 source=out/(name+'-first.nano')
 for kind in ('ordinary','selected-shadow'):
  chosen=source
  if kind=='selected-shadow':
   chosen=out/(name+'-selected.nano');chosen.write_text(OwnedNestedPathReconstruction.selected_shadow(source.read_text()))
  module=chosen.with_suffix('.rebuilt.nvm')
  for command in ([root/'bin/nano_virt',chosen,'--emit-nvm','-o',module],[root/'bin/nano_vm','--verify-only',module],[root/'bin/nano_vm',module]):
   r=subprocess.run(list(map(str,command)),capture_output=True,timeout=180)
   rows.append({'command':list(map(str,command)),'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()});(out/'regenerated-count-commands.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0,r.stderr
  r=subprocess.run([str(w/'bin/nanoisa_hl_facts'),str(module)],capture_output=True,timeout=30);assert r.returncode==0,r.stderr
  m=json.loads(r.stdout);counts=[f['locals'] for f in m['functions']];assert counts==([92,103] if kind=='ordinary' else [98,103]),counts
  print(name,kind,counts,'verify/VM PASS',flush=True)
