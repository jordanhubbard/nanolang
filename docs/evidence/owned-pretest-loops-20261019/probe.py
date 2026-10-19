import hashlib,json,subprocess
from pathlib import Path
root=Path('/Users/jkh/Src/nanolang-affine-borrowed-roots'); out=Path(__file__).parent
cases={
'preserve':('let mut i:int=0 while (< i count) {set i (+ i 1)} let Box{value,flag}=owner return (+ value i)',17),
'borrow-body':('let mut i:int=0 let mut sum:int=0 while (< i count) {set sum (+ sum owner.value) set i (+ i 1)} let Box{value,flag}=owner return (+ value sum)',35),
'borrow-condition':('let mut i:int=0 while (and owner.flag (< i count)) {set i (+ i 1)} let Box{value,flag}=owner return (+ value i)',17),
'fresh-body':('let mut i:int=0 let mut sum:int=0 while (< i count) {let fresh:Box=Box{value:2,flag:true} let Box{value,flag}=fresh set sum (+ sum value) set i (+ i 1)} let Box{value,flag}=owner return (+ value sum)',20),
'consume-entry':('let mut i:int=0 while (< i count) {let Box{value,flag}=owner set i (+ i 1)} return i',3),
'replace-owner':('let mut current:Box=owner let mut i:int=0 while (< i count) {let Box{value,flag}=current set current Box{value:(+ value 1),flag:flag} set i (+ i 1)} let Box{value,flag}=current return value',17),
'early-return':('let mut i:int=0 while (< i count) {if (== i 1) {let Box{value,flag}=owner return (+ value 100)} set i (+ i 1)} let Box{value,flag}=owner return value',114),
'body-return':('while (> count 0) {let Box{value,flag}=owner return (+ value 1)} let Box{value,flag}=owner return value',15),
}
receipt=[]
for name,(body,expected) in cases.items():
 source=out/(name+'.nano');source.write_text('resource struct Box {value:int,flag:bool}\nfn pick(owner:Box,count:int)->int {'+body+'}\nshadow pick {assert (== (main) '+str(expected)+')}\nfn main()->int {return (+ (pick Box{value:7,flag:true} 3) (pick Box{value:7,flag:true} 0))}\nshadow main {assert (== (main) '+str(expected)+')}\n')
 for compiler in ('bin/nano_virt','bin/nanoc_u8_context'):
  module=out/(name+'-'+Path(compiler).name+'.nvm')
  commands=[[root/compiler,source,'--emit-nvm','-o',module],[root/'bin/nano_vm','--verify-only',module],[root/'bin/nano_vm',module],[root/'bin/nanoisa_hl_facts',module]]
  for phase,cmd in zip(('produce','verify','execute','facts'),commands):
   p=subprocess.run(list(map(str,cmd)),cwd=root,capture_output=True,text=True,timeout=30)
   (out/(module.stem+'-'+phase+'.log')).write_text(p.stdout+p.stderr)
   receipt.append({'case':name,'provider':compiler,'phase':phase,'command':list(map(str,cmd)),'status':p.returncode,'expected':expected if phase=='execute' else 0,'stdout':p.stdout,'stderr':p.stderr})
   if phase=='produce' and p.returncode:break
   if phase=='verify' and p.returncode:break
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'providers.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('bin/nano_virt','bin/nanoc_u8_context','bin/nano_vm','bin/nanoisa_hl_facts')},indent=2)+'\n')
for r in receipt:
 if r['phase'] in ('produce','execute'):print(r['case'],r['provider'],r['phase'],r['status'],r['stderr'][:140].replace('\n',' '))
