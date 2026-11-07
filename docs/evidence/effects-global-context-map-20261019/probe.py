import ast,hashlib,json,shlex,struct,subprocess
from pathlib import Path
root=Path("/Users/jkh/Src/nanolang-51-harness");out=Path(__file__).parent
results=[]
for label,decl in (("immutable","let answer:int=7"),("mutable","let mut answer:int=7")):
 source=out/(label+".nano");source.write_text(decl+"\nfn read()->int{return answer}\nshadow read {assert (== (read) 7)}\nfn main()->int{return (- (read) 7)}\n")
 for driver in (root/"bin/nano_virt",root/"bin/nanoc_stage1"):
  target=out/(label+"-"+driver.name+".nvm");cmd=[str(driver),str(source),"--emit-nvm","-o",str(target)]
  p=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=120);(out/(target.stem+".log")).write_text(p.stdout+p.stderr)
  row={"case":label,"producer":str(driver),"producer_sha256":hashlib.sha256(driver.read_bytes()).hexdigest(),"command":cmd,"status":p.returncode}
  if p.returncode==0:
   d=subprocess.run([str(root/"bin/nanoisa"),"dump",str(target)],cwd=root,capture_output=True,text=True,timeout=30);d.check_returncode();(out/(target.stem+".nasm")).write_text(d.stdout)
   lines=d.stdout.splitlines();strings=[ast.literal_eval(x[8:]).encode("latin1") for x in lines if x.startswith(".string ")];functions=[shlex.split(x)[1] for x in lines if x.startswith(".function ")];facts={}
   for x in lines:
    if x.startswith(".metadata "):
     _,k,v=x.split()
     if strings[int(k)]==b"nano.effects.v1":
      index,mask=struct.unpack("<IB3x",strings[int(v)]);facts[functions[index]]=mask
   row["facts"]=facts
   run=subprocess.run([str(root/"bin/nano_vm"),str(target)],cwd=root,capture_output=True,text=True,timeout=30);row["vm_status"]=run.returncode;row["vm_output"]=run.stdout+run.stderr
  results.append(row)
(out/"receipt.json").write_text(json.dumps(results,indent=2)+"\n");print(json.dumps(results,indent=2))
