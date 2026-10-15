from pathlib import Path
import json,subprocess
root=Path.cwd();work=Path('/private/tmp/nl51-generic-union-parameters')
prefix='union Box<T> { Some {value:T}, None {} }\n'
fixtures={
'identity':prefix+'''fn identity(x:T)->T{return x}
shadow identity {assert (== (identity 7) 7)}
fn read(x:Box<int>)->int{match x{Some(p)=>{return p.value} None(p)=>{return 0}}}
shadow read {assert (== (read Box<int>.Some {value:7}) 7)}
fn main()->int{let x:Box<int> = (identity Box<int>.Some {value:7}) assert (== (read x) 7) return 0}
''',
'unbox':prefix+'''fn unbox(x:Box<T>,fallback:T)->T{match x{Some(p)=>{return p.value} None(p)=>{return fallback}}}
shadow unbox {assert (== (unbox Box<int>.Some {value:7} 0) 7)}
fn main()->int{assert (== (unbox Box<int>.Some {value:7} 0) 7) assert (== (unbox Box<string>.None {} "kept") "kept") return 0}
''',
'rebox':prefix+'''fn box(x:T)->Box<T>{return Box.Some {value:x}}
shadow box {assert (== (read (box 7)) 7)}
fn read(x:Box<int>)->int{match x{Some(p)=>{return p.value} None(p)=>{return 0}}}
shadow read {assert (== (read Box<int>.Some {value:7}) 7)}
fn main()->int{assert (== (read (box 7)) 7) return 0}
'''}
compilers={'c':root/'bin/nano_virt','native':root/'bin/nanoc_c','selfhost':Path('/private/tmp/nl51-functional-product-x5y7l6dd/nano-selfhost-native-product-w9so6fs6/compiler')}
results=[]
for name,source in fixtures.items():
 path=work/(name+'.nano');path.write_text(source+'shadow main {assert (== (main) 0)}\n')
 for producer,compiler in compilers.items():
  out=work/(name+'-'+producer);out.write_bytes(b'prior')
  command=[str(compiler),str(path),'-o',str(out)]
  if producer!='native':command+=['--emit-nvm']
  result=subprocess.run(command,capture_output=True,text=True,timeout=120)
  row={'case':name,'producer':producer,'exit':result.returncode,'diagnostic':result.stdout+result.stderr,'prior_preserved':out.read_bytes()==b'prior'}
  results.append(row);print(json.dumps(row),flush=True)
  if result.returncode==0:
   execution=[str(out)] if producer=='native' else [str(root/'bin/nano_vm'),str(out)]
   r=subprocess.run(execution,capture_output=True,text=True,timeout=20)
   row['execution_exit']=r.returncode;row['execution_diagnostic']=r.stdout+r.stderr
   print('execution',r.returncode,r.stdout+r.stderr,flush=True)
(work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
