from pathlib import Path
import json,subprocess,sys,tempfile
root=Path('/Users/jordanh/Src/nanolang')
compiler=Path(sys.argv[1]); records=[]
for payload, value in [('Handle','Handle {value:7}'),('array<Handle>','[Handle {value:7}]'),('List<Handle>','(list_Handle_new)'),('Wrapper','Wrapper {handle:Handle {value:7}}'),('Box<Handle>','Box<Handle>.Some {value:Handle {value:7}}')]:
 with tempfile.TemporaryDirectory() as tmp:
  source=Path(tmp)/'main.nano';output=Path(tmp)/'output'
  source.write_text('resource struct Handle {value:int}\nstruct Wrapper {handle:Handle}\nunion Box<T> {Some {value:T}}\n'
   f'fn hold(value:Box<{payload}>)->int {{return 0}}\nshadow hold {{assert true}}\nfn main()->int {{let value:Box<{payload}> = Box<{payload}>.Some {{value:{value}}} return (hold value)}}\nshadow main {{assert (== (main) 0)}}\n')
  output.write_bytes(b'prior artifact')
  result=subprocess.run([str(compiler),str(source),'--emit-nvm','-o',str(output)],cwd=root,capture_output=True,text=True,timeout=120)
  records.append({'payload':payload,'returncode':result.returncode,'diagnostic':result.stdout+result.stderr,'artifact_preserved':output.read_bytes()==b'prior artifact'})
print(json.dumps(records,indent=2))
assert all(x['returncode']!=0 and x['artifact_preserved'] and ('unsupported local type' in x['diagnostic'] or 'source borrow profile' in x['diagnostic'] or '[E0036]' in x['diagnostic']) for x in records)
