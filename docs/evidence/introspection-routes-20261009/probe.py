from pathlib import Path
import subprocess,json
root=Path('/private/tmp/nanolang-match-guards-20261009')
out=Path('/private/tmp/nanolang-introspection-routes-20261009')
lib=out/'probe.nano'
lib.write_text('module route_probe\npub fn answer() -> int { return 7 }\nshadow answer { assert (== (answer) 7) }\n')
common=f'module {json.dumps(str(lib))} as probe\nextern fn ___module_function_count_route_probe() -> int\n'
cases={
'function_value': 'fn main() -> int { let count: fn() -> int = ___module_function_count_route_probe assert (== (count) 1) return 0 }\nshadow main { assert (== (main) 0) }\n',
'owned_direct': 'resource struct Handle { value: int }\nfn consume(h: Handle) -> int { let Handle { value } = h return value }\nshadow consume { assert (== (consume Handle { value: 7 }) 7) }\nfn main() -> int { let h: Handle = Handle { value: 7 } assert (== (___module_function_count_route_probe) 1) assert (== (consume h) 7) return 0 }\nshadow main { assert (== (main) 0) }\n',
}
rows=[]
for name,body in cases.items():
 source=out/(name+'.nano');source.write_text(common+body)
 for producer,compiler in [('nano_virt',root/'bin/nano_virt'),('component',Path('/private/tmp/nanolang-introspection-lowering-20261009/compiler'))]:
  target=out/(name+'-'+producer+'.nvm');target.write_bytes(b'previous module')
  r=subprocess.run([str(compiler),str(source),'--emit-nvm','-o',str(target)],cwd=root,capture_output=True,text=True,timeout=120)
  row={'case':name,'producer':producer,'compile_exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'preserved':target.read_bytes()==b'previous module'}
  if r.returncode==0:
   r=subprocess.run([str(root/'bin/nano_vm'),str(target)],cwd=root,capture_output=True,text=True,timeout=15);row.update(vm_exit=r.returncode,vm_stdout=r.stdout,vm_stderr=r.stderr)
  rows.append(row);print(json.dumps(row),flush=True)
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
