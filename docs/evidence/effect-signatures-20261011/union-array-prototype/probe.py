from pathlib import Path
import subprocess,json
root=Path.cwd();w=Path('/private/tmp/nl51-union-arrays')
cases={
 'mixed_variants': ('union Value<T>{Number{n:T},Text{s:string}}', 'let xs:array<Value<int>> = [Value<int>.Number{n:7},Value<int>.Text{s:"hello"}] match (at xs 0){Number(p)=>{assert (== p.n 7)} Text(p)=>{assert false}} match (at xs 1){Number(p)=>{assert false} Text(p)=>{assert (== p.s "hello")}} return 7'),
 'generic_alias': ('union Box<T>{Some{value:T}}', 'let xs:array<Box<string>> = [Box<string>.Some{value:"retained"}] let ys:array<Box<string>> = xs match (at ys 0){Some(p)=>{assert (== p.value "retained")}} return 7'),
 'filled': ('union Box<T>{Some{value:T}}', 'let xs:array<Box<int>> = (array_new 3 Box<int>.Some{value:9}) match (at xs 2){Some(p)=>{assert (== p.value 9)}} return 7'),
}
results=[]
for name,(declaration,body) in cases.items():
 source=w/(name+'.nano');module=w/(name+'.nvm');native=w/(name+'.c');binary=w/name
 source.write_text(declaration+'\nfn main()->int{'+body+'}\nshadow main{assert (== (main) 7)}\n')
 steps=[ [str(root/'bin/nano_virt'),str(source),'--emit-nvm','-o',str(module)], [str(w/'nvm2c'),str(module),'-o',str(native)], ['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(native),'-o',str(binary),str(root/'bin/nano_aot_runtime.o'),'-lm'],[str(binary)]]
 for i,args in enumerate(steps):
  p=subprocess.run(args,capture_output=True,text=True,timeout=120)
  (w/(name+'-'+str(i)+'.log')).write_text(p.stdout+p.stderr)
  results.append([name,i,p.returncode]);print(results[-1],flush=True)
  if p.returncode != (7 if i==3 else 0):break
(w/'results.json').write_text(json.dumps(results,indent=2))
