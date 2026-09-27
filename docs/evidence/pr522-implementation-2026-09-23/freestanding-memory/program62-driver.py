from pathlib import Path
import subprocess,shlex,json,hashlib
prior=Path('/tmp/nano-record-array-llvm-vnjobctg');out=Path('/tmp/pr522-memory-62-check');out.mkdir(exist_ok=True)
build=shlex.split((prior/'wasm-O0-linked-runtime-build-command.txt').read_text())
runtime=out/'runtime.o';build[build.index('-o')+1]=str(runtime)
link=shlex.split((prior/'wasm-O0-linked-0062-link-command.txt').read_text())
link=[str(runtime) if x==str(prior/'wasm-O0-linked-runtime.o') else x for x in link]
wasm=out/'product-0062.wasm';link[link.index('-o')+1]=str(wasm)
script="const fs=require('fs');const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));if(WebAssembly.Module.imports(m).length)throw Error('imports');if(new WebAssembly.Instance(m).exports.nano_main())throw Error('program');"
commands={'build':build,'link':link,'wasmtime':['wasmtime','run','--invoke','nano_main',str(wasm)],'node':['node','-e',script,str(wasm)]}
for label,cmd in commands.items():
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
 (out/(label+'.json')).write_text(json.dumps({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
 assert r.returncode==0,(label,r.stderr)
 if label=='wasmtime':assert r.stdout=='0\n',r.stdout
print('I link and execute the unchanged program 62 in Node and Wasmtime with no imports.')
(out/'hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/nanoisa/managed_strings.c'),runtime,wasm]},indent=2)+'\n')
