from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path('/Users/jordanh/Src/nanolang');work=Path('/private/tmp/nl51-selected-final')
inputs=json.loads((work/'final-inputs.json').read_text())
report={'inputs':inputs,'steps':[]}
commands=[
 [str(work/'seed-v3'),str(root/'src_nano/nanoc_v06.nano'),'--emit-nvm','-o',str(work/'final-compiler.nvm')],
 [str(root/'bin/nano_vm'),'--verify-only',str(work/'final-compiler.nvm')],
 [str(root/'bin/nvm2c'),str(work/'final-compiler.nvm'),'-o',str(work/'final-compiler.c')],
 ['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',str(work/'final-compiler.c'),'-o',str(work/'final-compiler'),str(root/'bin/nano_aot_runtime.o'),'-lm'],
 [str(work/'final-compiler'),'--help'],
 [sys.executable,'-m','unittest','-v','tests.test_owned_effect_c_source'],
]
for index,command in enumerate(commands):
 log=work/f'product-{index:02}.log';print('start',index,command[0],flush=True);start=time.monotonic()
 with log.open('w') as output:
  result=subprocess.run(command,cwd=root,stdout=output,stderr=subprocess.STDOUT,env={**os.environ,'NANO_OWNED_EFFECT_DRIVER':str(work/'final-compiler'),'NANO_VM':str(work/'capture-vm')})
 report['steps'].append({'argv':command,'returncode':result.returncode,'seconds':time.monotonic()-start,'log':log.name})
 report['unchanged_inputs']=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==digest for p,digest in inputs.items())
 (work/'product-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('done',index,result.returncode,flush=True)
 if result.returncode or not report['unchanged_inputs']:raise SystemExit(result.returncode or 1)
