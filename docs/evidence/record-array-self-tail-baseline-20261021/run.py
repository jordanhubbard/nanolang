import ast,hashlib,json,os,pathlib,sys,time
root=pathlib.Path('/Users/jkh/Src/nanolang-51-harness');sys.path.insert(0,str(root))
from tests.native_process import run_captured
work=pathlib.Path(__file__).parent
parsed=ast.parse((root/'tests/test_flat_array_source_publication.py').read_text())
method=next(n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name=='test_recursive_tail_calls_keep_optional_array_authority_absent')
source=next(n for n in ast.walk(method) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('struct Box')).value
(work/'source.nano').write_text(source)
providers=[root/'bin'/n for n in ['nano_virt','nanoc_stage1','nanoc_stage2','nano_vm','nanoisa','nvm2c']]
driver=pathlib.Path('/private/tmp/owned-string-baseline-20261019/shadow_driver');cc=pathlib.Path('/opt/homebrew/opt/llvm/bin/clang');providers += [driver,cc]
def pins(): return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
before=pins(); receipt={'before':before,'commands':[]};(work/'receipt.json').write_text(json.dumps(receipt,indent=2))
def run(args,build=False):
 t=time.monotonic();r=run_captured(list(map(str,args)),cwd=root,text=True,timeout=900 if build else 180,build=build)
 receipt['commands'].append({'argv':list(map(str,args)),'exit':r.returncode,'seconds':time.monotonic()-t,'stdout':r.stdout,'stderr':r.stderr});(work/'receipt.json').write_text(json.dumps(receipt,indent=2));assert r.returncode==0,(args,r.stdout,r.stderr);return r.stdout
for name,producer in [('seed',providers[0]),('stage1',providers[1]),('stage2',providers[2]),('original-shadow',driver)]:
 module=work/(name+'.nvm')
 if name=='original-shadow':
  nasm=work/(name+'.nasm');nasm.write_text(run([producer,work/'source.nano',0,'raw']));run([root/'bin/nanoisa','asm',nasm,'-o',module])
 else:run([producer,work/'source.nano','--emit-nvm','-o',module])
 run([root/'bin/nano_vm','--verify-only',module]);dump=run([root/'bin/nanoisa','dump',module]);(work/(name+'.dump')).write_text(dump)
 assert '\n  TAIL_CALL ' in dump
 assert not any(l.startswith(('.ownership ','.layouts ')) for l in dump.splitlines())
 run([root/'bin/nano_vm',module]);native=work/(name+'.c');binary=work/(name+'.native')
 run([root/'bin/nvm2c',module,'-o',native]);run([cc,'-std=c11','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-lm','-o',binary],True);run([binary]);print(name,'PASS',flush=True)
receipt['after']=pins();assert receipt['after']==before;receipt['source_sha256']=hashlib.sha256((work/'source.nano').read_bytes()).hexdigest();receipt['status']='PASS';(work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
