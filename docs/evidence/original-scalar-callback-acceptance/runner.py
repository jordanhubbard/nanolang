from pathlib import Path
import hashlib,json,subprocess,tempfile,time
root=Path('/home/jkh/Src/nanolang-named-scalar-callbacks')
work=Path(tempfile.mkdtemp(prefix='nano-original-callback-acceptance-'))
source=work/'fresh_named_float_callbacks.nano'
source.write_text('''fn quotient(value:float)->float { return (/ value 0.0) }
shadow quotient { assert (== (float_to_bits (quotient 2.0)) 0) }
fn accumulate(total:float,value:float)->float { return (+ total value) }
shadow accumulate { assert (== (accumulate 2.0 3.0) 5.0) }
fn main()->int {
 let payload:int=9221120237041090602
 let input:array<float> = [(float_from_bits payload),2.0]
 let output:array<float> = (map input quotient)
 assert (== (float_to_bits (at output 0)) 0)
 assert (== (float_to_bits (at output 1)) 0)
 assert (== (float_to_bits (reduce input 0.0 accumulate)) 9221120237041090560)
 let reduced:float=(reduce input 0.0 accumulate)
 assert (== (float_to_bits reduced) 9221120237041090560)
 assert (== (float_to_bits (at input 0)) payload)
 assert (== (float_to_bits (at input 1)) 4611686018427387904)
 return 0
}
shadow main { assert (== (main) 0) }
''')
tools=[root/'bin'/x for x in ['nano','nanoc_c','nanoc_stage1','nanoc_stage2','nano_virt','nano_vm','nanoisa','nvm2c']]
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tools}
records=[]
def run(label,*args):
 start=time.monotonic();r=subprocess.run(list(map(str,args)),cwd=root,capture_output=True,text=True,timeout=240)
 (work/(label+'.log')).write_text(repr(list(map(str,args)))+'\n'+r.stdout+r.stderr)
 records.append({'label':label,'command':list(map(str,args)),'status':r.returncode,'seconds':round(time.monotonic()-start,3)})
 print(label,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(label+'\n'+r.stdout+r.stderr)
 return r
print('I retain fresh original-pattern acceptance at',work,flush=True)
run('interpreter',tools[0],source)
for compiler in ['nanoc_c','nanoc_stage1','nanoc_stage2']:
 exe=work/(compiler+'-legacy')
 run(compiler+'-legacy-build',root/'bin'/compiler,source,'-o',exe)
 run(compiler+'-legacy-run',exe)
for compiler in ['nano_virt','nanoc_stage1','nanoc_stage2']:
 module=work/(compiler+'.nvm')
 run(compiler+'-publish',root/'bin'/compiler,source,'--emit-nvm','-o',module)
 run(compiler+'-verify',root/'bin/nano_vm','--verify-only',module)
 run(compiler+'-vm',root/'bin/nano_vm',module)
 dump=run(compiler+'-dump',root/'bin/nanoisa','dump',module).stdout
 assert 'FUNCREF' not in dump and 'CALL_INDIRECT' not in dump
 c=module.with_suffix('.c');run(compiler+'-native-publish',root/'bin/nvm2c',module,'-o',c)
 for cc,flags in [('cc',[]),('clang',['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'])]:
  exe=work/(compiler+'-'+cc)
  run(compiler+'-'+cc+'-build',cc,*flags,'-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',exe)
  run(compiler+'-'+cc+'-run',exe)
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tools};assert before==after
result={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tool_hashes':before,'unchanged':True,'records':records,'scope':'I qualify the original named scalar callback pattern on fresh ordinary source; I do not replay the failed worker artifact or infer general callable support.'}
(work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
Path('/tmp/nanolang-original-callback-acceptance-result-path').write_text(str(work/'result.json')+'\n')
print('I pass every original-pattern producer route; result:',work/'result.json',flush=True)
