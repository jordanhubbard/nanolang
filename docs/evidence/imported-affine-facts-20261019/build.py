import hashlib,json,pathlib,shlex,subprocess
root=pathlib.Path('/Users/jkh/Src/nanolang-51-harness')
work=pathlib.Path('/private/tmp/nanolang-imported-affine-build'); work.mkdir(exist_ok=True)
lines=pathlib.Path('/private/tmp/nanolang-imported-affine-build-plan.log').read_text().replace('\\\n',' ').splitlines()
argv=shlex.split(next(line for line in lines if line.startswith('cc ') and ' -o /private/tmp/nanolang-imported-affine-bin/nano_virt ' in line))
flags=argv[:argv.index('-o')]
inputs={}; cloned={}
for i,arg in enumerate(argv):
 if arg.endswith(('.o','.a')):
  p=pathlib.Path(arg); p=p if p.is_absolute() else root/p
  if not p.is_file(): raise RuntimeError('missing retained input: '+str(p))
  inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  target=work/('input-'+str(i)+p.suffix)
  subprocess.run(['cp','-c','-p',str(p),str(target)],check=True)
  cloned[arg]=str(target); argv[i]=str(target)
(work/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
changed=work/'codegen.o'
compile_args=flags+['-c','src/nanovirt/codegen.c','-o',str(changed)]
argv[argv.index(cloned[str(root/'obj/nanovirt/codegen.o')])]=str(changed)
output=work/'nano_virt'; argv[argv.index('-o')+1]=str(output)
(work/'commands.json').write_text(json.dumps([compile_args,argv],indent=2)+'\n')
subprocess.run(compile_args,check=True)
subprocess.run(argv,check=True)
(work/'outputs.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [changed,output,pathlib.Path('src/nanovirt/borrow_codegen.inc'),pathlib.Path('src/nanovirt/codegen.c')]},indent=2)+'\n')
