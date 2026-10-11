import pathlib,subprocess,json,time
r=pathlib.Path('/Users/jordanh/Src/nanolang');w=pathlib.Path('/private/tmp/nl51-selfhost-functional');compiler=pathlib.Path('/private/tmp/nl51-callable-union-product-xm4t8i4y/nano-selfhost-native-product-zrrrmdsr/compiler');results=[]
for folder in ['callable-array-operations-20261010','generic-array-callbacks-20261010']:
 for source in sorted((r/'docs/evidence'/folder).glob('test_*.nano')):
  name=folder+'-'+source.stem;m=w/(name+'.nvm');c=w/(name+'.c');b=w/name
  commands=[[compiler,source,'--emit-nvm','-o',m],[r/'bin/nano_vm','--verify-only',m],[r/'bin/nano_vm',m],[r/'bin/nvm2c',m,'-o',c],['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-o',b],[b]]
  row={'source':str(source),'steps':[]}
  for i,command in enumerate(commands):
   t=time.monotonic();p=subprocess.run(list(map(str,command)),cwd=r,capture_output=True,text=True,timeout=120);(w/(name+f'.{i}.log')).write_text(p.stdout+p.stderr);row['steps'].append({'argv':list(map(str,command)),'exit':p.returncode,'seconds':time.monotonic()-t})
   if p.returncode:print(name,i,p.returncode,(p.stdout+p.stderr)[-1600:],flush=True);break
  else:print(name,'PASS',flush=True)
  results.append(row);(w/'results.json').write_text(json.dumps(results,indent=2))
