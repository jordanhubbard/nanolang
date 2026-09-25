from pathlib import Path
import json,subprocess,shlex,time
root=Path('/qualification');out=Path('/tmp/pr522-setter-build');fixtures=Path('/tmp/pr522-setter-fixtures')
runtime=out/'runtime';(runtime/'bin').mkdir(parents=True,exist_ok=True)
for name in ['src','include','modules','lib']:
 link=runtime/name
 if not link.exists():link.symlink_to(root/name,target_is_directory=True)
commands=subprocess.check_output(['make','-n','-W','src/transpiler.c','bin/nanoc_c'],cwd=root,text=True).splitlines()
compile_cmd=shlex.split(next(l for l in commands if ' -c src/transpiler.c ' in l))
compile_cmd[compile_cmd.index('src/transpiler.c')]=str(out/'transpiler.c');compile_cmd[compile_cmd.index('obj/transpiler.o')]=str(out/'transpiler.o')
link_line=next(l for l in commands if l.lstrip().startswith('cc ') and ' -o bin/nanoc_c ' in l and '-fsanitize' not in l and '-fprofile' not in l)
link_cmd=shlex.split(link_line.strip().rstrip('\\').strip().rstrip(';'))
link_cmd[link_cmd.index('obj/transpiler.o')]=str(out/'transpiler.o');link_cmd[link_cmd.index('bin/nanoc_c')]=str(runtime/'bin/nanoc_c')
with (out/'build.log').open('w') as log:
 for cmd in [compile_cmd,link_cmd]:subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
results=[]
for version,seed in [('before',root/'bin/nanoc_c'),('after',runtime/'bin/nanoc_c')]:
 for name in json.loads((fixtures/'manifest.json').read_text()):
  binary=out/(version+'-'+name.removesuffix('.nano'));start=time.monotonic()
  with (out/(version+'-'+name+'.log')).open('w') as log:
   build=subprocess.run([str(seed),str(fixtures/name),'-o',str(binary)],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   run=None if build.returncode else subprocess.run([str(binary)],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=20).returncode
  results.append({'version':version,'fixture':name,'compile_exit':build.returncode,'execution_exit':run,'seconds':time.monotonic()-start})
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
assert all(r['compile_exit']==0 and r['execution_exit']==0 for r in results if r['version']=='after')
