from pathlib import Path
import json,subprocess,time,shutil,sys
r=Path.cwd();w=Path(__file__).parent;rows=[]
base=['-std=c11','-D_DEFAULT_SOURCE','-D_GNU_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa']
array=['-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME'];prep=[*array,'-DNANO_RECORD_MAP_PRIVATE_PREPARATION'];runtime=[*prep,'-DNANO_RECORD_MAP_PRIVATE_HEAP','-DNANO_RECORD_MAP_PRIVATE_RUNTIME']
for compiler in ['/opt/homebrew/opt/llvm/bin/clang',shutil.which('gcc-16')]:
 assert compiler
 for name,macros,source in [('ordinary',[],'src/nanovm/vm.c'),('array',array,'src/nanovm/vm.c'),('preparation',prep,'src/nanovm/vm.c'),('runtime',runtime,'src/nanovm/vm.c'),('harness',runtime,'tests/nanoisa/test_record_map_runtime.c'),('observation',runtime,'tests/nanoisa/test_record_map_observation.c'),('heap',runtime,'src/nanovm/heap.c')]:
  label=Path(compiler).name+'-'+name;args=[compiler,*base,*macros,'-fsyntax-only',source];t=time.monotonic();p=subprocess.run(args,cwd=r,capture_output=True);(w/(label+'.stdout')).write_bytes(p.stdout);(w/(label+'.stderr')).write_bytes(p.stderr);rows.append({'name':label,'argv':args,'exit':p.returncode,'seconds':time.monotonic()-t});(w/'strict.json').write_text(json.dumps(rows,indent=2));print(label,p.returncode,flush=True)
  if p.returncode:print(p.stderr.decode());sys.exit(p.returncode)
