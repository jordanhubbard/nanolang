from pathlib import Path
import subprocess,json,time
w=Path(__file__).parent;rows=[]
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for source,flags in [('src/nanoisa/managed_array_shapes.c',[]),('tests/nanoisa/test_record_map_boundary.c',[]),('tests/nanoisa/test_record_map_boundary.c',['-DRA_WHITEBOX'])]:
  argv=[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-fsyntax-only',*flags,source]
  p=subprocess.run(argv,capture_output=True,text=True,timeout=60);rows.append(dict(argv=argv,exit=p.returncode,stdout=p.stdout,stderr=p.stderr));(w/'strict.json').write_text(json.dumps(rows,indent=2))
  if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
print('PASS six strict GCC/Clang provider/harness controls')
