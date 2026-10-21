import subprocess,json
from pathlib import Path
rows=[]
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src in ['src/nanoisa/managed_array_shapes.c','src/nanoisa/ownership_contracts.c','tests/nanoisa/test_record_map_image.c','tests/nanoisa/test_record_map_execution.c']:
  args=[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa','-fsyntax-only',src]
  p=subprocess.run(args,capture_output=True,text=True);rows.append(dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
  Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2))
  assert p.returncode==0,p.stderr
print('PASS eight strict Clang/GCC checks')
