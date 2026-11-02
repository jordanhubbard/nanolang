import subprocess,json
from pathlib import Path
rows=[]
common=['-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa','-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk/usr/include/ffi']
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for src,flags in [('src/nanoisa/managed_array_shapes.c',[]),('src/nanovm/vm.c',[]),('src/nanovm/vm.c',['-DNANO_RECORD_MAP_PRIVATE_PREPARATION']),('tests/nanoisa/test_record_map_preparation.c',['-DNANO_RECORD_MAP_PRIVATE_PREPARATION']),('tests/nanoisa/test_record_map_preparation.c',['-DNANO_RECORD_MAP_PRIVATE_PREPARATION','-DVM_RM_WHITEBOX'])]:
  args=[cc,*common,*flags,'-fsyntax-only',src]
  p=subprocess.run(args,capture_output=True,text=True);rows.append(dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
  Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2))
  assert p.returncode==0,p.stderr
print('PASS ten strict Clang/GCC checks')
