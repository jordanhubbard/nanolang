from pathlib import Path
import subprocess,json
p=Path(__file__).parent;s=json.loads((p/'focused.json').read_text())['retained_selection'];includes=[x for x in s['flags'] if x.startswith('-I')];rows=[]
for cc in [s['cc'][0],'/opt/homebrew/bin/gcc-16']:
 for mode,macros in [('default',[]),('array',['-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME']),('combined',['-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME','-DNANO_RECORD_MAP_PRIVATE_PREPARATION','-DNANO_RECORD_MAP_PRIVATE_RUNTIME','-DNANO_RECORD_MAP_PRIVATE_HEAP','-DNANO_RECORD_MAP_PRIVATE_NATIVE'])]:
  for dispatch in ['switch','goto']:
   flags=['-std=c11','-D_DEFAULT_SOURCE','-D_GNU_SOURCE','-Wall','-Wextra','-Werror','-fsyntax-only',*includes,*macros,*(['-DNANO_NO_COMPUTED_GOTO'] if dispatch=='switch' else [])]
   sources=['src/nanoisa/managed_array_shapes.c','src/nanoisa/verifier.c','src/nanoisa/ownership_contracts.c','src/nanovm/heap.c','src/nanovm/vm.c','src/nanoisa/nvm2c.c']
   if mode=='combined':sources+=['tests/nanoisa/test_record_map_delete.c','tests/nanoisa/test_record_map_native.c']
   for source in sources:
    name=Path(cc).name+'-'+mode+'-'+dispatch+'-'+Path(source).stem;r=subprocess.run([cc,*flags,source],capture_output=True,timeout=60);rows.append({'name':name,'argv':[cc,*flags,source],'status':r.returncode});(p/(name+'.stderr')).write_bytes(r.stderr);(p/'strict.json').write_text(json.dumps(rows,indent=2));assert r.returncode==0 and not r.stderr,(name,r.stderr)
print('PASS',len(rows),'Clang/GCC16 strict cases: default/array/combined x switch/goto, changed providers+harnesses')
