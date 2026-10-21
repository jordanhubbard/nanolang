from pathlib import Path
import subprocess,json,hashlib
p=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness');rows=[]
def run(name,args):
 r=subprocess.run(list(map(str,args)),capture_output=True,timeout=180);(p/(name+'.stdout')).write_bytes(r.stdout);(p/(name+'.stderr')).write_bytes(r.stderr);rows.append({'name':name,'argv':list(map(str,args)),'status':r.returncode});(p/'runtime-neighbors.json').write_text(json.dumps(rows,indent=2));assert r.returncode==0 and not r.stderr,(name,r.returncode,r.stderr);print(r.stdout.decode(),end='')
for kind,old in [('map',Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-runtime-6gsrp07d')),('array',Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-array-vm-jixx2xwc'))]:
 s=json.loads((old/'selection.json').read_text());flags=[x for x in s['flags'] if x!='-DNANO_NO_COMPUTED_GOTO'];cc=s['cc'];sources=s['providers'] if kind=='map' else s['rebuilt_providers'];common=s['objects'] if kind=='map' else s['ordinary_common_objects'];objects=[str(old/(('observed-switch-' if kind=='array' else '')+Path(x).stem+'.o')) for x in sources if not x.endswith('/vm.c')];objects +=[str(agg/x) if not Path(x).is_absolute() else x for x in common]
 a=str(agg/'lib/libnano_compiler_inputs.a');objects=[x for i,x in enumerate(objects) if x!=a or a not in objects[i+1:]]
 for source in ['src/nanoisa/managed_array_shapes.c','src/nanoisa/verifier.c','src/nanoisa/ownership_contracts.c','src/nanovm/heap.c']:
  obj=p/(kind+'-neighbor-'+Path(source).stem+'.o');run(obj.stem+'-build',cc+flags+['-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c',source,'-o',obj]);objects=[str(obj) if Path(x).name==('observed-switch-' if kind=='array' else '')+Path(source).stem+'.o' else x for x in objects]
 for dispatch in ['switch','goto']:
  exe=p/(kind+'-neighbor-'+dispatch);f=flags+(['-DNANO_NO_COMPUTED_GOTO'] if dispatch=='switch' else [])+(['-DVM_RA_WHITEBOX'] if kind=='array' else [])
  run(exe.name+'-build',cc+f+['tests/nanoisa/test_record_'+('map_runtime' if kind=='map' else 'array_vm')+'.c','tests/nanoisa/record_array_alloc.c',*objects,*s['ldflags'],'-o',exe]);run(exe.name+'-run',[exe])
print('PASS old private map and array dispatch neighbors with current changed providers')
