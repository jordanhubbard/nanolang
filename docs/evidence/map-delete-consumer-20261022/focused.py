from pathlib import Path
import subprocess,json,hashlib,os
out=Path(__file__).parent;old=Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-runtime-6gsrp07d');s=json.loads((old/'selection.json').read_text());agg=Path('/Users/jkh/Src/nanolang-51-harness')
flags=[x for x in s['flags'] if x!='-DNANO_NO_COMPUTED_GOTO'];cc=s['cc'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
objects=[str(old/(Path(p).stem+'.o')) for p in s['providers'] if not p.endswith('/vm.c')];objects += [str(agg/p) if not Path(p).is_absolute() else p for p in s['objects']]
a=str(agg/'lib/libnano_compiler_inputs.a');objects=[x for i,x in enumerate(objects) if x!=a or a not in objects[i+1:]]
receipt={'retained_selection':s,'pins':{p:sha(p) for p in objects},'commands':[]}
def run(name,args):
 r=subprocess.run(list(map(str,args)),capture_output=True,timeout=180,env=dict(os.environ,LSAN_OPTIONS=''));(out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr);receipt['commands'].append(dict(name=name,args=list(map(str,args)),status=r.returncode));(out/'focused.json').write_text(json.dumps(receipt,indent=2));assert r.returncode==0,(name,r.stderr);assert not any(x in r.stderr for x in [b'AddressSanitizer',b'runtime error:']),r.stderr;return r.stdout
for source in ['src/nanoisa/managed_array_shapes.c','src/nanoisa/verifier.c','src/nanoisa/ownership_contracts.c','src/nanovm/heap.c']:
 obj=out/('combined-'+Path(source).stem+'.o');run(obj.stem+'-build',cc+flags+['-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c',source,'-o',obj]);objects=[str(obj) if x==str(old/(Path(source).stem+'.o')) else x for x in objects]
for mode in ['switch','goto']:
 exe=out/('focused-'+mode);f=flags+(['-DNANO_NO_COMPUTED_GOTO'] if mode=='switch' else [])
 run('focused-'+mode+'-build',cc+f+['tests/nanoisa/test_record_map_delete.c','tests/nanoisa/record_array_alloc.c',*objects,*s['ldflags'],'-o',exe])
 print(run('focused-'+mode+'-run',[exe,*[out/(n+'-selected.nvm') for n in ['seed','canonical','c-shadow','raw-shadow']]]).decode())
assert receipt['pins']=={p:sha(p) for p in receipt['pins']}
