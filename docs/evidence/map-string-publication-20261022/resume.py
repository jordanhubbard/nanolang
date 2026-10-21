from pathlib import Path
import json,subprocess,shlex,hashlib,os,time,re
p=Path(__file__).parent;old=Path('/private/tmp/nanolang-map-paired-publication-20261022');current=Path('/private/tmp/nanolang-map-string-consumer-20261022');agg=Path('/Users/jkh/Src/nanolang-51-harness');root=Path.cwd();config=dict(x.split('=',1) for x in (old/'closure.txt').read_text().splitlines());cc=['/opt/homebrew/opt/llvm/bin/clang'];flags=shlex.split(config['CFLAGS'])+['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all'];ld=shlex.split(config['LDFLAGS']);objects=[str(agg/x) for x in shlex.split(config['OBJECTS'])];archive=str(agg/'lib/libnano_compiler_inputs.a');objects=[x for i,x in enumerate(objects) if x!=archive or archive not in objects[i+1:]]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
changed={str(agg/('obj/'+x+'.o')):str(current/(Path(x).name+'.o')) for x in ['nanoisa/managed_array_shapes','nanoisa/verifier','nanoisa/ownership_contracts','nanovm/heap','nanovm/vm','nanoisa/nvm2c']}
objects=[changed.get(x,x) for x in objects];pins={x:sha(x) for x in objects};pins.update({str(current/x):sha(current/x) for x in ['nanoisa','nano_vm','nvm2c']})
frozen=json.loads((current/'first.json').read_text());source_pins={str(f):sha(f) for d in ['src/nanoisa','src/nanovm'] for f in Path(d).iterdir() if f.suffix in ['.c','.h','.inc']};assert source_pins==frozen['source_pins'],'consumer source drift'
source_pins.update({x:sha(x) for x in ['src/nanovirt/codegen.c','src/nanovirt/ordinary_authority.inc','tests/nanovirt/test_map_publication.c','tests/unit/test_map_field_operations.nano']})
rows=[]
def run(name,args):
 start=time.monotonic();r=subprocess.run(list(map(str,args)),capture_output=True,timeout=180,env=dict(os.environ,LSAN_OPTIONS=''));(p/(name+'.stdout')).write_bytes(r.stdout);(p/(name+'.stderr')).write_bytes(r.stderr);rows.append({'name':name,'argv':list(map(str,args)),'status':r.returncode,'seconds':time.monotonic()-start});(p/'seed-status.json').write_text(json.dumps(rows,indent=2));assert not r.returncode,(name,r.returncode,r.stderr);assert not any(x in r.stderr for x in [b'AddressSanitizer',b'runtime error:']),r.stderr;return r.stdout
rows=json.loads((p/'seed-status.json').read_text())
for name in ['seed','c-shadow']:
 m=p/(name+'.nvm');dump=run(name+'-dump',[current/'nanoisa','dump',m]);assert b'.ownership "03000000' in dump and b'HM_DELETE' in dump
 own=bytes.fromhex(b''.join(re.findall(rb'\.ownership "([0-9a-f]+)"',dump)).decode())
 # Exact one-scalar SS pair and one layout0/field0 binding, schema revision1.
 expected=bytes.fromhex('05000100280000000100000005000000ffffffff01000000000000000000000001000000000000000000000000000000')
 assert own.endswith(expected),(name,own.hex())
 run(name+'-verify',[current/'nano_vm','--verify-only',m]);run(name+'-vm',[current/'nano_vm',m]);run(name+'-native',[current/'nvm2c',m,'-o',p/(name+'.c')])
assert pins=={x:sha(x) for x in pins};(p/'seed-provider-pins.json').write_text(json.dumps({'providers':pins,'sources':source_pins,'new_codegen':sha(p/'codegen.o'),'outputs':{x:sha(p/x) for x in ['seed.nvm','c-shadow.nvm','seed.c','c-shadow.c']}},indent=2));print('PASS current C ordinary + independent original-shadow unchanged Holder SS wire, query, verification, VM and native translation',flush=True)
