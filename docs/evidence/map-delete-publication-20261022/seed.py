from pathlib import Path
import json,subprocess,shlex,hashlib,os
p=Path(__file__).parent;old=Path('/private/tmp/nanolang-map-paired-publication-20261022');current=Path('/private/tmp/nanolang-map-delete-consumer-20261022');agg=Path('/Users/jkh/Src/nanolang-51-harness');root=Path.cwd();config=dict(x.split('=',1) for x in (old/'closure.txt').read_text().splitlines());cc=['/opt/homebrew/opt/llvm/bin/clang'];flags=shlex.split(config['CFLAGS'])+['-fsanitize=address,undefined','-fno-sanitize-recover=all'];ld=shlex.split(config['LDFLAGS']);objects=[str(agg/x) for x in shlex.split(config['OBJECTS'])];archive=str(agg/'lib/libnano_compiler_inputs.a');objects=[x for i,x in enumerate(objects) if x!=archive or archive not in objects[i+1:]]
changed={str(agg/('obj/'+x+'.o')):str(current/(Path(x).name+'.o')) for x in ['nanoisa/managed_array_shapes','nanoisa/verifier','nanoisa/ownership_contracts','nanovm/heap','nanovm/vm']};objects=[changed.get(x,x) for x in objects];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={x:sha(x) for x in objects};rows=[]
def run(name,args):
 r=subprocess.run(list(map(str,args)),capture_output=True,timeout=180,env=dict(os.environ,LSAN_OPTIONS=''));(p/(name+'.stdout')).write_bytes(r.stdout);(p/(name+'.stderr')).write_bytes(r.stderr);rows.append({'name':name,'argv':list(map(str,args)),'status':r.returncode});(p/'seed-status.json').write_text(json.dumps(rows,indent=2));assert not r.returncode,(name,r.returncode,r.stderr);assert not any(x in r.stderr for x in [b'AddressSanitizer',b'runtime error:']),r.stderr;return r.stdout
run('seed-link',cc+flags+objects+ld+['-o',p/'nano_virt'])
without_main=[x for x in objects if x!=str(agg/'obj/nanovirt/main.o')]
run('c-shadow-build',cc+flags+['tests/nanovirt/borrow_shadow_names.c',*without_main,*ld,'-o',p/'c-shadow'])
run('c-controls-build',cc+flags+['tests/nanovirt/test_map_publication.c',*[x for x in without_main if x!=str(agg/'obj/nanovirt/codegen.o')],*ld,'-o',p/'c-controls'])
print(run('c-controls',[p/'c-controls','tests/nanoisa/fixtures/map_record_fields.nano']).decode())
source=(root/'tests/unit/test_map_field_operations.nano').read_text();assert source.count('HashMap<string,string>')==1 and source.count('"kept"')==2;source=source.replace('HashMap<string,string>','HashMap<string,int>').replace('"kept"','42');(p/'delete.nano').write_text(source)
run('seed-delete',[p/'nano_virt',p/'delete.nano','--emit-nvm','-o',p/'seed.nvm'])
asm=run('shadow-delete',[p/'c-shadow',p/'delete.nano']);(p/'c-shadow.nasm').write_bytes(asm);run('shadow-asm',[current/'nanoisa','asm',p/'c-shadow.nasm','-o',p/'c-shadow.nvm'])
for name in ['seed','c-shadow']:
 m=p/(name+'.nvm');dump=run(name+'-dump',[current/'nanoisa','dump',m]);assert b'.ownership "03000000' in dump and b'HM_DELETE' in dump
 run(name+'-verify',[current/'nano_vm','--verify-only',m]);run(name+'-vm',[current/'nano_vm',m]);run(name+'-native',[current/'nvm2c',m,'-o',p/(name+'.c')])
assert pins=={x:sha(x) for x in pins};(p/'seed-provider-pins.json').write_text(json.dumps(pins,indent=2));print('PASS current C ordinary + independent original-shadow DELETE publication, verification, VM and native translation')
