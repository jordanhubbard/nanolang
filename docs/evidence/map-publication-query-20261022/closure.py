import json,os,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tests.native_process import run_captured
root=Path.cwd();out=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness')
s=json.loads((out/'inputs.json').read_text());flags=[x for x in s['flags'] if not x.startswith('-DNANO_')]
rows=[]
def run(name,args):
 p=run_captured(args,cwd=root,env=os.environ,timeout=180,build=name.endswith('build'));(out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,argv=args,status=p.returncode));(out/'closure-status.json').write_text(json.dumps(rows,indent=2)+'\n');assert p.returncode==0 and not p.stderr,p.stderr;return p.stdout
obj=out/'default-managed_array_shapes.o';run('default-managed-build',s['cc']+flags+['-c','src/nanoisa/managed_array_shapes.c','-o',str(obj)])
manifest=json.loads((root/'modules/nanoisa/module.json').read_text());objects=[];mapping=[]
for source in manifest['c_sources']:
 source=(root/'modules/nanoisa'/source).resolve().relative_to(root)
 path=agg/('obj/'+str(source)[4:-2]+'.o') if str(source).startswith('src/') else agg/'obj/nanoisa/nanoisa_facade.o'
 if source==Path('src/nanoisa/managed_array_shapes.c'):path=obj
 objects.append(str(path));mapping.append(dict(source=str(source),object=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
objects.append(str(agg/'obj/utf8.o'))
assert not any('nvm2c' in x for x in objects)
(out/'module-json-closure.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256((root/'modules/nanoisa/module.json').read_bytes()).hexdigest(),providers=mapping,extras=objects[-1:]),indent=2)+'\n')
exe=out/'module-closure';run('module-closure-build',s['cc']+flags+['tests/nanoisa/test_record_map_publication.c','tests/nanoisa/record_array_alloc.c']+objects+s['ldflags']+['-o',str(exe)])
print(run('module-closure-run',[str(exe)]+s['modules']).decode())
