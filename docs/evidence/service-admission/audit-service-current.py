import pathlib,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);e=pathlib.Path(sys.argv[2]);m=json.loads((e/'emitter-after.json').read_text())
for name,v in m.items():
 p=pathlib.Path(name);assert p.is_file() and p.stat().st_size==v['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
out={'current_paths':len(m),'all_hashes_equal':True,'selected_endpoint':str(e/'emitter-after.json'),'dependencies':{},'emitter':{}}
for name in ['nanovm/vm','nanoisa/verifier','nanoisa/service_bindings_module']:
 p=root/'obj'/(name+'.d');s=p.read_text();assert 'service_classification_private.h' in s;out['dependencies'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
p=e/'products/emitter.nvm';assert p.stat().st_size>0;out['emitter']={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(e/'current-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
