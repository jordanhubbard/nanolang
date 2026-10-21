from pathlib import Path
import hashlib,json,os,subprocess,struct,re
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification'); out=Path(__file__).parent
p=Path('/private/tmp/nanolang-map-delete-publication-20261022'); c=Path('/private/tmp/nanolang-map-delete-consumer-20261022')
tools={n:p/n for n in ['nano_virt','canonical','c-shadow']};tools['raw_driver']=p/'raw';tools.update({n:c/n for n in ['nano_vm','nanoisa','nvm2c']});tools['cc']=Path('/opt/homebrew/opt/llvm/bin/clang')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'base':'200c4f850','providers':{str(v):sha(v) for v in tools.values()},'commands':[]}
def run(name,args,fail=False):
 r=subprocess.run(list(map(str,args)),cwd=root,capture_output=True,timeout=180,env=dict(os.environ,NANO_VM=str(tools['nano_vm'])))
 (out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
 receipt['commands'].append(dict(name=name,argv=list(map(str,args)),status=r.returncode));(out/'receipt.json').write_text(json.dumps(receipt,indent=2))
 assert (r.returncode!=0)==fail,(name,r.stderr)
 assert not any(x in r.stderr for x in [b'AddressSanitizer',b'runtime error:']),r.stderr
 return r.stdout
base=(root/'tests/unit/test_map_field_operations.nano').read_bytes()
assert base.count(b'HashMap<string,string>')==1 and base.count(b'"kept"')==2
source=base
path=out/'unchanged.nano';path.write_bytes(source);receipt['source']=sha(path);receipt['original_source']=hashlib.sha256(base).hexdigest()
for name,tool in [('seed','nano_virt'),('canonical','canonical'),('c-shadow','c-shadow'),('raw-shadow','raw_driver')]:
 m=out/(name+'.nvm')
 if 'shadow' in name:
  asm=out/(name+'.raw.nasm');asm.write_bytes(run(name+'-source',[tools[tool],path]+(['0','raw'] if tool=='raw_driver' else [])))
  run(name+'-asm',[tools['nanoisa'],'asm',asm,'-o',m])
 else:run(name+'-source',[tools[tool],path,'--emit-nvm','-o',m])
 run(name+'-verify',[tools['nano_vm'],'--verify-only',m]);run(name+'-vm',[tools['nano_vm'],m])
 dump=run(name+'-dump',[tools['nanoisa'],'dump',m]).decode();assert '.ownership ' not in dump and '.layouts ' not in dump
 (out/(name+'.nasm')).write_text(dump)
 assert 'HM_DELETE' in dump
 native=out/(name+'.c');run(name+'-native',[tools['nvm2c'],m,'-o',native])
 for opt in ['O0','O2']:
  exe=out/(name+'-'+opt)
  run(name+'-build-'+opt,[tools['cc'],'-std=c11','-'+opt,'-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-lm','-o',exe])
  run(name+'-run-'+opt,[exe])
 # Independent exact Holder.values schema, not inferred from executable operations.
 layout=struct.pack('<IBBHI',1,0,0,1,0xffffffff)+struct.pack('<B3xII',13,0xffffffff,0xffffffff)
 owned=struct.pack('<IIB3x',3,1,1)
 functions=re.findall(r'^\.function \S+ (\d+) (\d+) (\d+) (\w+) (\d+)',dump,re.M)
 assert functions
 params={int(i):t.split() for i,t in re.findall(r'^\.parameters (\d+)(.*)$',dump,re.M)}
 tags={'void':0,'int':1,'u8':2,'float':3,'bool':4,'string':5,'struct':8,'hashmap':13}
 desc=lambda t:struct.pack('<BBHI',t,0,0,0xffffffff)
 owned+=struct.pack('<I',len(functions))
 for i,(arity,locals_,_,result,results) in enumerate(functions):
  arity=int(arity);locals_=int(locals_);owned+=struct.pack('<HH',locals_,arity)+desc(tags[result] if int(results) else 0)
  for j in range(locals_):owned+=desc(tags[params[i][j]] if j<arity else 0)
 payload=struct.pack('<I',1)+desc(5)+struct.pack('<III',1,0,0)+struct.pack('<I',1)+struct.pack('<IHHI',0,0,0,0)
 owned+=struct.pack('<IIIHHI',4,0,1,5,1,len(payload))+payload
 annotated=out/(name+'-selected.nasm');annotated.write_text(dump+'\n.layouts "'+layout.hex()+'"\n.ownership "'+owned.hex()+'"\n')
 prior=out/(name+'-prior.nvm');prior.write_bytes(b'prior artifact')
 run(name+'-selected-refusal',[tools['nanoisa'],'asm',annotated,'-o',prior],True);assert prior.read_bytes()==b'prior artifact'
 receipt.setdefault('modules',{})[name]=sha(m)
receipt['after']={str(v):sha(v) for v in tools.values()};assert receipt['after']==receipt['providers']
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS four unchanged originals verified/VM/native O0/O2; independent selected refusal and prior outputs preserved; schema validation separately pending')
