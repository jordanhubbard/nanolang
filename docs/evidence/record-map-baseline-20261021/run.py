from pathlib import Path
import hashlib,json,os,sys,time
child=Path.cwd();sys.path.insert(0,str(child))
from tests.native_process import run_captured,native_compile_timeout
root=Path('/Users/jkh/Src/nanolang-51-harness');out=Path(__file__).parent
emitter=Path('/private/tmp/nanolang-self-tail-proof-20261021/emitter')
producers=[('seed',root/'bin/nano_virt'),('c-hosted-nano',emitter),('stage1',root/'bin/nanoc_stage1'),('stage2',root/'bin/nanoc_stage2')]
tools={name:root/'bin'/name for name in ['nano_vm','nanoisa','nvm2c']};cc=Path('/opt/homebrew/opt/llvm/bin/clang')
fixtures=[('string-int','tests/nanoisa/fixtures/map_record_fields.nano'),('string-string','tests/unit/test_map_field_operations.nano')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[p for _,p in producers]+list(tools.values())+[cc]
receipt={'base':'878c2bb83','before':{str(p):sha(p) for p in paths},'sources':{},'commands':[],'cases':[]}
# The C-hosted emitter was built from identical current Nano source bytes and current proof.
previous=json.loads(Path('/private/tmp/nanolang-self-tail-proof-20261021/emitter-pins.json').read_text())
for relative,digest in previous.items():
 if relative.startswith('src_nano/') or relative=='src/nanoisa/managed_array_shapes.c':
  assert sha(root/relative)==digest,relative
receipt['c_hosted_provenance']='record-array-self-tail-20261021/emitter-build.json; source/proof hashes rechecked against current aggregate'
def save(): (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
def run(name,args,build=False):
 start=time.monotonic();r=run_captured(list(map(str,args)),cwd=child,timeout=native_compile_timeout(default=900) if build else 180,build=build)
 (out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
 receipt['commands'].append({'name':name,'argv':list(map(str,args)),'exit':r.returncode,'seconds':time.monotonic()-start});save();return r
for label,relative in fixtures:
 source=out/(label+'.nano');source.write_bytes((root/relative).read_bytes());receipt['sources'][relative]=sha(source)
 for provider,compiler in producers:
  name=label+'-'+provider;module=out/(name+'.nvm');case={'fixture':relative,'provider':provider};receipt['cases'].append(case)
  r=run(name+'-compile',[compiler,source,'--emit-nvm','-o',module]);case['compile']=r.returncode
  if r.returncode or not module.exists():save();continue
  for phase,args in [('verify',[tools['nano_vm'],'--verify-only',module]),('vm',[tools['nano_vm'],module]),('dump',[tools['nanoisa'],'dump',module])]:
   r=run(name+'-'+phase,args);case[phase]=r.returncode
   if phase=='dump' and not r.returncode:
    case['layouts_present']=b'.layouts ' in r.stdout;case['ownership_present']=b'.ownership ' in r.stdout
   if r.returncode:break
  if r.returncode:save();continue
  c=out/(name+'.c');r=run(name+'-translate',[tools['nvm2c'],module,'-o',c]);case['translate']=r.returncode
  if r.returncode:save();continue
  binary=out/(name+'.native');r=run(name+'-native-build',[cc,'-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',binary],True);case['native_build']=r.returncode
  if not r.returncode:case['native']=run(name+'-native',[binary]).returncode
  save()
receipt['after']={str(p):sha(p) for p in paths};assert receipt['before']==receipt['after'];save()
print(json.dumps(receipt['cases'],indent=2))
