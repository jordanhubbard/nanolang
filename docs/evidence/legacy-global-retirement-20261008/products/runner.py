import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/private/tmp/nanolang-retirement-20261008');out=Path('/private/tmp/nanolang-retired-global-products-20261008');out.mkdir()
records=[]
env={**os.environ,'NANOLANG_ROOT':str(root),'NANO_BUILD_CACHE':str(root/'obj/module_cache')}
def run(name,args,expected=0):
 start=time.monotonic();r=subprocess.run([str(a) for a in args],cwd=root,env=env,capture_output=True)
 (out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
 records.append({'name':name,'command':[str(a) for a in args],'exit_code':r.returncode,'seconds':time.monotonic()-start})
 (out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 assert (r.returncode==0)==(expected==0),(name,r.returncode,r.stdout,r.stderr)
for tool in ('nano_virt','nanoc_stage1','nanoc_stage2'):
 for name in ('module_aliases','module_alias_targets','func_aliases','func_alias_targets'):
  prefix=tool+'-'+name;source=out/(prefix+'.nano');module=out/(prefix+'.nvm')
  body='fn main()->int {return (array_length '+name+')}\nshadow main {assert (== (main) 0)}\n'
  source.write_text(body);module.write_bytes(b'preserve me')
  run(prefix+'-refuse',[root/'bin'/tool,source,'--emit-nvm','-o',module],1)
  assert module.read_bytes()==b'preserve me'
  source.write_text('let '+name+':array<string> = []\n'+body)
  run(prefix+'-emit',[root/'bin'/tool,source,'--emit-nvm','-o',module])
  run(prefix+'-vm',[root/'bin/nano_vm',module])
  c=out/(prefix+'.c');binary=out/prefix
  run(prefix+'-translate',[root/'bin/nvm2c',module,'-o',c])
  run(prefix+'-cc',['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-O0','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',c,root/'bin/nano_aot_runtime.o','-lm','-o',binary])
  env['ASAN_OPTIONS']='detect_leaks=1'
  run(prefix+'-native',[binary])
print('PASS',len(records),'commands',flush=True)
(out/'runner.py').write_text(Path(__file__).read_text())
(out/'tools.json').write_text(json.dumps({tool:hashlib.sha256((root/'bin'/tool).read_bytes()).hexdigest() for tool in ('nano_virt','nanoc_stage1','nanoc_stage2','nvm2c')},indent=2)+'\n')
