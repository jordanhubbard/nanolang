from pathlib import Path
import os,json,subprocess,hashlib,time
root=Path('/private/tmp/nanolang-match-guards-20261009');os.chdir(root)
out=Path('/private/tmp/nanolang-selfhost-union-seed-cache');out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
boot=json.loads(Path('bin/nanoc_bootstrap.json').read_text())
prior=json.loads(Path('/private/tmp/nanolang-selfhost-union-supplement/manifest.json').read_text())
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_BUILD_CACHE':str(root/'obj/module_cache'),'NANOLANG_ROOT':str(root),'NANO_VM':str(root/'bin/nano_vm'),'NANO_MODULE_PATH':str(root/'modules')}
keys=('NANO_BUILD_CACHE','NANOLANG_ROOT','NANO_VM','NANO_MODULE_PATH','CC')
r={'environment':{k:env[k] for k in keys},'steps':[],'before':{p:sha(p) for p in prior['after']}}
seed=out/'corrected-seed.nvm'
for args in [['bin/nano_virt','src_nano/nanoc_v06.nano','--emit-nvm','--strip-debug','-o',str(seed)],['bin/nano_vm','--verify-only',str(seed)]]:
 start=time.time();log=out/f'{len(r["steps"])}.log'
 with log.open('w') as f: result=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
 r['steps'].append({'argv':args,'exit':result.returncode,'seconds':time.time()-start})
 (out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
r.update(seed_bytes_equal=seed.read_bytes()==Path('bin/nanoc_seed.nvm').read_bytes(),bootstrap_consumers_unchanged=all(sha('bin/'+n)==boot['tools'][n] for n in ('nano_vm','nvm2c')),hosts_unchanged=all(sha(p)==d for p,d in boot['hosts'].items()),unchanged=all(sha(p)==d for p,d in r['before'].items()),seed_sha256=sha(seed))
(out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:v for k,v in r.items() if k not in ('before','steps')})
raise SystemExit(0 if all(r[k] for k in ('seed_bytes_equal','bootstrap_consumers_unchanged','hosts_unchanged','unchanged')) else 1)
