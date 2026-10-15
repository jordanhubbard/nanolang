from pathlib import Path
import subprocess,time,json,hashlib,os
root=Path('/private/tmp/nanolang-match-guards-20261009');os.chdir(root)
out=Path('/private/tmp/nanolang-selfhost-union-supplement');out.mkdir(exist_ok=True)
initial=json.loads(Path('/private/tmp/nanolang-selfhost-union-final/manifest.json').read_text())
assert len(initial['steps'])==5 and [s['exit'] for s in initial['steps']]==[0,0,0,0,1] and initial['unchanged']
paths=list(initial['after'])+['tests/test_affine_scalar_union_source.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory():return {p:sha(p) for p in paths}
report={'before':inventory(),'steps':[],'started':time.time(),'initial_manifest':'/private/tmp/nanolang-selfhost-union-final/manifest.json','original_seed':sha('bin/nanoc_seed.nvm')}
report['changed_since_initial']=[p for p,d in initial['after'].items() if report['before'][p]!=d]
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_SELECTED_SOURCE_ARTIFACTS':str(out/'c-source')}
seed=out/'corrected-seed.nvm'
commands=[
 ['/opt/homebrew/bin/gcc-16','-Wall','-Wextra','-Werror','-std=c99','-g','-O3','-ftree-vectorize','-fPIC','-Isrc','-D_GNU_SOURCE','-c','src/nanovirt/codegen.c','-o',str(out/'codegen.o')],
 ['make','nano_virt','test-nanovirt','test-borrow-contract-allocation','CC=/opt/homebrew/opt/llvm/bin/clang'],
 ['python3','-m','unittest','-v','tests.test_owned_union_c_source','tests.test_owned_global_source','tests.test_owned_union_boundaries.UnionBoundaryC'],
 ['python3','-m','unittest','-v','tests.test_affine_scalar_union_source.AffineScalarUnionSource.test_refusals_preserve_prior_output_on_every_frontend'],
 ['bin/nano_virt','src_nano/nanoc_v06.nano','--emit-nvm','--strip-debug','-o',str(seed)],
 ['bin/nano_vm','--verify-only',str(seed)]]
code=0
for index,argv in enumerate(commands):
 log=out/f'{index}.log';start=time.time();print('I run supplement step',index,argv,flush=True)
 with log.open('w') as output:result=subprocess.run(argv,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 report['steps'].append({'argv':argv,'exit':result.returncode,'seconds':time.time()-start,'log':str(log)})
 report['after']=inventory();report['unchanged']=report['before']==report['after'];report['seconds']=time.time()-report['started']
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('I finish supplement step',index,'exit',result.returncode,'unchanged',report['unchanged'],flush=True)
 if result.returncode or not report['unchanged']:code=result.returncode or 1;break
if code==0:
 boot=json.loads(Path('bin/nanoc_bootstrap.json').read_text())
 report['corrected_seed']=sha(seed);report['seed_bytes_equal']=seed.read_bytes()==Path('bin/nanoc_seed.nvm').read_bytes()
 report['bootstrap_consumers_unchanged']=all(sha('bin/'+name)==boot['tools'][name] for name in ('nano_vm','nvm2c'))
 report['nano_sources_unchanged']=all(report['after'][p]==d for p,d in initial['after'].items() if p.startswith(('src_nano/','modules/','stdlib/')))
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print({key:report[key] for key in ('seed_bytes_equal','bootstrap_consumers_unchanged','nano_sources_unchanged')},flush=True)
 if not all(report[key] for key in ('seed_bytes_equal','bootstrap_consumers_unchanged','nano_sources_unchanged')):code=1
raise SystemExit(code)
