from pathlib import Path
import subprocess,time,json,hashlib,os
root=Path('/private/tmp/nanolang-match-guards-20261009');os.chdir(root)
stem=Path('/private/tmp/nanolang-selfhost-union-final');stem.mkdir(exist_ok=True)
paths=subprocess.check_output(['git','ls-files','src','src_nano','stdlib','modules'],text=True).splitlines()
paths+=['Makefile.gnu','tests/test_owned_union_producer.py','tests/test_owned_union_installed.py','tests/test_owned_union_boundaries.py','tests/test_owned_global_producer.py','tests/test_generic_selected_ownership.py','tests/test_owned_union_c_source.py']
def inventory():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if Path(p).is_file()}
report={'before':inventory(),'steps':[],'started':time.time(),'base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang',
 'NANO_GLOBAL_PRODUCER_ARTIFACTS':str(stem/'raw'),
 'NANO_SELECTED_SOURCE_ARTIFACTS':str(stem/'installed')}
commands=[
 ['python3','-m','unittest','-v','tests.test_owned_union_boundaries.UnionBoundaryProducer','tests.test_owned_union_boundaries.UnionBoundaryC'],
 ['make','bootstrap','nanoisa_emit','CC=/opt/homebrew/opt/llvm/bin/clang'],
 ['python3','-m','unittest','-v','tests.test_owned_union_installed','tests.test_owned_global_installed','tests.test_owned_union_boundaries.UnionBoundaryStage1','tests.test_owned_union_boundaries.UnionBoundaryStage2'],
 ['python3','-m','unittest','-v','tests.test_owned_union_producer','tests.test_owned_global_producer','tests.test_owned_union_c_source','tests.test_owned_global_source'],
 ['python3','-m','unittest','-v','tests.test_affine_scalar_union_source','tests.test_canonical_match_guards','tests.test_nanoisa_match_guards']]
code=0
for index,argv in enumerate(commands):
 log=stem/f'{index}.log';start=time.time()
 print('I run step',index,argv,'with log',log,flush=True)
 with log.open('w') as output:
  result=subprocess.run(argv,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 report['steps'].append({'argv':argv,'exit':result.returncode,'seconds':time.time()-start,'log':str(log)})
 report['after']=inventory();report['unchanged']=report['before']==report['after'];report['seconds']=time.time()-report['started']
 (stem/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('I finish step',index,'with exit',result.returncode,'unchanged sources',report['unchanged'],flush=True)
 if result.returncode or not report['unchanged']:code=result.returncode or 1;break
raise SystemExit(code)
