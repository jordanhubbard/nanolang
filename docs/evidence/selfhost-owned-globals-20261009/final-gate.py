from pathlib import Path
import subprocess,time,json,hashlib,os
root=Path.cwd(); stem=Path('/private/tmp/nanolang-selfhost-globals-final')
paths=subprocess.check_output(['git','ls-files','src','src_nano','stdlib','modules'],text=True).splitlines()
paths+=['Makefile.gnu','tests/test_owned_global_producer.py','tests/test_owned_global_installed.py','tests/test_owned_union_c_source.py']
def inventory():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if Path(p).is_file()}
report={'before':inventory(),'steps':[],'started':time.time()}
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang',
 'NANO_GLOBAL_PRODUCER_ARTIFACTS':str(stem/'raw'),
 'NANO_SELECTED_SOURCE_ARTIFACTS':str(stem/'installed')}
commands=[['make','bootstrap','nanoisa_emit','CC=/opt/homebrew/opt/llvm/bin/clang'],
 ['python3','-m','unittest','-v','tests.test_owned_global_producer','tests.test_owned_global_installed']]
code=0
for index,argv in enumerate(commands):
 log=Path(str(stem)+f'-{index}.log');start=time.time()
 with log.open('w') as output:
  result=subprocess.run(argv,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 report['steps'].append({'argv':argv,'exit':result.returncode,'seconds':time.time()-start,'log':str(log)})
 if result.returncode:code=result.returncode;break
report['after']=inventory();report['unchanged']=report['before']==report['after'];report['seconds']=time.time()-report['started']
Path(str(stem)+'.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'exit':code,'unchanged':report['unchanged'],'seconds':report['seconds']}),flush=True)
raise SystemExit(code or (0 if report['unchanged'] else 1))
