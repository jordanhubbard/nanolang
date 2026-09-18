import pathlib,subprocess,os,time,json,hashlib
root=pathlib.Path('/home/jkh/Src/nanolang-source-owned-temporary-arguments');out=pathlib.Path('/tmp/nanolang-temporary-owner-828edaac');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin.startswith('828edaac');assert not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def snapshot():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());before=snapshot();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'tools-before.json').write_text(json.dumps(tools(),indent=2)+'\n')
smoke='''from pathlib import Path
from tests.test_source_borrow_emission import SourceBorrowEmission, ROOT
case=SourceBorrowEmission()
case.work=Path('/tmp/nanolang-temporary-owner-828edaac/smoke-artifacts')
case.work.mkdir()
source=case.work/'ordered.nano'
source.write_text(case.temporary_owner_fixture())
baseline=None
for compiler in ('nano_virt','nanoc_stage1','nanoc_stage2'):
 module=case.work/(compiler+'.nvm')
 case.command(ROOT/'bin'/compiler,source,'--emit-nvm','-o',module)
 actual=case.command(ROOT/'bin/nanoisa','dump',module).stdout
 if baseline is None: baseline=actual
 case.assertEqual(actual,baseline)
 case.execute_pair(module,expected_output=b'ABCABCABC')
 print(compiler, 'exact canonical module, mandatory shadows and VM/native order PASS',flush=True)
'''
(out/'smoke.py').write_text(smoke)
report={'source':pin,'steps':[]};status=1;env=os.environ.copy();env['NANO_BUILD_CACHE']=str(root/'obj/module_cache');env['CC']='/bin/gcc';env.pop('NANOC',None)
steps=[('rebuild',['make','-j2','bin/nanoc_c','nano_virt','nano_vm','nvm2c','nanoisa_dump']),('smoke',['python3','-c',smoke]),('graph',['make','test-owned-value-graphs'])]
try:
 for name,cmd in steps:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  status=result.returncode;report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=snapshot();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'tools-after.json').write_text(json.dumps(tools(),indent=2)+'\n');report['source_unchanged']=before==after;report['unchanged_head']=git('rev-parse','HEAD')==pin;report['tracked_tree_clean']=not git('status','--porcelain','--untracked-files=no');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');(out/'status').write_text(str(status)+'\n')
raise SystemExit(status)
