import pathlib,subprocess,os,time,json,hashlib
root=pathlib.Path('/home/jkh/Src/nanolang-source-owned-temporary-arguments');out=pathlib.Path('/tmp/nanolang-temporary-owner-78f0f41f');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin.startswith('78f0f41f');assert not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def snapshot():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());before=snapshot();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'tools-before.json').write_text(json.dumps(tools(),indent=2)+'\n')
methods=['test_temporary_owner_actual_refusals_preserve_output','test_temporary_owner_actuals_preserve_order_and_roots','test_temporary_owner_factory_failure_cleans_prepared_arguments','test_temporary_owners_restore_original_pattern_sources'];cmd=['python3','-m','unittest','-v',*[f'tests.test_source_borrow_emission.SourceBorrowEmission.{m}' for m in methods]]
report={'source':pin,'steps':[]};status=1
try:
 for name,cc in [('gcc','/bin/gcc'),('clang','/bin/clang-18')]:
  env=os.environ.copy();env['CC']=cc;env['NANO_BUILD_CACHE']=str(root/'obj/module_cache');env.pop('NANOC',None)
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  status=result.returncode;report['steps'].append({'name':name,'CC':cc,'compiler_sha256':sha(pathlib.Path(cc).resolve()),'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=snapshot();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'tools-after.json').write_text(json.dumps(tools(),indent=2)+'\n');report['source_unchanged']=before==after;report['unchanged_head']=git('rev-parse','HEAD')==pin;report['tracked_tree_clean']=not git('status','--porcelain','--untracked-files=no');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');(out/'status').write_text(str(status)+'\n')
raise SystemExit(status)
