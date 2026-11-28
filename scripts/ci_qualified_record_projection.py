"""I retain my qualified-record projection repair and exact old-code controls."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,tempfile,time,traceback,unittest
from ci_original_qualification_policy import require_original_timeout_defaults
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,required=True);p.add_argument('--product',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
require_original_timeout_defaults(os.environ)
assert os.environ.get('GITHUB_ACTIONS')=='true'
assert (os.environ['RUNNER_OS'],os.environ['RUNNER_ARCH'],platform.machine()) in [('Linux','X64','x86_64'),('macOS','ARM64','arm64')]
old=a.old.resolve();product=a.product.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
tmp=out/'tmp';tmp.mkdir();os.environ['TMPDIR']=str(tmp);tempfile.tempdir=None
sys.path.insert(0,str(product))
from tests import native_process
from tests import nanolang_translator_routes as routes
from tests import test_nanovirt_qualified_record_projection as regression
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(whole_release=False,qualification=False,terminal=False,old='e2ad13e3ee129b926a5cab9b15f22bf61c0ac5d8',product='a52e44117f2623acc401d0fe0c10024a29b130dd',observer=os.environ['OBSERVER_REVISION'],host=dict(system=platform.system(),machine=platform.machine()),commands=[])
def save():(out/'result.json').write_text(json.dumps(q,indent=2)+'\n')
def source_map(root):
 return {str(root/n):sha(root/n) for n in subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0') if n and not n.startswith('docs/') and (root/n).is_file()}
def snapshot(table):return {name:sha(name) if Path(name).is_file() else None for name in table}
def record_files(table,category):
 directory=out/'inputs';directory.mkdir(exist_ok=True)
 for name,h in table.items():
  if h is None:q.setdefault('missing',[]).append(name);continue
  target=directory/h
  if not target.exists():shutil.copyfile(name,target)
  assert sha(target)==h
  q.setdefault('retained_inputs',[]).append(dict(category=category,original=name,sha256=h,retained=str(target)))
original=native_process.run_captured
def observed(args,**kwargs):
 index=len(q['commands']);row=dict(argv=list(map(str,args)),cwd=str(kwargs.get('cwd',product)),timeout=kwargs['timeout'],build=kwargs.get('build',False));row['effective_timeout']=native_process.native_compile_timeout() if row['build'] else row['timeout'];q['commands'].append(row);start=time.monotonic();save()
 try:
  result=original(args,**kwargs);row['exit']=result.returncode
  for stream in ['stdout','stderr']:
   data=getattr(result,stream);assert isinstance(data,bytes);path=out/(f'{index:04d}.'+stream);path.write_bytes(data);row[stream]=dict(path=str(path),sha256=sha(path))
  return result
 except BaseException as e:row['exception']=repr(e);raise
 finally:row['seconds']=time.monotonic()-start;save()
native_process.run_captured=routes.run_captured=observed
original_temp=tempfile.TemporaryDirectory
class RetainedTemporary(original_temp):
 def cleanup(self):self._finalizer.detach()
tempfile.TemporaryDirectory=RetainedTemporary
try:
 for root,revision in [(old,q['old']),(product,q['product'])]:
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==revision
  assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root)
 q['sources_before']={**source_map(old),**source_map(product)}
 for rel in ['tests/nvm2c_representation_joins_driver.nano','src_nano/nvm2c/representation_joins.nano','src_nano/nvm2c/representation_stack.nano']:
  assert (old/rel).read_bytes()==(product/rel).read_bytes()
 tools={str(Path(shutil.which(n)).resolve()):sha(Path(shutil.which(n)).resolve()) for n in ['cc','as','ld','make','python3']}
 config=product/'.test_output/retirement/capture-toolchain.json'
 if config.is_file():
  q['capture_toolchain']=json.loads(config.read_text())
  for row in q['capture_toolchain']['tools'].values():assert sha(row['path'])==row['sha256'];tools[row['path']]=row['sha256']
 q['tools_before']=tools;record_files(q['sources_before'],'source');record_files(tools,'tool');save()
 for root,targets in [(old,['nano_virt','nano_vm']),(product,['bin/nanoc_c','nano_virt','nano_vm','nvm2c','nvm2c-runtime'])]:
  result=observed(['make','-f','Makefile.gnu','-j2','CC=cc',*targets],cwd=root,timeout=600,build=True);assert result.returncode==0
 images={str(root/'bin'/n):sha(root/'bin'/n) for root,names in [(old,['nano_virt','nano_vm']),(product,['nanoc_c','nano_virt','nano_vm','nvm2c','nano_aot_runtime.o'])] for n in names}
 q['images_before']=images;record_files(images,'image');save()
 negative=out/'old-negative';negative.mkdir();(negative/'projection_provider.nano').write_text(regression.PROVIDER)
 source=negative/'main.nano';source.write_text(regression.fixture('left: int, right: int, tags: int'));target=negative/'prior.nvm';sentinel=b'I retain my previous output.\n';target.write_bytes(sentinel)
 result=observed([old/'bin/nano_virt',source,'-o',target],cwd=old,timeout=60,build=True)
 assert result.returncode!=0 and b'AGG_GET field 2 is unavailable' in result.stderr and target.read_bytes()==sentinel
 q['minimal_old_code_refused']=True;save()
 target=negative/'original-driver.nvm';target.write_bytes(sentinel)
 result=observed([old/'bin/nano_virt',old/'tests/nvm2c_representation_joins_driver.nano','-o',target],cwd=old,timeout=60,build=True)
 assert result.returncode!=0 and b'AGG_GET field 2 is unavailable' in result.stderr and b'push_slot' in result.stderr and target.read_bytes()==sentinel
 q['original_old_shadow_refused']=True;save()
 suite=unittest.defaultTestLoader.loadTestsFromModule(regression);assert suite.countTestCases()==2
 result=unittest.TextTestRunner(verbosity=2).run(suite);q.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped));assert result.wasSuccessful() and not result.skipped
 result=observed([product/'bin/nano_virt',product/'tests/nvm2c_representation_joins_driver.nano','-o',out/'current-driver.nvm'],cwd=product,timeout=60,build=True)
 assert result.returncode==0
 q['original_current_driver_shadows_passed']=True;q['functional_gate_passed']=True
except BaseException as e:
 q['exception']=repr(e);(out/'exception.log').write_text(traceback.format_exc())
finally:
 for category in ['sources','tools','images']:
  if category+'_before' in q:q[category+'_after']=snapshot(q[category+'_before'])
 q['guards_match']=all(q.get(k+'_before')==q.get(k+'_after') for k in ['sources','tools','images']) and all(k+'_before' in q for k in ['sources','tools','images'])
 q['terminal']=True;save()
sys.exit(0 if q.get('functional_gate_passed') and q['guards_match'] else 1)
