"""I retain my unchanged full callable tests, products, commands and source guards."""
from pathlib import Path
import tempfile, hashlib, json, os, sys, time, unittest, subprocess, traceback
import argparse,platform,re
from ci_original_qualification_policy import require_original_timeout_defaults
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--product',type=Path,required=True);p.add_argument('--consumer',type=Path,required=True)
p.add_argument('--native-observations',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);args=p.parse_args()
require_original_timeout_defaults(os.environ)
assert os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('RUNNER_ARCH')=='ARM64'
assert sys.platform=='linux' and platform.machine()=='aarch64'
assert re.fullmatch('[0-9a-f]{40}',os.environ['C_REPAIR_OBSERVER_REVISION'])
root=args.observations.resolve();root.mkdir(parents=True,exist_ok=False)
temporary=root/'tmp';temporary.mkdir();os.environ['TMPDIR']=str(temporary);tempfile.tempdir=None
tree=args.consumer.resolve();os.chdir(tree);sys.path.insert(0,str(tree))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=[n for n in subprocess.check_output(['git','ls-files','-z'],cwd=tree).decode().split('\0') if n and not n.startswith('docs/evidence/') and (tree/n).is_file() and (n.endswith(('.nano','.c','.h','.inc','.py','.yaml','.json')) or n in ('Makefile.gnu','Makefile'))]
names += [n for n in ('tests/test_module_assembly_workers.c','tests/test_module_assembly_workers.py','tests/test_native_array_write_inference.py') if n not in names]
if (tree/'modules/index.json').is_file() and 'modules/index.json' not in names:names.append('modules/index.json')
qualified=args.product.resolve()
qualification=args.native_observations.resolve()/'original'
receipt=qualified/'bin/nanoc_bootstrap.json';accepted=qualification/'line-regression/result.json'
bootstrap=json.loads(receipt.read_text());gates={name:json.loads((qualification/name/'result.json').read_text()) for name in ('bootstrap','native-suites','line-regression')}
for name,row in gates.items():
 assert row['qualification'] and not row['whole_release'] and row['exit']==0 and not row['command_timeout'] and row['guards_match'] and row['before']==row['after']
 assert row['source_head']=='ddd3b5a4ff35678d1906c07c08fc6bbc39b041de'
text=(qualification/'native-suites/stderr').read_text();assert 'Ran 36 tests' in text and 'Ran 12 tests' in text and 'skipped=' not in text
assert str(qualified/'bin/nanoc_stage2')==os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER']
assert bootstrap['raw_stage1_stage2_equal'] and len(bootstrap['steps'])==17 and all(v['exit_code']==0 for v in bootstrap['steps'])
assert bootstrap['stage_timeout_seconds']==1800 and bootstrap['shadow_timeout_seconds']==30
assert (qualified/'bin/nanoc_stage1.nvm').read_bytes()==(qualified/'bin/nanoc_stage2.nvm').read_bytes()
assert sha(qualified/'modules/index.json')==bootstrap['sources']['modules/index.json']
image=(qualified/'bin/nanoc_stage2').read_bytes();assert image[:6]==b'\x7fELF\x02\x01' and int.from_bytes(image[18:20],'little')==183
for name,h in bootstrap['artifacts'].items():assert sha(qualified/'bin'/name)==h
for name,h in bootstrap['hosts'].items():assert sha(name)==h
baseline=gates['line-regression']['after']
prerequisite={'actual_gate_results':gates,'sources_after':{str(qualified/name):h for name,h in baseline['sources'].items()},'tools_after':baseline['tools_and_witness'],'bootstrap_receipt_sha256':sha(receipt)}
for name,h in prerequisite['sources_after'].items():assert sha(name)==h
for name,h in prerequisite['tools_after'].items():assert sha(name)==h
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()=='ddd3b5a4ff35678d1906c07c08fc6bbc39b041de'
(root/'qualified-bootstrap-receipt.json').write_bytes(receipt.read_bytes())
(root/'qualified-native-result.json').write_bytes(accepted.read_bytes())
tools=[str(Path(__file__).resolve()),str(Path(__file__).with_name('ci_original_qualification_policy.py').resolve()),'/usr/bin/make','/usr/bin/cc','/usr/bin/as','/usr/bin/ld','/usr/bin/python3',os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER']]
tools += list(bootstrap['hosts'])
tools += [str(receipt),str(accepted),str(qualified/'modules/index.json')]
images=[tree/'bin'/n for n in ('nanoc_c','nano_virt','nano_vm','nvm2c','nano_aot_runtime.o')]
images += [p for d in ('obj','lib') for p in (tree/d).rglob('*') if p.is_file() and (p.suffix in ('.o','.a') or p.name in ('nvm2c_affine_analysis_reference','nvm2c_affine_public_reference','nvm2c_affine_analysis_cache_reference'))]
q=dict(observer_revision=os.environ['C_REPAIR_OBSERVER_REVISION'],hosted_arm_architecture=platform.machine(),prerequisite_current_stage2=True,prerequisite_reconstructed=False,prerequisite_receipt_sha256=sha(receipt),prerequisite_native_sha256=sha(accepted),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),observer_sha256=sha(Path(__file__)),terminal=False,whole_release=False,temporary_roots=[],commands=[],source_before={n:sha(tree/n) for n in names},tools_before={n:sha(n) for n in tools},images_before={str(n):sha(n) for n in images})
last_snapshot=[0.0]
def save():
 now=time.monotonic()
 if q.get('terminal') or now-last_snapshot[0]>=1:
  (root/'retained-callables-result.json').write_text(json.dumps(q,indent=2)+'\n');last_snapshot[0]=now
def event(kind,row):
 with (root/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,index=len(q['commands'])-1,row=row))+'\n')
def after_sha(p):
 try:return sha(p)
 except FileNotFoundError:q.setdefault('missing_after',[]).append(str(p));return None

original_mkdtemp=tempfile.mkdtemp
def retained_mkdtemp(*args,**kwargs):
 name=original_mkdtemp(*args,**kwargs);q['temporary_roots'].append(name);save();return name
tempfile.mkdtemp=retained_mkdtemp
original_temporary=tempfile.TemporaryDirectory
class RetainedTemporary(original_temporary):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);q['temporary_roots'].append(self.name);save()
 def cleanup(self):self._finalizer.detach()
tempfile.TemporaryDirectory=RetainedTemporary
from tests import native_process
original_run=native_process.run_captured
streams=root/'retained-callables-streams';streams.mkdir(exist_ok=False)
def observed(args,**kwargs):
 index=len(q['commands']);row=dict(argv=list(map(str,args)),timeout=kwargs['timeout'],build=kwargs.get('build',False));row['effective_timeout']=native_process.native_compile_timeout() if row['build'] else row['timeout'];q['commands'].append(row);event('start',row);save();start=time.monotonic()
 try:
  r=original_run(args,**kwargs);row['exit']=r.returncode
  for name in ('stdout','stderr'):
   p=streams/f'{index:04d}.{name}';value=getattr(r,name)
   if isinstance(value,str):
    import locale
    encoding=kwargs.get('encoding') or locale.getpreferredencoding(False)
    value=value.encode(encoding,kwargs.get('errors') or 'strict');row.setdefault('text_streams',{})[name]=encoding
   p.write_bytes(value);row[name]=dict(path=str(p),sha256=sha(p))
  return r
 except BaseException as e:row['exception']=repr(e);raise
 finally:row['seconds']=time.monotonic()-start;event('finish',row);save()
native_process.run_captured=observed
start=time.monotonic()
try:
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tests.test_nanolang_translator_affine_analysis','tests.test_nanolang_translator_affine_public','tests.test_nanolang_translator_affine_analysis_cache','tests.test_nanolang_translator_shape_solver','tests.test_nanolang_translator_callables','tests.test_native_nonscalar_cast','tests.test_native_array_write_inference','tests.test_native_record_pointer_storage'))
 assert not unittest.defaultTestLoader.errors,unittest.defaultTestLoader.errors
 assert suite.countTestCases()==20,suite.countTestCases()
 q['discovered_methods']=suite.countTestCases()
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 q.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),functional_gate_passed=result.wasSuccessful() and not result.skipped)
except BaseException as e:q['exception']=repr(e);(root/'retained-callables-exception.log').write_text(traceback.format_exc())
finally:
 q.update(seconds=time.monotonic()-start,source_after={n:after_sha(tree/n) for n in names},tools_after={n:after_sha(n) for n in tools},images_after={str(n):after_sha(n) for n in images})
 q['qualified_sources_after']={n:after_sha(n) for n in prerequisite['sources_after']};q['qualified_tools_after']={n:after_sha(n) for n in prerequisite['tools_after']};q['qualified_guards_match']=q['qualified_sources_after']==prerequisite['sources_after'] and q['qualified_tools_after']==prerequisite['tools_after'];q['all_guards_match']=all(q[k+'_before']==q[k+'_after'] for k in ('source','tools','images')) and q['qualified_guards_match'];q['terminal']=True;save()
sys.exit(0 if q.get('functional_gate_passed') and q['all_guards_match'] else 1)
