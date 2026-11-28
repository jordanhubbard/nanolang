"""I retain my complete representation join and prerequisite comparison."""
from pathlib import Path
import argparse,hashlib,json,os,platform,subprocess,sys,tempfile,time,traceback,unittest
from ci_original_qualification_policy import require_original_timeout_defaults
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--product',type=Path,required=True);p.add_argument('--observations',type=Path,required=True)
p.add_argument('--producer-admission',type=Path,required=True);p.add_argument('--qualified-report',type=Path,required=True)
a=p.parse_args();require_original_timeout_defaults(os.environ)
assert os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('RUNNER_ARCH')=='X64' and platform.machine()=='x86_64'
root=a.observations.resolve();root.mkdir(parents=True,exist_ok=False)
temporary=root/'tmp';temporary.mkdir();os.environ['TMPDIR']=str(temporary);tempfile.tempdir=None
tree=a.product.resolve();os.chdir(tree);sys.path.insert(0,str(tree))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
assert head==os.environ['PRODUCT_REVISION']=='e2ad13e3ee129b926a5cab9b15f22bf61c0ac5d8'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=tree)
admission=json.loads(a.producer_admission.read_text());verified=json.loads(a.qualified_report.read_text());assert len(verified)==1
verified=verified[0];assert verified['full_original_native_gate_passed'] and verified['bootstrap_steps']==17 and verified['native_methods']==36 and verified['emitter_methods']==12
assert admission['source_provider']=='ddd3b5a4ff35678d1906c07c08fc6bbc39b041de' and not admission['subject_stage_alias_created']
assert os.environ['NANOLANG_TRANSLATOR_SOURCE_COMPILER']==admission['actual_native']
assert sha(admission['actual_native'])==admission['actual_native_sha256']
qualified_sources=admission['source_inputs'];qualified_tools={**admission['tools'],**admission['actual_hosts'],admission['actual_native']:admission['actual_native_sha256'],admission['receipt']:admission['receipt_sha256']}
for name,h in {**qualified_sources,**qualified_tools}.items():assert sha(name)==h
prerequisite={'sources_after':qualified_sources,'tools_after':qualified_tools}
names=[n for n in subprocess.check_output(['git','ls-files','-z'],cwd=tree).decode().split('\0') if n and not n.startswith('docs/') and (tree/n).is_file()]
if (tree/'modules/index.json').is_file() and 'modules/index.json' not in names:names.append('modules/index.json')
tools=[str(Path(__file__).resolve()),str(Path(__file__).with_name('ci_original_qualification_policy.py').resolve()),'/usr/bin/cc','/usr/bin/as','/usr/bin/ld','/usr/bin/make','/usr/bin/python3',admission['actual_native'],str(a.producer_admission),str(a.qualified_report)]
tools+=list(admission['actual_hosts'])
images=[tree/'bin'/n for n in ['nanoc_c','nano_virt','nano_vm','nvm2c','nano_aot_runtime.o']]
images += [p for d in ['obj','lib'] for p in (tree/d).rglob('*') if p.is_file() and p.suffix in ('.o','.a')]
q=dict(qualification_execution=True,whole_release=False,subject=head,source_head=head,observer_revision=os.environ['OBSERVER_REVISION'],prior_qualified_producer=admission,terminal=False,temporary_roots=[],commands=[],source_before={n:sha(tree/n) for n in names},tools_before={n:sha(n) for n in tools},images_before={str(n):sha(n) for n in images})
last_snapshot=[0.0]
def save():
 now=time.monotonic()
 if q.get('terminal') or now-last_snapshot[0]>=1:
  (root/'retained-component-result.json').write_text(json.dumps(q,indent=2)+'\n');last_snapshot[0]=now
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
streams=root/'retained-component-streams';streams.mkdir(exist_ok=False)
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
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tests.test_nanolang_translator_representation_facts','tests.test_nanolang_translator_representation_shapes','tests.test_nanolang_translator_representation_stack','tests.test_nanolang_translator_shape_graph','tests.test_nanolang_translator_shape_solver','tests.test_nanolang_translator_representation_joins'))
 assert not unittest.defaultTestLoader.errors,unittest.defaultTestLoader.errors
 assert suite.countTestCases()==32,suite.countTestCases()
 q['discovered_methods']=suite.countTestCases()
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 q.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),functional_gate_passed=result.wasSuccessful() and not result.skipped)
except BaseException as e:q['exception']=repr(e);(root/'retained-component-exception.log').write_text(traceback.format_exc())
finally:
 q.update(seconds=time.monotonic()-start,source_after={n:after_sha(tree/n) for n in names},tools_after={n:after_sha(n) for n in tools},images_after={str(n):after_sha(n) for n in images})
 q['qualified_sources_after']={n:after_sha(n) for n in prerequisite['sources_after']};q['qualified_tools_after']={n:after_sha(n) for n in prerequisite['tools_after']};q['qualified_guards_match']=q['qualified_sources_after']==prerequisite['sources_after'] and q['qualified_tools_after']==prerequisite['tools_after'];q['all_guards_match']=all(q[k+'_before']==q[k+'_after'] for k in ('source','tools','images')) and q['qualified_guards_match'];q['terminal']=True;save()
sys.exit(0 if q.get('functional_gate_passed') and q['all_guards_match'] else 1)
