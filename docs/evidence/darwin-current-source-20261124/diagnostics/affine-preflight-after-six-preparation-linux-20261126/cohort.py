from pathlib import Path
import os,sys,json,time,hashlib,subprocess,signal,traceback,unittest,tempfile,shlex,shutil
from cohort_journal import CohortJournal
j=Path(__file__).resolve().parent;root=j/'source';os.chdir(root);sys.path.insert(0,str(root))
s=json.loads((j/'result.json').read_bytes());admission=json.loads((j/'admission-result.json').read_bytes())
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and admission['accepted']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANOLANG_TRANSLATOR_SOURCE_MODULE']:assert name not in os.environ,name
os.environ.update(TMPDIR=str(j/'tmp'),NANOLANG_TRANSLATOR_SOURCE_COMPILER=s['producer'],NANOLANG_AFFINE_FRAME_REFERENCE=str(j/'products/reference'))
tempfile.tempdir=str(j/'tmp')
from tests import nanolang_translator_routes as routes
from tests import test_nanolang_translator_affine_preflight as preflight_ops
q=dict(terminal=False,component_only=True,whole_release=False,commands=[],source_pin=s['source_base'],producer_source_pin=s['bootstrap_source_pin'],producer_sha256=s['producer_sha256'],source_before=s['source_after'],images_before=s['images_after'],tools_before=s['tools_after'],hosts_before=s['hosts_after'],providers={},working_overlay=True,overlay_sha256=s['overlays'],query_evidence=s['query_evidence'],global_array_evidence=s['global_array_evidence'],foundation_evidence=s['foundation_evidence'],original_methods=0,new_methods=1,original_frame_cases=1001,new_cases=dict(preflight=228),ordinary_seconds=10)
compiler_argv=shlex.split(os.environ.get('CC','cc'));assert len(compiler_argv)==1;compiler_path=shutil.which(compiler_argv[0]);assert compiler_path and Path(compiler_path).samefile('/usr/bin/cc');assert sha(compiler_path)==s['tools_after']['/usr/bin/cc'];q['compiler_binding']=dict(argv=compiler_argv,resolved=compiler_path,sha256=sha(compiler_path),environment={n:os.environ.get(n) for n in ['CC','PATH','CFLAGS','LDFLAGS']})
q['six_evidence']=s['six_evidence'];q['report_fix_commit']=s['report_fix_commit'];q['report_fix_manifest_sha256']=s['report_fix_manifest_sha256'];q['cache_fix_commit']=s['cache_fix_commit'];q['cache_key_seed_manifest_sha256']=s['cache_key_seed_manifest_sha256'];q['corrected_three_evidence']=s['corrected_three_evidence'];q['cache_full_evidence']=s['cache_full_evidence']
q['observer_before']={str(j/n):sha(j/n) for n in ['cohort.py','cohort_journal.py','setup.py','admit.py']}
assert q['observer_before'][str(j/'cohort_journal.py')]==s['observer_source_sha256']
journal=CohortJournal(j,q)
def save(*,metadata=False):journal.save(metadata=metadata)
def observe_module(path):
 key=str(path)
 if key in q['providers']:return
 args=[str(root/'bin/nanoisa'),'dump',key];r=subprocess.run(args,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
 index=len(q['providers']);(j/f'provider-{index}.stdout').write_bytes(r.stdout);(j/f'provider-{index}.stderr').write_bytes(r.stderr);assert r.returncode==0
 bindings=[shlex.split(line) for line in r.stdout.decode().splitlines() if line.startswith('.import ')]
 libraries=sorted({row[1] for row in bindings if row[1] and Path(row[1]).is_absolute()})
 q['providers'][key]=dict(module_sha256=sha(path),bindings=bindings,libraries_before={n:sha(n) for n in libraries},read_only=True,before_translation=True);save(metadata=True)
def checked(command,*,build=False,timeout=60):
 argv=list(map(str,command));budget=600 if build else timeout;number=len(q['commands'])+1
 row=journal.row(number=number,argv=argv,build=build,timeout=budget);q['commands'].append(row);save()
 if Path(argv[0]).name=='nvm2c':observe_module(Path(argv[1]))
 clock=time.monotonic()
 with (j/f'command-{number:04d}.stdout').open('wb') as out,(j/f'command-{number:04d}.stderr').open('wb') as err:
  p=subprocess.Popen(argv,cwd=root,stdout=out,stderr=err,start_new_session=True);row['pid']=p.pid;save()
  try:row['exit']=p.wait(timeout=budget)
  except BaseException as exc:
   row['interrupted_or_timeout']=repr(exc)
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   row['exit']=p.wait(timeout=10);raise
  finally:row['seconds']=time.monotonic()-clock;save()
 assert row['exit']==0,(row,(j/f'command-{number:04d}.stderr').read_text()[-2000:])
 return (j/f'command-{number:04d}.stdout').read_text()
routes.checked=checked
try:
 # I keep the audited strict flags, immutable actual-C objects and state diagnostic TU.
 template=json.loads(Path('/var/tmp/nanolang-affine-preflight-c-audit-20261125-ua02nodh/result.json').read_bytes())['commands'][0]['argv']
 q['object_inputs']={v:sha(v) for v in template if v.endswith(('.o','.a'))};q['reference_argv']={};save(metadata=True)
 argv=[str(root/'tests/nvm2c_affine_preflight_reference.c') if v.endswith('/tests/nvm2c_affine_preflight_reference.c') else str(root/'tests/nvm2c_affine_owners_state.c') if v.endswith('/tests/nvm2c_affine_owners_state.c') else v for v in template]
 argv[argv.index('-o')+1]=str(j/'products/preflight-reference')
 q['reference_argv']['preflight']=argv;save(metadata=True);checked(argv,timeout=60)
 os.environ['NANOLANG_AFFINE_PREFLIGHT_REFERENCE']=str(j/'products/preflight-reference')
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(preflight_ops)]);assert suite.countTestCases()==1
 with (j/'permanent-test.stderr').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 q['tests']=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),passed=result.wasSuccessful())
 assert result.testsRun==1 and result.wasSuccessful() and not result.skipped;q['accepted_component']=True
except BaseException as exc:q['exception']=repr(exc);(j/'exception.log').write_text(traceback.format_exc());raise
finally:
 for group in ['source','images','tools','hosts']:q[group+'_after']={n:sha(root/n if group in ['source','images'] else n) for n in q[group+'_before']}
 q['objects_after']={n:sha(n) for n in q.get('object_inputs',{})};q['producer_after']=sha(s['producer'])
 for module,record in q['providers'].items():record['module_after']=sha(module);record['libraries_after']={n:sha(n) for n in record['libraries_before']}
 q['all_guards_match']=all(q[n+'_before']==q[n+'_after'] for n in ['source','images','tools','hosts']) and q.get('object_inputs',{})==q['objects_after'] and q['producer_after']==s['producer_sha256'] and all(v['module_after']==v['module_sha256'] and v['libraries_before']==v['libraries_after'] for v in q['providers'].values())
 q['observer_after']={n:sha(n) for n in q['observer_before']};assert q['observer_before']==q['observer_after']
 q['terminal']=True;save()
if not q.get('accepted_component') or not q['all_guards_match']:raise SystemExit(1)
