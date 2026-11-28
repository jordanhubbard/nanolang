"""I retain my pinned hosted ARM bootstrap and original native gates."""
from pathlib import Path
import subprocess,json,hashlib,os,time,signal
import argparse,platform,sys,re
from ci_original_qualification_policy import require_original_timeout_defaults
p=argparse.ArgumentParser(description='I qualify my pinned original ARM native workloads.')
p.add_argument('--product',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);args=p.parse_args()
require_original_timeout_defaults(os.environ)
assert os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('RUNNER_ARCH')=='ARM64'
assert sys.platform=='linux' and platform.machine()=='aarch64'
assert re.fullmatch('[0-9a-f]{40}',os.environ['C_REPAIR_OBSERVER_REVISION'])
b=args.observations.resolve();b.mkdir(parents=True,exist_ok=False);w=args.product.resolve();d=b/'original';d.mkdir(mode=0o700,exist_ok=False);temporary=d/'tmp';temporary.mkdir(mode=0o700);witness=temporary/'namespace-witness';witness.write_text('I retain my current original-limit qualification namespace.\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip();assert head=='ddd3b5a4ff35678d1906c07c08fc6bbc39b041de'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=w)
names=subprocess.check_output(['git','ls-files'],cwd=w,text=True).splitlines();paths=[w/p for p in names if not p.startswith('docs/') and (w/p).is_file()];tools=[Path(n).resolve() for n in ['/usr/bin/cc','/usr/bin/as','/usr/bin/ld','/usr/bin/python3','/usr/bin/make']]+[witness,Path(__file__).resolve(),Path(__file__).with_name('ci_original_qualification_policy.py').resolve()]
def guards():return {'sources':{str(p.relative_to(w)):sha(p) for p in paths},'tools_and_witness':{str(p):sha(p) if p.exists() else None for p in tools}}
env={**os.environ,'TMPDIR':str(temporary)}
metadata={'qualification':True,'whole_release':False,'pid':os.getpid(),'source_head':head,'TMPDIR':str(temporary),'started_epoch':time.time(),'repair_component':'ddd3b5a4ff35678d1906c07c08fc6bbc39b041de','original_prior_arm_attempt':'I retain sourcef8 ARM17/36/12/20 PASS separately and require every original workload again at the changed directional-flow candidate ddd3b5a4.','hosted_execution_admission':{'run_id':os.environ['GITHUB_RUN_ID'],'job':os.environ['GITHUB_JOB'],'runner_name':os.environ['RUNNER_NAME'],'architecture':platform.machine(),'observer_revision':os.environ['C_REPAIR_OBSERVER_REVISION'],'observer_sha256':sha(Path(__file__))}};(d/'admission.json').write_text(json.dumps(metadata,indent=2)+'\n')
def run(label,argv,limit,extra_env=None):
 out=d/label;out.mkdir();before=guards();started=time.monotonic()
 with (out/'stdout').open('wb') as stdout,(out/'stderr').open('wb') as stderr:
  child=subprocess.Popen(argv,cwd=w,env={**env,**(extra_env or {})},stdout=stdout,stderr=stderr,start_new_session=True);timed=False
  (out/'process.json').write_text(json.dumps({'pid':child.pid,'argv':argv,'started_epoch':time.time()},indent=2)+'\n')
  try:child.wait(timeout=limit)
  except subprocess.TimeoutExpired:timed=True;os.killpg(child.pid,signal.SIGKILL);child.wait()
 after=guards();record={'qualification':True,'whole_release':False,'source_head':head,'argv':argv,'exit':child.returncode,'command_timeout':timed,'outer_observation_seconds':limit,'seconds':time.monotonic()-started,'before':before,'after':after,'guards_match':before==after,'TMPDIR':str(temporary)};(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(label,child.returncode,round(record['seconds'],3),flush=True);assert before==after;return child.returncode
make=['make','-f','Makefile.gnu','-j1','CC=/usr/bin/cc']
status=run('shape-flow-controls',['/usr/bin/python3','-m','unittest','-v','tests.test_shape_flow_scaling'],600,{'NANO_SHAPE_FLOW_ARTIFACT_DIR':str(d/'shape-flow-products')})
if status:raise SystemExit(status)
status=run('original-shapes',make+['test-nvm2c-shapes'],300)
if status:raise SystemExit(status)
__import__('shutil').copy2(w/'obj/test_nvm2c_shape',d/'original-shapes/test_nvm2c_shape')
status=run('bootstrap',make+['bootstrap3'],8000)
if status:raise SystemExit(status)
receipt=json.loads((w/'bin/nanoc_bootstrap.json').read_text());assert receipt['phase']=='stage2' and receipt['raw_stage1_stage2_equal'] and len(receipt['steps'])==17 and all(s['exit_code']==0 for s in receipt['steps']);assert receipt['stage_timeout_seconds']==1800 and receipt['shadow_timeout_seconds']==30;assert (w/'bin/nanoc_stage1.nvm').read_bytes()==(w/'bin/nanoc_stage2.nvm').read_bytes();(d/'bootstrap-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
status=run('native-suites',make+['test-native-shadows','test-native-shadow-emitter'],1800)
if status:raise SystemExit(status)
text=(d/'native-suites/stderr').read_text();assert 'Ran 36 tests' in text and 'Ran 12 tests' in text and 'skipped=' not in text
raise SystemExit(run('line-regression',make+['test-gc-registry-working-set','test-gc-registry-working-set-sanitizers','test-gc-shutdown-sanitizers','test-eval-caller-root-handoff','test-eval-string-split','test-compiler-line-splitting','test-native-root-insertion'],600,{'NANOLANG_TRANSLATOR_SOURCE_COMPILER':str(w/'bin/nanoc_stage2')}))
