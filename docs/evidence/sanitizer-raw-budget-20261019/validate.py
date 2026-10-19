from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('docs/evidence/sanitizer-raw-budget-20261019');root.mkdir(exist_ok=True)
src=Path('/private/tmp/nanolang-sanitizer-stage1-12201640931/obj/bootstrap-nanoisa/run-hl9ute_5/manifest.json');shutil.copy2(src,root/'failed-manifest.json')
shutil.copy2('/private/tmp/nanolang-sanitizer-budget-checkpoint.log',root/'checkpoint-tests.log')
log=Path('/private/tmp/nanolang-sanitizer-stage1-118094674820.log')
lines=log.read_text().splitlines();selected=[i for i,l in enumerate(lines) if '7200' in l or 'TimeoutExpired' in l or 'timed out' in l];keep=set()
for i in selected:keep.update(range(max(0,i-4),min(len(lines),i+5)))
(root/'failure-excerpt.log').write_text('\n'.join(lines[i] for i in sorted(keep))+'\n')
def yaml(s):
 return json.loads(subprocess.check_output(['ruby','-rjson','-ryaml','-e','puts JSON.generate(YAML.load(STDIN.read))'],input=s,text=True))
workflow=Path('.github/workflows/ci.yml');new=yaml(workflow.read_text());old=yaml(subprocess.check_output(['git','show','HEAD:.github/workflows/ci.yml'],text=True));changes={}
for name in ['sanitizers-stage1','sanitizers']:
 a=old['jobs'][name];b=new['jobs'][name]
 assert b['env']['BOOTSTRAP_NANOISA_TIMEOUT']=='10800'
 for key in a['env']:
  if key!='BOOTSTRAP_NANOISA_TIMEOUT':assert a['env'][key]==b['env'][key],key
 assert a['needs']==b['needs'] and a['if']==b['if']
 assert len(a['steps'])==len(b['steps'])
 for x,y in zip(a['steps'],b['steps']):
  assert {k:v for k,v in x.items() if k!='timeout-minutes'}=={k:v for k,v in y.items() if k!='timeout-minutes'}
 changes[name]={'job_minutes':b['timeout-minutes'],'generation_step_minutes':210,'raw_command_seconds':10800}
assert new['jobs']['sanitizers-stage1']['env']==new['jobs']['sanitizers']['env']
for name in new['jobs']:
 if name not in changes:assert new['jobs'][name]==old['jobs'][name],name
(root/'workflow-validation.json').write_text(json.dumps({'changes':changes,'unchanged_tests_commands_flags_and_dependencies':True,'identical_checkpoint_environments':True,'other_jobs_unchanged':True},indent=2)+'\n')
files=[src,log,Path('/private/tmp/nanolang-sanitizer-stage1-12201640931.zip'),workflow]
(root/'input-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
(root/'README.md').write_text('''# I retain the instrumented raw-generation deadline failure

I retain hosted run39261195595/job118094674820 at0d8cd446f: seed generation passes in164.403s, seed verification in1.918s, and actual instrumented `nano_vm nanoc_seed.nvm` Stage1 self-generation reaches7200.057s. Its Stage1 log is empty; I have no sanitizer diagnostic or progress trace proving the cause. Artifact12201640931 and the official manifest are pinned here. This is distinct from the earlier combined-job cancellation during native Stage2 compilation.

The earlier5bb06c298 instrumented generations completed in6616.891s and6776.897s (retained in ../sanitizer-stage-timeout-20261019/). Both already use most of7200s. Between those source pins, Nano source bytes grow by1.75%; the new seed and seed-verification timings grow26.2% and30.7%. These are different source/tool/runner executions, not a controlled slowdown measurement. Read-only tracing finds a new absent-context array allocation in the nested/lambda resource-check wrapper, not every top-level function; top-level calls use context directly and the repeated no-resource struct scan predates this change. I do not change compiler semantics based on that unmeasured candidate.

I bound each sanitizer raw-generation command at10800s (180min). Both generation steps allow210min, leaving30min for seed/verification/translation/native/smoke work; the earlier Stage1 nonraw work totals about592s. Stage1 job270min leaves60min for setup/build/pack/upload. Continuation job360min contains210min generation,120min unit allowance and30min combined setup/restore/negative/diagnostic reserve. These ceilings are not guarantees of completion.

I preserve flags, instrumentation, tests, exact checkpoint validation, shadow300s and component compilation1800s. Both transferred-checkpoint environments remain identical. YAML structural comparison confirms only these two jobs' budgets change and all commands/dependencies remain unchanged; ten existing checkpoint tests pass. Local Python lacks PyYAML, so the YAML structural check uses installed Ruby YAML; hosted setup already installs PyYAML. Exact hosted completion and full release acceptance remain open.
''')
print(json.dumps(changes,indent=2))
