from pathlib import Path
import os,subprocess,time,json,hashlib
root=Path('/private/tmp/nanolang-match-guards-20261009');os.chdir(root)
out=Path('/private/tmp/nanolang-selfhost-union-fresh-installed');out.mkdir(exist_ok=True)
boot=json.loads(Path('bin/nanoc_bootstrap.json').read_text())
assert Path(boot['work']).name=='run-_5jehk9f' and boot['raw_stage1_stage2_equal']
assert all(s.get('exit_code')==0 for s in boot['steps'])
prior=json.loads(Path('/private/tmp/nanolang-selfhost-union-supplement/manifest.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={p:sha(p) for p in prior['after']}
assert before==prior['after']
args=['python3','-m','unittest','-v','tests.test_owned_union_installed','tests.test_owned_global_installed','tests.test_owned_union_boundaries.UnionBoundaryStage1','tests.test_owned_union_boundaries.UnionBoundaryStage2']
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_SELECTED_SOURCE_ARTIFACTS':str(out/'installed')}
start=time.time()
with (out/'tests.log').open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
report={'argv':args,'exit':r.returncode,'seconds':time.time()-start,'before':before,'unchanged':all(sha(p)==v for p,v in before.items()),'bootstrap':boot['work']}
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='before'})
raise SystemExit(r.returncode or (not report['unchanged']))
