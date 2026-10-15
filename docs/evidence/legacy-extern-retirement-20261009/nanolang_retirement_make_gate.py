from pathlib import Path
import subprocess,os,json,time,hashlib
root=Path('/private/tmp/nanolang-match-guards-20261009');os.chdir(root)
out=Path('/private/tmp/nanolang-legacy-retirement-make');out.mkdir(exist_ok=True)
paths=subprocess.check_output(['git','ls-files','src','src_nano','modules','std','stdlib'],text=True).splitlines()+['Makefile.gnu','tests/test_nanoisa_extern_declarations.py','tests/nanoisa/fixtures/historical_extern_helpers.nano.txt','tests/nanoisa/fixtures/historical_extern_assertions.nano.txt']
def inv():return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if Path(p).is_file()}
r={'before':inv(),'base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'argv':['make','test-nanoisa-extern-declarations','CC=/opt/homebrew/opt/llvm/bin/clang']}
env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_EXTERN_DECLARATION_ARTIFACTS':str(out/'routes')}
(out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
start=time.time()
with (out/'make.log').open('w') as f:p=subprocess.run(r['argv'],env=env,stdout=f,stderr=subprocess.STDOUT)
r.update(exit=p.returncode,seconds=time.time()-start,after=inv());r['unchanged']=r['before']==r['after']
(out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ('before','after')})
raise SystemExit(p.returncode or (not r['unchanged']))
