import pathlib,subprocess,os,json,hashlib,time,shutil,sys
r=pathlib.Path.cwd();d=pathlib.Path(sys.argv[1]);d.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files=subprocess.check_output(['git','ls-files','src','src_nano','tests','scripts','modules','Makefile','GNUmakefile','Makefile.gnu'],text=True).splitlines()
def inv():return {p:sha(p) for p in files if pathlib.Path(p).is_file()}
cc='/opt/homebrew/opt/llvm/bin/clang' if sys.platform=='darwin' else '/usr/bin/gcc'
paths={n:str(pathlib.Path(shutil.which(n)).resolve()) for n in ['make','python3',cc]}
tools={n:{'path':p,'sha256':sha(p),'version':subprocess.check_output([p,'--version'],text=True,stderr=subprocess.STDOUT)} for n,p in paths.items()}
before=inv();(d/'before.json').write_text(json.dumps(before,indent=2));(d/'tools.json').write_text(json.dumps(tools,indent=2))
env={**os.environ,'NANO_NATIVE_TEST_CC':cc};cmd=['make','-j8','CC='+cc,'test-nanocore'];start=time.monotonic()
with (d/'gate.log').open('w') as f:
 f.write(str(cmd)+'\n');f.flush();p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
after=inv();(d/'after.json').write_text(json.dumps(after,indent=2));ta={n:sha(p) for n,p in paths.items()};(d/'tools-after.json').write_text(json.dumps(ta,indent=2))
result={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':p.returncode,'seconds':time.monotonic()-start,'source_unchanged':before==after,'tools_unchanged':all(ta[n]==v['sha256'] for n,v in tools.items()),'command':cmd,'native_test_cc':cc,'uname':list(os.uname())};(d/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));sys.exit(p.returncode)
