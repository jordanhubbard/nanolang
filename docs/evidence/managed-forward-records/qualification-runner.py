import json,os,subprocess,hashlib,time,shlex,sys
from pathlib import Path
r=Path.cwd();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
files=subprocess.check_output(['git','ls-files','src','src_nano','tests','scripts','modules','Makefile.gnu'],text=True).splitlines()
files += ['bin/nvm2c','bin/nvm2llvm','bin/nvm2wasm','bin/nanoisa','bin/nano_vm','obj/nanoisa/managed_runtime_ir.h','obj/nanoisa/managed_runtime_ir.json','obj/test_verifier_profiles','obj/managed_record_reentry']
def inventory():return {f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files if (r/f).is_file()}
before=inventory();(out/'before.json').write_text(json.dumps(before,indent=2)+'\n')
dry=subprocess.check_output(['make','-n','test-llvm-managed-records'],text=True)
line=next(x for x in dry.splitlines() if x.startswith('NMA_LINK_OBJECTS='));objects=shlex.split(line)[0].split('=',1)[1]
env={**os.environ,'NMA_LINK_OBJECTS':objects,'NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','NMR_ARTIFACTS':str(out/'artifacts'),'NMS_RUNTIME_CLANG':'clang-18','NMS_RUNTIME_OPT':'opt-18'}
nrp=subprocess.check_output(['make','-n','test-managed-record-plan'],text=True)
env['NRP_LINK_OBJECTS']=shlex.split(next(x for x in nrp.splitlines() if x.startswith('NRP_LINK_OBJECTS=')))[0].split('=',1)[1]
noa=subprocess.check_output(['make','-n','test-ordinary-record-authority'],text=True)
env['NOA_LINK_OBJECTS']=shlex.split(next(x for x in noa.splitlines() if x.startswith('NOA_LINK_OBJECTS=')))[0].split('=',1)[1]
tests=sys.argv[2:];start=time.monotonic()
with (out/'test.log').open('w') as log:
 p=subprocess.run(['python3','-m','unittest','-v',*tests],env=env,stdout=log,stderr=subprocess.STDOUT)
res={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tests':tests,'status':p.returncode,'seconds':time.monotonic()-start}
(out/'result.json').write_text(json.dumps(res,indent=2)+'\n');after=inventory();(out/'after.json').write_text(json.dumps(after,indent=2)+'\n')
print(json.dumps({**res,'identities':len(before),'unchanged':before==after}));sys.exit(p.returncode)
