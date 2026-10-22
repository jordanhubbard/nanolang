from pathlib import Path
import os,sys,json,hashlib,time,subprocess,shlex
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
w=Path(__file__).parent;e=json.loads((r/'docs/evidence/map-declarations-20261021/environment.json').read_text());env=os.environ.copy()
for k in ['CC','CFLAGS','OBJECTS','LDFLAGS']:env['RECORD_ARRAY_'+k]=e['DECLARATION_'+k]
(w/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('RECORD_ARRAY_')},indent=2))
env['RECORD_MAP_MODULES']=' '.join(str(Path('/private/tmp/nanolang-map-baseline-20261021')/('string-int-'+p+'.nvm')) for p in ['seed','c-hosted-nano','stage1','stage2'])
root=Path('/Users/jkh/Src/nanolang-51-harness')
recipe='include Makefile.gnu\n.PHONY: image-link\nimage-link:\n\t@echo $(sort $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS))\n\t@echo $(LDFLAGS)\n'
sel=subprocess.run(['make','--no-print-directory','-s','-f','-','image-link'],input=recipe,text=True,cwd=root,capture_output=True,check=True).stdout.splitlines()
for k in ['CC','CFLAGS']:env['RECORD_ARRAY_VM_'+k]=env['RECORD_ARRAY_'+k]
env['RECORD_ARRAY_VM_OBJECTS']=' '.join(str(root/x) for x in shlex.split(sel[0]))
env['RECORD_ARRAY_VM_LDFLAGS']=sel[1]
(w/'vm-environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('RECORD_ARRAY_VM_')},indent=2))
args=[sys.executable,'-m','unittest','-f','-v','tests.test_record_array_vm'];t=time.monotonic();p=run_captured(args,cwd=r,env=env,timeout=900)
(w/'vm-corrected.stdout').write_bytes(p.stdout);(w/'vm-corrected.stderr').write_bytes(p.stderr);(w/'vm-corrected.json').write_text(json.dumps({'argv':args,'exit':p.returncode,'seconds':time.monotonic()-t},indent=2));print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
