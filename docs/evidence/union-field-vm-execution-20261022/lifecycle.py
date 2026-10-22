import pathlib,subprocess,json,hashlib,shlex,os,time
root=pathlib.Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=pathlib.Path('/private/tmp/nanolang-union-vm-execution-20261022/lifecycle');out.mkdir(exist_ok=True);e=root/'docs/evidence/union-field-vm-preparation-20261022'
status={}
def run(name,args,env=None):
 (out/(name+'.argv.json')).write_text(json.dumps(args,indent=2));t=time.monotonic()
 p=subprocess.run(args,cwd=root,env=env,stdout=(out/(name+'.stdout')).open('w'),stderr=(out/(name+'.stderr')).open('w'),timeout=180)
 status[name]={'code':p.returncode,'seconds':time.monotonic()-t};(out/'first-status.json').write_text(json.dumps(status,indent=2));print(name,status[name],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
old=pathlib.Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-union-field-preparation-wwnmnogu')
cmd=shlex.split((old/'linked-harness-build-command.txt').read_text());cmd[cmd.index('tests/nanoisa/test_union_field_preparation.c')]='tests/nanoisa/test_union_field_vm.c';cmd.remove('tests/nanoisa/record_array_alloc.c');cmd.remove(str(old/'linked-vm.o'));cmd.insert(1,'-DUNION_EXECUTION_WHITEBOX');cmd[-1]=str(out/'first')
files=[x for x in cmd if pathlib.Path(x).is_file()]+['src/nanovm/vm.c','src/nanovm/union_field_vm_context.inc','src/nanovm/union_field_vm_run.inc','src/nanovm/union_field_runtime_private.h']
pins={x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in files};(out/'pins-before.json').write_text(json.dumps(pins,indent=2))
run('first-build',cmd)
mods=shlex.split(json.loads((e/'target-env.json').read_text())['UNION_PREPARATION_MODULES'])
env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1';run('first-six',[str(out/'first'),*mods],env)
(out/'pins-after.json').write_text(json.dumps({x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in files},indent=2))
