import pathlib,shlex,subprocess,json,hashlib,os,time
root=pathlib.Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=pathlib.Path('/private/tmp/nanolang-union-halt-baseline-20261022');old=pathlib.Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-union-field-preparation-wwnmnogu')
cmd=shlex.split((old/'linked-harness-build-command.txt').read_text());cmd[cmd.index('tests/nanoisa/test_union_field_preparation.c')]=str(out/'baseline.c');cmd.remove('tests/nanoisa/record_array_alloc.c');cmd[-1]=str(out/'baseline')
pins={x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in cmd if pathlib.Path(x).is_file()};(out/'pins-before.json').write_text(json.dumps(pins,indent=2));(out/'build.argv.json').write_text(json.dumps(cmd,indent=2))
def run(name,args,env=None):
 t=time.monotonic();p=subprocess.run(args,cwd=root,env=env,stdout=(out/(name+'.stdout')).open('w'),stderr=(out/(name+'.stderr')).open('w'),timeout=120);(out/(name+'.json')).write_text(json.dumps({'code':p.returncode,'seconds':time.monotonic()-t}));print(name,p.returncode,flush=True);return p.returncode
if run('build',cmd):raise SystemExit(1)
env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1';code=run('baseline',[str(out/'baseline'),str(out)],env)
(out/'pins-after.json').write_text(json.dumps({x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in pins},indent=2));raise SystemExit(code)
