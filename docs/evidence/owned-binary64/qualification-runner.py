import pathlib,subprocess,time,json,hashlib,sys
root=pathlib.Path.cwd();pin=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();label=sys.argv[1];out=pathlib.Path('/tmp/nanolang-owned-binary64-'+pin[:8]);out.mkdir(exist_ok=True)
paths=subprocess.check_output(['git','ls-files','src','tests','Makefile.gnu'],text=True).splitlines();before={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if (root/p).is_file()};(out/(label+'-before.json')).write_text(json.dumps(before,sort_keys=True));start=time.monotonic()
with (out/(label+'.log')).open('w') as f:
 try:status=subprocess.run(sys.argv[2:],stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
 except subprocess.TimeoutExpired:status=124
(out/(label+'-status.json')).write_text(json.dumps({'pin':pin,'command':sys.argv[2:],'status':status,'seconds':time.monotonic()-start}));after={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in before};(out/(label+'-after.json')).write_text(json.dumps(after,sort_keys=True));print(out,status,'unchanged',before==after,flush=True)
