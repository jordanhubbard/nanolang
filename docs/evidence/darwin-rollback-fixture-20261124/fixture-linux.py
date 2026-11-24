from pathlib import Path
import json,subprocess,shlex,tempfile,os,time,hashlib
r=Path.cwd();j=r/'.test_output/darwin-current-source-20261124';base=json.loads((r/'.test_output/fresh-provider-builds-20261124/repaired/result.json').read_text());w=Path(tempfile.mkdtemp(prefix='nano-1246-',dir='/var/tmp'));q=dict(work=str(w),profiles=[],source_sha256=hashlib.sha256((r/'tests/test_module_rejection_rollback.c').read_bytes()).hexdigest());env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
for profile in base['profiles']:
 tree=Path(profile['tree']);mode=profile['mode'];line=next(x for x in (tree/'focused.stdout').read_text().splitlines() if '-o obj/test_module_rejection_rollback ' in x);args=shlex.split(line);args[args.index('-o')+1]=str(w/mode);args[args.index('tests/test_module_rejection_rollback.c')]=str(r/'tests/test_module_rejection_rollback.c');row=dict(mode=mode,commands=[]);q['profiles'].append(row)
 for label,argv in [('build',args),('run',[str(w/mode)])]:
  start=time.monotonic()
  with (w/(mode+'-'+label+'.stdout')).open('wb') as out,(w/(mode+'-'+label+'.stderr')).open('wb') as err:p=subprocess.run(argv,cwd=tree,env=env,stdout=out,stderr=err,timeout=180)
  row['commands'].append(dict(label=label,args=argv,exit=p.returncode,seconds=time.monotonic()-start));assert p.returncode==0
 row['binary_sha256']=hashlib.sha256((w/mode).read_bytes()).hexdigest()
q['terminal']=True;(j/'fixture-linux-result.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
