from pathlib import Path
import os,subprocess,time,json
work=Path('/work/profile-corrected');work.mkdir(exist_ok=True)
env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0','NANO_SHADOW_TIMEOUT_SECONDS':'60','NANO_MODULE_PATH':'modules','NANO_BUILD_CACHE':'/work/profile-module-cache'}
started=time.monotonic()
with (work/'shadow-sampled.log').open('w') as log:
 run=subprocess.Popen(['bin/nanoc_pr522_profile','src_nano/nanoc_v06.nano','--verbose','-o','/work/profile-stage1-sampled'],cwd='/work',env=env,stdout=log,stderr=subprocess.STDOUT)
 child=None
 while run.poll() is None and time.monotonic()-started<300:
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    stat=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().split(b'\0')
    if int(stat[3])==run.pid and cmd and cmd[0].endswith(b'nanoc_pr522_profile'):child=int(p.name);break
   except (OSError,ValueError,IndexError):pass
  if child:break
  time.sleep(.2)
 print('shadow child',child,'after',round(time.monotonic()-started,2),flush=True)
 if child:
  for i in range(8):
   if run.poll() is not None or not Path(f'/proc/{child}').exists():break
   result=subprocess.run(['gdb','-q','-batch','-ex','set pagination off','-ex',f'attach {child}','-ex','bt 160','-ex','detach'],capture_output=True,text=True,timeout=10,env={**env,'DEBUGINFOD_URLS':''})
   (work/f'stack-{i:02}.txt').write_text(f'elapsed={time.monotonic()-started:.3f}\n'+result.stdout+result.stderr)
   print('sample',i,'status',result.returncode,flush=True)
   time.sleep(1)
 status=run.wait(timeout=300)
(work/'sample-result.json').write_text(json.dumps({'returncode':status,'elapsed':time.monotonic()-started,'child':child})+'\n')
print('compiler status',status,flush=True)
