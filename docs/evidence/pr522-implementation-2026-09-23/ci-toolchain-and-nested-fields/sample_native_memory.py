from pathlib import Path
import os,subprocess,time,json
root=Path('/work/ci-toolchain/native-memory');root.mkdir(parents=True,exist_ok=True)
env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:detect_stack_use_after_return=1', 'NANO_SHADOW_TIMEOUT_SECONDS':'60', 'NANO_CC':'clang','CC':'clang','NANO_LDFLAGS':'-fsanitize=address,undefined --rtlib=compiler-rt','DEBUGINFOD_URLS':''}
started=time.monotonic();samples=[]
with (root/'compiler.log').open('w') as log:
 run=subprocess.Popen(['bin/nanoc_stage1','src_nano/nanoc_v06.nano','-o','/work/ci-toolchain/diagnostic-stage2'],cwd='/work',env=env,stdout=log,stderr=subprocess.STDOUT)
 while run.poll() is None and time.monotonic()-started<120:
  try:
   status=Path(f'/proc/{run.pid}/status').read_text()
   rss=int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))
  except (OSError,StopIteration):break
  if rss>= (len(samples)+1)*500000:
   command=['gdb','-q','-batch','-ex','set pagination off','-ex',f'attach {run.pid}','-ex','bt 16','-ex','break gc_alloc','-ex','continue','-ex','bt 16','-ex','call (void)gc_print_stats()','-ex','call (int)fflush(0)','-ex','detach']
   sample=subprocess.run(command,capture_output=True,text=True,timeout=20,env=env)
   name=f'stack-{len(samples):02}.txt';(root/name).write_text(sample.stdout+sample.stderr)
   samples.append({'elapsed':time.monotonic()-started,'rss_kb':rss,'gdb_exit':sample.returncode,'file':name})
   print(samples[-1],flush=True)
  if len(samples)>=3 or rss>2500000:break
  time.sleep(1)
 if run.poll() is None:
  run.terminate()
 code=run.wait(timeout=20)
(root/'result.json').write_text(json.dumps({'returncode':code,'diagnostic_termination':True,'samples':samples},indent=2)+'\n')
