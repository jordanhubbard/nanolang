import hashlib,json,os,pathlib,signal,subprocess,time,sys
root=pathlib.Path('/home/jkh/Src/nanolang-shared-match-ordering-controls')
out=pathlib.Path('/tmp/nanolang-match-855-linux-first');out.mkdir(exist_ok=False)
def save(n,v): (out/n).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sources():
 names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
 return {n:sha(root/n) for n in names if n and (root/n).is_file()}
def providers():return {str(p.relative_to(root)):sha(p) for p in sorted((root/'obj').rglob('*.o'))}|{n:sha(root/n) for n in ['bin/nanoc_c','obj/test_interpreter_ffi_native.so']}
def run(name,args,seconds):
 start=time.monotonic()
 with (out/(name+'.log')).open('wb') as f:
  p=subprocess.Popen(args,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=seconds)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=124
 save(name+'.json',{'argv':args,'exit_code':code,'seconds':time.monotonic()-start})
 print(name,code,flush=True)
 if code:raise RuntimeError(name+' failed')
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert pin.startswith('5e063cad1')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root)
original=sources();save('source-before.json',original)
toolpaths=['/usr/bin/gcc','/usr/bin/make','/usr/bin/python3','/usr/bin/as','/usr/bin/ld',subprocess.check_output(['/usr/bin/gcc','-print-prog-name=cc1'],text=True).strip()]
tools={p:{'path':str(pathlib.Path(p).resolve()),'sha256':sha(p)} for p in toolpaths};save('tools-before.json',tools)
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
setup=out/'setup.mk'
setup.write_text('include Makefile.gnu\n.PHONY: root-match-build\nroot-match-build: stage1 $(OBJ_DIR)/test_interpreter_ffi_native.so $(OBJ_DIR)/eval_io_faults.o $(OBJ_DIR)/eval_clock_test.o\n\t$(CC) $(CFLAGS) -o '+str(out/'test_eval')+' tests/test_eval.c $(filter-out $(OBJ_DIR)/eval.o $(OBJ_DIR)/eval/eval_io.o,$(COMMON_OBJECTS)) $(OBJ_DIR)/eval_clock_test.o $(OBJ_DIR)/eval_io_faults.o $(RUNTIME_OBJECTS) $(LDFLAGS)\n')
status={'pin':pin,'success':False}
try:
 run('build',['/usr/bin/make','-j2','-f',str(setup),'CC=/usr/bin/gcc','root-match-build'],600)
 before=providers();save('providers-before.json',before);save('eval-binary.json',{'path':str(out/'test_eval'),'sha256':sha(out/'test_eval')})
 run('evaluator',[str(out/'test_eval')],120)
 run('totality',['/usr/bin/python3','-m','unittest','-v','tests.test_cseed_match_totality'],600)
 after=providers();save('providers-after.json',after);assert before==after
 status['success']=True
except Exception as e:status['error']=str(e)
finally:
 current=sources();save('source-after.json',current);status['sources_unchanged']=original==current
 now={p:{'path':str(pathlib.Path(p).resolve()),'sha256':sha(p)} for p in toolpaths};save('tools-after.json',now);status['tools_unchanged']=tools==now
 status['head_unchanged']=pin==subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 save('status.json',status);print(json.dumps(status),flush=True)
sys.exit(0 if status['success'] else 1)
