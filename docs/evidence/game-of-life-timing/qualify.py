import sys,subprocess,pathlib,json,hashlib,os,time,signal
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert pin=='69208feaca71fadd9abceacb3c9b1d3a179ee422'
paths=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines();before={p:sha(root/p) for p in paths if (root/p).is_file()};save('source-before.json',before)
reports=[]
for label,argv,limit in [('build',['make','-j2','bin/nanoc_c'],1200),('regressions',['make','test-examples-regressions'],180),('retained-example',[str(root/'bin/nanoc_c'),'examples/graphics/sdl_game_of_life.nano','-o',str(out/'game-of-life'),'--keep-c'],20)]:
 start=time.monotonic();state={'argv':argv,'limit':limit,'timeout':False}
 with (out/(label+'.log')).open('wb') as log:
  proc=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:state['returncode']=proc.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   state['timeout']=True;os.killpg(proc.pid,signal.SIGKILL);state['returncode']=proc.wait()
 state['seconds']=time.monotonic()-start;state['log_sha256']=sha(out/(label+'.log'));save(label+'-status.json',state);reports.append(state);print(label,state['returncode'],flush=True)
 if state['returncode'] or state['timeout']:break
after={p:sha(root/p) for p in paths if (root/p).is_file()};save('source-after.json',after);assert before==after
save('result.json',{'head':pin,'passed':len(reports)==3 and all(x['returncode']==0 and not x['timeout'] for x in reports),'limits':'Cseed strict compilation and selected shadows only; graphical window not run; original regression script deletes its temporary products'})
save('products.json',{str(p):sha(p) for p in [root/'bin/nanoc_c',*out.glob('game-of-life*')] if p.is_file()})
