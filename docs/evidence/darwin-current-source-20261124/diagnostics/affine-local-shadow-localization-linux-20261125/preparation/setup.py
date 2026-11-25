from pathlib import Path
import os,json,subprocess,hashlib,time,shutil,signal,traceback
j=Path(__file__).resolve().parent;integration=Path('/var/tmp/nanolang-darwin-publication-qkk55gi3');old=Path('/var/tmp/nanolang-hoisted-lists-cold-bootstrap-20261125-ia5_k8qp/source');root=j/'source';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();base='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8';q=dict(terminal=False,source_base=base,working_overlay=False,commands=[],ordinary_seconds=10,new_bootstrap=False,stage_alias=False,whole_release=False)
def save():(j/'result.json').write_text(json.dumps(q,indent=2)+'\n')
def run(label,args,cwd=root,timeout=180,input=None,env=None):
 row=dict(label=label,argv=list(map(str,args)),cwd=str(cwd),timeout=timeout);q['commands'].append(row);save();t=time.monotonic()
 with (j/(label+'.stdout')).open('wb') as out,(j/(label+'.stderr')).open('wb') as err:
  p=subprocess.Popen(row['argv'],cwd=cwd,env=env,stdin=subprocess.PIPE if input is not None else None,stdout=out,stderr=err,start_new_session=True);row['pid']=p.pid;save()
  if input is not None:p.stdin.write(input);p.stdin.close()
  try:row['exit']=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);row.update(exit=p.wait(),timed_out=True)
 row['seconds']=time.monotonic()-t;save();return row['exit']
try:
 proof=json.loads((j/'preflight-result.json').read_bytes());assert proof['terminal'] and proof['accepted'] and proof['source_pin']==base and sha(j/'cohort_journal.py')==proof['observer_helper_sha256']
 q['preflight_evidence']=proof
 q['overlays']={n:sha(integration/n) for n in []};q['tools_before']=json.loads((old.parent/'bootstrap-result.json').read_bytes())['tools_after'];assert all(sha(n)==h for n,h in q['tools_before'].items());q['cpu_rounds']=[]
 for _ in range(2):
  wall=time.monotonic();cpu=time.process_time();v=1
  while time.process_time()-cpu<1:
   for k in range(1024):v=(v*1664525+1013904223)&0xffffffff
  c=time.process_time()-cpu;w=time.monotonic()-wall;q['cpu_rounds'].append(dict(cpu=c,wall=w,share=c/w));assert c/w>=.75
 patterns=subprocess.check_output(['git','sparse-checkout','list'],cwd=old)
 assert run('worktree',['git','worktree','add','--detach','--no-checkout',root,base],integration)==0
 assert run('sparse',['git','sparse-checkout','set','--no-cone','--stdin'],input=patterns)==0
 assert run('checkout',['git','reset','--hard',base])==0
 for n in q['overlays']:
  shutil.copy2(integration/n,root/n);assert sha(root/n)==q['overlays'][n]
 q['images_before']={}
 for n in ['bin/nano_virt','bin/nano_vm','bin/nanoc_c','bin/nvm2c','bin/nanoisa','bin/nano_aot_runtime.o','bin/nano_as_capture.so','modules/index.json']:
  p=root/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/n,p);q['images_before'][n]=sha(p)
 qualified=json.loads((old.parent/'bootstrap-result.json').read_bytes())
 for n,h in q['images_before'].items():
  if n in qualified['images_after']:assert h==qualified['images_after'][n]
 for n,h in qualified['source_after'].items():
  if n.startswith('src/') and n.endswith(('.c','.h','.inc')):assert sha(root/n)==h,n
 q['bootstrap_source_inputs']=qualified['source_after'];q['bootstrap_source_pin']=qualified['source_pin'];q['bootstrap_result_sha256']=sha(old.parent/'bootstrap-result.json');q['bootstrap_receipt_sha256']=sha(old/'bin/nanoc_bootstrap.json');consumer=json.loads((old.parent/'native-gates-result.json').read_bytes());q['native_consumer_accepted']=consumer['accepted'];q['native_consumer_receipt_sha256']=sha(old.parent/'native-gates-result.json');assert qualified['terminal'] and qualified['original_make_pass'] and qualified['exit']==0 and consumer['terminal'];assert isinstance(q['native_consumer_accepted'],bool)
 tracked=sorted(set(subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0'))|set(q['overlays']));q['source_before']={n:sha(root/n) for n in tracked if n and (root/n).is_file() and ((n.endswith(('.nano','.c','.h','.inc','.json','.py')) and not n.startswith('docs/evidence/')) or n in qualified['source_after'])};save()
 receipt=json.loads((old/'bin/nanoc_bootstrap.json').read_bytes());assert receipt['phase']=='stage2' and receipt['raw_stage1_stage2_equal'];subject=root/'bin/nanoc_emitter_phase';shutil.copy2(old/'bin/nanoc_stage2',subject);q['producer']=str(subject);q['producer_sha256']=sha(subject);assert q['producer_sha256']==receipt['artifacts']['nanoc_stage2'];q['hosts_before']=receipt['hosts'];assert all(sha(n)==h for n,h in q['hosts_before'].items());q['cache_state']='new exact source root; actual7b9f08 bootstrap Stage2 and immutable host closure retained; native consumer outcome recorded separately; no historical timing equivalence';q['producer_kind']='original-Make-bootstrap-own-Stage2';q['producer_origin_evidence']=os.environ['NANOLANG_BOOTSTRAP_EVIDENCE_COMMIT'];assert len(q['producer_origin_evidence'])==40 and all(c in '0123456789abcdef' for c in q['producer_origin_evidence']);q['producer_source_pin']=q['bootstrap_source_pin'];q['admitted_setup']=True
 q['source_after']={n:sha(root/n) for n in q['source_before']};q['images_after']={n:sha(root/n) for n in q['images_before']};q['tools_after']={n:sha(n) for n in q['tools_before']};q['hosts_after']={n:sha(n) for n in q['hosts_before']};assert q['hosts_after']==q['hosts_before'];assert q['source_before']==q['source_after'] and q['images_before']==q['images_after'] and q['tools_before']==q['tools_after'];q['all_guards_match']=True
except BaseException as e:q['exception']=repr(e);(j/'exception.log').write_text(traceback.format_exc());raise
finally:q['terminal']=True;save()
