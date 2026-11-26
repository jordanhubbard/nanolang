from pathlib import Path
import os,json,subprocess,hashlib
j=Path(__file__).resolve().parent;s=json.loads((j/'result.json').read_bytes());assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=dict(terminal=False,free_bytes=os.statvfs(j).f_bavail*os.statvfs(j).f_frsize,loadavg=os.getloadavg(),affinity=sorted(os.sched_getaffinity(0)),other_jobs=[])
try:
 prior=Path('/var/tmp/nanolang-affine-typed-five-20261125-udz2o6c_/cohort-result.json');assert sha(prior)==s['foundation_evidence']['result_sha256']
 assert s['foundation_evidence']['producer_sha256']==s['producer_sha256']
 assert sha('/var/tmp/nanolang-path-copy-global-array-five-20261125-jsj5x9iu/cohort-result.json')==s['global_array_evidence']['result_sha256']
 assert s['global_array_evidence']['producer_sha256']==s['producer_sha256']
 assert sha(j/'cohort_journal.py')==s['observer_source_sha256']
 r['filesystem']=subprocess.check_output(['stat','-f','-c','%T',str(j)],text=True).strip();assert r['filesystem']!='nfs' and r['free_bytes']>15*1024**3
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:name=Path(os.readlink(p/'exe')).name;cwd=os.readlink(p/'cwd')
  except OSError:continue
  if name in ['nano_vm','nano_virt','nvm2c'] or name.startswith('nanoc'):r['other_jobs'].append(dict(pid=int(p.name),image=name,cwd=cwd))
 assert not r['other_jobs'],r['other_jobs']
 assert s['source_base']=='67d99e5121521428740d385d10071b9679e66013' and not (j/'source/bin/nanoc_stage2').exists() and not (j/'source/bin/nanoc_bootstrap.json').exists() and not list((j/'source').glob('.bootstrap*.built'))
 assert sha(s['producer'])==s['producer_sha256']
 for group in ['source_after','images_after']:
  assert all(sha(j/'source'/n)==h for n,h in s[group].items())
 for group in ['tools_after','hosts_after']:assert all(sha(n)==h for n,h in s[group].items())
 assert all(v['share']>=.75 for v in s['cpu_rounds'])
 r['output_directories']={name:(j/name).is_dir() for name in ['products','tmp']};assert all(r['output_directories'].values());r['accepted']=True
finally:r['terminal']=True;(j/'admission-result.json').write_text(json.dumps(r,indent=2)+'\n')
