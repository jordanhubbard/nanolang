"""I verify a guarded unstarted four-leaf qualification preparation."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'7e1211745efffb1718bb5e8acd9a999b461577989dacd8f9bff38a3736ee89da');sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('preparation/preparation-result.json')
assert m['preparation_only'] and not any(m[k] for k in ['started','full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_preparation'] and not q['started'] and q['negative_exit']==1 and q['negative_before_commands']
assert m['source_commit']==q['source_pin']=='121d1b58b06d788a94c4654faf79bdd3414df26c'
assert q['cases']==dict(owners=14542,aggregates=15222,effects=8128,value_calls=15644)
assert q['total_new_cases']==53536 and q['total_complete_reports']==345240
for name,h in q['scripts'].items():assert rows['preparation/'+name]['sha256']==h
assert rows['preparation/cohort_journal.py']['sha256']=='4e5a202045b7d4576c7cfa8ed9845cc895e5b2bf4cd2e727e68534a2a34c652d'
negative=js('negative-control/result.json');assert negative['terminal'] and not negative['commands'] and not negative.get('admitted_setup')
assert 'terminal qualified five-method foundation' in negative['exception']
assert not any(n.startswith('negative-control/source/') or n.startswith('preparation/source/') for n in rows)
for name,h in q['scripts'].items():assert rows['negative-control/'+name]['sha256']==h
probe=js('preparation/ordinary-shadows/result.json');assert probe['terminal'] and probe['all_guards_match']
assert probe['source_before']==probe['source_after'] and probe['tool_before']==probe['tool_after']==rows['external/nano_virt']['sha256']
assert len(probe['commands'])==4
for c in probe['commands']:
 assert c['exit']==0 and c['timeout']==600 and c['ordinary_shadow_seconds']==10
 assert c['argv'][2:4]==['--emit-nvm','-o']
 for name,h in probe['source_before'].items():assert rows['preparation/source-inputs/'+name]['sha256']==h
setup=data('preparation/setup.py').decode();collector=data('preparation/cohort.py').decode();admit=data('preparation/admit.py').decode()
assert "base='121d1b58b06d788a94c4654faf79bdd3414df26c'" in setup
assert setup.index("foundation.get('terminal')")<setup.index("global_result.get('terminal')")<setup.index("run('foundation-reader'")<setup.index("run('global-array-reader'")<setup.index("run('worktree'")
for expected in ["global_result['tests']==dict(tests=2,failures=0,errors=0,skips=0,passed=True)","foundation['tests']==dict(tests=5,failures=0,errors=0,skips=0,passed=True)","global_result['producer_sha256']==foundation['producer_sha256']","global_result['observer_before']==global_result['observer_after']","foundation['observer_before']==foundation['observer_after']","NANOLANG_AFFINE_GLOBAL_ARRAY_EVIDENCE_COMMIT","NANOLANG_AFFINE_FOUNDATION_EVIDENCE_COMMIT"]:assert expected in setup
for expected in ["global_entry['sha256']","entry['sha256']","'merge-base','--is-ancestor'","global_reader.read_bytes()==subprocess.check_output","reader.read_bytes()==subprocess.check_output"]:assert expected in setup
for expected in ["s['global_array_evidence']['result_sha256']","s['foundation_evidence']['result_sha256']","assert not r['other_jobs']","r['free_bytes']>15*1024**3","v['share']>=.75"]:assert expected in admit
for expected in ["suite.countTestCases()==4","result.testsRun==4","not result.skipped","for leaf in ['owners','aggregates','effects','value_calls']","budget=600 if build else timeout","q['observer_before']==q['observer_after']"]:assert expected in collector
assert q['no_second_guard_execution_claim']
for n in ['setup.py','admit.py','cohort.py','cohort_journal.py']:ast.parse(data('preparation/'+n))
# I bind unchanged permanent methods and the original batch/build budgets to Git.
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in q['source_methods'].items():
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n'
 assert sha(raw)==h==rows['preparation/source-inputs/'+name]['sha256']
 if name.endswith('.py'):
  assert b'timeout=600 if group==\'original\' else 60' in raw
  assert b'paths[i:i+8]' in raw and b'scalar_batches(paths)' in raw
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; exact four unchanged methods/counts, ordinary imported shadows, immutable journal/scripts, live-foundation refusal before commands/checkout, and both published-success dependency guards. Preparation only; second guard execution, component execution and release acceptance remain pending.'%len(rows))
