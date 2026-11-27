"""I retain completed assertions separately from the failed exit deadline."""
from pathlib import Path
import argparse,hashlib,json
from decimal import Decimal
p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
root=a.artifact/'.test_output/c-repair-original-gates'
q=json.loads((root/'emitter-setup-diagnostic.json').read_bytes())
assert q['exit_code']==1 and not q['command_timeout'] and not q['qualification'] and q['shadow_seconds']==10
rows=[]
for line in (root/'emitter-setup-diagnostic.stderr').read_text().splitlines():
 if line.startswith('{"event":'):rows.append(json.loads(line))
starts=[v for v in rows if v['event']=='shadow-start'];done=[v for v in rows if v['event']=='shadow-complete']
assert len(starts)==len(done)==1139
for x,y in zip(starts,done):
 for field in ('name','source','line'):assert x[field]==y[field]
 assert x['failures']==y['failures']==0
phases=[v for v in rows if v['event']=='cseed-shadow-phase'];names=[v['phase'] for v in phases]
assert names==['dispatch-start','report-start','report-returned','dispatch-returned','completion-written','flush-start','flush-end','supervisor-status'],names
for row in phases:assert row['clock_ok']
clock=lambda row:row['monotonic_seconds']*1000000000+row['monotonic_nanoseconds']
assert [clock(v) for v in phases]==sorted(clock(v) for v in phases)
assert all(v['pid']==phases[0]['pid'] for v in phases[:-1])
assert phases[0]['value']==10 and phases[1]['value']==1139
assert all(v['value']==1 for v in phases[2:5])
status=phases[-1];assert status['child_pid']==phases[0]['pid']
assert status['reaped'] and status['completed'] and status['parent_timeout'] and status['parent_start_clock_ok']
assert status['exit_code']==0 and status['signal']==0 and not status['clock_failure'] and status['deadline_seconds']==10
start=status['parent_start_seconds']*1000000000+status['parent_start_nanoseconds']
flush=clock(phases[-2])-start;observed=clock(status)-start
assert flush==9719997000 and observed==10007673000
assert flush<10000000000<observed
print(json.dumps({'diagnostic_exit_code':q['exit_code'],'diagnostic_command_timeout':q['command_timeout'],'selected_shadows_started':len(starts),'selected_shadows_completed':len(done),'completed_shadow_failures':0,'sum_completed_shadow_seconds':str(sum((Decimal(str(v['elapsed_seconds'])) for v in done),Decimal(0))),'flush_end_after_parent_start_ns':flush,'parent_status_after_parent_start_ns':observed,'flush_to_parent_observation_ns':observed-flush,'child_reaped_zero':True,'parent_timeout':True,'actual_child_exit_time_known':False,'exclusive_tail_cause_known':False,'qualification':False,'whole_release':False}))
