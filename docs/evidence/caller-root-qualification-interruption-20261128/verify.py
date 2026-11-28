"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile,textwrap
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='0079fa293adc7935ae164971a08124946ed9eff6357f1d83daeec42ccf585667'
m=json.loads(raw)
with tempfile.TemporaryDirectory(prefix='nano-arm-receipt-',dir=args.workspace) as d:
 root=Path(d)
 for row in m['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
  output=root/name;output.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with output.open('wb') as stream:
   for key in row['chunks']:
    packed=(b/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed)
    assert len(data)==meta['bytes'] and sha(data)==key
    stream.write(data);h.update(data);size+=len(data)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 load=lambda n:json.loads((root/n).read_text())
 assert not m['qualification'] and not m['whole_release'] and m['interrupted_original_invocation']
 result=load('native/original/bootstrap/result.json')
 assert result['source_head']==m['source_head'] and result['exit']==-15 and not result['command_timeout']
 assert result['guards_match'] and result['before']==result['after']
 assert result['outer_observation_seconds']==8000
 stopped=load('native/resource-admission-interruption.json')
 assert stopped['source_head']==m['source_head'] and stopped['coordinator_status']=='busy' and stopped['coordinator_fence']==11
 assert not stopped['qualification_acceptance'] and not stopped['whole_release']
 assert load('queue/queue-state.json')['terminal'] and 'suite_exit' not in load('queue/queue-state.json')
 original=(root/'queue/queue.py').read_text();split=original.index('b=Path(__file__)')
 expected=original[:split]+'\ndef main():\n'+textwrap.indent(original[split:],'    ')+'\nif __name__ == "__main__":\n    main()\n'
 corrected=(root/'corrected/wait_for_native.py').read_text();assert corrected==expected
 control=load('corrected/control-result.json')
 assert control['returncode']==1 and control['no_build_or_consumer_execution'] and not control['import_side_effects']
 assert not control['qualification'] and not control['whole_release']
 assert control['queue_origin'].startswith('/usr/lib/python3.') and control['queue_origin'].endswith('/queue.py')
 assert sha(corrected.encode())==control['controller_sha256'] and sha(original.encode())==control['original_controller_sha256']
 assert control['actual_failed_prerequisite_sha256']==sha((root/'native/original/bootstrap/result.json').read_bytes())
 terminal=control['terminal_state'];assert terminal==load('corrected/queue-state.json') and terminal['terminal']
 assert 'build' not in terminal and 'suite_exit' not in terminal
 trace=(root/'corrected/refusal.stderr').read_text();assert trace.count('Traceback (most recent call last):')==1 and 'Error in sys.excepthook' not in trace
 old_trace=(root/'queue/controller.log').read_text();assert 'from apport.ui import HookUI, NoninteractiveHookUI' in old_trace and 'import queue' in old_trace
 resolution=load('native/controller-import-control.json')['rows'];assert resolution[0]['origin'].endswith('/caller-root-consumers-20261128/queue.py')
 assert resolution[1]['origin'].startswith('/usr/lib/python3.') and resolution[1]['origin'].endswith('/queue.py')
print('I verified the intentional worker-admission interruption, unchanged guards, failed-prerequisite refusal and import-safe controller correction. No bootstrap/native/consumer or release acceptance follows.')
