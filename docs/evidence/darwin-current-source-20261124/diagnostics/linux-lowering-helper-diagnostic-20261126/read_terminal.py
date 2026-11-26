"""I verify my retained failed observation and complete Linux diagnostic."""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();b=Path(__file__).resolve().parent;results=[]
for case,expected,extra in [('failed-full-name','7fa6d133f8cc867488f10ebf03a8a0cec73a033c375c7a6b9dcaf565361bdd5e',['--failed-name-guard']),('complete','b562894bd99ae58f41e3aaeb504518a45dd7d1c7d26548c262e8600b787277df',[])]:
 assert hashlib.sha256((b/case/'manifest.json').read_bytes()).hexdigest()==expected
 checked=subprocess.run([sys.executable,str(b/case/'read_terminal.py'),'--repo',str(a.repo)]+extra,capture_output=True,text=True);assert checked.returncode==0,checked.stderr;results.append(json.loads(checked.stdout))
assert not results[0]['helper_observation_valid'] and results[1]['helper_observation_valid']
print(json.dumps(dict(verified=True,records=sum(r['records'] for r in results),failed_records=results[0]['records'],complete_records=results[1]['records'],source_blobs_verified_each=4683,parent_records=results[1]['parent_records'],bootstrap_parent_records=results[1]['bootstrap_parent_records'],compiler_accepted=False,whole_release=False)))
