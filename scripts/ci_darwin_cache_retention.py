"""I require a collected payload and every original gate; I retry no product."""
import argparse, json, os
from pathlib import Path

GATES = ('pin','observer','build','regression','array','cache','guards')
OUTCOMES = {'','success','failure','skipped','cancelled'}

def retention_status(attempts, gates):
    if len(attempts) != 3 or any(value not in OUTCOMES for value in attempts):
        raise ValueError('I require exactly three known retention outcomes.')
    if set(gates) != set(GATES) or any(value not in OUTCOMES for value in gates.values()):
        raise ValueError('I require every original gate outcome.')
    return any(value == 'success' for value in attempts) and all(value == 'success' for value in gates.values())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observe-only',action='store_true')
    a=p.parse_args()
    attempts=[os.environ.get('RETENTION_'+str(i),'') for i in range(1,4)]
    gates={name:os.environ.get('CACHE_GATE_'+name.upper(),'') for name in GATES}
    accepted=retention_status(attempts,gates)
    record=dict(product_revision=os.environ['PRODUCT_REVISION'],observer_revision=os.environ['GITHUB_SHA'],
        run_id=os.environ['GITHUB_RUN_ID'],retention_attempts=attempts,original_gates=gates,
        terminal=not a.observe_only,collected_and_original_gates_report_success=accepted,
        qualification_granted=False,independent_reader_required=True,whole_release=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,sort_keys=True))
    if not a.observe_only and not accepted:
        raise SystemExit('I retain failure: I need a successful upload and every original gate.')

if __name__=='__main__': main()
