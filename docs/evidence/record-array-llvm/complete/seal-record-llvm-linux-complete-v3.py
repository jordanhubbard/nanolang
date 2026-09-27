from pathlib import Path
import json,subprocess,sys,shutil,time
new=Path('/run/user/1000/nanolang-llvm-o2-aed76-20260921');out=Path('/run/user/1000/nanolang-record-llvm-linux-sanitizer-seal');state=Path('/tmp/nanolang-record-llvm-linux-sanitizer-finalization-v3.json')
record={'state':'sealing','started':time.time(),'source':'7804258431ed8fba21fa33c5507b06ad9a1bbd94','fixture_overlay':'aed76cdb3','output':str(out)}
state.write_text(json.dumps(record,indent=2))
try:
 assert shutil.disk_usage(out.parent).free>2644147981+2147483648
 retention=json.loads((new/'durable-retention.json').read_text());assert retention['gate_returncode']==0 and retention['corpus_correspondence']=='PASS' and retention['local']==retention['remote']
 histories=[new/'reports',Path('/tmp/nanolang-record-llvm-78042-linux-sanitizers'),Path('/tmp/nanolang-record-llvm-78042-linux-clang-continuation-r2')]
 expected=[ [('configuration',0),('clang-sanitizer-native',0)], [('configuration',0),('sanitizer-emission',0),('sanitizer-native',0)], [('configuration',0),('clang-sanitizer-emission',0),('clang-sanitizer-native',-15)]]
 for root,wanted in zip(histories,expected):assert [(x['phase'],x['status']) for x in json.loads((root/'results.json').read_text())]==wanted
 audit=json.loads(Path('/tmp/record-llvm-clang-o0-complete-audit.json').read_text());assert audit['all_O0_statuses_clean'] and audit['all_O0_statuses']==1365 and audit['coverage_records']==73
 args=[sys.executable,'/tmp/seal-record-llvm-local-copies.py','--output',str(out)]
 for p in histories:args+=['--history',str(p)]
 extras=[new/n for n in ['outer.json','outer.log','launch.json','durable-retention.json','corpus-correspondence.json','source-overlay.json','copied-providers.json','original-current.json','nanolang-record-llvm-o2-driver.py','nanolang-record-llvm-o2-launch.py','verify-record-llvm-o2-archive.py']]
 extras += [p for p in Path('/tmp/nanolang-record-llvm-78042-linux-clang-continuation').rglob('*') if p.is_file() and 'artifacts' not in p.parts and 'controls' not in p.parts]
 extras += [Path('/tmp')/n for n in ['record-llvm-clang-o0-complete-audit.json','nanolang-record-llvm-gcc-child-audit.json','nanolang-record-llvm-o2-final-current.json','nanolang-record-llvm-linux-capacity-current.json','nanolang-record-llvm-linux-sanitizer-capacity.json','nanolang-record-llvm-linux-clang-continuation-r2-capacity.json','nanolang-record-llvm-linux-sanitizer-finalization.json','nanolang-record-llvm-linux-clang-continuation-finalization.json','nanolang-record-llvm-78042-source.json','nanolang-record-llvm-78042-linux-sanitizers-outer.json','nanolang-record-llvm-78042-linux-sanitizers-outer.log','nanolang-record-llvm-78042-linux-clang-continuation-outer.json','nanolang-record-llvm-78042-linux-clang-continuation-outer.log','nanolang-record-llvm-78042-linux-clang-continuation-r2-outer.json','nanolang-record-llvm-78042-linux-clang-continuation-r2-outer.log']]
 assert len({p.name for p in extras})==len(extras),'extra basename collision'
 for p in extras:args+=['--extra',str(p)]
 subprocess.run(args,check=True,timeout=1800)
 record.update(state='sealed',durable_O2_retention=retention)
except Exception as e:record.update(state='stopped',error=repr(e));raise
finally:record['finished']=time.time();state.write_text(json.dumps(record,indent=2)+'\n')
