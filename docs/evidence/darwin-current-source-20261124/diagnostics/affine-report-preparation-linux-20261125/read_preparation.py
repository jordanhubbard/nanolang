"""I verify a private formatter preparation, without a benchmark claim."""
from pathlib import Path
import hashlib,json,subprocess,sys,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'1508a78c02b8719a9e247721fabee708731d82238abdc279386a67baf2b05a71');sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['preparation_only'] and m['private_overlay'] and not any(m[k] for k in ['started','full_five_routes','new_bootstrap','whole_release'])
q=js('preparation/preparation.json');assert q['terminal'] and q['private_candidate'] and not q['executed_benchmark'] and q['all_guards_match']
assert m['source_commit']==q['source_pin']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8'
assert q['source_before']==q['source_after'] and q['tool_before']==q['tool_after']==rows['external/nano_virt']['sha256']
assert len(q['commands'])==2 and [c['exit'] for c in q['commands']]==[0,0]
assert [c['timeout'] for c in q['commands']]==[60,600] and all(c['ordinary_shadow_seconds']==10 for c in q['commands'])
for name,h in q['source_before'].items():assert rows['preparation/source/'+name]['sha256']==h
old=data('preparation/old-driver.nano');new=data('preparation/source/tests/nvm2c_affine_locals_driver.nano')
left=b'        for arm in (range 0 256) { set text (+ text (flag (at item.excluded arm))) }';right=b'        set text (+ text (report_text.exclusions item.excluded))'
assert old.count(left)==1 and new==b'module "tests/nvm2c_affine_report.nano" as report_text\n'+old.replace(left,right)
assert [line for line in new.splitlines() if b'assert ' in line]==[line for line in old.splitlines() if b'assert ' in line]
helper=data('preparation/source/tests/nvm2c_affine_report.nano')
assert rows['preparation/source/tests/nvm2c_affine_report.nano']['sha256']==q['helper_sha256']
assert b'for arm in (range 0 256) { set text (+ text (flag (at bits arm))) }' in helper
assert b'extern fn' not in helper and b'array_set bits' in helper.split(b'shadow exclusions',1)[1]
proof=js('preparation/admission-preparation.json');assert proof['terminal'] and proof['accepted_preparation'] and not proof['started'] and proof['before_commands'] and proof['exit']==1
assert proof['script_sha256']==proof['script_after']==rows['preparation/benchmark.py']['sha256']==rows['negative-control/benchmark.py']['sha256']
negative=js('negative-control/benchmark-result.json');assert negative['terminal'] and not negative['commands'] and not negative.get('accepted_experiment')
assert 'complete qualified foundation success' in negative['exception']
assert not any(name.startswith('preparation/round-') or name.startswith('preparation/benchmark-result') for name in rows)
script=data('preparation/benchmark.py').decode();ast.parse(script)
for required in ["f.get('terminal') and f.get('accepted_component') and f.get('all_guards_match')","tests=5,failures=0,errors=0,skips=0,passed=True","NANOLANG_AFFINE_FOUNDATION_EVIDENCE_COMMIT","manifest['full_five_routes']","'merge-base','--is-ancestor'","assert not q['other_jobs']","assert c/w>=.75","run('translate',translate,600)","run('native-build',native,600)","budget=60 if group=='locals' else 600","for round_ in range(2)","q.get('observer_before',{})==q['observer_after']"]:assert required in script
assert script.index("f.get('terminal')")<script.index("run('foundation-reader'")<script.index("run('translate'")
baseline=js('baseline/measurement.json');assert baseline['terminal'] and baseline['source_pin']==q['source_pin'] and baseline['live_collector_unchanged']
assert baseline['observer_before']==baseline['observer_after']
assert baseline['case']=='load-4-True-True-256' and baseline['report_bytes']==73279
assert baseline['completed_original_command']['number']==17696 and baseline['completed_original_command']['exit']==0 and baseline['completed_original_command']['timeout']==60
assert baseline['completed_original_command']['seconds']==20.56080517800001
assert baseline['expected_report_sha256']==rows['baseline/report.stdout']['sha256']=='105d47468028d19616571ddc26ae261844f6bd03a328d10e4951268a63bd3a6e'
assert baseline['source_sha256']==q['old_driver_sha256']==sha(old) and data('baseline/driver.nano')==old
cases=js('preparation/benchmark-cases.json');assert len(cases)==9
assert [c['name'] for c in cases]==['load-4-True-True-256']+['exclusion-'+str(n) for n in [0,63,64,127,128,191,192,255]]
for case in cases:
 name='preparation/inputs/'+Path(case['input']).name;assert rows[name]['sha256']==case['input_sha256']
assert (cases[0]['expected']+'\n').encode()==data('baseline/report.stdout') and cases[0]['input_sha256']==rows['baseline/input.bin']['sha256']
builds=js('preparation/original-builds.json');assert set(builds)=={'translate','native'}
assert all(c['exit']==0 and c['build'] and c['timeout']==600 for c in builds.values())
assert '-fsanitize=address,undefined' in builds['native']['argv'] and '-fno-sanitize-recover=all' in builds['native']['argv']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in q['source_before'].items():
 if name=='tests/nvm2c_affine_report.nano':continue
 if name=='tests/nvm2c_affine_locals_driver.nano':h=q['old_driver_sha256']
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; historical original-limit full-stack C report/timing, exact private formatter-only delta with all assertions/product sources retained, ordinary imported shadows, nine staged C reports, immutable controller and real incomplete-foundation refusal. No benchmark, isolated cause, speedup, driver update or release acceptance.'%len(rows))
