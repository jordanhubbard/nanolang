"""I verify formatter-only integration and guarded fresh consumer preparations."""
from pathlib import Path
import hashlib,json,subprocess,sys,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'87b05a2bfe09518a03d36eac0b616fd749cd73bd5c46c2f861d961237ec11522');sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('proof/preparation.json');integration=js('integration/integration.json');ready=js('integration/readiness/result.json');info=q['roots']
assert m['preparation_only'] and not any(m[k] for k in ['started','full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_preparation'] and not q['started'] and m['source_commit']==info['source_pin']=='5d809aa3d14d11b25adc2ee2ae391893ae5cf0f2'
assert integration['terminal'] and integration['exact_formatter_delta'] and integration['old_assertions_unchanged'] and integration['product_source_unchanged']
assert len(integration['files'])==8
left=b'        for arm in (range 0 256) { set text (+ text (flag (at item.excluded arm))) }';right=b'        set text (+ text (report_text.exclusions item.excluded))'
for name,record in integration['files'].items():
 old=data('integration/before/'+name);new=data('integration/after/'+name)
 assert sha(old)==record['before'] and sha(new)==record['after']==rows['source-inputs/'+name]['sha256']
 assert new==b'module "tests/nvm2c_affine_report.nano" as report_text\n'+old.replace(left,right)
 assert [line for line in old.splitlines() if b'assert ' in line]==[line for line in new.splitlines() if b'assert ' in line]
assert ready['terminal'] and ready['all_guards_match'] and ready['source_before']==ready['source_after'] and ready['tool_before']==ready['tool_after']==rows['external/nano_virt']['sha256']
assert len(ready['commands'])==8 and all(c['exit']==0 and c['timeout']==600 and c['ordinary_shadow_seconds']==10 for c in ready['commands'])
for name,h in ready['source_before'].items():assert rows['integration/readiness/source/'+name]['sha256']==h
for label in ['global','six']:
 assert q['controls'][label]['exit']==1 and q['controls'][label]['before_commands'] and q['controls'][label]['before_checkout']
 negative=js(label+'-negative-control/result.json');assert negative['terminal'] and not negative['commands'] and not negative.get('admitted_setup')
 assert ('NANOLANG_AFFINE_FOUNDATION_EVIDENCE_COMMIT' if label=='global' else 'terminal qualified global/array success') in negative['exception']
 for name,h in q['scripts'][label].items():assert rows[label+'/'+name]['sha256']==h==rows[label+'-negative-control/'+name]['sha256'];ast.parse(data(label+'/'+name))
 assert data(label+'/cohort.py')==data('prior-'+label+'/cohort.py') and data(label+'/cohort_journal.py')==data('prior-'+label+'/cohort_journal.py')
 assert not any(n.startswith(label+'/source/') for n in rows)
pin=info['source_pin'];old_global=info['prior_global_root'];new_global=info['global_root']
assert data('global/setup.py')==data('prior-global/setup.py').replace(b"base='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8'",("base='"+pin+"'").encode())
assert data('global/admit.py')==data('prior-global/admit.py').replace(b"s['source_base']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8'",("s['source_base']=='"+pin+"'").encode())
for name in ['setup.py','admit.py']:
 old=data('prior-six/'+name).decode();expected=old.replace('884887793345685872a206edf3c49c97ceb6834c',pin).replace(old_global,new_global).replace("global_result['source_pin']==foundation['source_pin']","global_result['source_pin']==base").replace('diagnostics/affine-global-array-typed-five-linux-20261125','diagnostics/affine-global-array-report-five-linux-20261125')
 assert data('six/'+name).decode()==expected
assert q['cases']==dict(globals=3688,arrays=17056,owners=14542,aggregates=15222,effects=8128,value_calls=15644,call_context=834,analysis_cache=3317)
assert q['global_complete_reports']==136476 and q['six_complete_reports']==382158
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
protected=[':(top)src/',':(top)src_nano/',':(top)Makefile.gnu',':(top,glob)tests/test_nanolang_translator_affine*.py',':(top,glob)tests/nvm2c_affine*_reference.c',*[':(top)tests/nvm2c_affine_'+n+'_driver.nano' for n in ['frame','scalar','locals']]]
assert not subprocess.check_output(['git','diff','--name-only',integration['old_pin'],pin,'--',*protected],cwd=repo).strip()
benchmark='docs/evidence/darwin-current-source-20261124/diagnostics/affine-report-benchmark-linux-20261125/manifest.json'
bm=json.loads(subprocess.check_output(['git','show',info['benchmark_evidence']+':'+benchmark],cwd=repo))
assert next(r for r in bm['files'] if r['path']=='comparison/source/tests/nvm2c_affine_report.nano')['sha256']==integration['helper_sha256']==ready['source_before']['tests/nvm2c_affine_report.nano']
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def check(commit,name,h):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h
for name,h in ready['source_before'].items():check(pin,name,h)
for name,h in q['source_methods'].items():assert rows['source-inputs/'+name]['sha256']==h;check(pin,name,h)
for name,r in integration['files'].items():check(integration['old_pin'],name,r['before'])
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; exact eight-driver formatter-only integration with all assertions/product/corpus/reference/older-driver sources retained, eight ordinary imported shadows, unchanged two/six-method collectors/counts/budgets, source-pin/dependency bindings and real missing-publication/global-success refusals before commands/checkout. Preparation only; current full-route/paired/release acceptance remains pending.'%len(rows))
