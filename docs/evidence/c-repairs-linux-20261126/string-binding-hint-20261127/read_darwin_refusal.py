"""I check my selected strict-C99 Darwin harness refusal."""
from pathlib import Path
import argparse,hashlib,json,subprocess
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='ec0ef664bdc83093051f783240c7e3b1e3b17f992c75fc7a3653caf978490658';m=json.loads(raw);assert m['version']==1 and not m['qualification'] and not m['whole_wrapper_reconstructed']
for row in m['files']:
 name=Path(row['path']);assert len(name.parts)==1;data=(a.archive/name).read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256']
job=json.loads((a.archive/'darwin-bootstrap-job.json').read_text());assert job['id']==m['job']==140652282110 and job['head_sha']==m['source'] and job['conclusion']=='failure'
steps={s['name']:s['conclusion'] for s in job['steps']};assert steps['Build fresh production tools']==steps['Verify source and selected tool guards']=='success'
assert steps['Check managed STRING publication and exact alias retention']=='failure' and steps['Execute the original bootstrap with its default limits']=='skipped' and steps['Execute the original native compiler and emitter suites']=='skipped'
log=(a.archive/'darwin-bootstrap.log').read_text();assert "call to undeclared function 'mkdtemp'" in log and '-Wimplicit-function-declaration' in log
old=(a.archive/'before.c').read_bytes();new=(a.archive/'after.c').read_bytes();assert new==b'#ifndef _DARWIN_C_SOURCE\n#define _DARWIN_C_SOURCE 1\n#endif\n'+old
for rev,data in [(m['source'],old),(m['test_fix'],new)]:assert data==subprocess.check_output(['git','show',rev+':tests/test_env_string_binding_hint.c'],cwd=a.repo)
print(json.dumps({'verified':True,'job':m['job'],'files':len(m['files']),'cold_production_tools_passed':True,'original_bootstrap_not_reached':True,'test_feature_declaration_only':True,'product_execution':False,'qualification':False}))
