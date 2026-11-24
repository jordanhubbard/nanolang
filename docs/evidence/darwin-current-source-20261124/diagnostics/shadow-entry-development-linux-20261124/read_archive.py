"""I verify unsuccessful development qualification without admitting a compiler."""
from pathlib import Path
import sys,json,hashlib,subprocess,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'567ce08c2cb03dfa0524e021e2f48f5f358ac91ccf3d09357d5dae201dadc8bd');sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('candidate/derivation-result.json');p=js('candidate/preflight-result.json');s=js('candidate/setup-result.json');n=js('entry-neighbors/result.json');r=js('refused-reuse-setup/setup-result.json')
assert len(rows)==19886 and m['known_issue']==1253 and not m['development_compiler_accepted'] and not m['new_bootstrap_in_archive'] and not m['production_admission'] and not m['whole_release']
assert q['terminal'] and q['development_derivation'] and not q['new_bootstrap'] and not q['stage_alias'] and not q['development_compiler_accepted'] and not q['products'] and q['ordinary_shadow_seconds']==10
assert q['source_pin']==m['source_commit'] and q['source_producer_pin']==m['source_producer_commit']
assert len(q['commands'])==1;c=q['commands'][0];assert c['timeout']==600 and c['timed_out'] and c['exit']==-9 and c['seconds']>=600 and c['label']=='source-generation'
assert data('candidate/source-generation.stdout')==data('candidate/source-generation.stderr')==b''
assert q['source_before']==q['source_after']==p['source_before']==p['source_after']==s['source_before']==s['source_after']
assert q['producer_before']==q['producer_after']==p['producer_before']==p['producer_after'] and q['tools_before']==q['tools_after']==p['tools_before']==p['tools_after'] and q['hosts_before']==q['hosts_after']==s['immutable_hosts']
assert p['terminal'] and p['accepted_preflight'] and [c['exit'] for c in p['commands']]==[0,0,0,1]
assert b'3628 static fixture fragments; 0 reserved names; 154 lexer-refused' in data('candidate/identifier-corpus.stdout')
assert b'Ran 5 tests' in data('candidate/identifier-controls.stderr') and b'Ran 6 tests' in data('candidate/entry-baseline-red.stderr') and b'FAILED (failures=4)' in data('candidate/entry-baseline-red.stderr')
for name,h in q['producer_before'].items():assert rows['candidate/source/'+name]['sha256']==h
for name,h in q['tools_before'].items():assert rows['tool/'+name.lstrip('/')]['sha256']==h
for name,h in q['hosts_before'].items():
 rel=str(Path(name).relative_to('/var/tmp/nanolang-bootstrap-core-supervisors-20261124-ndvs104a'))
 parent=json.loads((base.parent/'core-supervisor-bootstrap-linux-20261124/manifest.json').read_bytes());pr={v['path']:v for v in parent['files']};assert pr['bootstrap/'+rel]['sha256']==h
assert n['terminal'] and n['guards_match'] and n['source_pin']==m['neighbor_source_commit'] and not n['production_edits']
assert n['source_before']==n['source_after'] and n['hosts_before']==n['hosts_after'] and n['tools_before']==n['tools_after'] and n['producer_before']==n['producer_after']==q['producer_before']['bin/nanoc_stage2']
for i,c in enumerate(n['cases']):
 folder='entry-neighbors/'+c['name']+'/';original='localization/processes/%03d/before/'%i
 lm=json.loads((base.parent/'native-shadow-localization-linux-20261124/manifest.json').read_bytes());lr={v['path']:v for v in lm['files']}
 src=data(folder+'input.nano');assert sha(src)==c['source_sha256'] and c['only_added_entry'] and c['exit']==1 and c['timeout']==90 and c['prior_preserved']
 assert src.endswith(b'fn main() -> int { return 0 }\n');assert sha(src[:-len(b'\nfn main() -> int { return 0 }\n')])==c['original_source_sha256']
 assert lr[original+'input.nano']['sha256']==c['original_source_sha256']
 assert rows[folder+c['name']+'.c']['sha256']==c['helper_sha256']==lr[original+c['name']+'.c']['sha256']
 assert data(folder+'program')==b'prior output' and data(folder+'stderr')==b'' and b'unsupported extern result or symbol' in data(folder+'stdout')
assert r['terminal'] and r['source_pin']==m['refused_setup_source_commit']==m['harness_commit'] and not r.get('admitted_setup',False) and r['exception']=='AssertionError()'
assert r['qualified_image_relinks_in_plan']==['nano_vm','nvm2c','nanoc_c','nanoc_c','nanoc_c']
assert all(c['exit']==0 for c in r['commands']) and not any('bootstrap3' in c['args'] for c in r['commands'])
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
for prefix,pin,guard in [('candidate/source/',m['source_commit'],q['source_before']),('entry-neighbors/source/',m['neighbor_source_commit'],n['source_before']),('refused-reuse-setup/source/',m['refused_setup_source_commit'],r['source_before'])]:
 for name,h in guard.items():assert sha(blob(pin,name))==h==rows[prefix+name]['sha256'],(prefix,name)
for name in ['src_nano/compiler/nanoisa_codegen.nano','src_nano/nanoc_v06.nano']:
 assert data('candidate/source/'+name)==data('candidate/harness-source/'+name)==blob(m['harness_commit'],name)
name='tests/test_native_shadows.py'
a=ast.parse(blob(m['source_commit'],name));b=ast.parse(data('candidate/harness-source/'+name))
def methods(tree):return [ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
assert methods(a)==methods(b)
proc.stdin.close();assert proc.wait()==0
print('PASS integrity: %d records; ordinary600 derivation TIMEOUT/no compiler; baseline6 FAIL4; two explicit-main extern refusals; reuse preflight refused before bootstrap. No candidate/current-native/release acceptance.'%len(rows))
