"""I verify the current strict callable audit and retained aggregate leak."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
base=Path(__file__).resolve().parent;repo=base.parents[2];m=json.loads((base/'manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files']);verified={};images=set()
for row in rows.values():
 p=(base/row['artifact']).resolve();assert p.is_relative_to(repo/'docs/evidence');identity=(row['sha256'],row['compressed_sha256'],row['bytes'])
 if p in verified:assert verified[p]==identity;continue
 z=p.read_bytes();assert sha(z)==row['compressed_sha256'];raw=gzip.decompress(z);assert sha(raw)==row['sha256'] and len(raw)==row['bytes'];verified[p]=identity
 if raw[:6]==b'\x7fELF\x02\x01' and int.from_bytes(raw[16:18],'little') in (2,3) and int.from_bytes(raw[18:20],'little')==183:images.add(row['sha256'])
assert {p.resolve() for p in (base/'objects').glob('*.gz')}=={(base/r['artifact']).resolve() for r in rows.values() if r['artifact'].startswith('objects/')}
assert sha((base/m['base_manifest']['path']).read_bytes())==m['base_manifest']['sha256']
def data(name):return gzip.decompress((base/rows[name]['artifact']).read_bytes())
def js(name):return json.loads(data('journal/'+name+'.json'))
first=js('result');matrix=js('matrix-result');assert first['source_pin']==matrix['source_pin']==m['source_pin']=='8949313b0a56c0c3bd52509fb363edb4eefba3e4'
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
source={}
for row in m['git_sources']:
 name=row['path'];proc.stdin.write((m['source_pin']+':'+name+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob';raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==row['sha256']==rows['source/'+name]['sha256'];source[name]=row['sha256']
proc.stdin.close();assert proc.wait()==0 and len(source)==3520
assert first['terminal'] and not first.get('exception') and not first['source_edited']
assert first['before']==first['after']==source and first['providers']==first['providers_after']
assert rows['tool/cc']['sha256']==first['tool_before']==first['tool_after']
assert all(c['exit']==0 for c in first['commands'])
assert matrix['terminal'] and matrix['exception']=="AssertionError(('san', 'record-escape'))"
assert [(p['mode'],[(c['case'],c['command']['exit']) for c in p['cases']]) for p in matrix['profiles']]==[('normal',[('record-escape',0),('expression-ownership',0),('borrowed-values',0)]),('san',[('audit',0),('record-escape',1)])]
terminal={p['mode']:p for p in js('terminal-guards')}
for p in matrix['profiles']:
 assert p['before']==terminal[p['mode']]['source_after']==source and p['providers']==terminal[p['mode']]['providers_after']
 for name,h in p['providers'].items():assert rows['provider/'+p['tree']+'/'+name]['sha256']==h
 for c in p['cases']:
  assert rows['product/'+c['command']['args'][0]]['sha256']==c['binary_sha256'] and c['binary_sha256'] in images
  assert c['command']['cwd'].startswith(matrix['work'])
for name,h in first['providers'].items():assert rows['provider/'+first['provider_root']+'/'+name]['sha256']==h
assert rows['product/'+first['audit']['args'][0]]['sha256']==first['binary_sha256'] and first['binary_sha256'] in images
expected=['direct','bound','declaration_return','binding_return','recursive_scalar','recursive_return','invoke','forward','computed']
for path in [first['work']+'/audit.stdout',matrix['work']+'/san-audit.stdout']:
 raw=data('product/'+path).decode();matches=re.findall(r'I observe (\w+): (\d+) names, (\d+) releases, (\d+) outstanding after teardown\.',raw);assert [r[0] for r in matches]==expected and len(matches)==9
 assert all(int(a)==int(b) and int(a)>0 and int(n)==0 for _,a,b,n in matches)
 assert raw.endswith('I retain 0 callable names across nine 64-call routes.\n')
err=data('product/'+matrix['work']+'/san-record-escape.stderr');out=data('product/'+matrix['work']+'/san-record-escape.stdout')
assert b'SUMMARY: AddressSanitizer: 3600 byte(s) leaked in 153 allocation(s).' in err
assert b'ERROR: LeakSanitizer: detected memory leaks' in err and b'ERROR: AddressSanitizer:' not in err and b'runtime error:' not in err
assert b'I preserve record and union callbacks after their defining calls and run every fixture shadow.' in out
# The unsuccessful wrapper stops here; I do not infer later SAN or O3 results.
assert not any(p['mode']=='production' for p in matrix['profiles'])
print('PASS: %d records, %d Git sources, %d native images; normal/SAN strict audit zero, full SAN record leak retained.'%(len(rows),len(source),len(images)))
