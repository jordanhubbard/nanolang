"""I bind existing traces to immutable source spans; I execute no compiler."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b.parent))
from read_chunks import read_archive
sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=(b/'failure-path.json').read_bytes();assert sha(raw)=='d8ed64bb8bd1d16467d00dd9ae17875584ce41f936a26eed658a015947266715'
p=json.loads(raw);assert p['interpretation_only'] and p['new_compiler_commands']==0 and not any(p[k] for k in ['accepted_repair','accepted_component','whole_release'])
source=subprocess.check_output(['git','show',p['source_commit']+':'+p['source_path']],cwd=b);assert sha(source)==p['source_sha256']
assert len(p['source_associations'])==2 and all(row['sha256']==p['source_sha256'] for row in p['source_associations'])
lines=source.splitlines(keepends=True);spans={}
for row in p['spans']:
 raw=b''.join(lines[row['start']-1:row['end']]);assert sha(raw)==row['sha256'] and raw.decode()==row['text'];spans[row['label']]=raw.decode()
assert set(spans)=={'kinds','word-storage','known-receiver','array-deferral','fallback','inference-loop'}
kinds={key:int(value) for key,value in re.findall(r'#define NVM2C_VK_(\w+) (\d+)',spans['kinds'])}
assert {k:kinds[k] for k in ['INT','UNK','ARR','VALUE']}==dict(INT=0,UNK=2,ARR=3,VALUE=8)
assert 'b->canonical_records && arr.kind == NVM2C_VK_UNK' in spans['array-deferral']
assert '(!facts->final && val.kind == NVM2C_VK_INT)' in spans['array-deferral']
assert 'NVM2C_VK_BARR : NVM2C_VK_ARR' in spans['fallback'] and 'NVM2C_VK_BARR ? NVM2C_VK_BOOL : NVM2C_VK_INT' in spans['fallback']
assert 'I require the matching scalar shape at this array write' in spans['fallback']
assert 'if (!classify_function(' in spans['inference-loop'] and 'goto fail;' in spans['inference-loop']
m,records,data,js=read_archive(b,p['reproducer_manifest_sha256'])
d=js('reduction/localization/result.json');assert d['terminal'] and d['read_only_debugger'] and len(d['traces'])==2
for row in d['traces']:
 text=data('reduction/localization/'+row['producer']+'.stdout').decode()
 for key,want in [('idx',row['function']),('start',42),('val.kind',kinds['ARR']),('arr.kind',kinds['UNK']),('written_array',kinds['ARR']),('facts->final',0),('b->track_shapes',0)]:
  found=re.search(r'^'+re.escape(key)+r' = (\d+)',text,re.M);assert found and int(found[1])==want
 assert 'b->canonical_records = false' in text and row['before']==row['after']
assert p['expected_trace']==dict(receiver=2,value=3,written_array=3,offset=42,final=False,canonical_records=False)
print('PASS exact Git source spans and both existing nonrecursive traces: unknown receiver becomes integer-array storage before nonfinal scalar mismatch; no compiler execution or repair acceptance')
