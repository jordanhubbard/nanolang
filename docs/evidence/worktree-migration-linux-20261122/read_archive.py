#!/usr/bin/env python3
"""I verify exact migration acceptance and reclamation records without changing Git."""
from pathlib import Path
import argparse,gzip,hashlib,json,struct
here=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--shared',type=Path);args=parser.parse_args()
m=json.loads((here/'manifest.json').read_text());records={row['path']:row for row in m['files']};assert len(records)==m['counts']['records'];objects={row['artifact'] for row in records.values()};assert len(objects)==m['counts']['objects']
def read(name):
 r=records[name];packed=(here/r['artifact']).read_bytes();assert len(packed)==r['compressed_bytes'] and sha(packed)==r['compressed_sha256'];raw=gzip.decompress(packed);assert len(raw)==r['bytes'] and sha(raw)==r['sha256'];return raw
def data(name):return json.loads(read(name))
for name in records:read(name)
assert sum(r['bytes'] for r in records.values())==m['counts']['raw_bytes'];assert sum((here/name).stat().st_size for name in objects)==m['counts']['compressed_bytes']
summary=data('final-summary.json');verified=data('verify-result.json');source=data('source-hash-result.json');accepted=data('accepted.json');reclaim=data('reclaim-result.json')
assert summary['status']=='OWNER_ACCEPTED_AND_RECLAIMED' and verified['status']=='PASS' and verified['mismatches']==[]
assert verified['files']==source['regular_files']==689872 and verified['entries']==699686
assert verified['logical_bytes']==source['bytes']==34830929153 and verified['source_sparse_files']==2218
assert source['status']=='PASS' and source['manifest_sha256']=='595771cfa0a73f39850ce70f2b74c1d4ed0a26a5054f6cb1d224170c2bd15e6b'
assert data('final-source-result.json')['status']=='PASS' and data('final-source-result.json')['errors']==[]
assert accepted['status']=='OWNER_ACCEPTED' and reclaim['status']=='RECLAIMED_ACCEPTED_EXACT_ORIGINAL'
assert accepted['original_inode']==reclaim['original_inode']==2760847 and accepted['original_device']==reclaim['original_device']==66306
assert accepted['original_allocated_bytes']==reclaim['allocated_bytes_reclaimed']==summary['allocated_bytes_reclaimed']==37026091008
assert accepted['source']==summary['stable_entrypoint']==reclaim['source'];assert accepted['destination']==summary['physical_destination']==reclaim['destination']
ops=data('runtime-operations.json');assert len(ops)==accepted['runtime_operations']==17
for row in ops:
 assert row['returncode']==0
 for suffix in ['stdout','stderr']:assert sha(read(row['label']+'.'+suffix))==row[suffix+'_sha256']
post=data('post-reclaim-runtime.json');assert post['status']=='PASS' and post['original_absent'] and len(post['operations'])==2
for row in post['operations']:
 assert row['returncode']==0
 for suffix in ['stdout','stderr']:assert sha(read(row['label']+'.'+suffix))==row[suffix+'_sha256']
tools=data('runtime-tools-before.json');assert tools==data('runtime-tools-after.json')==data('runtime-expected-tools.json') and len(tools)==8
assert data('frozen-roots-before.json')==data('frozen-roots-after.json')
assert all(not row['is_symlink'] and row['device']==66306 for row in data('frozen-roots-after.json'))
base=here/m['base_manifest'];assert sha(base.read_bytes())==m['base_manifest_sha256'];base_manifest=json.loads(base.read_text());base_rows={r['path']:r for r in base_manifest['files']}
for name,digest in tools.items():
 r=base_rows['tools/'+name];packed=(base.parent/r['artifact']).read_bytes();assert sha(gzip.decompress(packed))==digest==r['sha256']
cache=data('index-stat-applied.json');identity=data('index-stat-identity.json');git=data('nas-git-rebound-result.json')
assert cache['status']=='APPLIED_STAT_ONLY' and identity['status']=='PASS' and git['status']=='PASS'
assert cache['entries']==identity['entries']==358851 and cache['source_staged_bytes_proved']==10769141387
assert identity['identity_sha256_before']==identity['identity_sha256_after'] and identity['header_and_extensions_identical']
assert git['staged_oid_mode_path_sha256']==cache['staged_oid_mode_path_sha256']==data('index-content-before.json')['sha256']
assert sha(read(cache['retained_applied_helper']))==cache['retained_applied_helper_sha256']
assert data('index-control-v3/result.json')['status']=='PASS'
native=data('index-control-v3/native-timestamp-control.json');assert native['index_refreshed_by'].startswith('actual Git status only') and native['same_size'] and native['same_mtime'] and native['git_returncode']==0
assert native['cached_ctime'][0]==native['actual_ctime'][0] and native['cached_ctime'][1]!=native['actual_ctime'][1] and native['git_stdout']==''
assert data('index-extension-refusals.json')['refused_extensions']==['ZZZZ','FSMN','UNTR','link','sdir']
assert data('pre-delete-evidence.json')['status']=='INTEGRATED_PRE_DELETE_GUARDS_PASSED'
assert sha(read('reclaim-accepted.py'))==data('pre-delete-evidence.json')['script_sha256']
assert data('nas-git-readonly-result.json')['returncode']==-15
assert data('nas-git-diagnostic-stop.json')['status']=='STOP_OWN_READONLY_PREFIX_FOR_VERIFIED_STAT_CACHE_REBIND'
shared_checked=False
if args.shared:
 for r in m['shared_large_records']:
  p=args.shared/r['shared'];assert p.stat().st_size==r['bytes'] and sha(p.read_bytes())==r['sha256'],r['path']
 before=gzip.decompress((args.shared/'delta-7713e9381/index-original.gz').read_bytes());after=gzip.decompress((args.shared/'delta-7713e9381/index-candidate.gz').read_bytes())
 assert sha(before)==identity['original_sha256'] and sha(after)==identity['candidate_sha256']
 def parts(raw):
  assert raw[:4]==b'DIRC' and struct.unpack_from('>I',raw,4)[0]==2 and hashlib.sha1(raw[:-20]).digest()==raw[-20:];count=struct.unpack_from('>I',raw,8)[0];pos=12;fingerprint=hashlib.sha256()
  for _ in range(count):
   end=raw.index(b'\0',pos+62);limit=pos+((end+1-pos+7)//8)*8;fingerprint.update(raw[pos+24:pos+28]+raw[pos+40:limit]);pos=limit
  return raw[:12],raw[pos:-20],fingerprint.hexdigest()
 assert parts(before)==parts(after) and parts(before)[2]==identity['identity_sha256_before'];shared_checked=True
print(json.dumps(dict(status='PASS',records=len(records),regular_files=verified['files'],entries=verified['entries'],runtime_operations=17,post_reclaim_operations=2,allocated_bytes_reclaimed=reclaim['allocated_bytes_reclaimed'],shared_large_records_checked=shared_checked,scope='I verify this exact migration only; no current-source release acceptance follows.'),indent=2))
