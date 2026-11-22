from pathlib import Path
import gzip,hashlib,json,subprocess
ROOT=Path.cwd();base=Path('/Users/jkh/.worktrees/nanolang/peer-query-evidence-20261122/docs/evidence/clang18-gnu-query-linux-20261122');manifest=base/'manifest.json';m=json.loads(manifest.read_text());cache={};records={}
def sha(b):return hashlib.sha256(b).hexdigest()
for e in m['files']:
 p=(base/e['artifact']).resolve();assert p.is_relative_to(base.resolve())
 if p not in cache:
  packed=p.read_bytes();raw=gzip.decompress(packed);cache[p]=(packed,raw)
 packed,raw=cache[p];assert len(packed)==e['compressed_bytes'] and sha(packed)==e['compressed_sha256'];assert len(raw)==e['bytes'] and sha(raw)==e['sha256'];records[e['path']]=raw
trees={}
def tree(pin):
 if pin not in trees:
  output=subprocess.run(['git','ls-tree','-rz','--full-tree',pin],capture_output=True,check=True,timeout=30).stdout;rows={}
  for row in output.split(b'\0'):
   if not row:continue
   head,name=row.split(b'\t',1);mode,kind,oid=head.split();rows[name.decode()]=(mode.decode(),kind.decode(),oid.decode())
  trees[pin]=rows
 return trees[pin]
sets={}
for label,s in m['source_sets'].items():
 for name,h in s['source_hashes'].items():
  raw=records[s['root']+'/'+name];assert sha(raw)==h
  pin=s.get('source_pins',{}).get(name,s['implementation']);mode,kind,oid=tree(pin)[name];assert kind=='blob';assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==oid,(label,name,pin)
 for name,h in s['tools'].items():assert sha(records[name])==h,(label,name)
 sets[label]=dict(checked_source_inputs=len(s['source_hashes']),checked_native_images=len(s['tools']),declared_overrides=s.get('source_pins',{}))
r=dict(peer_commit='77100354c501c2ac9ee105e6ae2e6ad5abde5cb4',manifest_sha256=sha(manifest.read_bytes()),logical_files=len(m['files']),unique_payloads=len(cache),raw_bytes=sum(e['bytes'] for e in m['files']),source_sets=sets,all_raw_compressed_lengths_hashes_and_git_source_blobs_match=True,scope='I independently verify the retained archive and declared Git/source/tool identities. I do not repeat or extend its focused native execution.')
(ROOT/'.test_output/capture-repair-20261122/peer-query-archive-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
