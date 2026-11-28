from pathlib import Path
import sys,json,gzip,hashlib,subprocess
sha=lambda raw:hashlib.sha256(raw).hexdigest()
results=[];manifests=[]
for name in sys.argv[1:]:
 root=Path(name);m=json.loads((root/'manifest.json').read_text());cache={}
 for e in m['files']:
  if e['artifact'] not in cache:
   packed=(root/e['artifact']).read_bytes();raw=gzip.decompress(packed)
   cache[e['artifact']]=(len(packed),sha(packed),len(raw),sha(raw))
  assert cache[e['artifact']]==(e['compressed_bytes'],e['compressed_sha256'],e['bytes'],e['sha256']),e['path']
 source=m['source_hashes'];names=list(source);queries=''.join(m['implementation']+':'+name+'\n' for name in names).encode();raw=subprocess.check_output(['git','cat-file','--batch'],input=queries);cursor=0
 for name in names:
  end=raw.index(b'\n',cursor);header=raw[cursor:end].split();assert header[1]==b'blob',header;size=int(header[2]);data=raw[end+1:end+1+size];assert len(data)==size and sha(data)==source[name],name;cursor=end+1+size;assert raw[cursor:cursor+1]==b'\n';cursor+=1
 assert cursor==len(raw)
 entries={e['path']:e for e in m['files']};suffix='src_nano/nvm2c/module.nano';prefixes=[path[:-len(suffix)] for path in entries if path.endswith('/'+suffix)];assert len(prefixes)==1,prefixes
 for name,value in source.items():assert entries[prefixes[0]+name]['sha256']==value,name
 results.append(dict(archive=str(root),implementation=m['implementation'],logical_files=len(m['files']),unique_artifacts=len(cache),raw_bytes=sum(e['bytes'] for e in m['files']),compressed_bytes=sum(value[0] for value in cache.values()),source_files=len(source),all_compressed_and_raw_hashes_verified=True,all_guarded_sources_match_git_pin=True));manifests.append(m)
if len(manifests)>1:
 first=manifests[0]
 for other in manifests[1:]:
  assert other['implementation']==first['implementation'] and other['source_hashes']==first['source_hashes']
  for key in ['cases','private_candidates','pending','independent_expectations','original_products','routes','labels']:assert other['conversion'][key]==first['conversion'][key],key
print(json.dumps(results,indent=2))
