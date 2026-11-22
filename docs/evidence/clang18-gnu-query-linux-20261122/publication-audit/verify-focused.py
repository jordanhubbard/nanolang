from pathlib import Path
import sys,json,gzip,hashlib,subprocess
root=Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();cache={};entries={e['path']:e for e in m['files']}
for e in m['files']:
 if e['artifact'] not in cache:
  data=(root/e['artifact']).read_bytes();raw=gzip.decompress(data);cache[e['artifact']]=(len(data),sha(data),len(raw),sha(raw))
 assert cache[e['artifact']]==(e['compressed_bytes'],e['compressed_sha256'],e['bytes'],e['sha256']),e['path']
for role,s in m['source_sets'].items():
 names=list(s['source_hashes']);query=''.join(s['source_pins'].get(n,s['implementation'])+':'+n+'\n' for n in names);data=subprocess.check_output(['git','cat-file','--batch'],input=query.encode());cursor=0
 for n in names:
  end=data.index(b'\n',cursor);head=data[cursor:end].split();assert head[1]==b'blob';size=int(head[2]);raw=data[end+1:end+1+size];assert sha(raw)==s['source_hashes'][n],(role,n);assert entries[s['root']+'/'+n]['sha256']==sha(raw);cursor=end+1+size;assert data[cursor:cursor+1]==b'\n';cursor+=1
 assert cursor==len(data)
 for p,digest in s['tools'].items():assert entries[p]['sha256']==digest
assert m['source_sets']['red']['source_pins']=={'src/module_builder.c':m['source_sets']['red']['source_pins']['src/module_builder.c']}
print(json.dumps(dict(logical_files=len(m['files']),unique_artifacts=len(cache),raw_bytes=sum(e['bytes'] for e in m['files']),compressed_bytes=sum(x[0] for x in cache.values()),source_sets={k:dict(files=len(v['source_hashes']),implementation=v['implementation'],overrides=v['source_pins']) for k,v in m['source_sets'].items()},all_raw_and_compressed_hashes_verified=True,all_sources_match_declared_git_pins=True,all_native_tool_images_retained=True),indent=2))
