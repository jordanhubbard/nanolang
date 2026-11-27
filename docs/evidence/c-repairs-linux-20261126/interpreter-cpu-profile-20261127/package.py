from pathlib import Path
import gzip,hashlib,json,re,subprocess
b=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang');work=b/'worktrees/cseed-interpreter-profile-20261127';out=b/'tmp/interpreter-profile-evidence-20261127';out.mkdir();(out/'objects').mkdir();sha=lambda x:hashlib.sha256(x).hexdigest();selected={}
for label,root in [('software',work/'tmp/interpreter-profile'),('hardware',work/'tmp/interpreter-profile-hardware'),('hardware-control',b/'tmp/c-repairs-followups-20261126/perf-hardware-control-20261127')]:
 for p in sorted(root.iterdir()):
  if p.is_file() and p.name!='perf.data':selected[label+'/'+p.name]=p
selected['sampled-binary/nanoc_c']=work/'bin/nanoc_c';selected['tools/perf']=Path('/usr/bin/perf')
script=(work/'tmp/interpreter-profile-hardware/script.stdout').read_text();paths=set(re.findall(r'\((/[^()]+)\)$',script,re.M))
for name in sorted(paths):
 p=Path(name)
 if p.is_file() and p!=work/'bin/nanoc_c':selected['mapped-dso/'+name.lstrip('/')]=p
rows=[];chunks={}
for name,p in selected.items():
 keys=[];h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for data in iter(lambda:f.read(2*1024*1024),b''):
   k=sha(data);h.update(data);size+=len(data);keys.append(k)
   if k not in chunks:
    packed=gzip.compress(data,mtime=0);(out/'objects'/(k+'.gz')).write_bytes(packed);chunks[k]={'bytes':len(data),'gzip_sha256':sha(packed)}
 rows.append({'path':name,'sha256':h.hexdigest(),'size':size,'chunks':keys})
m={'version':1,'product_source':'6e09c07a9f1651294e4669282cdc17ee14126775','source_tree':str(work),'qualification':False,'whole_release':False,'sampled_pid':2329556,'samples':828,'files':rows,'chunks':chunks};raw=(json.dumps(m,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw);print(json.dumps({'sha256':sha(raw),'files':len(rows),'chunks':len(chunks),'raw_bytes':sum(x['size'] for x in rows),'mapped_dsos':len(paths)}))
