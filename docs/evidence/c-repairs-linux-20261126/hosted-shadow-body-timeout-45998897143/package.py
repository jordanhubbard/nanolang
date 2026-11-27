from pathlib import Path
import gzip,hashlib,json,zipfile,collections
b=Path('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp');source=b/'c-repairs-followups-20261126/lifecycle-45998897143';out=b/'lifecycle-45998897143-evidence';out.mkdir();(out/'objects').mkdir();sha=lambda x:hashlib.sha256(x).hexdigest()
f=source/'files';rows=[]
for l in (f/'compiler.stderr').read_text().splitlines():
 if l.startswith('{'):
  try:rows.append(json.loads(l))
  except ValueError:pass
starts=[r for r in rows if r['event']=='shadow-start'];done=[r for r in rows if r['event']=='shadow-complete'];phases=[r for r in rows if r['event']=='cseed-shadow-phase'];times=collections.defaultdict(float)
for r in done:times[r['source'].split('/')[-1]]+=r['elapsed_seconds']
a={'version':1,'whole_release':False,'qualification':False,'starts':len(starts),'completions':len(done),'completed_failures':sum(r['failures'] for r in done),'completed_elapsed_sum':sum(r['elapsed_seconds'] for r in done),'unfinished_shadow':starts[-1],'phases':phases,'completed_elapsed_by_file':dict(times),'exclusive_cost_attribution':False,'flush_observed':False,'old_failure_exclusively_explained':False};(source/'analysis.json').write_text(json.dumps(a,indent=2)+'\n')
files=[];chunks={}
for n in ['artifact.zip','api-run.json','api-run-full.json','api-artifact.json','failed.log','failed-log.stderr','analysis.json']:
 p=source/n;keys=[];h=hashlib.sha256();size=0
 with p.open('rb') as stream:
  for data in iter(lambda:stream.read(2*1024*1024),b''):
   k=sha(data);keys.append(k);h.update(data);size+=len(data)
   if k not in chunks:
    packed=gzip.compress(data,mtime=0);(out/'objects'/(k+'.gz')).write_bytes(packed);chunks[k]={'bytes':len(data),'gzip_sha256':sha(packed)}
 files.append({'path':n,'sha256':h.hexdigest(),'size':size,'chunks':keys})
with zipfile.ZipFile(source/'artifact.zip') as z:members=[{'path':n,'bytes':len(z.read(n)),'sha256':sha(z.read(n))} for n in z.namelist()]
m={'version':1,'run':45998897143,'job':140505211888,'artifact':15203409836,'source_revision':'51c82debb858145371cd9f28005d683e0e79184f','product_revision':'6e09c07a9f1651294e4669282cdc17ee14126775','wrapper_zip_sha256':'bcfa246132b6d500023f22e51433c169682d30dab47a1c6fb43168b5cf9263fb','whole_release':False,'qualification':False,'whole_wrapper_zip_reconstructed':True,'files':files,'chunks':chunks,'members':members};raw=(json.dumps(m,indent=2)+'\n').encode();(out/'manifest.json').write_bytes(raw);print(json.dumps({'sha256':sha(raw),'files':len(files),'chunks':len(chunks),'members':len(members),'raw_bytes':sum(x['size'] for x in files)}))
