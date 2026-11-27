import gzip, hashlib, json, subprocess
from pathlib import Path
root=Path.cwd().resolve()
base=root/'docs/evidence/capture-complete-hosted-20261122'
manifest_path=base/'manifest.json'
read=lambda name:subprocess.check_output(['git','show','HEAD:'+str(name.relative_to(root))])
raw=read(manifest_path)
assert hashlib.sha256(raw).hexdigest()=='1e9c5d0da4b187f976607d487fcf8c6cdec203516060ac4d88c7817a6878c489'
base.mkdir(parents=True,exist_ok=True);manifest_path.write_bytes(raw)
manifest=json.loads(raw)
prefix='observer/hosted-fixed-45009129801/fixed-as246/'
names=['build.json','binutils-2.46.0.tar.xz','input-scrub-before.c',
       'input-scrub-fixed.c','bin/as','baseline-gas.sum','fixed-gas.sum',
       'witness/results.json']
names += ['witness/'+tool+'-'+case+'.o' for case in
          ['actual-markers','without-directory','without-predef','source-marker-only']
          for tool in ['baseline','fixed']]
records=[]
for name in names:
    rows=[r for r in manifest['files'] if r['path']==prefix+name]
    assert len(rows)==1
    row=rows[0];path=(base/row['artifact']).resolve()
    assert path.is_relative_to(root/'docs/evidence')
    packed=read(path);data=gzip.decompress(packed)
    assert len(packed)==row['compressed_bytes'] and len(data)==row['bytes']
    assert hashlib.sha256(packed).hexdigest()==row['compressed_sha256']
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(packed)
    records.append({'logical':row['path'],'path':str(path.relative_to(root)),
                    'sha256':row['compressed_sha256']})
Path('.test_output/stage2-emitter-controls/retained-fixtures.json').write_text(json.dumps(records,indent=2)+'\n')