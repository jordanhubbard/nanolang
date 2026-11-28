"""I verify local chunk archives before reading their declared outcomes."""
from pathlib import Path
from functools import lru_cache
import gzip,hashlib,json

def read_archive(base, expected):
    base=Path(base).resolve();anchor=base.parent.parent
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    assert sha((base/'manifest.json').read_bytes())==expected
    manifest=json.loads((base/'manifest.json').read_bytes())
    for parent in manifest.get('base_manifests',[]):
        path=(base/parent['path']).resolve();assert path.is_relative_to(anchor)
        assert sha(path.read_bytes())==parent['sha256']
    rows={r['path']:r for r in manifest['files']};assert len(rows)==len(manifest['files'])
    for name in [*rows,*[r['path'] for r in manifest['symlinks']]]:
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    @lru_cache(maxsize=16)
    def chunk(digest):
        info=manifest['chunks'][digest];path=(base/info.get('artifact','objects/'+digest+'.gz')).resolve()
        assert path.is_relative_to(anchor);zipped=path.read_bytes()
        assert len(zipped)==info['compressed_bytes'] and sha(zipped)==info['compressed_sha256']
        raw=gzip.decompress(zipped);assert len(raw)==info['bytes'] and sha(raw)==digest
        return raw
    for h in manifest['chunks']:chunk(h)
    for row in rows.values():
        digest=hashlib.sha256();size=0
        for h in row['chunks']:raw=chunk(h);digest.update(raw);size+=len(raw)
        assert size==row['bytes'] and digest.hexdigest()==row['sha256'],row['path']
    def data(name):return b''.join(chunk(h) for h in rows[name]['chunks'])
    def js(name):return json.loads(data(name))
    return manifest,rows,data,js
