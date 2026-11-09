"""I audit actual collection native products against an immutable hosted source pin."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zipfile


def audit(archive, pin, output):
    output.mkdir(parents=True, exist_ok=False)
    inventory, texts = [], {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
            digest = hashlib.sha256()
            with z.open(info) as stream:
                for block in iter(lambda: stream.read(65536), b''): digest.update(block)
            inventory.append({'path':info.filename,'bytes':info.file_size,'sha256':digest.hexdigest()})
            if (info.filename.startswith('string-builtin-operands/') or
                'collection-handles-' in info.filename) and Path(info.filename).suffix in ('.json','.log','.txt','.stdout','.stderr','.sha256'):
                texts[info.filename] = z.read(info).decode()
        roots = sorted({p.rsplit('/',1)[0]+'/' for p in texts if p.endswith('/operations.json') and 'collection-handles-' in p})
        assert len(roots)==1
        root=roots[0]
        operations=json.loads(texts[root+'operations.json'])
        labels=[o['label'] for o in operations]
        assert labels==['build-plain','run-plain','build-sanitized','run-sanitized','build-threaded','run-threaded']
        assert all(o['returncode']==0 for o in operations)
        inputs=json.loads(texts[root+'inputs-before-after.json'])
        for path, expected in inputs.items():
            actual=subprocess.check_output(['git','show',pin+':'+path])
            assert hashlib.sha256(actual).hexdigest()==expected, path
        products=[]
        for name in ('plain','sanitized','threaded'):
            data=z.read(root+name)
            expected=json.loads(texts[root+name+'.identity.json'])['sha256_before_after']
            assert hashlib.sha256(data).hexdigest()==expected
            if data[:4]==b'\xcf\xfa\xed\xfe':
                assert struct.unpack_from('<I',data,12)[0]==2
                kind='Mach-O MH_EXECUTE'
            else:
                assert data[:4]==b'\x7fELF' and data[5] in (1,2)
                assert struct.unpack_from('<H' if data[5]==1 else '>H',data,16)[0] in (2,3)
                kind='ELF executable or PIE, corroborated by actual run operation'
            build=next(o for o in operations if o['label']=='build-'+name)['command']
            assert '-Werror' in build and '-UNDEBUG' in build and '-pthread' in build
            if name=='sanitized':
                assert '-fsanitize=address,undefined' in build and b'__asan_init' in data and b'__ubsan_handle' in data
            if name=='threaded': assert '-fsanitize=thread' in build and b'__tsan_init' in data
            assert texts[root+'run-'+name+'.stderr']==''
            assert texts[root+'run-'+name+'.stdout'].startswith('I pass collection kind, alias, stale, repeated-free')
            products.append({'name':name,'kind':kind,'bytes':len(data),'sha256_before_after':expected})
        assert len({p['sha256_before_after'] for p in products})==3
    report={'source_pin':pin,'archive':{'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'entries':len(inventory),'all_entries_streamed_crc_checked':True},
        'native_executables':products,'checked_inputs':inputs,'operations':operations,
        'compiler_report':json.loads(texts[root+'compiler.json']),
        'scope':'I independently hash actual uploaded native bytes and every archive entry; match all selected inputs to the immutable Git pin; check executable headers, sanitizer symbols, exact commands and actual stdout/stderr. Compiler identity is a retained before/after report, not independent toolchain closure. Full component conclusions are recorded separately. I qualify neither source-field admission nor release.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name, value in [('texts',texts),('inventory',inventory)]:
        (output/(name+'.json.gz')).write_bytes(gzip.compress((json.dumps(value,sort_keys=True)+'\n').encode(),mtime=0))
    print(json.dumps({'source_pin':pin,'entries':len(inventory),'native_executables':len(products)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('source_pin');p.add_argument('output',type=Path);a=p.parse_args();audit(a.archive,a.source_pin,a.output)
