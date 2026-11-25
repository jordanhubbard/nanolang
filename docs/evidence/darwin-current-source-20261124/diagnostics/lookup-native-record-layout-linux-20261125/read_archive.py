"""I verify read-only generated-source and machine-code facts, not runtime cost."""
from pathlib import Path
import sys,json,hashlib,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'766fb96db19d5535dd70e97f38d411f07f09f1a8d62cb34375a80a7e74402388');q=js('inspection/result.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert q['terminal'] and q['read_only'] and not q['executes_product'] and q['guards_match'] and q['exit']==q['get_exit']==0
assert not q['production_edits'] and not q['performance_measurement'] and not q['historical_cause']
assert not m['executes_product'] and not m['performance_claim'] and not m['historical_cause'] and not m['whole_release']
assert q['source_pin']==m['source_commit']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
assert q['image_sha256']==q['image_after']==q['image_after_get']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert q['c_source_sha256']==q['c_source_after']
for name,key in [('image','image_sha256'),('c_source','c_source_sha256'),('tool','tool_sha256')]:assert rows['external/'+q[name].lstrip('/')]['sha256']==q[key]
source=data('external/'+q['c_source'].lstrip('/'))
for name in ['lookup-helper.c.txt','record-layout.c.txt','array-get.c.txt']:assert data('inspection/'+name).strip() in source
assert b'ra[0] = r[0].ra[27]' in data('inspection/lookup-helper.c.txt')
assert b'nrarr_get(ra[2], t[5])' in data('inspection/lookup-helper.c.txt')
assert b'return a->data[idx]' in data('inspection/array-get.c.txt')
assert b'int64_t f[80]' in data('inspection/record-layout.c.txt')
lookup=data('inspection/gdb.stdout').decode();get=data('inspection/get.stdout').decode()
assert '<nrarr_get>' in lookup and lookup.count('<memcpy@plt>')==q['lookup_entry_memcpy_calls']==2
assert re.search(r'mov\s+x2, #0x1230',get) and '<memcpy@plt>' in get and q['machine_record_copy_bytes']==int('1230',16)==4656
assert not q['type_query_available'] and q['type_query_error_retained']
assert b'No symbol "nrec_t" in current context.' in data('inspection/gdb.stderr')
for argv in [q['argv'],q['get_argv']]:assert not any(v in ['run','start','continue','attach'] for v in argv)
prior=base.parent/'lookup-hoisted-cold-bootstrap-linux-20261125/manifest.json';raw=prior.read_bytes();assert sha(raw)=='d232ed852a6f375dc76f1d34cac02b22c87aa05c5b220073180f8dbbe39d7e49'
old=json.loads(raw);oldrows={r['path']:r for r in old['files']};assert old['source_commit']==q['source_pin']
assert oldrows['bootstrap/source/bin/nanoc_stage2']['sha256']==q['image_sha256']
assert any(r['path'].endswith('/stage2.c') and r['sha256']==q['c_source_sha256'] for r in old['files'])
print('PASS %d records: exact5a537 image/C source and read-only GDB commands; 4656-byte nrarr_get return copy and helper call, two entry copies, initial unavailable-type query retained. No product execution, measured bottleneck, fix, historical cause, performance gain or release acceptance.'%len(rows))
