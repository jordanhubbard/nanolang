#!/usr/bin/env python3
"""I account for one observed cache delta without accepting my ordinary heap gate."""
from pathlib import Path
import gzip,hashlib,json,re,struct,subprocess
base=Path(__file__).resolve().parent;m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest();priorpath=base/m['base_manifest']['path'];assert h(priorpath.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(priorpath.read_text());rows={}
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'];b=gzip.decompress(z);assert h(b)==r['sha256'] and len(b)==r['bytes'];assert r['path'] not in rows;rows[r['path']]=b
for r in m['git_sources']:
 b=subprocess.check_output(['git','show',m['source_pin']+':'+r['path']],cwd=base);assert b==rows['source/'+r['path']] and h(b)==r['sha256']
def js(name):return json.loads(rows['journal/'+name+'.json'])
q=js('observed');image=q['image'];expected='01164f0cec7feee171ad7e1ef6affc31d7f2728b970687f5262af20f79d2d0b6';assert q['before']==q['after']==h(rows['provider/'+image])==expected;assert q['timeout']==90 and q['seconds']<90 and q['exit']==0 and not q['allocator_overrides_present']
old=next(r for r in prior['files'] if r['path']=='provider/'+image);assert old['sha256']==expected
old_driver=next(r for r in prior['files'] if r['path']=='baseline/native-allocation-driver.c');driver=rows['input/.test_output/carrier-string-scopes-20261123/native-allocation-driver.c'];assert h(driver)==old_driver['sha256'] and b'last_bytes>middle_bytes+1024' in driver
assert h(rows['input/tests/fixtures/carrier_active_roots.nano'])=='dca702feb197ee94747f4d1f0060764078f2b30cbff8481f6e550575b8a44c83'
for prefix in ['first-','second-']:
 assert b'optimized out' in rows['journal/'+prefix+'run.stderr'] or b'optimized out' in rows['journal/'+prefix+'run.stdout']
 assert b'TCACHE_SAMPLE ' not in rows['journal/'+prefix+'run.stdout'] and b'INFERIOR_EXIT {"exit": null}' in rows['journal/'+prefix+'run.stdout']
text=rows['journal/run.stdout'].decode();assert b'Python Exception' not in rows['journal/run.stderr']
cache=[json.loads(s.removeprefix('TCACHE_SAMPLE ')) for s in text.splitlines() if s.startswith('TCACHE_SAMPLE ')];assert [r['calls'] for r in cache]==[64,128]
for r in cache:
 seen=set();total=0
 for b in r['bins']:
  assert b['count']==len(b['chunks']) and 0<b['count']<=7 and 0<=b['index']<64
  for c in b['chunks']:
   address=int(c['address'],16);assert address%16==0 and address not in seen;seen.add(address);assert c['chunk_bytes']==32+16*b['index'];total+=c['chunk_bytes']
 assert total==r['total_chunk_bytes']
assert [r['total_chunk_bytes'] for r in cache]==[11024,12080]
a={b['index']:b['count'] for b in cache[0]['bins']};b={b['index']:b['count'] for b in cache[1]['bins']};assert [(i,a.get(i,0),b.get(i,0)) for i in sorted(a.keys()|b.keys()) if a.get(i,0)!=b.get(i,0)]==[(31,3,5)]
positive,before,middle,last,delta,calls=map(int,re.search(r'positive=(\d+) before=(\d+) after64=(\d+) after128=(\d+) delta=(\d+) calls=(\d+)',text).groups());assert positive==2101904 and calls==128 and delta==last-middle==cache[1]['total_chunk_bytes']-cache[0]['total_chunk_bytes']==1056 and delta>1024
assert text.count('INFERIOR_EXIT {"exit": 94}')==1 and 'after64=7 after128=7 peak=7 destroyed=3755' in text
script=rows['journal/snapshot.gdb'];assert b'set disable-randomization off' in script and b'set debuginfod enabled off' in script
assert b'encoded^(p>>12)' in script and b'assert p==0' in script and b'libc+0x1afd90' in script
libc=rows['provider//lib/aarch64-linux-gnu/libc.so.6'];assert libc.startswith(b'\x7fELF')
# I bind the TLS-slot calculation to the actual retained AArch64 malloc instructions.
phoff=struct.unpack_from('<Q',libc,32)[0];size,count=struct.unpack_from('<HH',libc,54)
for i in range(count):
 typ,flags,off,va,pa,fs,ms,align=struct.unpack_from('<IIQQQQQQ',libc,phoff+i*size)
 if typ==1 and va<=0x964cc and 0x964cc+12<=va+fs:
  assert libc[off+0x964cc-va:off+0x964cc-va+12]==bytes.fromhex('d40800b094ca46f956d03bd5');break
else:raise AssertionError('I did not find the retained malloc instructions')
assert m['acceptance']==dict(cache_delta_accounted_for_observed_run=True,ordinary_native_byte_gate=False,historical_1776_cause=False,default_activation=False,whole_release=False)
print(f'PASS {len(rows)} records; unchanged image; inferior exit94; cache +1056 bytes == observed native delta; ordinary/historical/default acceptance open')
