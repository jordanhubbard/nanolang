"""I check my local STRING repair evidence without running a product."""
import argparse,gzip,hashlib,json,re,subprocess,tempfile,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='57fc7fef460539654ce6dcb5c9ba16a97fdaa7c11fb88ebd2a0760955640a09a';m=json.loads(raw);assert m['version']==1 and not m['qualification'] and not m['whole_release']
assert m['candidate']=='d18fed86140b6a4cf983898ac32a079002f6b784' and m['product_commit']=='2006c758df3c67c27f623ff9f7e1e1baaf56c136'
with tempfile.TemporaryDirectory(prefix='string-binding-hint-reader-',dir=a.workspace) as temporary:
 root=Path(temporary);seen=set();used=set()
 for row in m['files']:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen;seen.add(str(rel));dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with dest.open('wb') as f:
   for k in row['chunks']:
    assert re.fullmatch('[a-f0-9]{64}',k);packed=(a.archive/'objects'/(k+'.gz')).read_bytes();meta=m['chunks'][k];assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed);assert len(data)==meta['bytes'] and sha(data)==k;f.write(data);h.update(data);size+=len(data);used.add(k)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 assert len(seen)==135 and len(used)==182 and used==set(m['chunks'])
 load=lambda n:json.loads((root/n).read_text())
 counter=load('counter/result.json');broad=load('checks/broad.json');failed=load('emitter-refusal/result.json');hardware=load('hardware/result.json');san=load('sanitizer/result.json');corrected=load('sanitizer/corrected/result.json')
 for q,label in [(counter,'counter'),(failed,'emitter-refusal'),(hardware,'hardware')]:
  assert q['terminal'] and not q['qualification'] and not q['whole_release'] and q['guards_match'];assert q['source_before']==q['source_after'] and q['tools_before']==q['tools_after']
  assert sha((root/label/'observe.py').read_bytes())==q['observer_sha256']
 assert counter['compiler_exit']==0 and counter['command_timeout_seconds']==300 and counter['shadow_seconds']==10 and counter['reverse_cost_is_upper_bound']
 assert failed['compiler_exit']==1 and hardware['compiler_exit']==hardware['sampler_exit']==0
 assert broad['terminal'] and broad['exit']==0 and broad['guards_match'] and broad['source_before']==broad['source_after']
 assert failed['source_before']==broad['source_before']
 assert san['terminal'] and san['exit']==2 and san['ASAN_OPTIONS']=='detect_leaks=1:halt_on_error=1'
 assert corrected['terminal'] and corrected['compile_exit']==corrected['exit']==0 and corrected['ASAN_OPTIONS']==san['ASAN_OPTIONS']
 assert (root/'sanitizer/corrected/stderr').read_bytes()==b''
 assert load('sanitizer/index-confirm.json')['exit']==0 and (root/'sanitizer/index-confirm.stderr').read_bytes()==b''
 assert '55 byte(s) leaked in 4 allocation(s)' in (root/'sanitizer/stderr').read_text()
 baseline=load('sanitizer/baseline-literal-probe/result.json');assert baseline['probe_exit']==1 and baseline['header_declaration_only_additions'] and baseline['initial_probe_typecheck_refusal_retained']
 assert all(c['exit']==0 for c in baseline['commands']);assert '47 byte(s) leaked in 3 allocation(s)' in (root/'sanitizer/baseline-literal-probe/stderr').read_text()
 assert 'Program must define' in (root/'sanitizer/baseline-literal-probe/initial-stderr').read_text()
 for label,code in [('first',2),('second',0),('loop-gate',2),('loop-corrected',0),('fixture-cleanup',0)]:assert load('checks/'+label+'.json')['exit']==code
 assert 'defined but not used' in (root/'checks/first.stderr').read_text() and 'implicit declaration' in (root/'checks/loop-gate.stderr').read_text()
 negative=load('checks/negative-markers/result.json');assert [q['test_exit'] for q in negative]==[-6,1]
 assert all(q['compile_exit']==q['link_exit']==0 for q in negative)
 assert 'env_string_binding_seen(symbol->value)' in (root/'checks/negative-markers/missing-import-marker.stderr').read_text()
 assert 'Assertion failed at line 3' in (root/'checks/negative-markers/missing-loop-markers.stderr').read_text()
 # I bind every unchanged source and each approved diagnostic overlay to actual Git blobs.
 cache={}
 def git_sources(revision,expected,replacements=None):
  replacements=replacements or {};blobs={};tree=subprocess.check_output(['git','ls-tree','-rz',revision],cwd=a.repo)
  for entry in tree.split(b'\0'):
   if entry:
    meta,name=entry.split(b'\t',1);mode,kind,oid=meta.split();name=name.decode()
    if name in expected:assert kind==b'blob';blobs[name]=oid.decode()
  assert set(blobs)==set(expected)
  batch=root/('git-'+revision)
  with batch.open('wb') as out:subprocess.run(['git','cat-file','--batch'],input=''.join(blobs[n]+'\n' for n in expected).encode(),stdout=out,cwd=a.repo,check=True,timeout=120)
  with batch.open('rb') as stream:
   for name,h in expected.items():
    oid,kind,size=stream.readline().split();assert oid.decode()==blobs[name] and kind==b'blob';data=stream.read(int(size));assert stream.read(1)==b'\n'
    if name in replacements:assert sha(replacements[name])==h
    else:assert sha(data)==h,(revision,name)
    cache[(revision,name)]=data
   assert stream.read()==b''
 replacements={n:(root/'counter/overlay'/n).read_bytes() for n in counter['overlay_only']};assert set(replacements)=={'src/env.c','src/eval.c'}
 git_sources(m['original_c_source'],counter['source_before'],replacements)
 for n in replacements:assert (root/'counter/original'/n).read_bytes()==cache[(m['original_c_source'],n)]
 patch=(root/'counter/diagnostic.patch').read_bytes();assert sha(patch)==counter['diagnostic_patch_sha256']
 inverse=root/'inverse';inverse.mkdir()
 for n,data in replacements.items():dest=inverse/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 subprocess.run(['git','apply','--reverse',str(root/'counter/diagnostic.patch')],cwd=inverse,check=True,timeout=30)
 for n in replacements:assert (inverse/n).read_bytes()==cache[(m['original_c_source'],n)]
 git_sources(broad['head'],broad['source_before']);git_sources(m['candidate'],hardware['source_before'])
 assert broad['head']=='f4e220ff1c69ce19000961de98caad3935919974'
 assert m['candidate'].encode() in (root/'hardware/observe.py').read_bytes()
 san_sources=load('sanitizer-source-after.json');assert san_sources['source_after']==broad['source_before'] and not san_sources['initial_prebuild_source_map_recorded']
 for n in ['src/env.c','src/eval.c','src/module.c','src/nanolang.h','src/runtime/gc.h','tests/test_env_string_binding_hint.c','Makefile.gnu','.github/workflows/c-repair-original-gates.yml']:
  assert (root/'source'/n).read_bytes()==cache[(m['candidate'],n)]
 assert (root/'sanitizer/corrected/source/tests/test_env_string_binding_hint.c').read_bytes()==(root/'source/tests/test_env_string_binding_hint.c').read_bytes()
 assert (root/'sanitizer/corrected/source/src/env.c').read_bytes()==(root/'source/src/env.c').read_bytes()
 for name in ['env','eval','module']:
  expected=cache[(m['original_c_source'],'src/'+name+'.c')]
  assert (root/'sanitizer/baseline-literal-probe'/(name+'.c')).read_bytes()==expected
 for name in ['module','eval']:assert (root/'checks/negative-markers'/(name+'.c')).read_bytes()==cache[(m['original_c_source'],'src/'+name+'.c')]
 original=cache[(m['original_c_source'],'src/runtime/gc.h')];new=(root/'source/src/runtime/gc.h').read_bytes()
 block=lambda b:b[b.index(b'typedef struct GCHeader {'):b.index(b'} GCHeader;')+len(b'} GCHeader;')]
 assert block(original)==block(new)
 assert b'GC_FLAG_ENV_STRING_BOUND UINT16_C(1)' in new
 for label,q,binary in [('counter',counter,'counter/nanoc_c'),('hardware',hardware,'product/nanoc_c')]:
  key=next(n for n in q['tools_before'] if n.endswith('/bin/nanoc_c'));assert sha((root/binary).read_bytes())==q['tools_before'][key]
 assert failed['tools_before'][next(n for n in failed['tools_before'] if n.endswith('/bin/nanoc_c'))]==hardware['tools_before'][next(n for n in hardware['tools_before'] if n.endswith('/bin/nanoc_c'))]
 def trace(directory):
  records=[]
  for line in (directory/'compiler.stderr').read_text().splitlines():
   if line.startswith('{'):
    try:records.append(json.loads(line))
    except ValueError:pass
  starts=[x for x in records if x['event']=='shadow-start'];done=[x for x in records if x['event']=='shadow-complete'];phases=[x for x in records if x['event']=='cseed-shadow-phase'];assert all(x['failures']==0 for x in done)
  return starts,done,phases,records
 cs,cd,cp,cr=trace(root/'counter');assert len(cs)==len(cd)==1139 and not cp[-1]['parent_timeout']
 previous={};snapshots=0
 for r in cr:
  if r['event']!='alias-scan-counts':continue
  operation=r['operation'];prior=previous.get(operation,{k:0 for k in r if k not in ['event','operation']})
  assert r['hits']+r['misses']==r['loops'] and r['forward_probes']<=r['total_slots'] and r['reverse_upper_probes']<=r['total_slots']
  assert all(r[k]>=prior[k] for k in prior);previous[operation]={k:v for k,v in r.items() if k not in ['event','operation']};snapshots+=1
 assert snapshots==2278
 assert [previous[0][n] for n in ['loops','hits','misses','forward_probes','reverse_upper_probes']]==[91472,883,90589,746387154,738957085]
 assert [previous[1][n] for n in ['loops','hits','misses','forward_probes','reverse_upper_probes']]==[8566,774,7792,72875558,66306452]
 fs,fd,fp,_=trace(root/'emitter-refusal');assert len(fs)==1128 and len(fd)==1127 and fp[-1]['parent_timeout'] and fp[-1]['signal']==14 and not fp[-1]['completed']
 hs,hd,hp,_=trace(root/'hardware');assert len(hs)==len(hd)==1139 and hp[-1]['completed'] and not hp[-1]['parent_timeout'] and hp[-1]['exit_code']==0
 ns=lambda v,p:v[p+'seconds']*10**9+v[p+'nanoseconds'];elapsed=ns(hp[-1],'monotonic_')-ns(hp[-1],'parent_start_');assert elapsed==9756433114
 events=collections.Counter();periods=collections.Counter();samples=0
 for item in (root/'hardware/script.stdout').read_text().strip().split('\n\n'):
  lines=item.splitlines();match=re.match(r'^\S+\s+(\d+)\s+[\d.]+:\s+(\d+)\s+(\S+):',lines[0]);assert match and int(match[1])==hardware['sampled_identity']['pid'];assert match[3] in ['armv8_pmuv3_0/cycles/u','armv8_pmuv3_1/cycles/u'];events[match[3]]+=1;samples+=1
  frames=[l.strip() for l in lines[1:] if l.strip() and '(inlined)' not in l];assert frames;periods[frames[0].split()[1].split('+')[0]]+=int(match[2])
 assert samples==823 and sum(periods.values())==17065939822
 assert round(100*periods['env_define_evaluated_var']/sum(periods.values()),2)==1.53
 assert 'Total Lost Samples: 0' in (root/'hardware/report-self.stdout').read_text()
 assert sha((root/'hardware/perf-readable.data').read_bytes())==hardware['perf_data_sha256']
 print(json.dumps({'verified':True,'manifest_sha256':sha(raw),'records':len(seen),'chunks':len(used),'counter_source_files':len(counter['source_before']),'candidate_source_files':len(hardware['source_before']),'counter_snapshots':snapshots,'declaration_misses':90589,'declaration_probes':746387154,'broad_gate_pass':True,'marker_removal_controls_refuse':True,'initial_sanitizer_failure_retained':True,'corrected_leak_enabled_sanitizer_pass':True,'baseline_literal_leak_bytes':47,'original_emitter_refusal_retained':True,'hardware_shadows_completed':1139,'hardware_samples':samples,'hardware_weighted_periods':sum(periods.values()),'binding_first_noninline_percent':1.53,'supervisor_elapsed_ns':elapsed,'header_layout_unchanged':True,'product_execution':False,'fresh_perf_decode':False,'hosted_qualification':False,'whole_release':False}))
