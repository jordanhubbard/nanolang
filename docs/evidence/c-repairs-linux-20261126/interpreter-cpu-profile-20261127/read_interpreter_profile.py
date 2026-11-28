"""I check retained interpreter sampling bytes and their measured boundaries."""
from pathlib import Path
import argparse,gzip,hashlib,json,re,subprocess,tempfile,collections
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='b4c28416c3d93b24b19e689183a8b45fe36568d802273ee0337ea8b9b6433165';m=json.loads(raw);assert m['version']==1 and not m['qualification'] and not m['whole_release'] and m['product_source']=='6e09c07a9f1651294e4669282cdc17ee14126775'
with tempfile.TemporaryDirectory(prefix='interpreter-profile-reader-',dir=a.workspace) as t:
 root=Path(t);seen=set();used=set()
 for row in m['files']:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen;seen.add(str(rel));dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with dest.open('wb') as f:
   for k in row['chunks']:
    assert re.fullmatch('[a-f0-9]{64}',k);meta=m['chunks'][k];packed=(a.archive/'objects'/(k+'.gz')).read_bytes();assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed);assert len(data)==meta['bytes'] and sha(data)==k;f.write(data);h.update(data);size+=len(data);used.add(k)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 assert len(seen)==57 and len(used)==59 and used==set(m['chunks'])
 s=root/'software';f=root/'hardware';q=json.loads((f/'result.json').read_text());prior=json.loads((s/'result.json').read_text());audit=json.loads((s/'terminal-audit.json').read_text())
 assert q['terminal'] and q['guards_match'] and not q['qualification'] and not q['whole_release'] and q['compiler_exit']==1 and q['sampler_exit']==0
 assert (q['original_shadow_seconds'],q['outer_command_timeout'],q['sampler_timeout_seconds'],q['sampler_kill_after_seconds'])==(10,300,15,5)
 assert q['source_before']==q['source_after']==prior['source_before'] and q['tools_before']==q['tools_after']==prior['tools_before']
 assert audit['terminal'] and audit['sources_match'] and audit['tools_match'] and audit['compiler_pid_absent'] and audit['sampler_pid_absent']
 assert audit['source_after']==q['source_before'] and audit['tools_after']==q['tools_before'] and audit['compiler_exit_not_retained'] and audit['sampler_recording_refused']
 assert not prior['terminal'];assert 'PermissionError' in (s/'controller.stderr').read_text() and 'Failure to open any events' in (s/'perf.stderr').read_text()
 assert sha((s/'perf-readable.data').read_bytes())==audit['perf_data_sha256']
 for label,d in [('software',prior),('hardware',q)]:assert sha((root/label/'observe.py').read_bytes())==d['observer_sha256']
 binary_key=m['source_tree']+'/bin/nanoc_c';assert sha((root/'sampled-binary/nanoc_c').read_bytes())==q['tools_before'][binary_key]
 assert sha((root/'tools/perf').read_bytes())==q['tools_before']['/usr/bin/perf']
 assert sha((f/'perf-readable.data').read_bytes())==q['perf_data_sha256']
 expected=q['source_before'];assert len(expected)==4676
 tree=subprocess.check_output(['git','ls-tree','-rz',m['product_source']],cwd=a.repo);blobs={}
 for entry in tree.split(b'\0'):
  if entry:
   meta,name=entry.split(b'\t',1);mode,kind,oid=meta.split();name=name.decode()
   if name in expected:assert kind==b'blob';blobs[name]=oid.decode()
 assert set(expected)==set(blobs)
 batch=root/'source-objects'
 with batch.open('wb') as out:subprocess.run(['git','cat-file','--batch'],input=''.join(blobs[n]+'\n' for n in expected).encode(),stdout=out,cwd=a.repo,check=True,timeout=120)
 env_source=None
 with batch.open('rb') as data:
  for name,h in expected.items():
   oid,kind,size=data.readline().split();assert oid.decode()==blobs[name] and kind==b'blob';b=data.read(int(size));assert data.read(1)==b'\n' and sha(b)==h
   if name=='src/env.c':env_source=b
  assert data.read()==b''
 assert b'for (int i = 0; i < env->symbol_count; i++)' in env_source
 for label in ['build','build-materialized','build-complete-inputs']:assert json.loads((s/(label+'.json')).read_text())['exit']==2
 assert 'Makefile.gnu: No such file' in (s/'build.stderr').read_text()
 assert 'schema/compiler_schema.json' in (s/'build-materialized.stderr').read_text()
 assert 'spec/nanoisa.yaml' in (s/'build-complete-inputs.stderr').read_text()
 build=json.loads((s/'build-full-source.json').read_text());assert build['exit']==0 and build['timeout']==300
 def phases(directory):
  rows=[]
  for line in (directory/'compiler.stderr').read_text().splitlines():
   if line.startswith('{'):
    try:rows.append(json.loads(line))
    except ValueError:pass
  starts=[v for v in rows if v['event']=='shadow-start'];done=[v for v in rows if v['event']=='shadow-complete'];phases=[v for v in rows if v['event']=='cseed-shadow-phase']
  assert len(starts)==len(done)==1139 and all(v['failures']==0 for v in done)
  assert [(r['source'],r['line'],r['name']) for r in starts]==[(r['source'],r['line'],r['name']) for r in done]
  assert [v['phase'] for v in phases]==['dispatch-start','report-start','report-returned','dispatch-returned','completion-written','flush-start','flush-end','supervisor-status']
  assert phases[-1]['reaped'] and phases[-1]['completed'] and phases[-1]['exit_code']==0 and phases[-1]['deadline_seconds']==10
  return phases
 soft=phases(s);hard=phases(f);assert not soft[-1]['parent_timeout'] and hard[-1]['parent_timeout']
 assert hard[0]['pid']==q['sampled_identity']['pid']==m['sampled_pid']==2329556 and q['sampled_identity']['parent']==q['compiler_pid']
 assert q['sampled_identity']['dispatch']==hard[0]
 def ns(v,prefix):return v[prefix+'seconds']*10**9+v[prefix+'nanoseconds']
 supervisor_ns=ns(hard[-1],'monotonic_')-ns(hard[-1],'parent_start_');tail_ns=ns(hard[-1],'monotonic_')-ns(hard[-2],'monotonic_')
 assert supervisor_ns==10122881403 and tail_ns==625706947
 periods=collections.Counter();count=0
 for block in (f/'script.stdout').read_text().strip().split('\n\n'):
  lines=block.splitlines();match=re.match(r'^\S+\s+(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+):',lines[0]);assert match and int(match[1])==m['sampled_pid'] and match[4]=='armv8_pmuv3_1/cycles/u';count+=1
  frames=[l.strip() for l in lines[1:] if l.strip() and '(inlined)' not in l];assert frames;name=frames[0].split()[1].split('+')[0];periods[name]+=int(match[3])
 assert count==m['samples']==828 and sum(periods.values())==19573565123
 assert round(100*periods['env_define_evaluated_var']/sum(periods.values()),2)==13.71
 assert round(100*periods['symbol_lookup']/sum(periods.values()),2)==10.12
 report=(f/'report-self.stdout').read_text();assert 'Total Lost Samples: 0' in report and 'Samples: 828' in report and '13.71%' in report
 annotation=(f/'annotate-env_define_evaluated_var.stdout').read_text();assert 'env->symbols[i].value.type == VAL_STRING' in annotation and '51.39' in annotation
 control=root/'hardware-control';c=json.loads((control/'readable-result.json').read_text());assert c['decode_exit']==0 and c['samples']==23 and c['privileged_original_read_copy']
 assert sha((control/'perf-readable.data').read_bytes())==c['perf_data_sha256']
 assert sum(1 for l in (control/'script-readable.stdout').read_text().splitlines() if re.match(r'^python3\s+\d+.*armv8_pmuv3_\d/cycles/u:',l))==23
 print(json.dumps({'verified':True,'manifest_sha256':sha(raw),'records':len(seen),'chunks':len(used),'git_sources':len(expected),'hardware_samples':count,'weighted_periods':sum(periods.values()),'sampled_pid':m['sampled_pid'],'env_binding_first_noninline_percent':13.71,'symbol_lookup_first_noninline_percent':10.12,'completed_shadows_each_run':1139,'sampler_refusal_and_terminal_audit_retained':True,'sampled_supervisor_elapsed_ns':supervisor_ns,'sampled_post_flush_elapsed_ns':tail_ns,'fresh_perf_decode':False,'product_execution':False,'darwin_timing':False,'qualification':False,'whole_release':False}))
