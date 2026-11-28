"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import subprocess,re,struct
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='cc74ad2109d32ed3c027a0b9b76a4252f7fae52fe7119721c1741c4a1bca9b26'
m=json.loads(raw)
with tempfile.TemporaryDirectory(prefix='nano-arm-receipt-',dir=args.workspace) as d:
 root=Path(d)
 for row in m['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
  output=root/name;output.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with output.open('wb') as stream:
   for key in row['chunks']:
    packed=(b/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed)
    assert len(data)==meta['bytes'] and sha(data)==key
    stream.write(data);h.update(data);size+=len(data)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 load=lambda n:json.loads((root/n).read_text())
 revision=m['revision'];qualified=root/'qualified-root'
 gates={name:load('observation/original/'+name+'/result.json') for name in ['bootstrap','native-suites','line-regression']}
 for name,g in gates.items():assert g['qualification'] and not g['whole_release'] and g['source_head']==revision and g['exit']==0 and not g['command_timeout'] and g['guards_match'] and g['before']==g['after']
 q=load('observation/original/bootstrap-receipt.json')
 assert q['phase']=='stage2' and q['stage_timeout_seconds']==1800 and q['shadow_timeout_seconds']==30 and q['raw_stage1_stage2_equal']
 labels=['seed','seed-verify','stage1','stage1-verify','stage1-translate','stage1-native','stage1-hello-compile','stage1-hello-execute','stage2','stage2-verify','stage2-translate','stage2-native','stage2-hello-compile','stage2-hello-execute','seed-final-verify','stage1-final-verify','stage2-final-verify']
 assert [s['label'] for s in q['steps']]==labels and all(s['exit_code']==0 and s['elapsed_seconds']<1800 for s in q['steps'])
 for s in q['steps']:assert sha((qualified/Path(s['log']).relative_to(q['root'])).read_bytes())==s['log_sha256']
 for table in ['tools','artifacts']:
  for name,h in q[table].items():assert sha((qualified/'bin'/name).read_bytes())==h
 for name,h in q['hosts'].items():assert sha((qualified/Path(name).relative_to(q['root'])).read_bytes())==h
 assert len(q['hosts'])==6
 for row in m['selected_tools']:assert sha((root/row['member']).read_bytes())==row['sha256']
 module=(qualified/'bin/nanoc_stage2.nvm').read_bytes();assert module==(qualified/'bin/nanoc_stage1.nvm').read_bytes() and len(module)==1064608
 assert sha(module)=='f0a6609eae2d9b81d70b5b8345ab096339e0f7b6e521099cf3ac75f1e2e24abe'
 native=(qualified/'bin/nanoc_stage2').read_bytes();assert native[:4]==b'\x7fELF' and struct.unpack_from('<H',native,18)[0]==183
 assert sha(native)=='bccfa964cb5d2cc8652bb456592c00f74f7bcb330725631e3f34e84ecd3a803a' and native==(qualified/'bin/nanoc').read_bytes()
 assert 'bin/nanoc succeeded with bin/nanoc_c removed' in (root/'observation/original/bootstrap/stdout').read_text()
 sources=list(q['sources']);request=''.join(revision+':'+n+'\n' for n in sources).encode();data=subprocess.check_output(['git','cat-file','--batch'],cwd=args.repo,input=request);offset=0;generated=[]
 for name in sources:
  assert not Path(name).is_absolute() and '..' not in Path(name).parts and '\n' not in name
  end=data.index(b'\n',offset);header=data[offset:end].split()
  if header[-1]==b'missing':generated.append(name);blob=(qualified/name).read_bytes();offset=end+1
  else:size=int(header[-1]);blob=data[end+1:end+1+size];offset=end+size+2
  assert sha(blob)==q['sources'][name],name
 assert offset==len(data) and len(sources)==1383 and generated==['modules/index.json']
 for name in ['tests/test_native_shadows.py','tests/test_shadow_runner.py','tests/test_make_timeouts.py','tests/test_native_shadow_only_entry.py','tests/test_native_shadow_helpers.py','tests/test_native_shadow_emitter.py']:
  assert subprocess.check_output(['git','show','6e09c07a9f1651294e4669282cdc17ee14126775:'+name],cwd=args.repo)==subprocess.check_output(['git','show',revision+':'+name],cwd=args.repo)
 log=(root/'observation/original/native-suites/stderr').read_text()
 for count in [36,12]:assert re.search(r'^Ran '+str(count)+r' tests in [^\n]+\n\nOK(?:\n|$)',log,re.M)
 assert 'FAILED' not in log and 'skipped=' not in log
 log=(root/'observation/original/line-regression/stderr').read_text();assert re.search(r'^Ran 2 tests in [^\n]+\n\nOK(?:\n|$)',log,re.M)
 lifetime=(root/'observation/original/line-regression/stdout').read_text()
 for marker in ['I preserve caller declarations across refusals, collect reachable children and hand off without owning borrowed results.','I preserve custody declarations through 6 allocation failures and recover.']:assert lifetime.count(marker)==1
 caller=[json.loads(line) for line in lifetime.splitlines() if line.startswith('{"case":') and 'expected_reclaims' in line]
 assert len(caller)==10 and len({row['case'] for row in caller})==10 and sum(row['actual_reclaims'] for row in caller)==84
 assert all(row['expected_reclaims']==row['actual_reclaims'] for row in caller)

 for label in ['empty','plain tail','newline boundaries','multi-byte delimiter','CRLF','UTF8 segments','empty-delimiter bytes','empty input and delimiter','nonoverlapping matches']:
  assert 'I retained '+label+' across input and frame retirement.' in lifetime
 rows=[json.loads(line) for line in lifetime.splitlines() if line.startswith('{"owners":')]
 assert len(rows)==12 and [row['owners'] for row in rows]==[16,64,256,1024,4096,16]*2
 assert all(row['within_budget'] and row['maximum_bucket']<row['bucket_budget']==row['owners']*8 for row in rows)
 for count in [15,16,17,31]:assert lifetime.count(f'I preserved both essential allocation failures around a {count}-owner boundary.')==2
 for name in ['tests/test_gc_registry_working_set.c','tests/test_gc_registry_growth_failure.c','tests/test_gc_allocation_failure.c','tests/test_gc_struct.c','src/runtime/gc.c','src/runtime/gc.h','Makefile.gnu','.github/workflows/c-repair-original-gates.yml','tests/test_eval_caller_root_handoff.c','tests/support/ephemeral_array_cases.inc','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','src/eval/carrier_values.c','src/eval/carrier_values.h','src/eval.c']:
  data=(root/'qualified-source'/name).read_bytes();assert sha(data)==gates['line-regression']['after']['sources'][name]
  assert data==subprocess.check_output(['git','show',revision+':'+name],cwd=args.repo)

 metrics=list(qualified.glob('.test_output/compiler-line-splitting/*/length-work.json'));assert len(metrics)==1
 assert json.loads(metrics[0].read_text())=={'input_bytes':5955,'observed':{'prior':{'calls':6123,'requested_bytes':18236773},'current':{'calls':28,'requested_bytes':5421}}}
 assert not m['whole_release'] and not m['qualification_execution']
print('I verified current ARM original17/raw equality/installed/no-C-seed/36/12, new normal/sanitized GC controls, nine lifetime cases and all-five-route regression with source/tool/host guards. Full paired consumers/platform/release remain separate.')
