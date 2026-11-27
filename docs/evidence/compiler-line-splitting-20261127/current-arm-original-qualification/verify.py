"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import subprocess,re,struct
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='2b689b71f0ba9f4f1b9d9b8095afb60905641932aa41f021f43be8fe5ed88c66'
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
 module=(qualified/'bin/nanoc_stage2.nvm').read_bytes();assert module==(qualified/'bin/nanoc_stage1.nvm').read_bytes() and len(module)==1064548
 assert sha(module)=='d3d18d9dbc4008837635031d25ecec1e17d085372147bff6c057fa6be22a4470'
 native=(qualified/'bin/nanoc_stage2').read_bytes();assert native[:4]==b'\x7fELF' and struct.unpack_from('<H',native,18)[0]==183
 assert sha(native)=='fbf5ff3b97097992772cad62273b25f7d678da25f28b5fb83affaba93ed0ec3e' and native==(qualified/'bin/nanoc').read_bytes()
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
 lifetime=(root/'observation/original/line-regression/stdout').read_text();assert lifetime.count('I retained ')==9
 metrics=list(qualified.glob('.test_output/compiler-line-splitting/*/length-work.json'));assert len(metrics)==1
 assert json.loads(metrics[0].read_text())=={'input_bytes':5955,'observed':{'prior':{'calls':6123,'requested_bytes':18236773},'current':{'calls':28,'requested_bytes':5421}}}
 assert not m['whole_release'] and not m['qualification_execution']
print('I verified current ARM original17/raw equality/installed/no-C-seed/36/12, nine direct lifetime cases and all-five-route regression with source/tool/host guards. Full paired consumers/platform/release remain separate.')
