from pathlib import Path
import subprocess,json,hashlib,time,sys
j=Path(__file__).resolve().parent;publication=Path('/var/tmp/nanolang-darwin-publication-qkk55gi3');pin='f588f39687b1863576472d5961baa71cef1a07f7';base='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8';sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=dict(terminal=False,source_pin=base,evidence_commit=pin,commands=[],receipts={},observer_helper_sha256=sha((j/'cohort_journal.py').read_bytes()),source_checkout_started=False)
try:
 assert not (j/'source').exists()
 assert subprocess.run(['git','merge-base','--is-ancestor',pin,'origin/feat/5.1-service-module-integration'],cwd=publication).returncode==0
 prefix='docs/evidence/darwin-current-source-20261124/diagnostics/'
 shared=prefix+'read_chunks.py';assert (publication/shared).read_bytes()==subprocess.check_output(['git','show',pin+':'+shared],cwd=publication)
 for name in ['affine-foundation-shadow-failure-linux-20261125','affine-local-shadow-localization-linux-20261125']:
  receipt=prefix+name;manifest_bytes=subprocess.check_output(['git','show',pin+':'+receipt+'/manifest.json'],cwd=publication);manifest=json.loads(manifest_bytes)
  assert (publication/receipt/'manifest.json').read_bytes()==manifest_bytes
  reader=publication/receipt/'read_archive.py';assert reader.read_bytes()==subprocess.check_output(['git','show',pin+':'+receipt+'/read_archive.py'],cwd=publication)
  if name.startswith('affine-foundation'):
   prior=Path('/var/tmp/nanolang-affine-recovery-five-20261125-d_u2vhbz/cohort-result.json');data=json.loads(prior.read_bytes());entry=next(r for r in manifest['files'] if r['path']=='comparison/cohort-result.json')
   assert data['terminal'] and data['all_guards_match'] and not data.get('accepted_component') and data['tests']==dict(tests=5,failures=1,errors=0,skips=0,passed=False)
   assert sha(prior.read_bytes())==entry['sha256'];q['failed_result_sha256']=entry['sha256']
  else:
   entries={r['path']:r for r in manifest['files']}
   assert q['observer_helper_sha256']==entries['observer-corrected/cohort_journal.py']['sha256']
   fixture=subprocess.check_output(['git','show',base+':src_nano/nvm2c/affine_locals.nano'],cwd=publication)
   assert sha(fixture)==entries['typed-fixture/src_nano/nvm2c/affine_locals.nano']['sha256'];q['typed_fixture_sha256']=sha(fixture)
  argv=[sys.executable,str(reader)];t=time.monotonic();p=subprocess.run(argv,cwd=publication,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180);(j/(name+'.reader.stdout')).write_bytes(p.stdout);(j/(name+'.reader.stderr')).write_bytes(p.stderr);q['commands'].append(dict(argv=argv,exit=p.returncode,seconds=time.monotonic()-t,timeout=180));assert p.returncode==0
  q['receipts'][name]=dict(manifest_sha256=sha(manifest_bytes),reader_sha256=sha(reader.read_bytes()))
 q['accepted']=True
finally:q['terminal']=True;(j/'preflight-result.json').write_text(json.dumps(q,indent=2)+'\n')
