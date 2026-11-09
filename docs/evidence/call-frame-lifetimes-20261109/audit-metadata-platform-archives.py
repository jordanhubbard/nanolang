from pathlib import Path
import zipfile,json,gzip,hashlib,subprocess,struct,functools,sys
compiler=sys.argv[1];artifact=int(sys.argv[2]);run=int(sys.argv[3]);pin='f92237c52a894eea7d4ee13de5dfc4df224a99f9'
a=Path(f'.test_output/call-metadata-complete-{compiler}-{run}.zip');out=Path(f'.test_output/call-metadata-complete-{compiler}-audit');out.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
@functools.cache
def gitsha(p):return sha(subprocess.check_output(['git','show',pin+':'+p]))
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names));inventory=[];texts={};hashes={};native=[];binary=[]
 for n in names:
  assert not n.startswith('/') and '..' not in Path(n).parts
  b=z.read(n);h=sha(b);hashes[n]=h;inventory.append(dict(path=n,bytes=len(b),sha256=h))
  if b[:4]==b'\x7fELF':
   kind=struct.unpack_from('<H' if b[5]==1 else '>H',b,16)[0];native.append(dict(path=n,sha256=h,kind=kind,format='ELF'))
  elif b[:4] in (bytes.fromhex('cffaedfe'),bytes.fromhex('feedfacf')):
   kind=struct.unpack_from('<I' if b[:4]==bytes.fromhex('cffaedfe') else '>I',b,12)[0];native.append(dict(path=n,sha256=h,kind=kind,format='Mach-O'))
  try:texts[n]=b.decode()
  except UnicodeDecodeError:binary.append(n)
 prefix='string-builtin-operands/'
 assert z.read(prefix+'source-tree.txt')==subprocess.check_output(['git','ls-tree','-r',pin])
 assert texts[prefix+'revision.txt'].strip()==pin
 providerhashes=dict(line.split(None,1)[::-1] for line in texts[prefix+'providers.sha256'].splitlines());providerhashes={p.strip():h for p,h in providerhashes.items()}
 provider_after=texts[prefix+'providers-after.log'].splitlines();assert len(provider_after)==len(providerhashes) and all(x.endswith(': OK') for x in provider_after)
 stable=0;native_products=0;scalar_products=0;git_inputs=set();uploaded_inputs=set();provider_inputs=set();unbound=set()
 for n in names:
  if not n.endswith('/inputs.json'):continue
  x=json.loads(z.read(n))
  if 'before' not in x:
   assert 'source_before' in x and 'provider_before' in x,n
   source=x.get('actual_source',x.get('source'));assert source,n
   if 'source_sha256' in x:assert x['source_sha256']==x['source_before'],n
   x['before']={source:x['source_before'],x['provider']:x['provider_before']}
   x['after']={source:x['source_after'],x['provider']:x['provider_after']}
  assert x['before']==x['after'],n;stable+=1
  for p,h in x['before'].items():
   if '/.test_output/' in p:
    relative=p.split('/.test_output/',1)[1]
    if relative in hashes:assert hashes[relative]==h,(n,p);uploaded_inputs.add(relative)
    else:unbound.add(p)
   else:
    relative=p.split('/nanolang/nanolang/',1)[-1]
    if relative in providerhashes:assert providerhashes[relative]==h,(n,p);provider_inputs.add(relative)
    elif relative.startswith(('src/','src_nano/','modules/','std/','stdlib/','scripts/','tests/')):
     assert gitsha(relative)==h,(n,p);git_inputs.add(relative)
    else:unbound.add(p)
  for product in x.get('native_products',[]):
   assert product['before']==product['after'];p=product['path'].split('/.test_output/',1)[1];assert hashes[p]==product['before'];native_products+=1
  if 'product_before_run' in x:
   if x['product_before_run'] is None and x['product_after_run'] is not None:
    assert n.startswith(('bstring-vm-contracts/test_wrong_builtin_types_preserve_output-','bstring-source-vm-contracts/test_wrong_builtin_types_preserve_output-','bstring-source-vm-contracts/test_binary_owner_operators_refuse_mixed_or_ordered_operands-')),n
    prior=b'I retain my previous binary-string program.\n';assert x['product_after_run']==sha(prior)
    assert z.read(str(Path(n).parent/'program.nvm'))==prior
    commands=[json.loads(line) for line in texts[str(Path(n).parent/'commands.jsonl')].splitlines()]
    assert len(commands)==1 and commands[0]['label']=='build' and commands[0]['returncode']!=0
   else:
    assert x['product_before_run']==x['product_after_run'],n
    if x['product_before_run'] is not None:assert x['product_before_run'] in hashes.values(),n;scalar_products+=1
 assert '=== 2458 passed, 0 failed ===' in texts[prefix+'translator.log']
 for name,obj in [('inventory',inventory),('texts',texts)]:
  (out/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(obj).encode(),mtime=0))
 report=dict(run=run,artifact=artifact,head=pin,compiler=compiler,archive_sha256=sha(a.read_bytes()),entries_crc_hash_checked=len(names),source_tree_git_matched=True,reported_stable_providers=len(providerhashes),stable_input_reports=stable,git_matched_unique_inputs=len(git_inputs),uploaded_hash_matched_unique_inputs=len(uploaded_inputs),hash_matched_provider_reports=len(provider_inputs),native_products_before_after_bytes_matched=native_products,scalar_products_before_after_bytes_matched=scalar_products,native_entries=native,binary_entries=binary,unbound_input_paths=sorted(unbound),original_translator_passed=2458,archive_retained=str(a),scope='I CRC/hash-check every retained archive entry, compare the complete source tree to Git, verify before/after input reports and available Git/provider/uploaded identities, and bind reported products to their uploaded bytes. I do not claim all omitted tools/providers are retained, every subprocess has an independent witness, source reliability, full SDK closure or release qualification.')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('native_entries','binary_entries','unbound_input_paths')}));print('native',len(native),'binary',len(binary),'unbound',len(unbound))
