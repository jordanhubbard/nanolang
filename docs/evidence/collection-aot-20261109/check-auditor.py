import copy,importlib.util,json,sys,tempfile,zipfile,traceback
from pathlib import Path
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-corrected-hosted.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
archive=Path(sys.argv[1])
results=[]
with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(prefix='nanolang-aot-auditor-') as tmp:
 root=next(n.removesuffix('products.json') for n in source.namelist() if n.startswith('collection-aot-') and n.endswith('/products.json'))
 products=json.loads(source.read(root+'products.json'))
 product=next(iter(products)).split('/.test_output/',1)[1]
 cases=[('native-bytes',product,'assert digest(data) == report'),('source-input',root+'inputs-tools-before-after.json','assert digest(subprocess'),('operation-status',root+'operations.json',"assert op['returncode']"),('missing-refusal',root+'test_companion_and_marker_origin_refusals-both-dependency-refuse-O0.stderr',"assert stdout == b'' and b'I stopped"),('prior-output',root+'nl_hm_new-result.c',"assert z.read(target)")]
 for i,(name,path,expected) in enumerate(cases):
  data=source.read(path)
  if name=='native-bytes': data=data[:-1]+bytes([data[-1]^1])
  elif name=='source-input':
   d=json.loads(data); key=next(iter(d['inputs'])); d['inputs'][key]='0'*64; data=json.dumps(d).encode()
  elif name=='operation-status':
   d=json.loads(data); d[0]['returncode']=9; data=json.dumps(d).encode()
  else: data=b''
  changed=Path(tmp)/f'{i}.zip'
  with zipfile.ZipFile(changed,'w') as dest:
   for info in source.infolist(): dest.writestr(copy.copy(info),data if info.filename==path else source.read(info))
  try: m.audit(changed,Path(tmp)/f'output{i}',Path(tmp)/f'retained{i}')
  except AssertionError:
   frame=traceback.extract_tb(__import__('sys').exc_info()[2])[-1]
   assert expected in frame.line,(name,frame.line)
   results.append({'case':name,'refused':True,'assertion':frame.line})
  else: raise AssertionError(name+' was accepted')
  changed.unlink()
Path(sys.argv[2]).write_text(json.dumps({'actual_archive_controls':results,'all_passed':True},indent=2)+'\n')
print(json.dumps({'controls':len(results),'all_passed':True}))
