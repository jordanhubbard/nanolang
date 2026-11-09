from pathlib import Path
import tempfile,zipfile,json,subprocess,hashlib,gzip
root=Path.cwd();archive=root/'.test_output/call-metadata-comparison-42608197477.zip';auditor=root/'docs/evidence/call-frame-lifetimes-20261109/audit-metadata-comparison.py';checks=[]
for label,entry in [('provider-image','cseed-source-comparison/providers/original/nanoc_c'),('source-input','cseed-source-comparison/shared-inputs.json'),('created-product','cseed-source-comparison/samples/03-modified/compiler')]:
 with tempfile.TemporaryDirectory(prefix='metadata-audit-refusal-',dir=root/'.test_output') as temp:
  work=Path(temp);bad=work/'tampered.zip';change={}
  with zipfile.ZipFile(archive) as src,zipfile.ZipFile(bad,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as dst:
   assert entry in src.namelist()
   for info in src.infolist():
    b=src.read(info)
    if info.filename==entry:
     change['before_sha256']=hashlib.sha256(b).hexdigest()
     if label=='source-input':
      x=json.loads(b);key=sorted(x)[0];change['key']=key;change['previous']=x[key];x[key]='0'*64;b=json.dumps(x).encode()
     else:change['offset']=len(b)-1;change['previous_byte']=b[-1];b=b[:-1]+bytes([b[-1]^1])
     change['after_sha256']=hashlib.sha256(b).hexdigest()
    dst.writestr(info.filename,b)
  command=['python3',str(auditor),str(bad),str(work/'audit'),str(work/'native')]
  r=subprocess.run(command,capture_output=True,text=True,timeout=45)
  assert r.returncode!=0 and 'AssertionError' in r.stderr,(label,r.returncode,r.stderr)
  checks.append(dict(label=label,entry=entry,change=change,argv=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
report=dict(scope='I mutate retained bytes and source-input identity independently. Every actual auditor invocation refuses; I do not rerun a native factory.',original_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),auditor_sha256=hashlib.sha256(auditor.read_bytes()).hexdigest(),collector_source=Path(__file__).read_text(),checks=checks)
(root/'docs/evidence/call-frame-lifetimes-20261109/metadata-comparison-audit/refusal-controls.json.gz').write_bytes(gzip.compress(json.dumps(report,indent=2).encode(),mtime=0));print(json.dumps(dict(actual_tamper_refusals=len(checks))))
