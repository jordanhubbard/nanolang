import json,pathlib,hashlib,zipfile,subprocess
root=pathlib.Path('/home/jkh/nanolang-qualification/pr948-third-hosted');out=root/'provider-instrumentation';out.mkdir(exist_ok=False)
a=next(a for a in json.loads((root/'artifact-retention.json').read_text()) if a['name']=='sanitizer-providers-units-01');nm=pathlib.Path('/usr/local/bin/llvm-nm').resolve();result={'head':'82c30f5ccf6ee95ec0bf70e05bca1ad6c25007ed','merge':'c2c813c291d28f9cfc5c58807ffae5bff84f5833','archive':a['path'],'archive_sha256':a['sha256'],'tool':str(nm),'tool_sha256':hashlib.file_digest(nm.open('rb'),'sha256').hexdigest(),'products':[]}
with zipfile.ZipFile(a['path']) as z:
 for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
  member='bin/'+name;m=next(x for x in a['members'] if x['path']==member);b=z.read(member);h=hashlib.sha256(b).hexdigest();assert h==m['sha256'] and len(b)==m['bytes'];p=out/name;p.write_bytes(b)
  c=subprocess.run([str(nm),str(p)],capture_output=True,timeout=60);(out/(name+'.nm.txt')).write_bytes(c.stdout);(out/(name+'.stderr')).write_bytes(c.stderr);assert c.returncode==0
  text=c.stdout.decode();result['products'].append({'member':member,'sha256':h,'bytes':len(b),'asan':'__asan_' in text,'ubsan':'__ubsan_' in text,'status':c.returncode,'symbol_report_sha256':hashlib.sha256(c.stdout).hexdigest()})
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
