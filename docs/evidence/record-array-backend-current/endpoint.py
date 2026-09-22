import pathlib,json,hashlib,sys
base=pathlib.Path(sys.argv[1]);report=base/'reports';source=base/'source'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
m=json.loads((base/'source.json').read_text());assert all(sha(source/p)==h for p,h in m['files'].items())
t=json.loads((report/'package-tools-after.json').read_text());assert all(sha(v['path'])==v['sha256'] for v in t.values())
p=json.loads((report/'package-artifacts.json').read_text());assert all(sha(k)==v['sha256'] for k,v in p.items())
result={'pin':m['pin'],'sources':len(m['files']),'tools':len(t),'products':len(p),'status':'PASS'}
(base/'current-endpoint.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
