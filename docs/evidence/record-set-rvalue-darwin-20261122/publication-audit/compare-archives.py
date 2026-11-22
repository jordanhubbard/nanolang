from pathlib import Path
import json,gzip,hashlib,sys
left,right=map(Path,sys.argv[1:3]);a=json.loads((left/'manifest.json').read_text());b=json.loads((right/'manifest.json').read_text());assert a['implementation']==b['implementation'] and a['source_hashes']==b['source_hashes']
for item in [a,b]:
 r=item['verification'];assert r['successful'] and r['discovered']==r['ran']==20 and not(r['failures'] or r['errors'] or r['skipped']);assert item['sanitizers']['passed']
def captured(root,m):
 # I select the final changed-source suite; retained first-pass logs stay separate.
 matches=[e for e in m['files'] if e['path'].endswith('/commands/commands.json') and ('/linux-typed/' in e['path'] or '/record-set-typed-20261122-control/' in e['path'])];assert len(matches)==1
 commands=json.loads(gzip.decompress((root/matches[0]['artifact']).read_bytes()));s={e['sha256'] for c in commands for e in c.get('input_files',[]) if e['path'].endswith('.nano')};return commands,s
x,xs=captured(left,a);y,ys=captured(right,b);assert xs==ys,(xs-ys,ys-xs)
result=dict(implementation=a['implementation'],source_files=len(a['source_hashes']),all_source_hashes_identical=True,methods_each=20,failures_each=0,errors_each=0,skips_each=0,all_captured_nano_input_bytes_identical=True,unique_captured_nano_inputs=len(xs),subprocess_commands_each=[len(x),len(y)],direct_sanitizers_pass_each=True,limits='I compare source inputs and complete successful method outcomes. Native/bytecode artifact bytes are platform-specific and I do not claim binary equality.')
print(json.dumps(result,indent=2))
