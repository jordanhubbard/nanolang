import os,json,shlex,subprocess,hashlib
from pathlib import Path
root=Path('/Users/jkh/Src/nanolang-51-platform-qualification');out=Path(__file__).parent
s=json.loads(Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-native-8twxf6mp/selection.json').read_text())
s['objects']=[str(Path('/Users/jkh/Src/nanolang-51-harness')/p) for p in shlex.split((out/'make-query-closure.txt').read_text())]
s['modules']=[str(Path('/private/tmp/nanolang-map-concat-baseline-20261022')/(kind+'-'+producer+'.nvm')) for kind in ['unchanged','concat'] for producer in ['seed','c-hosted','stage1','stage2']]+[str(out/'original-shadow.nvm')]
env=dict(os.environ,RECORD_ARRAY_CC=shlex.join(s['cc']),RECORD_ARRAY_CFLAGS=shlex.join(s['flags']),RECORD_ARRAY_OBJECTS=shlex.join(s['objects']),RECORD_ARRAY_LDFLAGS=shlex.join(s['ldflags']),RECORD_MAP_MODULES=shlex.join(s['modules']))
(out/'inputs.json').write_text(json.dumps(s,indent=2)+'\n')
pins={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in s['objects']+s['modules']}
(out/'pins-before.json').write_text(json.dumps(pins,indent=2)+'\n')
with (out/'final-query.log').open('w') as log:
 p=subprocess.run(['python3','-m','unittest','-v','tests.test_record_map_publication'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'final-query-status.json').write_text(json.dumps({'status':p.returncode})+'\n')
print((out/'final-query.log').read_text());raise SystemExit(p.returncode)
