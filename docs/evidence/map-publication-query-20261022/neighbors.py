from pathlib import Path
import json,os,shlex,subprocess
out=Path(__file__).parent;s=json.loads((out/'inputs.json').read_text());mods=[str(Path('/private/tmp/nanolang-map-concat-baseline-20261022')/('concat-'+p+'.nvm')) for p in ['seed','c-hosted','stage1','stage2']]
env=dict(os.environ,RECORD_ARRAY_CC=shlex.join(s['cc']),RECORD_ARRAY_CFLAGS=shlex.join(s['flags']),RECORD_ARRAY_OBJECTS=shlex.join(s['objects']),RECORD_ARRAY_LDFLAGS=shlex.join(s['ldflags']),RECORD_MAP_MODULES=shlex.join(mods))
with (out/'neighbors.log').open('w') as f:p=subprocess.run(['python3','-m','unittest','-f','-v','tests.test_record_map_image','tests.test_record_map_execution'],env=env,stdout=f,stderr=subprocess.STDOUT)
(out/'neighbors-status.json').write_text(json.dumps(dict(status=p.returncode))+'\n');print((out/'neighbors.log').read_text());raise SystemExit(p.returncode)
