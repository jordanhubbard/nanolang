from pathlib import Path
import json,hashlib,subprocess,io,tarfile,sys
r=Path('/private/tmp/nanolang-source-qual-20261124-tsw6pyeo');q=json.loads((r/'result.json').read_text());cfg=json.loads((r/'config.json').read_text());tree=r/'source-normal';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
guards={n:sha(tree/n) for n in cfg['sources']};assert guards==cfg['sources'];assert q['terminal'] and q['controller_exit']==2;assert q['tools_before']==q['tools_at_terminal']
outputs={str(p.relative_to(r)):dict(bytes=p.stat().st_size,sha256=sha(p)) for d in ['obj','lib','bin'] for p in sorted((tree/d).rglob('*')) if p.is_file()}
report=json.dumps(dict(source_pin=q['source_pin'],guards=guards,outputs=outputs,source_guards_match=True,tool_guards_match=True),indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 info=tarfile.TarInfo('terminal-guards.json');info.size=len(report);tar.addfile(info,io.BytesIO(report))
 for p in sorted(r.iterdir()):
  if p.is_file():tar.add(p,arcname=p.name,recursive=False)
 tar.add(tree/'tests/test_module_rejection_rollback.c',arcname='failed-fixture.c')
