import os,sys,json,time
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
out=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness')
config=dict(line.split('=',1) for line in (out/'closure.txt').read_text().splitlines())
env=dict(os.environ,PUBLICATION_CC=config['CC'],PUBLICATION_CFLAGS=config['CFLAGS'],PUBLICATION_LDFLAGS=config['LDFLAGS'],PUBLICATION_OBJECTS=' '.join(str(agg/x) for x in config['OBJECTS'].split()),PUBLICATION_SEED=str(out/'nano_virt'),PUBLICATION_NANOISA=str(out/'nanoisa'))
args=[sys.executable,'-m','unittest','-f','-v','tests.test_publication_allocation.PublicationAllocation.test_map_publisher_and_query_prefixes']
t=time.monotonic();p=run_captured(args,cwd=root,env=env,timeout=900)
(out/'allocation.stdout').write_bytes(p.stdout);(out/'allocation.stderr').write_bytes(p.stderr);(out/'allocation.json').write_text(json.dumps(dict(argv=args,exit=p.returncode,seconds=time.monotonic()-t,selectors={k:v for k,v in env.items() if k.startswith('PUBLICATION_')}),indent=2)+'\n')
print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
