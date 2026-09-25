import os,sys,json
sys.path.insert(0,'scripts')
import ci_sanitizer_partitions as p
phase=sys.argv[1]
worker={'id':'units-00','targets':[]}
command=p.command_for(worker,phase)
os.environ.update(CC=p.CC,NANO_CC=p.CC,NANOLANG_GUARD_SAN_CC=p.CC,NANO_LDFLAGS=p.NATIVE_LDFLAGS)
print(json.dumps({'argv':command,'CC':p.CC,'ASAN_OPTIONS':os.environ.get('ASAN_OPTIONS'),'NANO_SHADOW_TIMEOUT_SECONDS':os.environ.get('NANO_SHADOW_TIMEOUT_SECONDS')}),flush=True)
os.execvp(command[0],command)
