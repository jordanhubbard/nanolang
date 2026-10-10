from pathlib import Path
import shlex,sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_websocket_dispatch import WebSocketDispatch
for name,location in [('sanitized','mdb5d4p5'),('gcc','hg72ol0r')]:
    source=Path('/private/var/folders/9z/xpmfgw8j09l4g6wwxrxtt97w0000gn/T/nano-websocket-dispatch-'+location)
    case=WebSocketDispatch();case.artifacts=Path('/private/tmp/nl51-websocket-dispatch-timeout-'+name);case.artifacts.mkdir(exist_ok=True)
    command=shlex.split((source/'native-1-O0-build-command.txt').read_text())
    # I retain the exact compiled provider/object closure. Only the generated
    # negative controls change; no peer or existing passing program is repeated.
    at=next(i for i,arg in enumerate(command) if arg.endswith('/case-1.c'));flags=command[1:at]
    providers=command[at+2:command.index('-o')]
    case.tamper([command[0]],flags,providers,[],[],(source/'case-1.c').read_text(),source/'driver.c')
    print(name+' exact timeout and call controls passed',flush=True)
