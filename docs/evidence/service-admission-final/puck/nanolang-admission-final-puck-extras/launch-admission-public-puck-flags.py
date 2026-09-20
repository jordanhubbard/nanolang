import json,os
from pathlib import Path
os.environ.update(PATH='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
    CC='/usr/bin/clang',CARRIER_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',
    CARRIER_PHASES='configuration,discovery,ordinary',LSAN_OPTIONS='',
    NMS_RUNTIME_CLANG='/opt/homebrew/opt/llvm/bin/clang',
    NMS_RUNTIME_OPT='/opt/homebrew/opt/llvm/bin/opt',NMS_NATIVE_CLANG_FLAGS='',
    CARRIER_EXTRA_TOOLS=json.dumps({'llvm_opt':'/opt/homebrew/opt/llvm/bin/opt'}))
os.environ['NANO_FILE_RUNTIME_CFLAGS']=Path('/tmp/nanolang-admission-final-bc45-puck/cflags.txt').read_text().strip()
assert all(Path(p).is_file() for p in json.loads(os.environ['CARRIER_EXTRA_TOOLS']).values())
os.execv('/opt/homebrew/bin/python3',['/opt/homebrew/bin/python3',
    '/tmp/run-admission-final-public.py','/tmp/nanolang-admission-public-qualified-bc45',
    '/tmp/nanolang-admission-public-bc45-flags-puck'])
