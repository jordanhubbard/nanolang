#!/bin/bash
set -e
export PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
export CC="$(xcrun --find clang)"
export SDKROOT="$(xcrun --show-sdk-path)"
export CARRIER_SAN_CC=/opt/homebrew/opt/llvm/bin/clang
export LIBRARY_PATH=/opt/homebrew/opt/openssl@3/lib
export NANO_FILE_RUNTIME_CFLAGS="$(/opt/homebrew/bin/pkg-config --cflags libffi)"
unset NMS_NATIVE_CLANG_FLAGS
export CARRIER_EXTRA_TOOLS="$(python3 - <<'PY'
import json,os,pathlib,shlex
p=pathlib.Path(shlex.split(os.environ['NANO_FILE_RUNTIME_CFLAGS'])[0][2:])
tools={'pkg-config':'/opt/homebrew/bin/pkg-config','ffi.h':str(p/'ffi.h'),'ffitarget.h':str(p/'ffitarget.h'),'libffi-link-stub':str(pathlib.Path(os.environ['SDKROOT'])/'usr/lib/libffi.tbd')}
assert all(pathlib.Path(p).is_file() for p in tools.values())
print(json.dumps(tools))
PY
)"
python3 /tmp/nanolang-file-frames-gate-v2.py /tmp/nanolang-file-frames-041 /tmp/nanolang-file-frames-041-darwin
