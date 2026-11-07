#!/bin/sh
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update -qq
mkdir -p /tmp/packages /tmp/toolchain /tmp/replay/tests/nanoisa
cd /tmp/packages
apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 download make gcc-14-x86-64-linux-gnu cpp-14-x86-64-linux-gnu libgcc-14-dev libc6-dev linux-libc-dev libmpfr6 libmpc3 libgmp10 libisl23 libzstd1 zlib1g libcrypt-dev
for package in ./*.deb; do dpkg-deb -x "$package" /tmp/toolchain; done
export LD_LIBRARY_PATH=/tmp/toolchain/usr/lib/x86_64-linux-gnu
ln -s /workspace/src /tmp/replay/src
cp /old.c /tmp/replay/tests/nanoisa/test_disasm_roundtrip.c
mkdir -p /tmp/bin
printf '%s\n' '#!/bin/sh' 'exec /tmp/toolchain/usr/bin/x86_64-linux-gnu-gcc-14 --sysroot=/tmp/toolchain "$@"' > /tmp/bin/gcc-14
chmod +x /tmp/bin/gcc-14
export PATH=/tmp/bin:/tmp/toolchain/usr/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
cd /workspace
python3 -m unittest -v tests.test_fast_gate.FastGateGccControls.test_actual_glibc_disassembler_profiles_reject_old_source
