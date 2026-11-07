# My instrumented CI worker

I build this image once per workflow and transfer the exact Docker archive to
both sanitizer jobs and both coverage jobs. I pin the Ubuntu base by digest and
use only the signed `20261019T000000Z` Ubuntu snapshot. I install `python3-yaml`
from that archive rather than running mutable pip installs. My checksum-pinned
CA bootstrap package supplies the minimal base's missing TLS trust bundle; I
keep TLS and apt signature verification enabled. Only snapshot expiration is
ignored, because the package archive is deliberately historical. I fail if any
apt index cannot be fetched. [Ubuntu snapshot documentation](https://snapshot.ubuntu.com/)

`scripts/ci_container.py build` records the archive SHA256, image content ID,
config digest and platform. Each consumer downloads the producer's exact
artifact ID, verifies the archive before loading, and verifies the loaded image.
For Docker's classic image store, the image ID is the config digest. For the OCI
containerd store, I additionally verify the manifest digest and its binding to
the config digest. A loaded image must resolve to one of those checked addresses;
an arbitrary tag or a claimed environment variable is insufficient.

Before every step, I inspect the created container's actual image. I execute its
script with a read-only image root, a read-only identity receipt, fixed
`/workspace` and `/ci-temp` paths, disk-backed private `/tmp`, writable `/run`, and a
writable HOME under `/ci-temp`. I pass only explicit build/test environment
variables and remap GitHub output/summary files into the temp mount. I do not
mount a host compiler or library directory. Checkout and artifact transport
remain on the host; compilation, checkpoint operations, tests and coverage
reporting run inside the image. No registry publication permission is needed.

The checkpoint CLI now **requires this verified container receipt**. I migrate
all existing CI callers together; this change does not promise compatibility
with historical host-created checkpoints. I retain source/commit/root identity,
flags, actual compiler/support-tool hashes, dynamic-library hashes, the package
inventory, provider/archive bytes, nanosecond timestamps and continuation checks.
The outer runner's ImageOS/ImageVersion are diagnostic only. The host kernel and
hardware remain outside this userspace image identity.

`ci_container_smoke.py` exercises actual archive load, two fresh container
invocations with different outer runner labels, sanitizer and coverage programs,
provider archive restoration, read-only root refusal, changed identity/flags,
step output mapping and nonzero exit propagation. These bounded controls do not
replace the complete hosted amd64 bootstrap, sanitizer and coverage gates.
