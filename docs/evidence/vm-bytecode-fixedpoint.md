# My full VM bytecode fixed point

I measured a bytecode fixed point at clean source
`1277bce22e2134173446cb10b09bde07bf7731af` on Linux ARM64. I compiled the
complete `src_nano/nanoc_v06.nano` source through two successive NanoVM-executed
compiler generations. Their raw `.nvm` files are identical: 352,236 bytes,
SHA-256 `fe4f3a55d146bb3b38f7db8719039df265276cb17815c4f58bcb027b22c5198c`.

My labels matter:

1. **Seed** is compiler bytecode emitted by the C-seed NanoVirt frontend.
   This uses a different lowering implementation. I do not compare it with
   the subsequent self-hosted outputs to claim a fixed point.
2. **Stage 1** is that seed compiler executing in NanoVM and compiling the
   identical pinned compiler source. It completes in 300.47 seconds, exit
   zero, with peak RSS 237,228 KiB.
3. **Stage 2** is Stage 1 executing in NanoVM and compiling the same source.
   It completes in 298.41 seconds, exit zero, with peak RSS 237,216 KiB.
4. `cmp` compares Stage 1 and Stage 2 directly. I normalize no paths, tables,
   instructions or metadata. Both outputs verify; Stage 2 also compiles a
   hello product that verifies and executes in NanoVM.

Both generations use one unchanged checkout and the same three immutable
host libraries: `compiler_support`, `nanoisa`, and `std`. Their absolute
artifact paths and SHA-256 hashes, the capture-helper hash and the clean
source pin are retained in `vm-bytecode-fixedpoint-1277.json` and the
original `/tmp/nanolang-bootstrap-shared-manifest.json`.
These host/helper hashes and the source state remain unchanged after both
generations. The original seed is 372,208 bytes, SHA-256
`ab84354ae568fe3065cc006f02e0e2b5fa270d9ce999405708656ae17182b13e`.

The exact commands, run in `/home/jkh/Src/nanolang-vm-bootstrap-shared`, are:

```sh
make -j8 nano_virt nano_vm nanoisa_dump
export NANO_AS_CAPTURE_HELPER="$PWD/bin/nano_as_capture.so"
bin/nano_virt src_nano/nanoc_v06.nano --emit-nvm --strip-debug \
  -o /tmp/nanolang-bootstrap-shared-seed.nvm
bin/nano_vm /tmp/nanolang-bootstrap-shared-seed.nvm -- \
  src_nano/nanoc_v06.nano --emit-nvm -o /tmp/nanolang-bootstrap-shared-stage1.nvm
bin/nano_vm /tmp/nanolang-bootstrap-shared-stage1.nvm -- \
  src_nano/nanoc_v06.nano --emit-nvm -o /tmp/nanolang-bootstrap-shared-stage2.nvm
cmp /tmp/nanolang-bootstrap-shared-stage1.nvm /tmp/nanolang-bootstrap-shared-stage2.nvm
```

The retained stage commands were wrapped in `/usr/bin/time -v` and a
1,800-second diagnostic timeout. Logs and raw outputs use the
`/tmp/nanolang-bootstrap-shared-` prefix. Compiler shadow fixtures emit some
expected negative diagnostics; the compiler exits zero and publishes the
verified output after its normal shadow checks.

My permanent `make test-vm-bytecode-bootstrap` gate repeats seed emission,
two full VM generations, raw equality, host/hash stability, verification
and final-compiler product execution. It requires a clean source checkout,
retains a fresh evidence directory and records argv, duration and status for
every step. `NANOLANG_BOOTSTRAP_ROOT` selects an existing clean build for an
independent pinned-source run; `NANOLANG_BOOTSTRAP_EVIDENCE` selects a new empty
output directory. `NANOLANG_BOOTSTRAP_STAGE_TIMEOUT` defaults to 1,800 seconds
per command. A timeout fails the gate and preserves evidence.

The permanent gate passes against the same clean `1277bce2` source in
601.921 seconds, with Stage 1 at 297.05 seconds and Stage 2 at 296.60 seconds.
Its fresh manifest and logs are retained in
`/tmp/nanolang-vm-fixedpoint-gate-1277`; the test log is
`/tmp/nanolang-vm-fixedpoint-gate-1277.log`. This is a second full execution
of both compiler generations, using the committed gate logic.

This establishes the measured bytecode convergence and full VM source route.
It does not prove semantic correctness. My canonical driver still generates,
compiles and executes **native C shadows** during this route; the complete
NanoISA-only architecture remains unfinished. The native AOT compiler-source
route is also independent: its older pinned experiment reaches a separately
tracked parser/runtime failure after record-array growth. I retain that
failure rather than claiming this VM result repairs it.
