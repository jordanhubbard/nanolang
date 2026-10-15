# My mixed service transport checkpoint

I implement this batch from `e695a2ce7` under #990. My
[format contract](../../SERVICE_MULTI_TRANSPORT.md) describes a canonical v3
map for up to 64 separate File/TCP catalog instances. I preserve existing
single-catalog bytes, catalog-instance identity, exact nested layouts and
ownership declarations through both module bridges. I do not admit mixed
execution in this checkpoint.

I require unique active import/layout indices across instances. My query
keeps instance, catalog, catalog ordinal, global index and per-kind source
ordinal distinct. A repeated File catalog remains a separate nominal owner;
its Result cannot substitute an earlier same-shaped File from another instance.
No grant or runtime authority follows from successful transport validation.

## Qualification

I use `make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang` with the LLVM
binary directory on PATH for translator dependencies.

| Check | Result |
| --- | --- |
| `test-multi-nominal`, instrumented allocations | 12,998 checks pass |
| `test-multi-nominal`, linked query | 12,828 checks pass |
| TCP raw/nominal regression | 23,069 linked and 23,084 instrumented checks pass |
| File raw/nominal regression | 19,770 and 19,745 checks pass |
| TCP module/consumer regression | 696 allocation and 463 linked checks pass |
| File module/consumer regression | 633 allocation and 433 linked checks pass |
| C TCP source/publication corpus | Three methods pass, 11.005 seconds |
| C File source/publication corpus | Sixteen methods pass, 111.654 seconds |
| Fresh Nano driver plus C TCP controls | Three methods pass, 44.970 seconds; 116 CLI commands |
| GCC 16 strict syntax checks | New codec, query and C fixture pass |

My new nominal corpus covers all raw truncations at maximum extent, byte
mutations, every supported instance count, overlapping buffers, preserved
failure outputs, mixed/repeated catalogs, permuted indices, cross-instance
payload refusal, exact ownership flags/modes, corrected-CRC malformed wire,
owned query lifetime, both bridge directions and allocation-failure prefixes.
ASan/UBSan and leak detection cover the new codec/query and rebuilt transport
adapters. Other linked objects retain ordinary build flags; I do not call this
a fully instrumented whole-program test.

I build the fresh Nano driver with `bin/nano_virt src_nano/nanoc_v06.nano
--emit-nvm --strip-debug -o /private/tmp/nl51-multi-driver.nvm`, LLVM clang,
O2 host flags, a 30-second compiler-shadow limit and the retained host cache
`/private/tmp/nl51-multi`. Its TCP test uses `NANO_TCP_DRIVER_MODULE` and no
Nano-native override: this run exercises the two C drivers and the Nano driver
in NanoVM. It is not a new Stage1/Stage2 fixed-point check. The previous
[bootstrap receipt](../tcp-bootstrap-generations-20261010/README.md) remains
scoped to the preceding compiler revision.

## Retained failures

I retain the first overlap fixture's insufficient wire capacity, the bridge's
missing v3 ownership selection, a missing LLVM `opt` in PATH, and the File
harness selecting Apple cc instead of the requested leak-capable compiler.
My first compiler-forwarding edit omitted a shell separator and failed before
running tests. I correct each cause and retain the subsequent passing results.
The broad `qualified.log` contains passing mixed/raw tests followed by the PATH
failure; `module-regressions.log` contains passing TCP checks followed by the
File sanitizer-runtime failure. `adjacency-final.log` records the corrected
File module and both C source suites. I do not label those earlier commands
successful as a whole.

My checked-in archives retain commands, test logs and wire/source artifacts;
`source-hashes.json` pins the implementation and the unchanged user fixture.
Mixed flow, VM/native dispatch, paired source lowering, per-catalog grants,
DNS/WebSocket and full release/platform qualification remain open.

I preserve original log bytes in `raw-logs.tar.gz`. The adjacent text logs
normalize trailing whitespace so repository whitespace checks remain useful.
