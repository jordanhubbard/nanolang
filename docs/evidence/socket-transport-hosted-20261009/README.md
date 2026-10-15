# TCP module transport and hosted preparation

I retain this batch against parent `60dd088e4`. `source-sha256.txt` identifies
my changed implementation and test inputs. My compressed logs retain commands,
failures and passing results; they are evidence of local Darwin checks.

I transport the exact catalog2, 128-byte TCP nominal map and ownership metadata
through both module bridges and v2 serialization. I retain separate File/TCP
preparation APIs over shared cyclic, indirect-target, ownership and hosted
engines. I consume Endpoint from the actual catalog signature and retain its
pending runtime domain-check obligation. I do not execute TCP with these plans.

| Check | Retained result |
| --- | --- |
| TCP hosted preparation | 2,998 linked and 6,467 allocation-instrumented checks; LLVM ASan/UBSan |
| TCP module transport | 463 linked and 696 allocation-instrumented checks; two methods pass |
| File module transport | 433 linked and 633 allocation-instrumented checks; two methods pass |
| File cyclic and indirect adjacency | Five suites, ten methods pass with ordinary compiler flags |
| Apple Clang and GCC 16 TCP hosted matrix | Both linked and instrumented corpora pass, ordinary flags |
| Fresh host-library cache integration | Paired lowerer helper TCP refusal test passes in 45.841 seconds |

I retain the combined results in `tcp-hosted-transport.log.gz`, the File checks
in `file-hosted-adjacency.log.gz`, and the extra compiler commands and exit
statuses in `tcp-hosted-matrix.log.gz`. The hosted allocation hooks cover the
selected plan/flow providers, not every linked codec dependency. The File
allocation-instrumented runs are not all sanitizer runs.

My fresh dependency-closure check builds `obj/test_service_lowering`,
`nano_virt` and `nvm2c` with LLVM Clang on PATH, then runs:

```sh
NANO_BUILD_CACHE=/private/tmp/nl51-tcp-hosted-closure \
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
python3 -m unittest -v tests.test_service_lowering_nano.ServiceLoweringNano.test_tcp_checked_source_stops_before_wire_publication
```

I retain its build and test logs as `hosted-compiler-build.log.gz` and
`hosted-closure.log.gz`. This checks the independently built Nano helper and
updated native host dependency manifests; it is not a fresh compiler bootstrap.

I preserve two initial failures. `tcp-transport-initial.log.gz` records missing
`opt` on PATH; I corrected the command environment. `tcp-transport-corrected.log.gz`
records successful TCP serialization followed by decoder rejection of the
128-byte section. I added that exact section length and reran the round-trip,
malformed metadata and consumer-refusal checks. `hosted-shared-review.diff.gz`
records the normalized shared-engine review: the behavior changes are catalog
signature argument transfer and the independent Endpoint obligation inventory.

I still refuse generic VM, FFI, native C/LLVM/Wasm and File execution consumers
for this TCP profile. Paired source lowering, runtime carrier/dispatch, grants,
network shadows, full 5.1 semantics and platform/release qualification remain
open under #990 and the release umbrella. I have not published a 5.1 release.
