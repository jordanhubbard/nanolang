# Private map heap qualification

This checkpoint implements the reviewed heap contract93f4f71d4 above private preparation34d080e06. No VM handler, public admission, native generation, publisher, VmState or VmHeap declaration changes. Direct adapters and teardown require both private map-heap and existing array-runtime macros; they do not authorize execution of selected map modules.

The shared change is separately identifiable: hm_resize receives the heap and charges successful net bucket-capacity growth after commit, matching array_grow. The unchanged original growth probe now reports allocated7126/freed7126/objects0, versus its retained prior2646/7126 discrepancy. Failure leaves counters unchanged. Independent allocation tracking, rather than those counters alone, establishes balanced live payloads.

## Gates

- Direct ASan/UBSan suite: **1 method PASS0.994s, 10318 checks**. Command: `RECORD_ARRAY_VM_CFLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all' python3 -m unittest -v tests.test_record_map_heap`. The driver compiles current heap/cycles/value/ISA providers with consistent combined macros and allocation interception. Its private whitebox wrapper exposes the real shared lookup solely to compare exact bucket results against the charged validator.
- Counted-NUL keys, empty keys, missing VOID versus INT0, a64-key same-bucket chain, genuine separately allocated equal-content key, maximum-ref replacement without allocation/retain, insertion headroom refusal, transient/persistent NEW and growth failures, exact observation bound/one-below, nine restored physical corruptions,32 partial record/map construction failures plus independent retry. All tracked allocations return to zero. Maximum256-record→map shared-key release and old256-record+3-array release allocate nothing during teardown. Ordinary cyclic map collection remains exercised separately.
- Combined-macro preparation: **2 methods PASS16.529s**, all four retained original modules and873 allocation prefixes in both modes. Actual owner818472/heap6312/field20/layout16; retained oracle9275369 bytes for400-constant fixture. These are separate measurements from preparation-only owner812256/retained9269153. Full reserved overlap remains within unchanged caps. There is still no runtime execution claim.
- Old array-only private VM: **2 methods PASS30.146s**, switch/goto linked2390 and instrumented6197 checks. Six old budget rows match exactly, including Analysis39840.
- Unchanged ordinary heap-allocation harness passes, including INT→STRING replacement/growth failures and retry. The full ordinary VM harness with harness-only O0 passes **274654/274654**; providers retain their original sanitized O1 build. See the failed O1 attempt below.
- Eight strict Clang/GCC16 cases pass: ordinary, array-only, combined heap and direct harness. No source frontend or bootstrap build.

Required target `test-record-map-heap` is wired to test-units and builds its own four small providers without broad prerequisites. Aggregate Make execution is a subsequent integration gate. Commands, environment, provider/source hashes and representative artifacts are retained here. Original modules remain seed/prior C-hosted/retained installed stages; this is not fresh source-producer qualification.

## Retained failures and limits

1. First strict provider compilation passed but minimal harness linking omitted the real ISA provider, causing undefined isa_tag_name. Adding isa.c to the test closure fixed it; no product change.
2. Optional full ordinary test_vm.c at O1 aborted with ASan stack overflow before its first banner. Actual compiler disassembly shows main subtracting seven0xfff000 blocks plus0xce5000+0x220:130933280 bytes, in addition to saved registers/alignment. The retained prologue and sanitizer trace establish this compiler-generated test frame. Rebuilding only that harness at O0 with inlining disabled and unchanged sanitized providers passes; **O1 is not claimed green**. Direct selected heap tests and legacy allocation tests passed independently at O1.

Private fast adapters require closed constructor-owned topology, immutable counted strings and no callback/concurrent mutation. Physical validation checks duplicate/unreachable entries before any future boundary can authorize these adapters. Corrupt tables are restored before destruction and never used as fast mutation inputs. No runtime work budget, DELETE support, public route, or arbitrary-pointer validity claim is introduced.

Readable logs remove trailing whitespace; matching `.raw.gz` files preserve original bytes where needed. No binary bundle is included.
