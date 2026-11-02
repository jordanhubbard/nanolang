# Private union VM preparation receipt

I qualified preparation only after union image75f811 and design007ad59c4. I added a module-taking private owner with independent copied image, decoded/dispatch tables, stack/globals and heap constants. I did not add an executor, HALT outcome, public/native selector or publisher. I did not call vm_init or label private setup as public acceptance. The ordinary heap remains ordinary; no persistent DAG policy or old VM/heap/Analysis field changed.

## Actual final gate

`test-union-field-preparation` is a required test-units dependency. Its Python driver rebuilds the exact permanent runtime/query provider set, preserves common-provider attribution, uses a separate VM source-including instrumented harness, and accepts zero or all six original selectors. The component invocation was:

```
python3 -m unittest -f -v tests.test_union_field_preparation
```

Exact selectors, objects, includes and flags are in `target-env.json` and `permanent-selection.json`; actual argv/status logs are retained. This is the target's test command with explicit retained dependencies, not a claim that the component ran a broad default Make prerequisite rebuild. I retained the actual `make -n -f Makefile.gnu test-union-field-preparation` plan for aggregate review. No frontend/bootstrap ran.

Final terminal: **1 method,20.643s;160 linked checks/all six originals and47,489 instrumented checks**. Providers/harness used Homebrew Clang23, O1, ASan/UBSan and fatal sanitizer recovery. `compiler-pins.json` records exact versions/binary hashes. Current artifact directory is `/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-union-field-preparation-wwnmnogu`; binaries/providers are pinned separately. Original six modules are seed, C-shadow, qualified C-hosted canonical, its captured checked shadow, raw program and raw shadow from the retained unchanged union anchor. Each is independently annotated in the harness. Preparation survives destruction of the original module, wire bytes and annotation backing. None is executed by the new owner.

## Measured storage and limits

On this arm64 ABI I measured VmState797776B, VmHeap96B, NanoValue16B unchanged in the default/private-union configuration; new owner807232B and copied instruction/site144B. Stack capacity is524289. Both old headers and the old public analysis header are byte-identical to75f811; strict combined private macro compilation also passed.

Basic preparation reserves runtime9,267,301B/9,273,676 logical work and combined image76,471,196B/26,134,951 work. Its earlier source constructor separately reserves67,189,009B/16,859,481 work. The six originals fit unchanged caps; the largest measured original reservation is76,511,558B/26,222,280 work. These reservations are not actual payload or one shared budget across both sequential proofs.

Independent allocator measurements establish:

| Case | Image retained | Runtime retained | Total retained | Whole preparation peak |
|---|---:|---:|---:|---:|
| Basic |14,886|9,201,765|9,216,651|9,216,651|
|406 constants|26,388|9,227,411|9,253,799|9,257,034|

The image retention was measured separately after freeing its source plan. The runtime oracle sums actual owner, copied rows, final decoder capacities, dispatch maps, stack/global tables, constant pointer table, intern bucket table and every distinct interned allocation. It does not equate reservation counters with payload. Basic source-phase peak74,825B and independent image-phase peak89,711B are recorded separately. The406-constant counterparts are86,327B and112,715B. The whole preparation measurement includes the earlier source-plan phase; constant growth demonstrates live old/new bucket overlap. All destruction paths return tracked bytes and allocations to zero.

I swept **127 basic allocation positions** and **1,329 positions with406 constants**, each in transient and persistent denial modes with independent valid retry and a real prior owner preserved. The basic sweep includes decode/dispatch/stack failure before heap initialization, intern initialization, constant table and string failures. Exactly two transient optional bucket-growth failures permit success with valid constants; persistent denial later fails cleanly. Ordinary intern growth preserves its existing optional-allocation behavior. Duplicate constants retain separate index edges; empty/count-NUL constants and present-zero-arity parameter rows retain exact content and ownership.

A zero-string complete union module remains refused because counted names are required. The successful S=0 control is explicitly only actual vm_heap_init plus vm_module_constants_build on an empty primitive module:256 intern buckets, no constant pointer table. It is not a newly admitted source shape.

Private image positive-extra controls qualify measured exact bytes/work caps, one-above refusal, UINT64_MAX overflow, prior-image preservation and recovery. Initial-reserve overflow refuses before image allocation. Final-measured-cap+1 reaches54/55 reversible allocations before later snapshot/recipe LIMIT; it is not a zero-allocation claim. A real1MiB unused-string module independently passes source construction and image0/0, then full VM preparation returns LIMIT with output preserved, balances allocations and recovers on a valid module. No production test seam or cap increase was added.

## Correspondence and scratch

I directly exercised reversed payload ordinal1, both empty alternatives, ordinary CALL and acyclic TAIL returning union values, blocked callee refusal, VOID-initialized global storage, and reachable HALT followed by an unestablished sensitive site. These are copied-row/preparation checks only. HALT never runs. NOT_ESTABLISHED remains non-authorizing. Site layout, unused operand, fused dispatch, call-handle pointer, maximum-stack and boundary mutations refuse and recover. The all256 recipe audit preserves the exact96-opcode set.

I retained Clang O1 stack-usage reports for VM, decode, dispatch, heap, collector and ISA plus the actual preparation/coverage disassembly. Uninstrumented preparation is528B static and coverage464B static. ASan reports288B and576B respectively as **dynamic**: disassembly shows fixed0x440 and0x240 sanitizer fallback areas plus alignment, not an input-sized VLA. Conservative own-frame extents including both alignments are1,438B and1,214B. Decode's project call path is preparation→decode module(96B)→decode function(272B)→isa_decode(144B); dispatch module/function report128/208B, constants build/string creation128/144B. These reports describe project frames and actual fixed fallback areas, not a proven whole-process C stack bound or sanitizer/libc internal bound. The65536 scratch allowance remains explicitly provisional; earlier source/image proof scratch belongs to those phases' own accounting.

## Compatibility and failures retained

Image/constructor/proof neighbors passed3 methods in25.072s: image2032/6384 with55 allocation positions, recipes842/3504 with54 constructor/38 atomic positions, proof338/393. Existing0/0 byte/work figures remained exact. Freshly rebuilt six legacy allocation/work rows are byte-identical, with Analysis39840B. The initial eight Clang/GCC provider syntax controls passed; the final matrix additionally passed12 VM default/union/combined macro ×switch/goto checks and four final test syntax checks. Default VM symbol inspection contains no `vm_union_field_private_*`; private preparation exposes only prepare/destroy/copied statistics.

I retained three first failures:

1. Initial manual provider closure omitted the SDK ffi include, so vm_ffi.c compilation stopped at missing ffi.h before a harness ran. Exact SDK ffi/OpenSSL include flags from the retained closure corrected setup; no production inference.
2. The first instrumented harness used heap.num_objects instead of heap.stats.num_objects. Linked160/all-six had already passed. The member typo correction changed only the test.
3. The first exact final-cap+1 test incorrectly demanded no allocation. LIMIT/prior-output/live-balance were already correct; the corrected test distinguishes reversible late exhaustion from checked initial overflow.

Raw commands, errors, statuses and source hashes are retained. Source/module/provider pins show all retained input binaries/modules unchanged. No failed test assertion was removed to admit an unsupported source.

## Remaining scope

I leave actual union PACK/GET/SET/TAG execution, cumulative retain headroom, ordinary collector container states, root/ASSERT/tail scope and distinct HALT behavior to the next reviewed execution contract. Public borrowed-state invocation, native ordinal/canonical representation and publishers remain closed for these record-containing originals. Existing conservative None-match and historical field-alias precision boundaries remain follow-on work. This checkpoint is not broad union or #1001 completion.
