# My complete native record-width prerequisite

I retain the first148eb Puck failure under full source task8bbc. Fresh bootstrap
passed in321.274 seconds, including actual Stage1/Stage2, installed compiler smoke
and the C-seed independence check. The first ordinary companion method then failed
on actual Stage1 publisher source preparation with --allow-temporary-files. The
child terminated with SIGABRT(-6), not an accepted refusal; its output sentinel
remained unchanged. No later method or configuration ran.

The existing macOS crash report names dyn_array_push_struct called from
nl_collect_files_ast+3348. Its image UUID matches the frozen Stage1 binary.
Nonexecuting LLDB disassembly shows the call passing488 bytes. My current runtime
explicitly rejects struct_size greater than UINT8_MAX: DynArray.elem_size is a
uint8_t, and the pre-growth snapshot is a255-byte automatic buffer. The actual
ParsedDeclarations record contains the large Parser plus its declaration capture.
FileParsedOrigin embeds that same record and also exceeds the representation.
This is a deterministic representation limit, not evidence of host memory pressure.
I preserve the original report/disassembly outside the qualified source tree and
do not rerun the failed binary to rediscover this limit.

## My proposed representation correction

I propose native DynArray ABI2 with size_t elem_size, matching the existing C API
width parameter and allocation arithmetic. This is the native foreign array ABI,
not the shared ownership descriptor envelope. I do not change the latter's version
or claim compatibility between old and new native array objects. The owning header
defines the version and full struct layout. All compiler/runtime/module consumers
must rebuild together; an old ABI1 foreign export must fail the existing identity
guard before entry or data inspection. Missing and wrong-image declarations remain
refused. I preserve opaque array ownership and existing caller lifetime preconditions.

I retain checked count-by-width arithmetic, capacity doubling/reservation checks,
null/negative/invalid storage validation, and existing allocation-failure semantics.
Constructors/clones return NULL where already specified; insertion/reservation
cannot report recoverable failure and therefore retain terminal failure behavior.
I remove only the one-byte representation ceiling, not checked allocation bounds.

For struct insertion I keep bounded small-record automatic staging and use an
explicit temporary heap copy for records larger than255 bytes. I validate width,
current metadata and representable intended storage before copying. The temporary
copy is acquired before any growth or storage replacement, so a source pointing
into the same array remains valid across growth. Every successful insertion frees
the temporary once; every allocation-failure test observes the existing terminal
contract. I do not replace a borrowed source with a pointer into reallocated data.
I audit clone, reserve, slice/copy, get/set/pop, GC destruction, generated array
helpers, FFI adapters and consumers for narrow casts or copied layout assumptions.
This remains flat record-byte copying; owned child-graph semantics are not silently
expanded or declared complete by the width correction.

## My source and ABI audit before implementation

My initial repository search finds the layout/version in runtime/dyn_array.h and
the narrow staging/width check in runtime/dyn_array.c. Both C and Nano native
emitters already name the header's NANO_DYN_ARRAY_ABI_VERSION macro in their guard;
their generated helpers use arr->elem_size without a one-byte cast. I still audit
all direct struct definitions, initializers, native foreign export declarations,
FFI loader and VM array adapters, module/header/cache dependency manifests, and
installed headers before asserting complete closure. Existing tests hardcode
ABI1 success and255/256 width behavior and must be corrected to the actual new
version plus an explicit stale-ABI1 refusal, not weakened to accept any version.
VM heap arrays and managed LLVM/Wasm handles use separate representations; I do
not widen those by changing a similarly named local variable.

I keep the source parser and graph design intact. I do not split the retained
Parser into arbitrary fragments, replace actual AST retention with reparsing,
introduce unowned pointer tokens, delegate Nano lowering to a C bridge, or skip
the real publisher case. I audit every record stored by this new lane, including
ParsedDeclarations and FileParsedOrigin, against actual generated widths.

## My required complete acceptance

I first submit the complete production/ABI inventory and fixtures for review.
Focused C controls cover255,256,488,504 and a larger record width; exact contents,
borrowed-element insertion across growth, reserve-before-first-insertion, clone
independence, get/set/pop, overflow, zero width and injected staging/storage
allocation failures. Both producers compile actual large-record arrays with
mandatory shadows, including the real retained Parser graph. Existing C ABI
layout/export tests and native compiler/FFI loader/VM guarded routes require the
new version and reject actual stale-layout modules before entry. Scalar/string
array consumers and the existing allocation/alias suite remain required.

Because this changes the native layout, I require a fresh complete C-seed,
Stage1/Stage2, runtime and foreign module closure on both hosts. I cannot copy
the otherwise successful148eb compiler products across it. I retain those
products under their original successful bootstrap pin. The original seven
selected C configurations, full paired companion/provider owner controls,
publisher/source-plan/schema/parser/module/wrapper and public refusal neighbors
remain required. Installed/native array ABI acceptance must cover the changed
header and rebuilt modules, not merely a local struct-size check. Linux builds
remain held until capacity is stable. Full File source execution and release
publication remain open; this prerequisite does not discharge them.
