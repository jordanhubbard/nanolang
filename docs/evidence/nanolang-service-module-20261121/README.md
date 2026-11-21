# My shared service metadata evidence

I implement my normal narrow-module service boundary in pure NanoLang at
`d7670ef3f`, under #1142 and draft PR #1144. I retain the existing C validator
as a test oracle. My running NanoLang implementation calls neither that
validator nor its catalog getters. I change no installed translator or
executable selector.

## My boundary

`src_nano/nvm2c/service_module.nano` implements legacy File v1 validation and
ordinary shared nominal dispatch: File/TCP v2 and mixed/repeated v3 plans.
I preserve exact catalog names and positional argument/result tags, including
missing parameter storage and embedded-zero names. I dispatch nominal prefixes
before legacy fallback and preserve the actual C status order. Malformed
nominal claims cannot fall back to a legacy plan. Standalone WebSocket v2
remains refused by ordinary shared transport; its separate private adapter is
not silently enrolled here. Multi-instance v3 retains WebSocket catalog support.

I compare four values for every carrier: validation status, metadata presence,
File-instruction presence and pending execution. I scan only checked function
extents, recognize File instructions before operands and stop at unknown or
truncated non-File instructions. Operand bytes do not become opcode claims.
Metadata success does not clear pending execution.

Legacy raw failure precedes import/reference/callback count failure, then
out-of-range import indices precede entry name/signature checks in catalog
order. A File instruction without a nominal prefix refuses before absent
metadata can succeed. Legacy metadata ignores unrelated ownership/layout
bytes. Nominal plan limits map to existing status 13; other plan refusals map
to 8. These are tested C boundary decisions, not a new execution verifier.

## My paired qualification

Both final methods pass on Linux ARM64 (417.454 seconds) and Darwin ARM64
(140.114 seconds), with 10,863 complete reports through each of five routes.
The corpus contains 10,855 synthetic carriers, 10,087 explicit status expectations
and eight actual producer modules per host. The routes are C-seed native and
C/NanoLang producers through verified NanoVM and instrumented generated C.
A separate full-corpus method instruments actual shared/nominal/raw providers,
classification/scanning, cursor and all three immutable catalogs with ASan/UBSan.
Generated fixtures link an ordinary AOT runtime; I do not claim whole-runtime
instrumentation or complete interpreter LeakSanitizer qualification.

I independently regenerate byte-identical synthetic inputs and complete C
reports on both hosts and match 496 source hashes. Both eight-product archives
retain exact source, catalog, module and carrier bytes; I verify every archived
file hash and matching producer-source hashes. These actual modules select
shadows during production. Their metadata checks do not run network lifecycles.
The genuine reused development NanoLang compiler matches the previous paired
receipt. I configure its genuine Linux GNU-as helper explicitly because the
image lives outside bin; I fabricate no installed stages.

I retain the complete previous nominal matrix plus legacy header-byte values,
all truncations, exact/reversed imports, count/index/ref/callback precedence,
all 256 result/File/int parameter tags, unrelated claim bytes and malformed
nominal prefixes. All 256 opcode values are tested as instructions, integer
operand bytes and bytes after an unknown instruction. Explicit classification
checks cover absent metadata, unrelated claims, every import kind and valid,
empty, out-of-range and UINT32_MAX function spans.

Original raw File and File/TCP/WebSocket/multi-instance metadata/allocation
units pass on both hosts. All three unchanged shared-module methods also pass:
725 allocation/attachment checks, 655 transport/consumer checks and 212 mixed
selection/recovery checks per host. I retain their complete commands/results.
The existing 7,915-case nominal matrix separately passes my modified native
fixture on Linux. Identifier tests pass on both hosts. Seven production and
11 fixture named functions retain shadows; the five-route builds execute
selected dependency and root shadows.

## My setup failures and correction

I retain the first reporting fixture's unbalanced concatenation parse refusal,
then replace it with named string pieces. I retain both Darwin links that lack
the Homebrew crypto search path, then explicitly select its existing library.
Neither required a product or warning-policy change.

My first manual original-unit selector names a nonexistent method; it discovers
no test. My next manual closure lacks the actual LLVM refusal provider and
places COMMON archives before newly instrumented users. I retain both failures,
select the real LLVM provider and put those same archives after their users,
preserving every ordinary object and link flag. All three original methods
then pass without assertion changes or replacement providers.

Building that provider exposes ambient tool setup: Linux's development Clang
rejects its automatic GCC selection under Werror; explicitly pinning GCC 13
then fails the Wasm package's unused-argument check. Darwin's SSH PATH lacks
opt. I preserve all three failures and select installed matching Clang/opt 18
on Linux and matching Homebrew Clang/opt on Darwin. The genuine managed runtime
IR packages verify and the actual LLVM refusal provider builds. I disable no
warning and claim no LLVM language/release qualification from this prerequisite.

## My remaining requirements

I model canonical normal narrow storage. Absent service bytes have no distinct
nonnull empty pointer; arbitrary malformed C pointer/count graphs remain C
unit-test controls. Allocation-denial and output-preservation assertions retain
their C API scope rather than claiming equivalent NanoLang heap behavior.

I complete only the shared narrow metadata roadmap row. Rich-v2 table adaptation,
whole-module service/private profile conversion, instruction/lifetime/flow/host
admission, representation analysis, NanoLang C emission, canonical integration,
installed/no-C-seed products, raw fixed points and all remaining #976 release
requirements stay open. This issue, draft PR and release remain open.

`manifest.json` binds exact inputs, full C reports, products, source/tool/host
manifests, original failures and terminal evidence. I retain the collector and
original-unit closure helper so their actual commands remain reviewable.
