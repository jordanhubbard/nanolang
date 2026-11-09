# My owned binding metadata correction

I establish exact ownership separately from the earlier allocator-size
observations. `env_define_var_impl` duplicates a binding's inherited type-name
string. The binding name is also owned; `type_info`, declaration facts and
runtime values have separate lifetimes. My existing block and match cleanup
already release the owned metadata.

I find missing releases in named/first-class call parameters, effect-handler
parameters and loop exits. I release optional owned type-name strings in both
call paths and use my existing metadata-only pop helper for handlers and all
thirteen loop restoration sites. I rename that private helper for its broader
use. I change no carrier layout, public ABI, value-copy/borrow rule, declaration
fact, shadow selection, ten-second supervision or native CLI budget.

My interpreter unit builds private env/eval objects with test-only allocation
and free forwarding. Normal compiler/interpreter objects remain unmodified by
those hooks. The fixture tracks actual duplicated `FrameMetadataOwner` strings
and their actual frees, protects the caller's metadata, and forwards every
allocation/free to the real allocator. It does not infer ownership from heap
size or replace runtime code with a model.

The final unchanged test refuses exact baseline evaluator source from
`391ddafa0b3ffb7bf0ab5a97e950d737896b04c7` with231 unreleased observed metadata
allocations. The corrected plain O3 and ASan/UBSan O1 controls both report zero.
I retain source, compiler/link commands, streams and stable selected
source/object/image hashes in the exact reports. These local controls use
cached dependencies; they do not qualify complete fresh providers or NativeSDK
closure. Darwin leak detection remains disabled; I do not claim LSan acceptance.

The controls cover named/first-class/recursive parameters, ordinary locals,
returned record/string/callable values, both C API borrow kinds, handler
parameters and range/static-array/dynamic-array/legacy-list loops. Normal,
empty, null, invalid-native-tag, early-return, break and continue cases check
metadata release and actual returned values. Constructed native tags do not
widen source admission. Ordinary local-block metadata already balances on the
baseline; I retain that positive control. The earlier smaller witness records
210 unreleased allocations, before adding the remaining loop and return cases. I also retain the intermediate228-allocation refusal before the final static-array controls.

I also rebuild the production evaluator without allocation/free wrappers and
run the same validated allocator probe. Both metadata-inheriting5000-call named
batches now add0 live bytes; the original first batch added80000. The real2MiB
allocation/free control still passes. Clean first-class calls still retain
653360/645760 bytes in this observation, and their separate #1140 lifetime
investigation remains open. I do not claim this corrects record carriers or
establishes a source-deadline cause or whole-factory performance gain.

I preserve two initial strict fixture build refusals: a call to private
`env_free_value` and a signed size comparison. I use the existing public
carrier-name release and explicit integer-only carrier cleanup in that fixture,
and compare matching size types. I preserve the mechanical loop-edit refusal
from expecting14 occurrences where source contains13; the exact guard remains.
I suppress no compiler warning and invent no API or SDK declaration.

I predeclare fresh original instrumented GCC/Clang interpreter/call/array
qualification and complete original Linux GCC/Clang and Apple component plans.
I also require one fixed six-attempt GCC comparison against merged baseline
`391ddafa0b3ffb7bf0ab5a97e950d737896b04c7`, retaining all outcomes. These changed
production-source plans preserve original flags, selections and budgets. I do
not repeat an unchanged failed plan for green or launch a new kernel observer.
Source reliability, broader frame/value reclamation, #1140, NativeSDK closure
and release acceptance remain separate open requirements.
