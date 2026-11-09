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

My exact implementation head `f92237c52a894eea7d4ee13de5dfc4df224a99f9`
passes fresh instrumented GCC/Clang run42608183231. Both complete interpreter
units report zero outstanding observed metadata allocations;28 original Python
methods and both original array-alias routes pass. I CRC/hash-check all12
entries per archive, compare each full source tree to Git and retain the texts,
inventories and audit report. Seven stable providers are reported per job;
this upload contains their hash reports, not native provider/unit image bytes.

My complete Linux run42608187904, Apple run42608192902 and fixed GCC comparison
run42608197477 are separate plans. The Linux GCC source-emitter step refuses
its original ten-second shadow deadline after1198 starts,1197 completions and
zero recorded assertion failures, with9.864710 summed completed-body seconds.
The final started shadow is `service_wire_serialize`. I do not attribute that
unfinished shadow or historical failures to a proved teardown cause. Later
controls and the other plans remain running at this checkpoint; I do not call
complete original source/platform acceptance green or restart the failed plan.

All declared plans are now terminal. Complete Clang passes. Complete GCC and
Apple fail only their source-emitter factory step; both later original
translator controls pass2458 checks. Apple stops after1064 shadow starts and
1063 completions, with zero recorded assertion failures and9.856874 summed
completed-body seconds; the last started shadow is `nanoisa_emit_shadows_nasm`.
I preserve the terminal metadata and original logs separately. My full archive
audits check1822 GCC,6573 Clang and2023 Apple entries, complete Git trees,
476 stable-input reports,144 reported native products and72 other reported
products against uploaded bytes. I retain the complete archives locally;
provider/tool hash reports without uploaded bytes do not establish SDK closure.

My fixed six-attempt GCC comparison42608197477 finishes in its declared order:
original failure, modified failure, modified success, original failure,
original failure, modified failure. Baseline succeeds0/3; corrected metadata
code succeeds1/3. Every completed shadow reports zero assertion failures;
the second attempt stops after1281 starts/1280 completes, while the other
five complete1286/1286. Five factory failures retain the original ten-second
supervision refusal. Zero fully successful pairs support a whole-factory gain.
I preserve and independently audit all53 entries,197 shared Git inputs,
67 stable source/provider inputs per attempt, two actual C seed ELF images
and the one actual candidate-created ELF compiler. That compiler is created,
not executed by this factory comparison. Metadata correctness, broader callable
ownership and complete source acceptance remain distinct requirements.

I retain three initial archive-auditor refusals. I distinguish refused
compilation's exact prior bytes from an executed product's before/after hash,
enumerate the inherited source-provider negative fixtures and normalize their
explicit source/provider identity schema. I still require the same byte/hash
and refusal/no-execution evidence; I suppress no native test failure.

Three actual altered-archive controls refuse independently: changed seed-image
bytes, changed shared-source identity and changed candidate compiler bytes.
Their commands and complete refusal streams remain in
`../metadata-comparison-audit/refusal-controls.json.gz`. I retain six original
hosted job logs and terminal run metadata. These are archive audits; none starts
a new native factory.
