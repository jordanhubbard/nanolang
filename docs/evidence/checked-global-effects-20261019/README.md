# My checked global effects receipt

I qualify the bounded #998 checked-global bridge from `dfc8708b6`, with the
contract recorded first in `1f4509148`. I use the actual C seed and NanoVirt
providers pinned in `sha256.txt`, plus my newly built C-hosted canonical compiler.
I do not claim fresh raw Stage 1/Stage 2, Linux, or release qualification here.

## Product and source pins

I build `src_nano/nanoc_v06.nano` with the aggregate's `nanoc_c` into
`/private/tmp/nanolang-checked-global-effects-compiler`. `build.log` records a
successful exit 0 after both early-failure guards. The three changed compiler
sources and all runtime/tool binaries have exact SHA-256 pins in `sha256.txt`.
This build checks imported and root shadows, including exact-index identity,
bounded symbol lookup, checked/raw reset, failure and generic refusal controls.

My bridge runs checking, immutable summary capture, program lowering and selected
shadow lowering synchronously. Only actual checked bound globals feed new masks;
raw no-global inference remains unchanged. Original function index, owner and
saved epoch constrain use. I retain no Symbol arrays or persistent Parser state.
I return no text after checker or either lowering failure, and canonical runtime
shadow execution still precedes product publication.

## Retained initial failures

`initial-fixture-failures.log` records my first paired run: four new-fixture
failures, with existing effects cases passing. Both producers intentionally omit
advisory entries for the synthetic `__init__`; my new global fixtures now assert
that exact absence without changing existing metadata expectations. An imported
fixture omitted `pub` and correctly failed access checking; I corrected its
source. I also make all expected functions reachable and use whitespace before
`=` after a nested generic type so the negative fixture reaches actual shadow
lowering. Its canonical failure now asserts that diagnostic explicitly.

`corrected-effects.log` records all seven methods passing in 3.429 seconds with
the first C-hosted build. That build preceded two early-return compatibility
guards. It is diagnostic evidence; the final exact-source run is recorded
separately below.

## Qualification boundary

The paired effects corpus checks immutable/local reads (0), mutable/transitive
reads (2), same-spelled imported global owners, actual VM/native outcomes,
byte-exact assembly round trips, and absence of native-code authority from
stripped, forged or duplicate advisory metadata. Failure controls preserve the
prior output on checking, runtime shadow, and shadow-lowering failures.

My source adds `CheckedAdvisoryOutput`, `NanoisaCheckedText` and snapshots of
`array<ASTFunction>`. C-hosted success cannot establish their generated raw
compiler shapes. Fresh combined Stage 1/Stage 2 bootstrap and exact hosted
platform checks remain required in aggregate integration. I do not close #998.

## Final local results

I pass all 28 effects/purity/match/affine metadata methods with the final compiler
in `neighbors-initial.log`: seven effects, eight purity metadata, three match,
five affine completion, and five affine-use methods. Both C-seed NanoVirt and the
changed canonical compiler participate. The raw checker/reset fixtures build
actual native helpers with the pinned C seed; no stub supplies checker facts.

That 33-method run took 176.350 seconds and failed nine native-output subcases
across two declared-purity methods because my isolated checkout lacked
`bin/nano_aot_runtime.o`. Every failure records the same missing-file linker
error. I link the existing pinned aggregate runtime object into ignored `bin/`,
change no source or assertions, and rerun only the five declared-purity methods:
all pass in 16.388 seconds (`purity-final.log`). I retain the original failure;
this is a demonstrated missing test prerequisite, not an unexplained retry.

Reproduction uses `NANOLANG_EFFECTS_COMPILERS`, `NANOLANG_PURITY_COMPILERS`,
`NANOLANG_MATCH_COMPILERS`, `NANOLANG_AFFINE_COMPILERS` and
`NANOLANG_AFFINE_USE_COMPILERS`, each set to the absolute pinned `nano_virt` path
and the new compiler path separated by the platform path separator. I set
`NANOLANG_SELFHOST_COMPILER` to the new compiler. I invoke unittest modules
`tests.test_function_effects_metadata`, `tests.test_function_purity_metadata`,
`tests.test_match_coverage_metadata`, `tests.test_function_affine_metadata`,
`tests.test_affine_use_metadata` and `tests.test_purity_contract`. The last module
alone is the targeted prerequisite-corrected rerun.

An independent production review found no blocking issue in exact-index analysis,
raw fallback, early guards, owner/epoch checks or no-text failure handling.
The component is ready for aggregate qualification, not a release declaration.
