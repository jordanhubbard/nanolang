# My production integration review

I review merge f44712e0265064784d1bea1cb9109f3b4424fc89 against its actual
parents b481ba57a1c5ac5c0eb68e12b332b3d4ae9bd5b9 (native802 plus plan) and
243747e4fc2785137e64d266b355d3c5a2f8eefe (SDK). I run no compiler or runtime
product. The isolated canonical Python inventory generator is the only
execution in this supplemental review.

## My novel source decisions

My review bundle `/tmp/native-sdk-f447-review` contains `inventory.json`,
`novel-resolution-review.txt` (246 lines, text novel against both parents with
two context lines), `resolution-vs-auto-merge.patch`, and complete paired
`vs-native-parent.patch` / `vs-sdk-parent.patch`. I exclude docs, evidence and
tests from the production patches. I exclude 111 paths whose final bytes are
identical to either parent. Twenty-four paths combine both parents or contain
an explicit integration edit. The compact view intentionally elides inherited
spans and is not an applyable patch. I retain the paired patches for context.
`git show --remerge-diff f44712e02 -- src src_nano schema scripts modules Makefile.gnu`
reproduces the resolution comparison without unrelated inherited commits.

| Boundary | My integration decision |
| --- | --- |
| Environment lookup | I add source `str_split` to native indexed same-owner declaration preference; actual builtin identity stays authoritative. |
| C STRING result | I copy exact builtin STRING-array facts through NominalView, publish through the native checked binder, and return the stable retained row. |
| C contextual destinations | I preserve native original-owner recursion and coarse-plus-contextual comparison; the symmetric known STRING mismatch check uses complete facts before legacy inference. |
| Nano destinations | I keep native complete record, constructor, qualified-call and nested-array-store checks; SDK private visibility, STRING globals and concrete union comparisons remain. |
| Metadata ownership | I use one native tuple helper/cleanup owner, retain rollback wrappers, and remove older eager SDK array/map publication and duplicate union discovery loops. |
| File consumer guards | I place refusal inside native implementation wrappers so failed public entries retain rollback. |
| Native declarations | I preserve ordinary and opaque tuple/callback graph activation, complete children, source-order staging and global discovery before declaration output. |
| Installed closure | I add seven native includes/headers to required inputs; they belong to existing translation units, so the object list stays unchanged. |
| Module failure | I add Environment teardown preflight before the SDK list-generation failure clears its isolated cache. Existing generation leases remain. |
| Provider ownership | I retain the complete SDK runtime registry and move all three old cJSON identity shadow properties to its actual registry; no spelling-only suppression returns. |
| Signed primitive | I carry reviewed ecaf373f0 lowering and the exact reviewed4d1444560 shadow expectation correction. |

## My SDK opaque, tuple and ABI retention audit

I found no dropped SDK functionality in these source boundaries. This is a
static source conclusion, not a fresh compiler or runtime acceptance claim.

| SDK requirement | Retained implementation and ownership |
| --- | --- |
| ABI2 and wide values | `runtime/dyn_array.h`, `runtime/dyn_array.c`, `runtime/native_array_abi.h` are byte-identical to the SDK parent. Width stays `size_t`, ABI stays 2, borrowed insertion snapshots precede growth, and stale image declarations refuse before entry. |
| Typed array carriers | `transpiler_opaque_arrays.inc` is byte-identical to SDK: tuple/callback/opaque carriers retain exact typed `sizeof`, fresh empty width-zero rules, once-only source/index staging, and NULL getter refusal before a copied load. Nano equivalents remain in `native_array_load_code`, `native_array_same_type` and the actual array-call emitter. |
| Semantic versus C identity | `transpiler_opaque_names.inc` is byte-identical to SDK. Native owner-aware nominal resolution replaces only legacy spelling lookup; opaque identity remains separately framed in `env_opaque_keys.inc`. No C spelling becomes declaration authority. |
| Complete tuple annotations | `env.c` retains the complete child accessor, exact count/flat-view invariant, transactional refresh and one checked owned tuple binder. Native checked copying replaces the older duplicate SDK helper; complete child equality remains in `type_infos_equal`. |
| Tuple/callback graph | `native_derived_collect` visits complete tuple children and callback parameter/result trees. Native activation covers ordinary derived types in addition to the SDK opaque/array subset. Registry owners and declaration-versus-layout edges remain. |
| Metadata replacement | `typechecker_native_context.inc` owns independent materialized snapshots and append-only suffix rollback. Complete original NominalView context authorizes values; flattened emission facts do not. The native owner independently read the complete C/Nano adapter delta and reported scoped PASS. |
| Original SDK controls | `native_sdk_opaque_cases.py`, `test_native_sdk.py`, `test_native_array_abi.py`, `test_str_split_paired.py` and `str_split_result_context.c` are byte-identical to the SDK parent. I preserve source-prefix, callback/tuple, original Json.Json, tag/width and installed-only assertions. |

I preserve all original native tests and add the reviewed signed-length ninth
method. The only follow-up code edit after the merge is4d's existing shadow
expectation `strlen` to `nl_str_length` plus its legacy `str_len` assertion.
I do not relabel either lane's historical bootstrap as an integrated result.

## My generated inventory check

I copied the canonical generator and both JSON inputs into an isolated temporary
tree, invoked its actual `main` with `--root`, and compared both generated files
byte for byte. Both match the committed outputs. The SDK input delta is exactly:
`env_provider_leases.inc`, `env_record_lists.inc`, `env_signature_snapshot.inc`,
`runtime/native_record_list.h`, `typechecker_native_context.inc`,
`typechecker_nominal_arrays.inc`, and `typechecker_nominal_context.inc`, all under
`src/`. No input was removed. Full input/output hashes and sizes are in
NATIVE_SDK_GENERATED_INVENTORY_REPRODUCTION.json.

I await completed source review before any integrated bootstrap or runtime gate.
Full File execution, source-hidden SDK acceptance and release remain open.
