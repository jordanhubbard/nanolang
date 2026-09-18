# My public C integer conversion contract

I execute task_3a0411f257e4437ca446d2bbec0e5ed6 under public portability parent6ade.

I execute a bounded child of task_6ade6d62ef644390bb9645b15077c8df on main96cdb7c5. Static src/c_backend.c int_to_string emits a GNU statement expression with one mutable static buffer per callsite; repeated calls can overwrite retained aliases. I require exact INT builtin input, decimal signed int64 endpoints, once evaluation, binding/declaration/checked-signature precedence, portable C99/C11 helper emission and stable snapshots using the existing float conversion pool. A shared checked copy helper retains process-lifetime snapshots and frees them at exit; checked allocation/registration failures free any unlinked node, retain previous ownership and never mark failed registration successful. Shared allocation diagnostics may become scalar rather than float-specific, with existing float failure controls updated explicitly. I preserve formatting-specific errors and shared binary64 formatting unchanged. This is process-lifetime retention, not bounded memory or early reclamation.
I qualify fresh GCC/Clang C99/C11 O0/O2 ASan/UBSan/LSan ordinary signed endpoints, globals/returns/repeated-call aliases, mixed float/int snapshots, side effects once, namespace/binding collisions and exact non-INT/unknown refusal with prior-output retention and same-process recovery. Isolated generated-helper failure injection covers allocation, atexit registration and cleanup; no production fault hooks or historical failed artifacts. I freeze production/harness/tool hashes before gates and send production for independent review before fresh fixture execution. Concat allocation/operand ordering, expression blocks, match typeof, options/closures and whole public-target portability remain required separate scope.

My registry declares int_to_string(INT)->STRING. My interpreter returns a fresh decimal string for INT and a legacy fallback "0" for other runtime tags; this child preserves my public C exact-type refusal boundary rather than admitting that fallback. I do not change the interpreter or global type compatibility.

## My remaining inventory at main96cdb7c5

| Path | Static finding | Required boundary |
|---|---|---|
| c_backend.c AST_CALL int_to_string | GNU block, mutable static buffer, spelling-only interception | This child: exact INT, binding precedence, portable stable ownership |
| c_backend.c nano_strcat and STRING PLUS | unchecked size/allocation, untracked result, unordered C arguments | task_c075a8783181411d89c63a2e6f1950b0: ordered checked ownership contract and implementation |
| c_backend.c AST_BLOCK expression | GNU statement expression | Parent6ade: portable expression planning or explicit supported-context refusal |
| c_backend.c MATCH payload binding | __typeof__ of anonymous variant storage | Parent6ade: standard type representation or checked refusal |
| c_backend.h options | no_stdlib/static_strings not consumed; no_main only checked with globals | Parent6ade: implement documented modes and qualify output, or explicitly reject unsupported requests |
| c_backend.h capture description | captured closures described without an established lowering | Parent6ade: inventory callable/capture implementation and required acceptance |
| c_backend.h string description | static/no-heap claim contradicts conversion snapshots and concat | Parent6ade: document actual ownership by operation; no deletion of release obligations |

I keep the full parent open. These are static findings, not executed failure reproductions or whole-target acceptance. My literal normalization childcf871 is complete through canonical PR756 exactdee259e9 and its retained evidence.

My first frozen GCC/Clang runs pass three methods each and stop the endpoint method before generated C compilation: my harness namespace assertion also matches the intentionally declared user function. Generated private helper selection is correctly nano_cb_1. I retain both logs and correct only the assertion to inspect the private static const char pointer signature; production and all original frozen identities were unchanged.
