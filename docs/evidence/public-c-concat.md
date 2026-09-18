# My public C concat evidence

I qualify task_c075a8783181411d89c63a2e6f1950b0 at frozen source/harness `e9f6b6b891b7df81bb7d9afca0774512f4093b29`, production `b7fe9981`, after my [reviewed preimplementation contract](../PUBLIC_C_CONCAT_CONTRACT.md). I retain [21 source/provider/harness/tool hashes, two compiler hashes and six log hashes](public-c-concat.json). Every recorded identity remained unchanged after all gates terminated; my final evidence commit changes documentation only.

I lower exact STRING PLUS and builtin str_concat through one private helper with distinct left/right operand slots and portable comma sequencing. I preserve declaration, lexical, checked-signature and expression-callee precedence. I remove the internal nano_strcat inference shortcut and unprefixed helper collision. Unknown/mixed arguments are checked refusals. I preserve the existing first-NUL string boundary.

I share checked flexible-array allocation, registration and exit cleanup with integer/float snapshots. Header/payload/terminator size checks precede allocation; combined concat length checks precede addition. Inputs remain borrowed; each successful result is a fresh stable snapshot. Failed registration frees the unlinked node and leaves registration state unset. Existing scalar allocation/registration diagnostics remain unchanged. I retain results until process exit; I do not claim tracing, early reclamation or an RSS bound.

| Frozen gate | Result |
|---|---|
| GCC concat methods | 4 pass, 1.746s |
| Clang concat methods | 4 pass, 2.531s |
| GCC adjacent public C methods | 24 pass, 10.759s |
| Clang adjacent public C methods | 24 pass, 13.749s |
| Unchanged public C backend programs | 7/7 pass |

My concat methods compile actual generated C99/C11 at O0/O2 with ASan/UBSan and default Linux leak checking. They cover nested stateful left-to-right operands, once counts, equality slot interactions, globals/calls/returns, repeated loop-condition evaluation, 300 concatenations with earlier aliases retained, mixed scalar conversions and both private/old-helper namespace collisions. Empty, ordinary, UTF-8 long and first-NUL visible bytes pass public C, the ordinary interpreter and a verified VM module from fresh source. This interpreter/VM comparison qualifies the byte fixture; stateful ordering is checked by exact assertions in generated public C.

Isolated generated helpers exercise checked malloc/atexit refusal, freeing the unlinked node, cleanup of retained integer/float owners, NULL input refusal, length-sum/header-size refusal before allocation and idempotent cleanup. These are post-repair deterministic controls; I do not execute malformed modules, invalid memory or historical failure artifacts. API controls retain exact type/arity and binding boundaries, the first diagnostic, prior path/stream contents and valid same-process recovery. Existing scalar provider and ownership controls are unchanged and pass in the adjacent suite.

I make no bootstrap, Darwin or whole-public-target portability claim. Parent6ade remains open for GNU expression blocks, match typeof, advertised options/captures and their required implementation/acceptance. Canonical merge precedes task reconciliation.
