# I preserve optional scalar aggregate storage

My full product bootstrap at `9808acf6` exposed a conversion from optional to
integer aggregate storage after PR597. Stage1 and hello passed; Stage2 native
translation refused. I recorded task `task_497e1ba5b9544b81b3614ec37da90e98`
before changing production code.

I first extended directed storage conversion to inferred int/bool destinations,
keeping exact constraints intact. Fresh mixed-record controls then exposed a
second string-only assumption in record projection. My corrected projection
retains present integer/Boolean/string tags and existing tagged optional values.
I retained `/tmp/nanolang-optional-scalar-focused.log`, including its four failed
native controls, before extending projection and scalar field seeding.

The next product bootstrap at `6009d8cb` still refused an exactly constrained
scalar destination. Static review found that inferred local/parameter storage
and typed returns could turn a late-resolving projection into an exact source
constraint. I now use directed flow for inferred scalar storage and keep typed
return checks at the consumer. Constructors and array payload constraints remain
exact; the existing emitter checks tags before unboxing a scalar return.

Production `967cbec0`, with controls at `13a7a982`, passes five focused methods
covering 21 paired VM/native artifacts, including mixed present/optional record
arguments and projected scalar locals, calls and typed returns. GCC generated
ASan/UBSan/leak checks pass in 3.624 seconds; Clang passes in 4.772 seconds.
My 1,222 shape checks pass under ASan/UBSan/leak checks, including both conversion
orders, mismatched payload refusal, exact-destination refusal, recursive storage
and solver stability. Independent static review found no scoped blocker.

The preceding projection checkpoint passed all 2,422 native checks. At that
checkpoint, the final consumer change still required the full suite and complete
product bootstrap. The later results below supersede that pending suite check;
the complete product gate remains unmet. The distinct historical
export-shadow abort and the release publication hold remain open.

My subsequent static trace located the remaining constraint at the typed write
of `ASTLet.name` in `par_closed_function`. I retain its fresh compiler module and
trace under `/tmp/nanolang-product-current-compiler-storage.nvm` and
`/tmp/nanolang-optional-scalar-trace.log`. Scalar array writes now keep exact array
payloads separate from late-resolving projected sources. Already tagged values
retain their known payload constraints. First-write empty-array inference uses
the resulting array kind. I retained the intermediate 11-failure and 2-failure
native logs; I corrected the code without weakening those assertions.

Final production `727db160` passes all 2,422 native and 1,222 shape checks, and
five focused methods (21 VM/native artifacts) pass GCC in 3.633 seconds and
Clang in 4.702 seconds with generated-code sanitizers. The strengthened consumer
case includes typed array push/set after a projected scalar local overwrite.
The complete retained 350,016-byte compiler module translates successfully.

Full native compiler acceptance remains unmet: product `9d49de10`, which carried
an earlier write-separation checkpoint, built Stage2 but its ordinary hello
smoke aborted without a compiler diagnostic. I preserve the binary/log/hashes
in `/tmp/nanolang-product-startup-9d49de10/evidence.json`; I have not replayed it
or attributed its cause. PR601 stays draft and this task remains open. Later
focused and native gate success does not establish a repaired compiler startup.

At integration `0cf55548` through main PR615, I pass 1,269 shape checks
and twelve optional/numeric methods (3.892 and 0.885 seconds). Fresh product
`b8f3b842` passes Stage1 and hello, then builds Stage2; its ordinary hello
smoke stops at a native invariant in `parse_function_definition` at generated
C line 46847. I retain the exact artifact, log and SHA256 manifest in
`/tmp/nanolang-product-startup-b8f3b842/evidence.json`. I do not replay that
artifact or attribute its cause. The diagnostic enables static source review;
it does not satisfy the full product gate. PR601 remains draft.

## I pass the corrected storage bootstrap gate

Source-only emission at `b8f3b842` located the named guard: an exact string
projection required plain storage even though inferred storage can carry a
boxed present string. This identifies a static representation gap, not the
runtime contents of the preserved failed process. At `c7b95857`, I accept
plain int/bool/string storage or a boxed value with the exact expected payload
tag. I retain absent/wrong-tag refusal and check a string pointer before use.
Independent static review confirmed that record construction stores those
payloads in the same slots.

At `d8362cc4` (test-only addition after `c7b95857`), all 2,422 native checks
and 1,269 shape checks pass. Six focused methods, including nested records,
pass GCC in 4.519 seconds and Clang in 6.398 seconds with generated-code
ASan/UBSan/leak checks. Logs are `/tmp/nanolang-typed-projection-*`.

Fresh integrated product `85d2e294` passes both bootstrap stages, hello and
installed execution without the C seed. The ordinary product suite passes
27 of 28 methods in 11.665 seconds. Export-shadow compilation still aborts,
now with a diagnostic in `parser_set_let_var_type`; I retain that separate
artifact and log in `/tmp/nanolang-product-exports-85d2e294/evidence.json`.
This completes the storage task's fresh bootstrap requirement. It does not
complete export-shadow task dd74, product PR522 or release acceptance.
