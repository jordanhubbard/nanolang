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

The preceding projection checkpoint passed all 2,422 native checks. I am
requalifying the final consumer change against that full suite and the complete
product bootstrap before closing this task. The distinct historical
export-shadow abort and the release publication hold remain open.
