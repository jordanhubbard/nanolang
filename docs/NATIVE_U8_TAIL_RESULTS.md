# My native U8 tail-result contract

I track this repair as `task_9eb8a1d54fc441f582b180b398ac0a08`.
My ordinary non-self TAIL_CALL emits a call to a function with the same result
signature, then follows existing frame cleanup. My boxed U8 result already has
a checked RET and a native carrier, but the tail-call assignment whitelist omits
it. I must assign that result to nresult before cleanup instead of discarding it.

I change only that whitelist. I preserve signature checks, exact tag checks,
root cleanup, simultaneous self-tail staging and all other result kinds. I test
all 256 U8 values through multiple ordinary tail relays, comparison with direct
calls, both helper declaration orders, self-tail control, and retained wrong-tag
refusal. Generated GCC/Clang sanitizer execution checks ordinary cleanup.
I do not change enum, numeric, heap or source-language semantics.

I implement this on main `d66f9e73`, after pre-code contract `3af2cd53`.
Production `d8a3b304` changes one whitelist condition. Three focused VM/native
GCC methods pass in 10.831 seconds with ASan/UBSan/LSan. Twenty-four Clang
methods pass in 25.920 seconds: three new tail-result methods plus existing
enum, numeric-union and tagged-arithmetic controls, also with sanitizers.
I retain `/tmp/nanolang-u8-tail-{build,gcc,clang}.log`. I do not repeat the
full native gate already passed by the unchanged base for this single condition;
I test the changed route and adjacent scalar transports. I make no new compiler
bootstrap or startup claim and execute no retained failing product artifact.
