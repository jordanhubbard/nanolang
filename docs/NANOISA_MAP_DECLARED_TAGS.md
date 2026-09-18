# My declared map write contract

I track `task_b19f8bf0527d4a33911be26706629616`. HM_NEW records key and value
tags; HM_KEYS and HM_VALUES use those declarations for their array element
tags. My standalone native string-key maps already require the exact value tag.
I give ordinary VM HM_SET the same declared-content invariant.

Before mutation, I require the key tag and value tag to equal the map's declared
tags. I do not coerce bool, U8, enum or float to int. On refusal I report a type
error, release all three consumed values, leave other aliases and prior map
contents intact, and publish no result. Successful insertion and replacement
retain their existing ownership behavior. The direct internal heap API keeps
its existing caller precondition; this child checks the bytecode dispatcher.

This child does not change HM_GET/HM_HAS/HM_DELETE, add native non-string map
keys, or define new wildcard declarations. I test ordinary VM API lifecycle,
aliases, replacement and checked refusal, plus same-module native supported
string-to-int/string controls. I do not replay retained compiler failures or
use old malformed artifacts.

## My measured acceptance

At source `eff59e75` on main `780180cf`, my full VM gate passes 274,493 checks.
Seven paired methods pass, including unchanged map aliases/collection tests and
ordinary NanoLang string-to-int/string insertion and replacement with normal
shadows. Standalone native products retain strict O2 warnings and
ASan/UBSan/LSan. My focused dispatcher/heap/value sanitizer build passes 77 API
ownership checks: exact scalar refusals, string/array owner release, preserved
prior contents, and successful replacement after failure.

I retain `/tmp/nanolang-map-declared-final.log` and
`/tmp/nanolang-map-declared-sanitizers.log`. The first run at
`/tmp/nanolang-map-declared-vm.log` passed 274,492 checks and failed only my new
final object-count assertion: it expected zero instead of the VM initialization
baseline containing a cached module string. I corrected that test expectation
and retained the original log; I do not attribute it to map ownership.

A separate static allocation boundary remains under
`task_bc7264a337074246953284ef892785d2`: HM_NEW does not check a NULL heap
constructor result, and the void heap setter cannot report failed growth.
This child does not change that API or claim allocation-failure acceptance.

The same focused VM sanitizer target also passes with Clang, followed by all
seven paired methods in 3.824 seconds. I retain
`/tmp/nanolang-map-declared-clang.log`. Both compiler runs instrument the
changed dispatcher and the VM heap, cycle collector and value implementation;
other linked compiler/runtime objects retain their normal build flags.
