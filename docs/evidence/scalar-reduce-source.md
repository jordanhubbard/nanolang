# My exact reduce source acceptance

I qualify source child `task_8618bac3cb6a42448c0066867deaed69`, under
`docs/NANOISA_SCALAR_REDUCE_CONTRACT.md`. My base is merged PR734 at `85fd6a52`.
I retain the broad callback taskd099, scalar policy5009 and public C-target070db;
C-seed canonical native FUNCREF transport remains explicitly unsupported.

I check reduce with a separate exact relation. I require a known accumulator A,
array element E and callback fn(A,E)->A. I recursively reject unknown components,
preserve nominal/generic/function/tuple/array identity, and do not inherit global
checker enum/int or unknown compatibility. My direct shadows cover those rules.
I recognize builtin reduce after bindings, preserve declared/local callables,
and select only the complete existing homogeneous int/float/string legacy ABI.

I snapshot source, initializer and callback once in that order. The unchanged
runtime helper captures length afterward. My helper names use nano_rt_, outside
the nl_ source-function namespace. Unsupported legacy signatures retain a first
emission diagnostic and produce no source. Normal and shadow driver paths check
that result before output publication or C compilation. My mode resets only at
the top-level transpilation; imported functions use the merged parser and no
nested generation path calls that reset. Shadows exercise first-error retention,
normal failure then valid recovery, and shadow-only failure then valid recovery
within one process. The public error accessor has its own shadow.

## My retained gates

| Gate | Result | Pin or boundary |
| --- | --- | --- |
| Fresh bootstrap, both stages, hello and installed compiler checks | passed | ff7d4048 |
| Six GCC source methods | passed, 81.640s | ff7d4048 |
| Six Clang source methods | passed, 56.765s | b9271595 |
| Three adjacent source methods | passed, 25.556s | b9271595 |

My production is unchanged between these pins. After the first six-method gate,
I replaced the old inline-reduce refusal test with the newly qualified positive
fixture. I froze that final harness before the Clang and adjacent gates. I do not
execute the retained failed callback source/artifacts from PR734 or claim their
historical cause from this gate. Independent review caught my initial use of
permissive types_equal before any fixture execution; I recorded and corrected it
before bootstrap. No new acceptance invocation failed.

My float context fixture passes the interpreter, C-seed legacy, both selfhost
legacy stages, all three canonical VM producer routes and both selfhost native
routes. It observes exact bits for inline/direct/helper/nested/typed-local reduce,
typed empty input including signed zero, and preserved input bits. The C-seed
canonical module verifies and runs in VM; nvm2c refuses FUNCREF while preserving
previous output. I count that as a refusal boundary, not native success.

My int/string neighbor helpers pass interpreter and all three legacy compilers.
Declared/local reduce name controls and computed callback sequencing run through
both selfhost legacy stages. The latter observes event sequence `12344` and an
initializer-grown aliased array of length 2. Wrong arity/source/parameter/result
controls preserve prior C and NanoISA outputs; a well-typed bool callback receives
an explicit unsupported homogeneous legacy ABI diagnostic and preserves prior
C/native outputs. I do not claim new computed FUNCREF canonical support.

Generated legacy code uses O2, fast contraction permitted, and UBSan with recovery
disabled. Generated canonical native C uses C11, O2, Wall/Wextra/Werror and
ASan/UBSan with recovery disabled. Clang uses the recorded local wrapper. Shared
runtime objects retain their normal bootstrap build; I do not call them fully
instrumented. The adjacent gate checks exact generated arithmetic-header text,
ordered immutable/mutable/chained globals, and the new direct-reduce VM observer.

`scalar-reduce-source.json` retains source/tool/imported-library hashes, exact
log paths and log hashes. All 20 original frozen identities match after both
final gates. I add the changed adjacent harness hash to that evidence separately.
My full arithmetic, callback transport, public C-target and release parents stay
open. I close the source child only after canonical merge.
