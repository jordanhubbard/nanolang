# Native map lexical cleanup

I keep cleanup for native match-arm locals inside their lexical C block. This
is MAC `task_1edadd5eb33a445d9bf6516744bc405e`, recovered from a fleet attempt
whose build lacked `ffi.h`, and continued locally in PR #493.

## Repair and preserved boundary

My full map-context fixture previously compiled a selected-arm local named
`nested`, then referenced it during function cleanup outside the arm. The
validated compiler preceding this change still reports `nested undeclared` on
that fixture; `/tmp/nanolang-map-cleanup-baseline.log` preserves the result.
I now push a scope for each match arm, emit its statements and cleanup within
that arm, and pop the scope. Integer, guarded, wildcard, union and or-pattern
paths share the scoped emitter. The native fixture no longer removes the
selected-arm case; it runs unchanged alongside NanoVirt for all four scalar
key/value pairs.

I preserved the original recovered patch at
`/tmp/nanolang-recovered-map-cleanup.patch`. Its proposed early-return changes
are absent from this repair: my AST_RETURN emitter is unchanged from main.
Evaluating a return result after releasing its inputs and changing a shared
cleanup flag while compiling one conditional path are not valid general
cleanup rules.

My native HashMap specializations currently allocate with malloc/calloc.
`gc_release` does not deallocate those unmanaged maps. This change establishes
lexical visibility and preserves their tested alias/return behavior; it does
not establish complete map lifetime management. Early-return cleanup and
computed/borrowed-result retain/transfer semantics remain open in
`task_195cac35e7704e56805932977512ae02`.

A separate direct-expression limitation remains in
`task_e018b78bc20a47d18619fce55a20e567`: `map_get` on a call returning a
string/string map loses native result metadata. That ordinary baseline fixture
and log are preserved at `/tmp/nanolang-map-return-expression.nano` and
`/tmp/nanolang-map-return-expression-baseline.log`. Explicit typed locals
isolate the lexical cleanup gate without claiming the direct-expression case
works.

## Validation

At source checkpoint `21b22f9f`, a fresh normal-budget bootstrap passes.
The focused map/native-effect gate passed 16 methods in 29.857 seconds.
After bootstrap, all 36 map/native-effect/generic-selected-ownership methods
passed in 107.065 seconds.
My five map methods cover all four scalar tag pairs, invalid constructor
contexts with preserved artifacts, integer/guarded/or/wildcard match scopes,
map aliases reused after arm exit, both conditional return paths, scalar return
expressions and borrowed string results.

The runtime ownership control uses managed opaque objects: ten executions of
each of two selected arms finalize exactly 20 objects, with each finalizer
observed after its arm exits and before the surrounding block ends. This checks
actual cleanup timing without treating unmanaged map storage as GC-owned.
Existing native effect cleanup/transfer cases remain in the same gate.

The reproducible focused command is `make test-map-constructor-contexts`.
Local logs use `/tmp/nanolang-map-cleanup-`, including `lifetimes.log`,
`bootstrap.log` and `integrated.log`.
