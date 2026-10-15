# Conditional generic callback contexts

I continue #976 from c90d7498f. A declared `fn(int)->int` local initialized
from a conditional choice of generic functions failed both routes: my C
shadow compiler could not resolve the generic function, and my self-hosted
checker retained `fn(T)->T`. I retain the source and both baseline errors.

I apply the declared callback signature to every conditional branch, including
nested choices and the single-expression blocks used by the self-hosted parser.
My C checker registers each concrete generic callback before emission. My
self-hosted checker validates both branches, including inactive branches, and
my NanoISA emitter specializes each branch under the declared context while
preserving ordered condition evaluation and branch selection.

My six new methods cover locals, reassignment, returns, call arguments, record,
array and union storage, imported qualified/selective names, nested conditions,
observable condition order and incompatible inactive-branch output preservation.
I add meaningful checker and emitter shadows alongside their changed functions.

## Current checks

- `c-final.log` passes the first five methods plus eleven contextual regressions through C NanoISA, VM and sanitized C AOT. `imported-c.log` adds the sixth method.
- `native-tests.log` passes the first four methods plus eleven contextual regressions through C-native execution. `native-order.log` and `imported-native.log` add the other two methods.
- `component-gates.log` passes `make test-typechecker test-transpiler`.
- `seed-build.log` builds the updated self-hosted compiler, including dependency shadows. `seed-tests.log` passes all seventeen methods through its NanoISA output, verified VM execution and sanitized C AOT.

I keep #976, installed-stage/platform qualification and the full 5.1 scope open.
GitHub API access remains unavailable; the issue update is retained if posting fails.

My seven-step compiler-product audit passes in 328.602 seconds after the fresh seed build. The seed emits the updated compiler module with its shadows; I verify it, translate it without VM wrappers and link its native compiler. That product passes 35 callback/aggregate methods and 22 nominal/resource/layout methods. The four recorded changed-source inputs remain unchanged. `product-manifest.json` retains commands, pins, timings and logs. This is local product qualification, not a final candidate Linux/Darwin bootstrap.
