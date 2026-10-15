# Mixed service namespace grants

I reconcile the namespace regression with the public contract in
[Service multi-transport](../../SERVICE_MULTI_TRANSPORT.md#cli-publication-and-invocation).
Every declared File/TCP instance requires its grant when I execute selected
shadows or invoke a product, including unused instances. The old assertion
expected TCP lowering to remain disconnected after that integration landed.
My baseline retains the resulting assertion failure and missing-TCP diagnostic.

I require neither grant, File only, and TCP only to refuse compilation and
preserve prior output through both the native C seed and NanoVirt. With both
grants I require successful native and NanoISA publication. I then exercise
both products with each missing-grant combination and require success only
with both grants. Existing namespace identities, collisions, C query results,
independent Nano routes and sanitized C AOT execution remain in the suite.

My full three-method namespace suite passes (corrected.log), including both
independent Nano producers and sanitized C AOT probes. This validates the
namespace/capability boundary; it does not qualify a live TCP lifecycle or
refresh the installed Stage 2 compiler.
