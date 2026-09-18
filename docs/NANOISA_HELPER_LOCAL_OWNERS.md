# My helper-local ownership prerequisite

MAC `task_74170cc4b1784e68970017da2a3ec64a`, under `ed702`, `718` and `28f2`.

I retain the bounded entry0/helper1 profile and borrowed-only helper parameters.
I admit mode-zero owned record locals in the helper only after native reference
resolution distinguishes actual caller roots from helper-local roots. Each
reference retains its origin frame and invocation generation; reborrowing
preserves both. Repeated calls create fresh helper generations. VM references
already carry these facts; native descriptors must agree before admission.

My helper parameter value locals remain non-authoritative. I keep parameter
LOAD/STORE and ownership consumption refused, along with deeper calls,
recursion, imports, callbacks, aggregate results and escaping references.
Exact nominal layouts, affine exit consumption and reference/region joins
remain mandatory. A helper resolves every local owner before normal return.
On assertion/allocation failure, existing terminal cleanup releases actual
owners in both activations and clears reference state.

I require ordinary valid VM/native modules with same-numbered caller/helper
roots holding different values, shared/exclusive mutation, nested local
construction/destruction, repeated calls, and assertion/allocation cleanup.
Existing authority and lifecycle gates remain required. I do not execute
malformed modules or replay historical corruption artifacts. Source producer
guards remain unchanged until this runtime prerequisite is reviewed and merged.
